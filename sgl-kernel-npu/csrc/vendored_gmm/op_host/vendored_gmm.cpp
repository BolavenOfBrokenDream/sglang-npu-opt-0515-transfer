/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/*!
 * \file vendored_gmm.cpp
 * \brief vgmm1_main (vendored GMM main op) host implementation.
 *
 * host tiling = a replica of the stock GMMSetMMTiling
 * (grouped_matmul_tiling.cpp:1375-1428) non-quantized bf16/ND path:
 * MultiCoreMatmulTiling + SetAType/BType/CType + SetOrgShape/SetShape +
 * SetFixSplit + SetBufferSpace + GetTiling + the stock override block
 * (shareMode=0 / dbL0C=1 / baseM/baseN/baseK / stepKa/stepKb /
 * depthA1/depthB1 / stepM=stepN=1). Two deliberate deviations (documented in
 * the README "risks and known deviations" section):
 *   1. SetDim(1): stock does not call SetDim explicitly under the GE context;
 *      this op's kernel side uses MatmulImpl as "one single-core GEMM per
 *      block" (SetSubBlockIdx(0)+SetSingleShape runtime override), and
 *      SetDim(1) keeps the single-core problem description consistent with
 *      that (same approach as fused_norm_qkv_proj_scatter in
 *      tp_ascendc_fusion, validated on-device).
 *   2. baseM/baseN/baseK are computed on the spot by this package's
 *      VgmmComputeBaseTile, replicating the stock CalMMTiling non-quantized
 *      formula (the production tuningConfigOptional path does not apply to
 *      unit-test scenarios).
 */

#include <cstring>
#include <unordered_map>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "defines.h"
#include "torch_helper.h"
#include "common.h"
#include "vendored_gmm.h"
#include "vendored_gmm_host_common.h"

#include <register/tilingdata_base.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

// The full TCubeTiling definition comes from the tiling headers above (same
// convention as lora/sgemmc_tiling.h); vendored_gmm_tiling_data.h embeds it
// by value and must be included after them
#include "vendored_gmm_tiling_data.h"

#include "aclrtlaunch_vgmm1_main_aic.h"

// The launch return code propagates via the closure return value: the sync
// branch is checked by InnerRunOpApi's OPS_CHECK_ERROR
// (OpParamMaker.cpp:432-433); the async (TASK_QUEUE) branch surfaces ret via
// ExecFuncOpApi into the queue error channel (OpParamMaker.cpp:607-611 +
// NPUQueue.cpp ReadQueue).
// Iron rule: deferred closures must capture by value only — the old
// &launch_ret by-reference capture became a 4-byte async wild write into the
// producer thread's already-reclaimed stack frame under TASK_QUEUE=1 (root
// cause of the e2e capture-time stack smashing; see
// docs/bug-fix/npu-vgmm1-e2e-capture-crash-swiglu-host-investigation.md §8).
#define VGMM_EXEC_KERNEL_CMD_CHECKED(kernel_name, blockdim, ...)                                     \
    do {                                                                                           \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                            \
        auto converted_params = sglang::npu_kernel::TorchNpuHelper::ConvertTypes(__VA_ARGS__);     \
        auto acl_call = [acl_stream, blockdim, converted_params]() -> int {                        \
            int launch_ret = 0;                                                                    \
            std::apply(                                                                            \
                [&](auto &&...params) {                                                            \
                    launch_ret =                                                                   \
                        static_cast<int>(ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream,    \
                                                                          params...));             \
                },                                                                                 \
                converted_params);                                                                 \
            return launch_ret;                                                                     \
        };                                                                                         \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);                               \
    } while (false)

namespace sglang {
namespace npu_kernel {

namespace {
constexpr uint32_t TILING_PADDING_BYTE = 32U;
constexpr uint32_t MAX_CAPTURE_NUM = 1024U;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;  // grouped_matmul_host_util.h:104

uint32_t g_vgmmMainCaptureNum = 0;
std::unordered_map<uint64_t, uint32_t> g_vgmmMainCaptureMap;

uint64_t VgmmMainTilingHash(int64_t m, int64_t k, int64_t n)
{
    uint64_t h = 0;
    auto combine = [&h](int64_t v) {
        h ^= static_cast<uint64_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    combine(m);
    combine(k);
    combine(n);
    return h;
}
}  // namespace

HOST_API at::Tensor vgmm1_main_impl(const at::Tensor &x, const at::Tensor &w, const at::Tensor &block_table,
                                    const at::Tensor &total_blocks)
{
    TORCH_CHECK(x.defined() && w.defined() && block_table.defined() && total_blocks.defined(),
                "vgmm1_main: all tensors must be defined");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                "vgmm1_main: x/w must be bf16");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 3, "vgmm1_main: x must be [M,K], w must be [E,N,K]");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && total_blocks.scalar_type() == at::kInt,
                "vgmm1_main: block_table/total_blocks must be int32");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && block_table.is_contiguous() &&
                    total_blocks.is_contiguous(),
                "vgmm1_main: inputs must be contiguous (w must be the production [E,N,K] ND-contiguous layout, not a transpose view)");
    int64_t totalM = x.size(0);
    int64_t k = x.size(1);
    int64_t groupNum = w.size(0);
    int64_t n = w.size(1);
    TORCH_CHECK(w.size(2) == k, "vgmm1_main: w [E,N,K] K mismatch with x");
    TORCH_CHECK(totalM > 0, "vgmm1_main: totalM must be positive (filter empty-counts shapes first)");

    VgmmHwInfo hw = VgmmQueryHw();
    VgmmBaseTile tile = VgmmComputeBaseTile(totalM, hw);

    VgmmTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    tilingData.baseParams.coreNum = hw.aicNum;
    tilingData.baseParams.baseM = tile.baseM;
    tilingData.baseParams.baseN = tile.baseN;
    tilingData.baseParams.k = static_cast<uint32_t>(k);
    tilingData.baseParams.n = static_cast<uint32_t>(n);
    tilingData.baseParams.isOutputDisableL2Cache = 0;

    // ---- stock GMMSetMMTiling replica (non-quantized bf16, ND, transB handled by the kernel template) ----
    auto ascendc_platform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling mm(ascendc_platform);
    // Deviation 1: SetDim. Stock does not call SetDim (defaults to full
    // platform cores); tp_ascendc_fusion proved SetDim(aic_num)'s multi-core
    // auto-dispatch breaks the manual blocking, so default to SetDim(1) for a
    // single-core problem description.
    // VGMM_SETDIM=full -> SetDim(aicNum); =none -> do not call it at all
    // (stock as-is); debug knob
    const char *setdimEnv = std::getenv("VGMM_SETDIM");
    if (setdimEnv == nullptr) {
        mm.SetDim(1);
    } else if (std::strcmp(setdimEnv, "full") == 0) {
        mm.SetDim(static_cast<int32_t>(hw.aicNum));
    } else if (std::strcmp(setdimEnv, "none") != 0) {
        TORCH_CHECK(false, "vgmm1_main: VGMM_SETDIM must be unset / full / none");
    }
    // Deviation 2 fix: stock's no-bias branch only calls SetBias(false)
    // (grouped_matmul_tiling.cpp:1277-1278), not SetBiasType; the original
    // EnableBias(false)+SetBiasType combo was not stock semantics
    mm.SetBias(false);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16, false);  // stock verbatim (host passes false)
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                matmul_tiling::DataType::DT_FLOAT16);          // stock verbatim (:1383)
    mm.SetOrgShape(static_cast<int32_t>(totalM), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mm.SetShape(static_cast<int32_t>(totalM), static_cast<int32_t>(tile.baseN),
                static_cast<int32_t>(k));  // stock: SetShape(mInMM, baseN_, maxK_)
    mm.SetFixSplit(static_cast<int32_t>(tile.baseM), static_cast<int32_t>(tile.baseN),
                   static_cast<int32_t>(tile.baseK));
    mm.SetBufferSpace(static_cast<int64_t>(hw.l1Size), static_cast<int64_t>(hw.l0cSize),
                      static_cast<int64_t>(hw.ubSize));
    TORCH_CHECK(mm.GetTiling(tilingData.cubeTiling) != -1, "vgmm1_main: matmul GetTiling failed");

    // Stock override block (grouped_matmul_tiling.cpp:1410-1424).
    // Note: stock's mmTilingData is a TILING_DATA_FIELD_DEF_STRUCT field (with
    // set_ accessors); this package's cubeTiling is a classic plain TCubeTiling
    // struct, so write the fields directly (set_baseM <-> baseM one-to-one)
    tilingData.cubeTiling.shareMode = 0;
    tilingData.cubeTiling.dbL0C = 1;
    tilingData.cubeTiling.baseM = static_cast<int32_t>(tile.baseM);
    tilingData.cubeTiling.baseN = static_cast<int32_t>(tile.baseN);
    tilingData.cubeTiling.baseK = static_cast<int32_t>(tile.baseK);
    tilingData.cubeTiling.stepKa = static_cast<int32_t>(tile.stepKa);
    tilingData.cubeTiling.stepKb = static_cast<int32_t>(tile.stepKb);
    tilingData.cubeTiling.depthA1 =
        static_cast<int32_t>(tile.stepKa * DOUBLE_BUFFER_STEPKA_STEPKB);  // stepM=1
    tilingData.cubeTiling.depthB1 =
        static_cast<int32_t>(tile.stepKb * DOUBLE_BUFFER_STEPKA_STEPKB);  // stepN=1
    tilingData.cubeTiling.stepM = 1;
    tilingData.cubeTiling.stepN = 1;

    // VGMM_STAGE: staged kernel power-down debugging, delivered via
    // baseParams.reserved[0]
    // (0=full default; 1=CopyTiling only; 2=+table read; 3=+mm.Init; 4=+address
    //   computation without running IterateAll;
    //   5=kernel runs full as usual, but host over-allocates y by 8*baseM
    //   guard rows — distinguishes C copyout overrunning the end of y
    //   (absorbed by the guard, no crash) vs a broken fixp descriptor itself
    //   (guard useless, still crashes))
    const char *stageEnv = std::getenv("VGMM_STAGE");
    tilingData.baseParams.reserved[0] =
        stageEnv != nullptr ? static_cast<uint32_t>(std::atoi(stageEnv)) : 0;

    if (std::getenv("VGMM_DEBUG") != nullptr) {
        fprintf(stderr,
                "[vgmm1_main] M=%ld K=%ld N=%ld E=%ld aic=%u baseM=%u baseN=%u baseK=%u "
                "stepKa=%u stepKb=%u l1=%llu l0c=%llu ub=%llu stage=%u\n",
                totalM, k, n, groupNum, hw.aicNum, tile.baseM, tile.baseN, tile.baseK, tile.stepKa,
                tile.stepKb, (unsigned long long)hw.l1Size, (unsigned long long)hw.l0cSize,
                (unsigned long long)hw.ubSize, tilingData.baseParams.reserved[0]);
        const auto &ct = tilingData.cubeTiling;
        fprintf(stderr,
                "[vgmm1_main] cubeTiling: usedCoreNum=%d M=%d N=%d Ka=%d Kb=%d singleCoreM=%d "
                "singleCoreN=%d singleCoreK=%d baseM=%d baseN=%d baseK=%d depthA1=%d depthB1=%d "
                "stepM=%d stepN=%d isBias=%d transLength=%d iterateOrder=%d shareMode=%d "
                "shareL1=%d shareL0C=%d shareUb=%d stepKa=%d stepKb=%d dbL0A=%d dbL0B=%d dbL0C=%d\n",
                ct.usedCoreNum, ct.M, ct.N, ct.Ka, ct.Kb, ct.singleCoreM, ct.singleCoreN,
                ct.singleCoreK, ct.baseM, ct.baseN, ct.baseK, ct.depthA1, ct.depthB1, ct.stepM,
                ct.stepN, ct.isBias, ct.transLength, ct.iterateOrder, ct.shareMode, ct.shareL1Size,
                ct.shareL0CSize, ct.shareUbSize, ct.stepKa, ct.stepKb, ct.dbL0A, ct.dbL0B,
                ct.dbL0C);
    }

    // VGMM_STAGE=5: append 8*baseM guard rows at the tail of y (kernel
    // unaware, runs full as usual). Slice back to [totalM, n] before return
    // to keep the unit-test shape contract; if the guard absorbs the overrun
    // and no crash occurs, C copyout overrunning the end of y is confirmed
    // (bounded overrun); if it still crashes, the fixp descriptor/compute
    // stage itself is broken (check the cubeTiling dump).
    // VGMM_STAGE=6 (production canary stage, for the 2026-08-28 e2e segfault
    // localization): same guard over-allocation as 5, plus fill the guard
    // area with canary value 3.140625 before launch and **return y in full**
    // (no slicing) — the python wrapper (SGLANG_NPU_VGMM1_DEBUG=1,
    // non-capture) sync-checks the position/value distribution of non-canary
    // bytes in the guard area, then slices back to [totalM, n] for
    // downstream. Probes must not use 6 (use 5).
    const uint32_t dbgStageVal = tilingData.baseParams.reserved[0];
    const bool stage5Guard = dbgStageVal == 5;
    const bool stage6Canary = dbgStageVal == 6;
    at::Tensor y = (stage5Guard || stage6Canary)
                       ? at::empty({totalM + 8 * static_cast<int64_t>(tile.baseM), n}, x.options())
                       : at::empty({totalM, n}, x.options());
    if (stage6Canary) {
        y.slice(0, totalM, totalM + 8 * static_cast<int64_t>(tile.baseM)).fill_(3.140625);
    }

    int32_t tilingSize = (static_cast<int32_t>(sizeof(VgmmTilingData)) +
                          static_cast<int32_t>(TILING_PADDING_BYTE) - 1) /
                         static_cast<int32_t>(TILING_PADDING_BYTE) * static_cast<int32_t>(TILING_PADDING_BYTE);

    static auto globalTilingBuffer =
        at::empty({static_cast<int64_t>(tilingSize) * MAX_CAPTURE_NUM},
                  at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(VgmmTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    uint64_t hashValue = VgmmMainTilingHash(totalM, k, n);
    at::Tensor tilingTensor;
    auto iter = g_vgmmMainCaptureMap.find(hashValue);
    if (iter != g_vgmmMainCaptureMap.end()) {
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * iter->second),
                                     tilingSize, at::kByte);
    } else if (g_vgmmMainCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        g_vgmmMainCaptureMap[hashValue] = g_vgmmMainCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer
            .slice(0, g_vgmmMainCaptureNum * tilingSize, g_vgmmMainCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        g_vgmmMainCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() +
                                         (tilingSize * g_vgmmMainCaptureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    uint32_t libWsSize = ascendc_platform.GetLibApiWorkSpaceSize();
    int64_t wsBytes = static_cast<int64_t>(libWsSize) > 64 ? static_cast<int64_t>(libWsSize) : 64;
    auto workspaceTensor = at::empty({wsBytes}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    // blockDim must land in a local variable first: the lambda inside the
    // macro captures by name, and member access hw.aicNum cannot be captured
    const uint32_t blockDim = hw.aicNum;
    VGMM_EXEC_KERNEL_CMD_CHECKED(vgmm1_main_aic, blockDim, x, w, block_table, total_blocks, y, workspaceTensor,
                                 tilingTensor);

    return stage5Guard ? y.slice(0, 0, totalM) : y;
}

}  // namespace npu_kernel
}  // namespace sglang
