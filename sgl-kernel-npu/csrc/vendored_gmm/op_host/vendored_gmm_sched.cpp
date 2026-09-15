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
 * \file vendored_gmm_sched.cpp
 * \brief Host implementation of vgmm1_sched (offsets + block-schedule-table
 *        front kernel) and vgmm1_query_tile (pure host query). The tiling
 *        cache copies the hash + static device buffer pattern of
 *        gmm_scan_probe/tp_fusion_probe (graph-capture pointer stability);
 *        launch carries a return-code check (lesson from tp_fusion_probe
 *        ut_result_6/probe_2 silent failures).
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

// vendored_gmm_tiling_data.h embeds TCubeTiling by value, so its full
// definition must be included first
// (same convention as lora/sgemmc_tiling.h; sched itself has no matmul, this
// is only for the shared header)
#include <register/tilingdata_base.h>
#include "tiling/tiling_api.h"

#include "vendored_gmm_tiling_data.h"

#include "aclrtlaunch_vgmm1_sched_pre.h"

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

uint32_t g_vgmmSchedCaptureNum = 0;
std::unordered_map<uint64_t, uint32_t> g_vgmmSchedCaptureMap;

uint64_t VgmmSchedTilingHash(int64_t groupNum, int64_t baseM, int64_t baseN, int64_t n, int64_t maxBlocks)
{
    uint64_t h = 0;
    auto combine = [&h](int64_t v) {
        h ^= static_cast<uint64_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    combine(groupNum);
    combine(baseM);
    combine(baseN);
    combine(n);
    combine(maxBlocks);
    return h;
}
}  // namespace

HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor> vgmm1_sched_impl(
    const at::Tensor &group_list, int64_t total_m, int64_t n, int64_t k, int64_t max_blocks, int64_t base_m,
    int64_t base_n)
{
    TORCH_CHECK(group_list.defined(), "vgmm1_sched: group_list must be defined");
    TORCH_CHECK(group_list.scalar_type() == at::kLong, "vgmm1_sched: group_list must be int64");
    TORCH_CHECK(group_list.is_contiguous(), "vgmm1_sched: group_list must be contiguous");
    int64_t groupNum = group_list.numel();
    TORCH_CHECK(groupNum >= 1 && groupNum <= 4096, "vgmm1_sched: E must be in [1, 4096]");
    TORCH_CHECK(n > 0 && max_blocks > 0, "vgmm1_sched: n/max_blocks must be positive");
    TORCH_CHECK(total_m > 0, "vgmm1_sched: total_m must be positive (= total rows of x, only used to clamp baseM)");

    // When base_m/base_n <= 0, compute them on the spot with the stock
    // formula (same function and same totalM input as vgmm1_main,
    // guaranteeing the two kernels agree)
    VgmmHwInfo hw = VgmmQueryHw();
    VgmmBaseTile tile = VgmmComputeBaseTile(total_m, hw);
    uint32_t baseM = base_m > 0 ? static_cast<uint32_t>(base_m) : tile.baseM;
    uint32_t baseN = base_n > 0 ? static_cast<uint32_t>(base_n) : tile.baseN;

    auto opts = group_list.options();
    at::Tensor block_table = at::empty({max_blocks * static_cast<int64_t>(VGMM_BLOCK_ENTRY_INT32)},
                                       opts.dtype(at::kInt));
    at::Tensor row_offsets = at::empty({groupNum}, opts.dtype(at::kInt));
    at::Tensor total_blocks = at::empty({8}, opts.dtype(at::kInt));

    VgmmSchedTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    tilingData.groupNum = static_cast<uint32_t>(groupNum);
    tilingData.baseM = baseM;
    tilingData.baseN = baseN;
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.maxBlocks = static_cast<uint32_t>(max_blocks);

    int32_t tilingSize = (static_cast<int32_t>(sizeof(VgmmSchedTilingData)) +
                          static_cast<int32_t>(TILING_PADDING_BYTE) - 1) /
                         static_cast<int32_t>(TILING_PADDING_BYTE) * static_cast<int32_t>(TILING_PADDING_BYTE);

    static auto globalTilingBuffer =
        at::empty({static_cast<int64_t>(tilingSize) * MAX_CAPTURE_NUM},
                  at::TensorOptions().dtype(at::kByte).device(opts.device()));

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(VgmmSchedTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    uint64_t hashValue = VgmmSchedTilingHash(groupNum, baseM, baseN, n, max_blocks);
    at::Tensor tilingTensor;
    auto iter = g_vgmmSchedCaptureMap.find(hashValue);
    if (iter != g_vgmmSchedCaptureMap.end()) {
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * iter->second),
                                     tilingSize, at::kByte);
    } else if (g_vgmmSchedCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        g_vgmmSchedCaptureMap[hashValue] = g_vgmmSchedCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer
            .slice(0, g_vgmmSchedCaptureNum * tilingSize, g_vgmmSchedCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        g_vgmmSchedCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() +
                                         (tilingSize * g_vgmmSchedCaptureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    auto workspaceTensor = at::empty({64}, at::TensorOptions().dtype(at::kByte).device(opts.device()));

    // AIV single-core table build (table building is O(E + totalBlocks)
    // scalar work, one core is enough).
    // blockDim must land in a local variable first: the lambda inside the
    // macro captures by name; literals/member access cannot be captured
    constexpr uint32_t blockDim = 1;
    VGMM_EXEC_KERNEL_CMD_CHECKED(vgmm1_sched_pre, blockDim, group_list, block_table, row_offsets, total_blocks,
                                 workspaceTensor, tilingTensor);

    return std::make_tuple(block_table, row_offsets, total_blocks);
}

HOST_API at::Tensor vgmm1_query_tile_impl(int64_t m, int64_t k, int64_t n)
{
    (void)k;
    (void)n;
    VgmmHwInfo hw = VgmmQueryHw();
    VgmmBaseTile tile = VgmmComputeBaseTile(m, hw);
    auto out = at::empty({6}, at::TensorOptions().dtype(at::kInt));
    int32_t *p = out.data_ptr<int32_t>();
    p[0] = static_cast<int32_t>(tile.baseM);
    p[1] = static_cast<int32_t>(tile.baseN);
    p[2] = static_cast<int32_t>(tile.baseK);
    p[3] = static_cast<int32_t>(tile.stepKa);
    p[4] = static_cast<int32_t>(tile.stepKb);
    p[5] = static_cast<int32_t>(hw.aicNum);
    return out;
}

}  // namespace npu_kernel
}  // namespace sglang
