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
 * \file vendored_gmm_sched_partial.cpp
 * \brief Host implementation of vgmm1_sched_partial (C1: sched absorbs the
 *        cumsum).
 *        Input = v22 rank_hist's partials int32[partNum, partStride] (partial
 *        histogram); output = block_table + row_offsets + total_blocks
 *        (bit-identical to vgmm1_sched) + counts int64[E] (bit-identical to
 *        v22 partials_cumsum, integer column reduction).
 *        The Triton partials_cumsum node disappears from the chain entirely.
 *        Structure copies vendored_gmm_sched.cpp: tiling cache hash + static
 *        device buffer (graph-capture pointer stability) + launch return-code
 *        check (value-capture closure iron rule, see that file's header).
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

#include "aclrtlaunch_vgmm1_sched_partial_pre.h"

// The launch return code propagates via the closure return value (iron rule:
// deferred closures must capture by value only — a &launch_ret by-reference
// capture is an async wild write into a reclaimed stack frame under
// TASK_QUEUE=1; see
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

uint32_t g_vgmmSchedPartialCaptureNum = 0;
std::unordered_map<uint64_t, uint32_t> g_vgmmSchedPartialCaptureMap;

uint64_t VgmmSchedPartialTilingHash(int64_t groupNum, int64_t partNum, int64_t partStride, int64_t baseM,
                                    int64_t baseN, int64_t n, int64_t maxBlocks)
{
    uint64_t h = 0;
    auto combine = [&h](int64_t v) {
        h ^= static_cast<uint64_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    combine(groupNum);
    combine(partNum);
    combine(partStride);
    combine(baseM);
    combine(baseN);
    combine(n);
    combine(maxBlocks);
    return h;
}
}  // namespace

HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> vgmm1_sched_partial_impl(
    const at::Tensor &partials, int64_t num_experts, int64_t total_m, int64_t n, int64_t k, int64_t max_blocks,
    int64_t base_m, int64_t base_n)
{
    TORCH_CHECK(partials.defined(), "vgmm1_sched_partial: partials must be defined");
    TORCH_CHECK(partials.scalar_type() == at::kInt, "vgmm1_sched_partial: partials must be int32");
    TORCH_CHECK(partials.dim() == 2 && partials.is_contiguous(),
                "vgmm1_sched_partial: partials must be [partNum, partStride] contiguous");
    const int64_t partNum = partials.size(0);
    const int64_t partStride = partials.size(1);
    TORCH_CHECK(partNum >= 1 && partStride >= 1, "vgmm1_sched_partial: partials shape must be non-empty");
    TORCH_CHECK(num_experts >= 1 && num_experts <= partStride,
                "vgmm1_sched_partial: num_experts must be in [1, partStride]");
    TORCH_CHECK(n > 0 && max_blocks > 0, "vgmm1_sched_partial: n/max_blocks must be positive");
    TORCH_CHECK(total_m > 0, "vgmm1_sched_partial: total_m must be positive (= total rows of x, only used to clamp baseM)");

    // When base_m/base_n <= 0, compute them on the spot with the stock
    // formula (same function and same totalM input as vgmm1_main,
    // guaranteeing the two kernels agree)
    VgmmHwInfo hw = VgmmQueryHw();
    VgmmBaseTile tile = VgmmComputeBaseTile(total_m, hw);
    uint32_t baseM = base_m > 0 ? static_cast<uint32_t>(base_m) : tile.baseM;
    uint32_t baseN = base_n > 0 ? static_cast<uint32_t>(base_n) : tile.baseN;

    auto opts = partials.options();
    at::Tensor block_table = at::empty({max_blocks * static_cast<int64_t>(VGMM_BLOCK_ENTRY_INT32)},
                                       opts.dtype(at::kInt));
    at::Tensor row_offsets = at::empty({num_experts}, opts.dtype(at::kInt));
    at::Tensor total_blocks = at::empty({8}, opts.dtype(at::kInt));
    at::Tensor counts = at::empty({num_experts}, opts.dtype(at::kLong));

    VgmmSchedPartialTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    tilingData.groupNum = static_cast<uint32_t>(num_experts);
    tilingData.partNum = static_cast<uint32_t>(partNum);
    tilingData.partStride = static_cast<uint32_t>(partStride);
    tilingData.baseM = baseM;
    tilingData.baseN = baseN;
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.maxBlocks = static_cast<uint32_t>(max_blocks);

    int32_t tilingSize = (static_cast<int32_t>(sizeof(VgmmSchedPartialTilingData)) +
                          static_cast<int32_t>(TILING_PADDING_BYTE) - 1) /
                         static_cast<int32_t>(TILING_PADDING_BYTE) * static_cast<int32_t>(TILING_PADDING_BYTE);

    static auto globalTilingBuffer =
        at::empty({static_cast<int64_t>(tilingSize) * MAX_CAPTURE_NUM},
                  at::TensorOptions().dtype(at::kByte).device(opts.device()));

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(VgmmSchedPartialTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    uint64_t hashValue =
        VgmmSchedPartialTilingHash(num_experts, partNum, partStride, baseM, baseN, n, max_blocks);
    at::Tensor tilingTensor;
    auto iter = g_vgmmSchedPartialCaptureMap.find(hashValue);
    if (iter != g_vgmmSchedPartialCaptureMap.end()) {
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * iter->second),
                                     tilingSize, at::kByte);
    } else if (g_vgmmSchedPartialCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        g_vgmmSchedPartialCaptureMap[hashValue] = g_vgmmSchedPartialCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer
            .slice(0, g_vgmmSchedPartialCaptureNum * tilingSize,
                   g_vgmmSchedPartialCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        g_vgmmSchedPartialCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() +
                                         (tilingSize * g_vgmmSchedPartialCaptureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    auto workspaceTensor = at::empty({64}, at::TensorOptions().dtype(at::kByte).device(opts.device()));

    // AIV single core: column reduction (log2(partNum) rounds of whole-row
    // Add) + O(E + totalBlocks) scalar table build.
    // blockDim must land in a local variable first: the lambda inside the
    // macro captures by name; literals/member access cannot be captured
    constexpr uint32_t blockDim = 1;
    VGMM_EXEC_KERNEL_CMD_CHECKED(vgmm1_sched_partial_pre, blockDim, partials, block_table, row_offsets,
                                 total_blocks, counts, workspaceTensor, tilingTensor);

    return std::make_tuple(block_table, row_offsets, total_blocks, counts);
}

}  // namespace npu_kernel
}  // namespace sglang
