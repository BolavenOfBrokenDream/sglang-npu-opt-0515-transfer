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
 * \file fused_tail.cpp
 * \brief fused_tail (MoE layer-tail fin+add+AR+norm fusion, tp_ascendc_fusion
 *        v4 ported to the 0515 baseline) host-side implementation. Three ops:
 *          fused_fin_ar_norm  fin(+skip1)+spin AIV AR+add+gemma rmsnorm single kernel
 *          fused_fin_add      fin(+skip1) local front stage (stock HCCL AR variant)
 *          fused_tail_zero    in-kernel MTE3 zeroing of a local GM region
 *        Mechanism and layouts: see op_kernel/fused_tail_kernel_lib.h header.
 *
 * Tiling cache follows the hash-key + static device buffer pattern (graph
 * capture pointer stability); the launch macro is value-capture only (deferred
 * closures must never capture by reference).
 */

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"

#include "defines.h"
#include "torch_helper.h"
#include "fused_tail.h"
#include "fused_tail_tiling_data.h"

#include "aclrtlaunch_fused_fin_ar_norm.h"
#include "aclrtlaunch_fused_fin_add.h"
#include "aclrtlaunch_fused_tail_zero.h"

// Launch macro with return-code check: launch_ret is produced inside the
// closure and propagated via its return value; the synchronous branch is
// checked by InnerRunOpApi's OPS_CHECK_ERROR, the asynchronous (TASK_QUEUE)
// branch surfaces ret through the queue error channel. Hard rule: deferred
// closures capture by value only.
#define FT_EXEC_KERNEL_CMD_CHECKED(kernel_name, blockdim, ...)                                     \
    do {                                                                                         \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                          \
        auto converted_params = sglang::npu_kernel::TorchNpuHelper::ConvertTypes(__VA_ARGS__);   \
        auto acl_call = [acl_stream, blockdim, converted_params]() -> int {                      \
            int launch_ret = 0;                                                                  \
            std::apply(                                                                          \
                [&](auto &&...params) {                                                          \
                    launch_ret =                                                                 \
                        static_cast<int>(ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream,  \
                                                                          params...));           \
                },                                                                               \
                converted_params);                                                               \
            return launch_ret;                                                                   \
        };                                                                                       \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);                             \
    } while (false)

namespace sglang {
namespace npu_kernel {

namespace {
constexpr uint32_t TILING_PADDING_BYTE = 32U;
constexpr uint32_t MAX_CAPTURE_NUM = 1024U;
constexpr uint64_t CYCLES_PER_US = 50UL;

constexpr int64_t MAX_M = 4096;   // guard rail
constexpr int64_t MAX_K = 16;     // topk guard rail
constexpr int64_t MAX_NCORES = 48;
constexpr int64_t MAX_WORLD = 8;  // addr_tab capacity
constexpr int64_t TILE_BYTES = 8192;

uint32_t g_ftCaptureNum = 0;
std::unordered_map<uint64_t, uint32_t> g_ftCaptureMap;
// Pinned-staging keep-alive pool for map entries (<=MAX_CAPTURE_NUM); see
// PrepareTiling comment.
std::vector<at::Tensor> g_ftPinnedKeepAlive;

// Tiling cache key = FNV-1a hash over the whole TilingData struct (whole-struct
// hashing guards against forgetting a field).
uint64_t HashTilingData(const FusedTailTilingData &td)
{
    const uint32_t *p = reinterpret_cast<const uint32_t *>(&td);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < sizeof(FusedTailTilingData) / sizeof(uint32_t); ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

// Common guards: shapes/dtypes/core counts/rank. op is used in error strings.
void CheckCommon(const char *op, int64_t m, int64_t h, int64_t k, int64_t ncores, int64_t has_skip1,
                 int64_t use_eri, int64_t eri_va, const at::Tensor &x, const at::Tensor &scales,
                 const at::Tensor &skip1)
{
    TORCH_CHECK(h == 2048 || h == 4096, op, ": H must be 2048 or 4096");
    TORCH_CHECK(m >= 1 && m <= MAX_M, op, ": M must be in [1, 4096]");
    TORCH_CHECK(k >= 1 && k <= MAX_K, op, ": K(topk) must be in [1, 16]");
    TORCH_CHECK(ncores >= 1 && ncores <= MAX_NCORES, op, ": ncores must be in [1, 48]");
    TORCH_CHECK(has_skip1 == 0 || has_skip1 == 1, op, ": has_skip1 must be 0/1");
    TORCH_CHECK(use_eri == 0 || use_eri == 1, op, ": use_eri must be 0/1");
    TORCH_CHECK(use_eri == 0 || eri_va != 0, op, ": use_eri=1 requires eri_va != 0");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() >= m * k * h,
                op, ": x=xp/xexp must be contiguous bf16 [M*K,H]");
    TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous() && scales.numel() >= m * k,
                op, ": scales must be contiguous fp32 [M,K]");
    if (has_skip1 != 0) {
        TORCH_CHECK(skip1.scalar_type() == at::kBFloat16 && skip1.is_contiguous() && skip1.numel() >= m * h,
                    op, ": has_skip1=1 requires skip1 contiguous bf16 [M,H]");
    }
}

// Fill common tiling fields + cache reuse (graph-capture pointer stable).
// Returns the reused/new device tiling tensor. Flag layout derivatives are
// derived host-side with the same formulas as the python wrapper (single
// source of truth; counter/dfx offset check = bilateral consistency guard).
at::Tensor PrepareTiling(FusedTailTilingData &tilingData, const at::Device &device)
{
    int32_t tilingSize = (static_cast<int32_t>(sizeof(FusedTailTilingData)) +
                          static_cast<int32_t>(TILING_PADDING_BYTE) - 1) /
                         static_cast<int32_t>(TILING_PADDING_BYTE) * static_cast<int32_t>(TILING_PADDING_BYTE);

    static auto globalTilingBuffer = at::empty({static_cast<int64_t>(tilingSize) * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(device));

    uint64_t hashValue = HashTilingData(tilingData);
    // Graph-capture safety: this op's tiling hash covers eri_va/norm_w_va, so
    // every (bs tier x layer) entry is first created inside capture and goes
    // through the copy chain below. CopyTensorHostToDevice's non-blocking H2D
    // (pin_memory -> .to(non_blocking=true)) gets RECORDED INTO THE GRAPH by
    // ACL — torch_npu has no capture-state check on the H2D path — and at
    // replay the memcpy node re-reads the pinned source address baked at
    // capture; once the temporary pinned staging is returned to the host
    // allocator it can be reused, so replay would read garbage tiling and the
    // kernel would run MTE out of bounds. Fix: pinned staging of map entries
    // (<=1024, ~100KB total) is kept alive statically so the recorded H2D
    // always re-reads correct, immutable content. The >1024 fallback is
    // eager-only (per-(tier x layer) entries are bounded: ~2 warmup + 1
    // capture each), so it is not kept alive to avoid a long-run leak.
    auto copyTilingToDevice = [&](bool keepPinnedAlive) {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(FusedTailTilingData));
        auto pinned = cpuTiling.pin_memory();
        if (keepPinnedAlive) {
            g_ftPinnedKeepAlive.emplace_back(pinned);
        }
        return pinned.to(device, pinned.scalar_type(), /*non_blocking=*/true, /*copy=*/true);
    };

    at::Tensor tilingTensor;
    auto iter = g_ftCaptureMap.find(hashValue);
    if (iter != g_ftCaptureMap.end()) {
        tilingTensor =
            at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * iter->second), tilingSize, at::kByte);
    } else if (g_ftCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice(/*keepPinnedAlive=*/false);
    } else {
        g_ftCaptureMap[hashValue] = g_ftCaptureNum;
        auto deviceTiling = copyTilingToDevice(/*keepPinnedAlive=*/true);
        globalTilingBuffer
            .slice(0, g_ftCaptureNum * tilingSize, g_ftCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        g_ftCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() +
                                         (tilingSize * g_ftCaptureMap[hashValue]),
                                     tilingSize, at::kByte);
    }
    return tilingTensor;
}

void FillCommonTiling(FusedTailTilingData &td, int64_t m, int64_t h, int64_t k, int64_t ncores,
                      int64_t has_skip1, int64_t use_eri, int64_t eri_va)
{
    std::memset(&td, 0, sizeof(td));
    td.eri_va = static_cast<uint64_t>(eri_va);
    td.m = static_cast<uint32_t>(m);
    td.h = static_cast<uint32_t>(h);
    td.k = static_cast<uint32_t>(k);
    td.ncores = static_cast<uint32_t>(ncores);
    td.tile_bytes = static_cast<uint32_t>(TILE_BYTES);
    td.rows_per_tile = static_cast<uint32_t>(TILE_BYTES / (h * 2));
    td.payload_bytes = static_cast<uint32_t>(m * h * 2);
    td.has_skip1 = static_cast<uint32_t>(has_skip1);
    td.use_eri = static_cast<uint32_t>(use_eri);
}
}  // namespace

HOST_API void fused_fin_ar_norm_impl(
    const at::Tensor &x, const at::Tensor &scales, const at::Tensor &skip1, const at::Tensor &residual,
    at::Tensor &add_out, at::Tensor &norm_out, const at::Tensor &addr_tab,
    int64_t eri_va, int64_t norm_w_va,
    int64_t m, int64_t h, int64_t k, int64_t ncores, int64_t rank, int64_t world,
    int64_t has_skip1, int64_t use_eri, double eps, int64_t cycle_limit_us,
    int64_t slot_stride, int64_t ring_stride, int64_t max_tiles,
    int64_t counter_offset, int64_t dfx_offset)
{
    TORCH_CHECK(x.defined() && scales.defined() && skip1.defined() && residual.defined() && add_out.defined() &&
                    norm_out.defined() && addr_tab.defined(),
                "fused_fin_ar_norm: all tensor args must be defined (pass dummies for unused slots)");
    TORCH_CHECK(addr_tab.scalar_type() == at::kLong && addr_tab.numel() >= 18 && addr_tab.is_contiguous(),
                "fused_fin_ar_norm: addr_tab must be contiguous int64[>=18] "
                "([0..7] data VA, [8..15] flag VA, [17]=cell area VA)");
    TORCH_CHECK(world >= 1 && world <= MAX_WORLD, "fused_fin_ar_norm: world must be in [1, 8]");
    TORCH_CHECK(rank >= 0 && rank < world, "fused_fin_ar_norm: rank must be in [0, world)");
    CheckCommon("fused_fin_ar_norm", m, h, k, ncores, has_skip1, use_eri, eri_va, x, scales, skip1);
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() && residual.numel() >= m * h,
                "fused_fin_ar_norm: residual must be contiguous bf16 [M,H]");
    TORCH_CHECK(add_out.scalar_type() == at::kBFloat16 && add_out.is_contiguous() && add_out.numel() >= m * h,
                "fused_fin_ar_norm: add_out must be contiguous bf16 [M,H]");
    TORCH_CHECK(norm_out.scalar_type() == at::kBFloat16 && norm_out.is_contiguous() && norm_out.numel() >= m * h,
                "fused_fin_ar_norm: norm_out must be contiguous bf16 [M,H]");
    TORCH_CHECK(norm_w_va != 0, "fused_fin_ar_norm: norm_w_va must be non-zero (norm w bf16[H] VA)");
    // Row-aligned tiles: H2048 -> 2 rows/tile, H4096 -> 1 row/tile; payload
    // must divide evenly into tiles.
    TORCH_CHECK((m * h * 2) % TILE_BYTES == 0,
                "fused_fin_ar_norm: requires m*h*2 % 8192 == 0 (m even for H2048)");
    TORCH_CHECK((m * h * 2) / TILE_BYTES <= max_tiles,
                "fused_fin_ar_norm: tile count exceeds flag area capacity max_tiles");
    // slot/ring layout consistency + slot must hold this payload (otherwise
    // the push would write out of bounds into the peer's symmem).
    TORCH_CHECK(slot_stride > 0 && ring_stride == world * slot_stride,
                "fused_fin_ar_norm: slot/ring stride mismatch (ring_stride must be world*slot_stride)");
    TORCH_CHECK(m * h * 2 <= slot_stride,
                "fused_fin_ar_norm: payload exceeds slot capacity (slot_stride must be >= m*h*2)");
    // cell area layout (plain GM; counter/dfx offsets = in-area byte offsets
    // 0 / 48*128).
    TORCH_CHECK(counter_offset == 0 && dfx_offset == 48 * 128,
                "fused_fin_ar_norm: counter/dfx offsets inconsistent with cell layout (0 / 48*128)");

    const int64_t flag_ring_stride = max_tiles * world * 32;   // item stride pinned 32B
    const int64_t flag_phase_stride = 4 * flag_ring_stride;    // RING(4) generations

    FusedTailTilingData tilingData;
    FillCommonTiling(tilingData, m, h, k, ncores, has_skip1, use_eri, eri_va);
    tilingData.norm_w_va = static_cast<uint64_t>(norm_w_va);
    tilingData.rank = static_cast<uint32_t>(rank);
    tilingData.world = static_cast<uint32_t>(world);
    tilingData.n_tiles = static_cast<uint32_t>((m * h * 2) / TILE_BYTES);
    tilingData.slot_stride = static_cast<uint32_t>(slot_stride);
    tilingData.ring_stride = static_cast<uint32_t>(ring_stride);
    tilingData.max_tiles = static_cast<uint32_t>(max_tiles);
    tilingData.flag_ring_stride = static_cast<uint32_t>(flag_ring_stride);
    tilingData.flag_phase_stride = static_cast<uint32_t>(flag_phase_stride);
    tilingData.counter_offset = static_cast<uint32_t>(counter_offset);
    tilingData.dfx_offset = static_cast<uint32_t>(dfx_offset);
    tilingData.eps = static_cast<float>(eps);
    uint64_t cycleLimit = static_cast<uint64_t>(cycle_limit_us) * CYCLES_PER_US;
    tilingData.cycle_limit =
        static_cast<uint32_t>((cycleLimit > 0xFFFFFFFFULL) ? 0xFFFFFFFFULL : cycleLimit);

    at::Tensor tilingTensor = PrepareTiling(tilingData, x.device());
    uint32_t blockDim = static_cast<uint32_t>(ncores);
    // The dummy workspace slot gets addr_tab (kernel ignores it; no valid
    // parameter goes into the workspace slot — its argument is
    // deterministically corrupted on arrival, unresolved).
    FT_EXEC_KERNEL_CMD_CHECKED(fused_fin_ar_norm, blockDim, x, scales, skip1, residual, add_out, norm_out,
                               addr_tab, addr_tab, tilingTensor);
}

HOST_API void fused_fin_add_impl(
    const at::Tensor &x, const at::Tensor &scales, const at::Tensor &skip1, at::Tensor &out,
    int64_t eri_va, int64_t m, int64_t h, int64_t k, int64_t ncores,
    int64_t has_skip1, int64_t use_eri)
{
    TORCH_CHECK(x.defined() && scales.defined() && skip1.defined() && out.defined(),
                "fused_fin_add: all tensor args must be defined (pass dummies for unused slots)");
    CheckCommon("fused_fin_add", m, h, k, ncores, has_skip1, use_eri, eri_va, x, scales, skip1);
    // Row-granular core split; no tile/flag concept (no symmem, no m*h*2%8192
    // alignment requirement).
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() >= m * h,
                "fused_fin_add: out must be contiguous bf16 [M,H]");

    FusedTailTilingData tilingData;
    FillCommonTiling(tilingData, m, h, k, ncores, has_skip1, use_eri, eri_va);

    at::Tensor tilingTensor = PrepareTiling(tilingData, x.device());
    uint32_t blockDim = static_cast<uint32_t>(ncores);
    FT_EXEC_KERNEL_CMD_CHECKED(fused_fin_add, blockDim, x, scales, skip1, out, out, tilingTensor);
}

HOST_API void fused_tail_zero_impl(at::Tensor &x, int64_t nbytes, int64_t ncores)
{
    TORCH_CHECK(x.defined() && x.is_contiguous() && x.numel() >= nbytes,
                "fused_tail_zero: x must be contiguous with numel >= nbytes");
    TORCH_CHECK(nbytes > 0 && nbytes % 32 == 0,
                "fused_tail_zero: nbytes must be a positive multiple of 32 (DataCopyPad alignment)");
    TORCH_CHECK(ncores >= 1 && ncores <= MAX_NCORES, "fused_tail_zero: ncores must be in [1, 48]");

    FusedTailTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    tilingData.ncores = static_cast<uint32_t>(ncores);
    tilingData.tile_bytes = static_cast<uint32_t>(TILE_BYTES);
    tilingData.total_bytes = static_cast<uint32_t>(nbytes);

    at::Tensor tilingTensor = PrepareTiling(tilingData, x.device());
    uint32_t blockDim = static_cast<uint32_t>(ncores);
    FT_EXEC_KERNEL_CMD_CHECKED(fused_tail_zero, blockDim, x, x, tilingTensor);
}

}  // namespace npu_kernel
}  // namespace sglang
