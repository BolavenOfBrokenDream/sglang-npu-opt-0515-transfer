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
 * \brief fused_tail (fin+add+AR+norm layer-tail fusion, tp_ascendc_fusion_v4.1)
 *        host-side implementation. Three ops:
 *          fused_fin_ar_norm  fin(+skip1)+spin AIV AR+add+gemma rmsnorm single kernel
 *          fused_fin_add      fin(+skip1) local front stage (first half of the stock HCCL AR route)
 *          fused_tail_zero    in-kernel MTE3 zeroing of a local GM region (flag symmem reset)
 *        Mechanism and layouts: see the header comment of
 *        op_kernel/fused_tail_kernel_lib.h.
 *        [v4.1] eri/norm_w changed from tiling VA scalars to tensor
 *        parameters (tiling hash degenerates to shape level, eliminating the
 *        per-layer in-graph H2D+D2D copy pair); scales supports direct bf16
 *        feeding (tiling.scales_bf16, exact in-kernel widening, eliminating
 *        the wrapper cast).
 *
 * Tiling cache follows tp_oneshot_ar_probe's hash-key + static device buffer
 * pattern (graph-capture pointer stability); the launch macro carries a
 * return-code check (pure value capture — deferred closures must capture by
 * value only; the vgmm1 &launch_ret wild-write lesson).
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

// Launch macro with return-code check (same as tp_oneshot_ar_probe's
// OAR_EXEC_KERNEL_CMD_CHECKED; round-24 fixed form: launch_ret lives inside
// the closure, pure value capture, ret propagates via the closure return
// value — the sync branch is checked by InnerRunOpApi's OPS_CHECK_ERROR, the
// async branch surfaces ret via ExecFuncOpApi into the queue error channel.
// Iron rule: deferred closures must capture by value only).
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
// Keep-alive pool for the pinned staging of map entries (<=MAX_CAPTURE_NUM);
// see the PrepareTiling comment.
std::vector<at::Tensor> g_ftPinnedKeepAlive;

// Tiling cache key = FNV-1a hash of the whole TilingData struct contents
// (whole-struct hashing guards against forgotten fields).
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

// Common guards: shape/dtype/core count/rank. The op name goes into error
// strings.
// [v4.1] eri is now a tensor parameter (with use_eri=0 the caller passes a
// dummy; contents not validated); scales accepts fp32 or bf16 (bf16 is
// widened exactly in-kernel, see the scales_bf16 tiling flag).
void CheckCommon(const char *op, int64_t m, int64_t h, int64_t k, int64_t ncores, int64_t has_skip1,
                 int64_t use_eri, const at::Tensor &eri, const at::Tensor &x, const at::Tensor &scales,
                 const at::Tensor &skip1)
{
    TORCH_CHECK(h == 2048 || h == 4096, op, ": H must be 2048 or 4096");
    TORCH_CHECK(m >= 1 && m <= MAX_M, op, ": M must be in [1, 4096]");
    TORCH_CHECK(k >= 1 && k <= MAX_K, op, ": K(topk) must be in [1, 16]");
    TORCH_CHECK(ncores >= 1 && ncores <= MAX_NCORES, op, ": ncores must be in [1, 48]");
    TORCH_CHECK(has_skip1 == 0 || has_skip1 == 1, op, ": has_skip1 must be 0/1");
    TORCH_CHECK(use_eri == 0 || use_eri == 1, op, ": use_eri must be 0/1");
    if (use_eri != 0) {
        TORCH_CHECK(eri.scalar_type() == at::kInt && eri.is_contiguous() && eri.numel() >= m * k,
                    op, ": use_eri=1 requires eri contiguous int32 [M*K] (wild-address read risk, host must block)");
    }
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() >= m * k * h,
                op, ": x=xp/xexp must be contiguous bf16 [M*K,H]");
    TORCH_CHECK((scales.scalar_type() == at::kFloat || scales.scalar_type() == at::kBFloat16) &&
                    scales.is_contiguous() && scales.numel() >= m * k,
                op, ": scales must be contiguous fp32 or bf16 [M,K]");
    if (has_skip1 != 0) {
        TORCH_CHECK(skip1.scalar_type() == at::kBFloat16 && skip1.is_contiguous() && skip1.numel() >= m * h,
                    op, ": has_skip1=1 requires skip1 contiguous bf16 [M,H] (wild-address read risk, host must block)");
    }
}

// Fills common tiling fields + cache reuse (graph-capture pointer stability).
// Returns the reused/newly created device tiling tensor. Flag-layout
// derivatives are derived host-side with the same formulas as the python
// wrapper (single source of truth, preventing two-sided drift; the
// counter/dfx offset check is the two-sided consistency guard).
at::Tensor PrepareTiling(FusedTailTilingData &tilingData, const at::Device &device)
{
    int32_t tilingSize = (static_cast<int32_t>(sizeof(FusedTailTilingData)) +
                          static_cast<int32_t>(TILING_PADDING_BYTE) - 1) /
                         static_cast<int32_t>(TILING_PADDING_BYTE) * static_cast<int32_t>(TILING_PADDING_BYTE);

    static auto globalTilingBuffer = at::empty({static_cast<int64_t>(tilingSize) * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(device));

    uint64_t hashValue = HashTilingData(tilingData);
    // Graph-capture safety (root-cause fix for the relax e2e round-9 crash;
    // see docs/bug-fix/npu-tp-fusion-v4-hccl-prefill-aiv-mte-oob.md round 9):
    // [v4.1] tiling no longer contains any VA (eri/norm_w now travel as
    // tensor parameters); the hash degenerates to shape level — 40 layers
    // share 1 entry, and entries created inside capture drop from "per
    // (tier x layer)" to "at most 1 per bs tier" (0 if eager warmup already
    // ran the same m). But creation inside capture can still happen
    // (non-warmed tiers), and the copy chain below gets **recorded into the
    // graph** by ACL during capture — CopyTensorHostToDevice's non-blocking
    // H2D (pin_memory -> .to(non_blocking=true)) is recorded as a memcpy
    // node during capture; torch_npu performs no capture-state check on the
    // H2D path (AsyncTaskQueueInterface.cpp:106 issues aclrtMemcpyAsync
    // straight onto the capture stream), and at replay that memcpy node
    // re-reads the pinned source address baked in at capture; meanwhile a
    // temporary pinned staging buffer is reused after being returned to the
    // host allocator (its record_event deferred-free protection is itself an
    // illegal call on the capture stream and does nothing), so after the
    // pinned-allocation churn of a large prefill it holds garbage -> replay
    // pours garbage tiling into the static slot -> kernel MTE DDR overrun.
    // Hence the pinned keep-alive fix must be kept (not dead code):
    // for entries that go into the map (<=1024) the pinned staging is kept
    // alive statically, so the graph-recorded H2D re-reads correct and
    // immutable contents on every replay.
    // The >1024 fallback path is reachable only in eager (post-v4.1 the
    // capture-time entry count is bounded by the number of bs tiers <<1024);
    // no keep-alive there, to avoid long-run leaks.
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
                      int64_t has_skip1, int64_t use_eri, bool scalesBf16)
{
    std::memset(&td, 0, sizeof(td));  // reserved_va0/1 (v4's eri_va/norm_w_va) stay always-0
    td.m = static_cast<uint32_t>(m);
    td.h = static_cast<uint32_t>(h);
    td.k = static_cast<uint32_t>(k);
    td.ncores = static_cast<uint32_t>(ncores);
    td.tile_bytes = static_cast<uint32_t>(TILE_BYTES);
    td.rows_per_tile = static_cast<uint32_t>(TILE_BYTES / (h * 2));
    td.payload_bytes = static_cast<uint32_t>(m * h * 2);
    td.has_skip1 = static_cast<uint32_t>(has_skip1);
    td.use_eri = static_cast<uint32_t>(use_eri);
    td.scales_bf16 = scalesBf16 ? 1U : 0U;
}
}  // namespace

HOST_API void fused_fin_ar_norm_impl(
    const at::Tensor &x, const at::Tensor &scales, const at::Tensor &skip1, const at::Tensor &residual,
    at::Tensor &add_out, at::Tensor &norm_out, const at::Tensor &addr_tab,
    const at::Tensor &eri, const at::Tensor &norm_w,
    int64_t m, int64_t h, int64_t k, int64_t ncores, int64_t rank, int64_t world,
    int64_t has_skip1, int64_t use_eri, double eps, int64_t cycle_limit_us,
    int64_t slot_stride, int64_t ring_stride, int64_t max_tiles,
    int64_t counter_offset, int64_t dfx_offset)
{
    TORCH_CHECK(x.defined() && scales.defined() && skip1.defined() && residual.defined() && add_out.defined() &&
                    norm_out.defined() && addr_tab.defined() && eri.defined() && norm_w.defined(),
                "fused_fin_ar_norm: all tensor args must be defined (pass dummies for unused slots)");
    TORCH_CHECK(addr_tab.scalar_type() == at::kLong && addr_tab.numel() >= 18 && addr_tab.is_contiguous(),
                "fused_fin_ar_norm: addr_tab must be contiguous int64[>=18] ([0..7] data VA, "
                "[8..15] flag VA, [17]=cell area VA)");
    TORCH_CHECK(world >= 1 && world <= MAX_WORLD, "fused_fin_ar_norm: world must be in [1, 8]");
    TORCH_CHECK(rank >= 0 && rank < world, "fused_fin_ar_norm: rank must be in [0, world)");
    CheckCommon("fused_fin_ar_norm", m, h, k, ncores, has_skip1, use_eri, eri, x, scales, skip1);
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() && residual.numel() >= m * h,
                "fused_fin_ar_norm: residual must be contiguous bf16 [M,H]");
    TORCH_CHECK(add_out.scalar_type() == at::kBFloat16 && add_out.is_contiguous() && add_out.numel() >= m * h,
                "fused_fin_ar_norm: add_out must be contiguous bf16 [M,H]");
    TORCH_CHECK(norm_out.scalar_type() == at::kBFloat16 && norm_out.is_contiguous() && norm_out.numel() >= m * h,
                "fused_fin_ar_norm: norm_out must be contiguous bf16 [M,H]");
    TORCH_CHECK(norm_w.scalar_type() == at::kBFloat16 && norm_w.is_contiguous() && norm_w.numel() >= h,
                "fused_fin_ar_norm: norm_w must be contiguous bf16 [H]");
    // Row-aligned tiles: H2048 -> 2 rows/tile, H4096 -> 1 row/tile; payload
    // must be divisible by tile
    TORCH_CHECK((m * h * 2) % TILE_BYTES == 0,
                "fused_fin_ar_norm: requires m*h*2 % 8192 == 0 (m must be even for H2048)");
    TORCH_CHECK((m * h * 2) / TILE_BYTES <= max_tiles,
                "fused_fin_ar_norm: tile count exceeds flag area capacity max_tiles");
    // slot/ring layout consistency + slot capacity must hold this payload
    // (otherwise pushing overruns into the peer's symmem)
    TORCH_CHECK(slot_stride > 0 && ring_stride == world * slot_stride,
                "fused_fin_ar_norm: slot/ring stride mismatch (ring_stride must be world*slot_stride)");
    TORCH_CHECK(m * h * 2 <= slot_stride,
                "fused_fin_ar_norm: payload exceeds slot capacity (slot_stride must be >= m*h*2)");
    // Cell area layout (plain GM; counter/dfx offsets = in-area byte offsets
    // 0 / 48*128)
    TORCH_CHECK(counter_offset == 0 && dfx_offset == 48 * 128,
                "fused_fin_ar_norm: counter/dfx offsets inconsistent with cell area layout (0 / 48*128)");

    const int64_t flag_ring_stride = max_tiles * world * 32;   // item stride pinned to 32B
    const int64_t flag_phase_stride = 4 * flag_ring_stride;    // RING(4) gens

    FusedTailTilingData tilingData;
    FillCommonTiling(tilingData, m, h, k, ncores, has_skip1, use_eri,
                     scales.scalar_type() == at::kBFloat16);
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
    // The dummy workspace slot passes addr_tab (kernel ignores that
    // parameter; no valid parameter goes into the workspace slot — probes
    // proved that slot's argument is deterministically corrupted on arrival
    // at the kernel, the aux +out_offset unresolved case).
    // [v4.1] eri/norm_w are proper tensor parameters (located before the
    // workspace slot, same channel as existing parameters like x/scales).
    FT_EXEC_KERNEL_CMD_CHECKED(fused_fin_ar_norm, blockDim, x, scales, skip1, residual, add_out, norm_out,
                               addr_tab, eri, norm_w, addr_tab, tilingTensor);
}

HOST_API void fused_fin_add_impl(
    const at::Tensor &x, const at::Tensor &scales, const at::Tensor &skip1, at::Tensor &out,
    const at::Tensor &eri, int64_t m, int64_t h, int64_t k, int64_t ncores,
    int64_t has_skip1, int64_t use_eri)
{
    TORCH_CHECK(x.defined() && scales.defined() && skip1.defined() && out.defined() && eri.defined(),
                "fused_fin_add: all tensor args must be defined (pass dummies for unused slots)");
    CheckCommon("fused_fin_add", m, h, k, ncores, has_skip1, use_eri, eri, x, scales, skip1);
    // Row-granular core split, no tile/flag concept (no symmem, no
    // m*h*2%8192 alignment requirement)
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() >= m * h,
                "fused_fin_add: out must be contiguous bf16 [M,H]");

    FusedTailTilingData tilingData;
    FillCommonTiling(tilingData, m, h, k, ncores, has_skip1, use_eri,
                     scales.scalar_type() == at::kBFloat16);

    at::Tensor tilingTensor = PrepareTiling(tilingData, x.device());
    uint32_t blockDim = static_cast<uint32_t>(ncores);
    // The dummy workspace slot passes out (kernel ignores that parameter,
    // same discipline as ar_norm).
    FT_EXEC_KERNEL_CMD_CHECKED(fused_fin_add, blockDim, x, scales, skip1, out, eri, out, tilingTensor);
}

HOST_API void fused_tail_zero_impl(at::Tensor &x, int64_t nbytes, int64_t ncores)
{
    TORCH_CHECK(x.defined() && x.is_contiguous() && x.numel() >= nbytes,
                "fused_tail_zero: x must be a contiguous tensor with numel >= nbytes");
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
