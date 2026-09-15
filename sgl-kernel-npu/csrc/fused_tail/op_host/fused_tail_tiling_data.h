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
 * \file fused_tail_tiling_data.h
 * \brief fused_tail (fin+add+AR+norm layer-tail fusion, tp_ascendc_fusion_v4.1)
 *        host/kernel shared tiling struct. Scalar fields only (all-AIV
 *        vector kernels, no cube parts); sizeof stays a multiple of 4 bytes
 *        (kernel_utils::CopyTiling copies by uint32).
 *
 * Differences from the probe version (tp_oneshot_ar_probe_tiling_data.h):
 *   - [v4.1] eri / norm w are now passed as proper kernel tensor parameters
 *     (GM_ADDR formal parameters) — v4 baked eri_va/norm_w_va into tiling as
 *     uint64 scalars (the production version of the probe's addr_tab[16]/[18]
 *     VA channel workaround for the aux-parameter-slot unresolved case), at
 *     the cost of VAs in the tiling hash => every (bs tier x layer) entry
 *     had to be born during capture and its pinned->H2D->static-slot copy
 *     chain had to enter the graph (the round-9 root-cause chain), with
 *     replay paying two wasted memcpy nodes per layer (~3us). After
 *     parameterization the tiling degenerates to shape level (40 layers
 *     share 1 entry) with <=1 in-graph copy pair per tier; eri is a
 *     graph-pool static buffer and norm w a parameter, both addresses
 *     stable for the graph's lifetime, same channel and same safety level
 *     as existing tensor parameters like x/scales (none goes into the
 *     workspace slot).
 *   - [v4.1] new scales_bf16 flag: scales may be fed directly as bf16
 *     (widened to fp32 exactly in-kernel, bit-equivalent to a host-side
 *     .float()), eliminating the wrapper's cast small-op.
 *   - Flag protocol pinned to v2 (phase-A without self item, target R-1,
 *     remote double-write, pure MTE3 clearing), item stride pinned to 32B —
 *     the probe's flag_mode/flag_item_bytes/dcci_mode/addr_dump/sub/mode/
 *     out_offset/rows_base/rows_rem/bw_bytes/rtt_iters are all deleted.
 */

#ifndef FUSED_TAIL_TILING_DATA_H
#define FUSED_TAIL_TILING_DATA_H

#include <cstdint>

namespace sglang {
namespace npu_kernel {

#pragma pack(push, 1)
struct FusedTailTilingData {
    uint64_t reserved_va0;  // v4's eri_va (since v4.1 eri travels as a tensor parameter, always 0)
    uint64_t reserved_va1;  // v4's norm_w_va (since v4.1 norm w travels as a tensor parameter, always 0)
    uint32_t m;           // row count M
    uint32_t h;           // hidden dim H (bf16 elements/row; only 2048/4096)
    uint32_t k;           // topk K ([1,16])
    uint32_t rank;        // this rank
    uint32_t world;       // R ([1,8])
    uint32_t ncores;      // launched core count C (= blockDim)
    uint32_t tile_bytes;        // = 8192 (H2048 -> 2 rows/tile, H4096 -> 1 row/tile)
    uint32_t rows_per_tile;     // = tile_bytes/(h*2)
    uint32_t n_tiles;           // = payload_bytes/tile_bytes (used by fin_ar_norm)
    uint32_t payload_bytes;     // = m*h*2
    uint32_t slot_stride;       // slot[r] capacity (bytes) = max_payload
    uint32_t ring_stride;       // = world*slot_stride
    uint32_t max_tiles;         // tile capacity per ring of the flag area (derived from Mmax)
    uint32_t flag_ring_stride;  // = max_tiles*world*32
    uint32_t flag_phase_stride; // = RING(4)*flag_ring_stride
    uint32_t counter_offset;    // byte offset of the ring counter area within the cell area (=0)
    uint32_t dfx_offset;        // byte offset of the DFX area within the cell area (=48*128)
    uint32_t has_skip1;         // 1=acc initialized with fp32(skip1[m]); 0=initialized with 0
    uint32_t use_eri;           // 1=x is unpermuted xp, read indirectly via eri; 0=x already laid out
    uint32_t scales_bf16;       // [v4.1] 1=scales is bf16 (widened exactly in-kernel); 0=fp32
    float eps;                  // gemma rmsnorm eps
    uint32_t cycle_limit;       // spin cycle limit (host precomputes us*CYCLES_PER_US)
    uint32_t total_bytes;       // fused_tail_zero: bytes to clear (multiple of 32)
    uint32_t pad[1];            // reserved/alignment
};
#pragma pack(pop)

}  // namespace npu_kernel
}  // namespace sglang

#endif  // FUSED_TAIL_TILING_DATA_H
