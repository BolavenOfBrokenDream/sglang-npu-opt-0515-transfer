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
 * \brief fused_tail (MoE layer-tail fin+add+AR+norm fusion) host/kernel shared
 *        tiling struct. Scalar fields only (all-AIV kernels, no cube parts);
 *        sizeof stays a multiple of 4 bytes (kernel_utils::CopyTiling copies
 *        by uint32).
 *
 * eri_va / norm_w_va travel as uint64 scalar tiling fields: each layer's norm
 * weight has a different VA and eri is a graph-static buffer VA, so baking
 * them into tiling at capture time is equivalent to tensor parameters (their
 * addresses never change for the graph's lifetime) and keeps them out of the
 * workspace slot (whose argument is deterministically corrupted on arrival —
 * unresolved; valid parameters never go into that slot).
 * Flag protocol pinned to v2 (phase-A without self item, spin target R-1,
 * remote double-write, pure MTE3 clearing), item stride pinned to 32B.
 */

#ifndef FUSED_TAIL_TILING_DATA_H
#define FUSED_TAIL_TILING_DATA_H

#include <cstdint>

namespace sglang {
namespace npu_kernel {

#pragma pack(push, 1)
struct FusedTailTilingData {
    uint64_t eri_va;      // use_eri=1: VA of eri int32[M*K] (flat slot -> expanded row);
                          // use_eri=0: ignored (x already laid out by (m,k))
    uint64_t norm_w_va;   // fused_fin_ar_norm: VA of norm weight bf16[H] (others ignore)
    uint32_t m;           // rows M
    uint32_t h;           // hidden H (bf16 elems/row; 2048/4096 only)
    uint32_t k;           // topk K ([1,16])
    uint32_t rank;        // this rank
    uint32_t world;       // R ([1,8])
    uint32_t ncores;      // launch cores C (= blockDim)
    uint32_t tile_bytes;        // = 8192 (H2048 -> 2 rows/tile, H4096 -> 1 row/tile)
    uint32_t rows_per_tile;     // = tile_bytes/(h*2)
    uint32_t n_tiles;           // = payload_bytes/tile_bytes (fin_ar_norm)
    uint32_t payload_bytes;     // = m*h*2
    uint32_t slot_stride;       // slot[r] capacity in bytes = max_payload
    uint32_t ring_stride;       // = world*slot_stride
    uint32_t max_tiles;         // flag area tile capacity per ring (derived from Mmax)
    uint32_t flag_ring_stride;  // = max_tiles*world*32
    uint32_t flag_phase_stride; // = RING(4)*flag_ring_stride
    uint32_t counter_offset;    // byte offset of ring counter area in cell area (=0)
    uint32_t dfx_offset;        // byte offset of DFX area in cell area (=48*128)
    uint32_t has_skip1;         // 1=acc initialized with fp32(skip1[m]); 0=zero-init
    uint32_t use_eri;           // 1=x is unpermuted xp read indirectly via eri; 0=x direct
    float eps;                  // gemma rmsnorm eps
    uint32_t cycle_limit;       // spin cycle limit (host computes us*CYCLES_PER_US)
    uint32_t total_bytes;       // fused_tail_zero: bytes to clear (multiple of 32)
    uint32_t pad[2];            // reserved/alignment
};
#pragma pack(pop)

}  // namespace npu_kernel
}  // namespace sglang

#endif  // FUSED_TAIL_TILING_DATA_H
