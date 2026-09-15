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
 * \file vgmm_sched_partial_pre_kernel.cpp
 * \brief vgmm1_sched_partial_pre: front kernel of the C1 route (sched
 *        absorbs the cumsum) — directly consumes the partials
 *        [partNum, partStride] (int32 partial histogram) produced by v22
 *        rank_hist, and a single AIV core computes in one pass:
 *          - counts int64[E]: column reduction (= v22 partials_cumsum's
 *            axis=0 sum; i32 addition is exact and bit-identical to the
 *            Triton version — integer addition is associative);
 *          - row_offsets int32[E]: exclusive cumsum (= v22's excl,
 *            bit-identical);
 *          - block_table / total_blocks: entry formula character-identical
 *            to vgmm1_sched_pre (a pure integer deterministic function of
 *            counts; bit-identical ⟹ vgmm1_main y unchanged bit for bit).
 *        The Triton partials_cumsum node (~4.8us in-graph, including 1.2us
 *        node tax) disappears from the chain entirely; net gain = full cost
 *        of the cumsum node minus this kernel's extra column reduction
 *        (~0.3us scale).
 *        Design decision: see
 *        docs/bug-fix/npu-vgmm1-t1-build-table-static-loop-slow.md §5
 *        (C1 promoted after two measured rounds of T1 loss-cutting).
 *
 * Column reduction implementation: partials rows are fold-added pairwise
 * (Add whole-row vectors, log2(partNum) rounds), odd rows carry over
 * naturally; partNum <= 63 (M<=512, bm>=8), partStride=256, work
 * <= 2x63x256 i32 ~= 32K elements, sub-us scale.
 *
 * Discipline (same as vgmm_sched_pre_kernel.cpp): outputs always go back
 * via DataCopy/MTE3, never scalar SetValue directly to GM; UB scalar
 * read/write uses LocalTensor.SetValue/GetValue.
 * One kernel per file (registration iron rule, tp_fusion_probe README §0.2).
 */

#include "kernel_operator.h"
#include "common_tiling_kernel.h"
#include "../op_host/vendored_gmm_tiling_data.h"

using namespace AscendC;
using sglang::npu_kernel::VGMM_BLOCK_ENTRY_INT32;
using sglang::npu_kernel::VgmmSchedPartialTilingData;

extern "C" __global__ __aicore__ void vgmm1_sched_partial_pre(GM_ADDR partials, GM_ADDR block_table,
                                                              GM_ADDR row_offsets, GM_ADDR total_blocks,
                                                              GM_ADDR counts, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) {
        return;  // single-core build
    }
    (void)workspace;

    VgmmSchedPartialTilingData td;
    kernel_utils::CopyTiling(&td, tiling);
    const uint32_t groupNum = td.groupNum;
    const uint32_t partNum = td.partNum;
    const uint32_t partStride = td.partStride;
    const uint32_t baseM = td.baseM;
    const uint32_t baseN = td.baseN;
    const uint32_t n = td.n;
    const uint32_t maxBlocks = td.maxBlocks;
    const uint32_t blockDimN = (n + baseN - 1) / baseN;
    // outUb layout (int32 units): [0, rowOffZone) = table zone; rowOffZone =
    // row_offsets zone; tbZone = total_blocks zone (8x int32); countsZone =
    // counts zone (each i64 written as lo/hi int32 pair, little-endian lo
    // first; counts <= 512 so hi is always 0). countsZone stays a multiple
    // of 8 (rowOffZone and the row_offsets zone are both multiples of 8), so
    // i64 elements are naturally aligned.
    const uint32_t rowOffZone = maxBlocks * VGMM_BLOCK_ENTRY_INT32;
    const uint32_t tbZone = rowOffZone + ((groupNum + 7) / 8 * 8);
    const uint32_t countsZone = tbZone + 8;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQue;   // partials in
    TQue<TPosition::VECIN, 1> outQue;  // table + rowOffsets + totalBlocks + counts out
    pipe.InitBuffer(inQue, 1, (partNum * partStride * sizeof(int32_t) + 31) / 32 * 32);
    pipe.InitBuffer(outQue, 1, (countsZone + groupNum * 2) * sizeof(int32_t));

    GlobalTensor<int32_t> partGm;
    partGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(partials));
    LocalTensor<int32_t> partUb = inQue.AllocTensor<int32_t>();
    DataCopyExtParams copyInParams{1, partNum * partStride * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(partUb, partGm, copyInParams, padParams);
    inQue.EnQue(partUb);
    partUb = inQue.DeQue<int32_t>();

    // Column reduction: [partNum, partStride] rows fold-added pairwise; row 0
    // ends up = counts (i32).
    // Pairing is (i, i+ceil(len/2)); with odd len the middle row carries over
    // directly to the next round.
    // Instructions in the same vector pipe issue in order, so inter-round
    // dependencies are naturally ordered.
    for (uint32_t len = partNum; len > 1; len = (len + 1) / 2) {
        const uint32_t half = len / 2;
        const uint32_t off = len - half;  // = ceil(len/2)
        for (uint32_t i = 0; i < half; ++i) {
            Add(partUb[i * partStride], partUb[i * partStride], partUb[(i + off) * partStride], partStride);
        }
    }

    LocalTensor<int32_t> outUb = outQue.AllocTensor<int32_t>();
    int32_t rowOff = 0;
    uint32_t blk = 0;
    for (uint32_t g = 0; g < groupNum; ++g) {
        int32_t m = partUb.GetValue(g);  // after column reduction, row 0 = this expert's count
        outUb.SetValue(rowOffZone + g, rowOff);      // exclusive: record start row first, accumulate later
        outUb.SetValue(countsZone + 2 * g, m);       // i64 lo
        outUb.SetValue(countsZone + 2 * g + 1, 0);   // i64 hi
        if (m <= 0) {
            continue;  // empty group produces no blocks (naturally absorbs sparse semantics)
        }
        uint32_t um = static_cast<uint32_t>(m);
        uint32_t bm = (um + baseM - 1) / baseM;
        for (uint32_t i = 0; i < bm; ++i) {
            for (uint32_t j = 0; j < blockDimN; ++j) {
                if (blk >= maxBlocks) {
                    break;  // defense: driver allocates with the total_M + E upper bound; unreachable
                }
                uint32_t o = blk * VGMM_BLOCK_ENTRY_INT32;
                outUb.SetValue(o + 0, static_cast<int32_t>(g));
                outUb.SetValue(o + 1, rowOff);
                outUb.SetValue(o + 2, static_cast<int32_t>(i));
                outUb.SetValue(o + 3, static_cast<int32_t>(j));
                uint32_t blockRows = um - i * baseM;
                outUb.SetValue(o + 4, static_cast<int32_t>(blockRows < baseM ? blockRows : baseM));
                uint32_t curN = n - j * baseN;
                outUb.SetValue(o + 5, static_cast<int32_t>(curN < baseN ? curN : baseN));
                outUb.SetValue(o + 6, m);
                outUb.SetValue(o + 7, 0);
                ++blk;
            }
        }
        rowOff += m;
    }
    for (uint32_t t = 0; t < 8; ++t) {
        outUb.SetValue(tbZone + t, t == 0 ? static_cast<int32_t>(blk) : 0);
    }
    outQue.EnQue(outUb);
    outUb = outQue.DeQue<int32_t>();

    GlobalTensor<int32_t> tableGm;
    tableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_table));
    GlobalTensor<int32_t> rowOffGm;
    rowOffGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(row_offsets));
    GlobalTensor<int32_t> tbGm;
    tbGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(total_blocks));
    GlobalTensor<int32_t> countsGm32;  // int32 view of counts (lo/hi pairs), same layout as UB
    countsGm32.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(counts));
    DataCopy(tableGm, outUb, maxBlocks * VGMM_BLOCK_ENTRY_INT32);  // 32B x maxBlocks, naturally aligned
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(groupNum * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(rowOffGm, outUb[rowOffZone], copyOutParams);
    DataCopy(tbGm, outUb[tbZone], 8);  // 32B
    DataCopyExtParams copyCntParams{1, static_cast<uint32_t>(groupNum * 2 * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(countsGm32, outUb[countsZone], copyCntParams);

    inQue.FreeTensor(partUb);
    outQue.FreeTensor(outUb);
}
