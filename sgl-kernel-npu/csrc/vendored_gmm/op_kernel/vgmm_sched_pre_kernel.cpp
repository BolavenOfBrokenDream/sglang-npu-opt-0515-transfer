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
 * \file vgmm_sched_pre_kernel.cpp
 * \brief vgmm1_sched_pre: offsets + block-schedule-table front kernel (the
 *        "front mini-kernel" of plan B).
 *
 * A single AIV core computes from counts (int64[E]) in one pass:
 *   - row_offsets[E]: exclusive cumsum (each expert's start row in packed
 *     x/y, i.e. offsets)
 *   - block_table[totalBlocks]: global block id -> {groupId, row0group,
 *     mIdx, nIdx, blockRows, curSingleN, groupRows, 0} (8x int32/block,
 *     32B aligned); block order = stock's group-major global block numbering
 *     (m first, then n within a group); the main kernel picks blocks round
 *     robin by b % coreNum
 *   - total_blocks[0]: total block count
 * Work is O(E + totalBlocks), scalar work, one core suffices (the persistent
 * line's similar cumsum front kernel measured ~4.9us, same order here).
 *
 * Discipline (four-round dropout lesson from the probe, gmm_scalar_bound_probe
 * README §0.8): outputs always go back via DataCopy/MTE3, never scalar
 * SetValue directly to GM; UB scalar read/write uses
 * LocalTensor.SetValue/GetValue (same precedent as alloc_extend).
 * One kernel per file (registration iron rule, tp_fusion_probe README §0.2).
 */

#include "kernel_operator.h"
#include "common_tiling_kernel.h"
#include "../op_host/vendored_gmm_tiling_data.h"

using namespace AscendC;
using sglang::npu_kernel::VGMM_BLOCK_ENTRY_INT32;
using sglang::npu_kernel::VgmmSchedTilingData;

extern "C" __global__ __aicore__ void vgmm1_sched_pre(GM_ADDR group_list, GM_ADDR block_table, GM_ADDR row_offsets,
                                                      GM_ADDR total_blocks, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) {
        return;  // single-core build
    }
    (void)workspace;

    VgmmSchedTilingData td;
    kernel_utils::CopyTiling(&td, tiling);
    const uint32_t groupNum = td.groupNum;
    const uint32_t baseM = td.baseM;
    const uint32_t baseN = td.baseN;
    const uint32_t n = td.n;
    const uint32_t maxBlocks = td.maxBlocks;
    const uint32_t blockDimN = (n + baseN - 1) / baseN;
    const uint32_t rowOffZone = maxBlocks * VGMM_BLOCK_ENTRY_INT32;  // row_offsets zone start (int32 units)
    const uint32_t tbZone = rowOffZone + ((groupNum + 7) / 8 * 8);   // total_blocks zone start (32B aligned)

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQue;   // counts in
    TQue<TPosition::VECIN, 1> outQue;  // table + rowOffsets + totalBlocks out (same VECIN dequeue convention as alloc_extend)
    pipe.InitBuffer(inQue, 1, (groupNum * sizeof(int64_t) + 31) / 32 * 32);
    pipe.InitBuffer(outQue, 1, (tbZone + 8) * sizeof(int32_t));

    GlobalTensor<int64_t> groupListGm;
    groupListGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(group_list));
    LocalTensor<int64_t> cntUb = inQue.AllocTensor<int64_t>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(groupNum * sizeof(int64_t)), 0, 0, 0};
    DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
    DataCopyPad(cntUb, groupListGm, copyInParams, padParams);
    inQue.EnQue(cntUb);
    cntUb = inQue.DeQue<int64_t>();

    LocalTensor<int32_t> outUb = outQue.AllocTensor<int32_t>();
    int32_t rowOff = 0;
    uint32_t blk = 0;
    for (uint32_t g = 0; g < groupNum; ++g) {
        int32_t m = static_cast<int32_t>(cntUb.GetValue(g));
        outUb.SetValue(rowOffZone + g, rowOff);  // exclusive: record start row first, accumulate later
        if (m <= 0) {
            continue;  // empty group produces no blocks (naturally absorbs direction-A sparse semantics)
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
    DataCopy(tableGm, outUb, maxBlocks * VGMM_BLOCK_ENTRY_INT32);  // 32B x maxBlocks, naturally aligned
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(groupNum * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(rowOffGm, outUb[rowOffZone], copyOutParams);
    DataCopy(tbGm, outUb[tbZone], 8);  // 32B

    inQue.FreeTensor(cntUb);
    outQue.FreeTensor(outUb);
}
