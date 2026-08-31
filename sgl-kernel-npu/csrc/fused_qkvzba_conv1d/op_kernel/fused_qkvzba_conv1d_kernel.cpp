// fused_qkvzba_conv1d kernel entry (GDN decode: fused split + causal_conv1d)
/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_qkvzba_conv1d_kernel.cpp
 * \brief fused_qkvzba_conv1d kernel entry: task 1 = causal_conv1d UPDATE (decode), task 2 = z/b/a split copy
 *
 * Bit-exactness rationale: the conv part reuses NsCausalConv1d::RunCausalConv1dUpdate (same device class,
 * same tiling struct); the only difference is that x GM reads use row stride tilingData->xRowStride
 * (= qkvz.stride(0), == size(1) when contiguous) instead of dim, and conv consumes only the first
 * dim = qkvWidth columns of each row, identical in content to the old path's split-out mixed_qkv.
 * y writeback and conv_states read/write strides remain dim, unchanged.
 * The z/b/a copy is pure data movement like the old Triton split kernel: a z row = zWidth elements of the
 * qkvz row starting at offset qkvWidth (zWidth carried explicitly in tiling, ba row stride carried as
 * baRowStride -- row-stride views of packed GEMM output are supported; padding in the row stride does not
 * affect z width or b/a content); a b/a row = the first / second numVHeads elements of a mixed_ba row.
 * The two tasks touch disjoint GM regions, are independent, and need no synchronization.
 */

// Relative include by contract: the csrc tree has two headers named causal_conv1d_update.h
// (causal_conv1d_update/op_kernel/ belongs to the old standalone op and lacks NsCausalConv1d::RunCausalConv1dUpdate;
//  causal_conv1d/op_kernel/ is the intended one). workspace_kernel's -I order places the old op's directory
// first, so a bare include would resolve to the wrong header; a relative include resolves from this file's
// directory and avoids the -I ambiguity.
#include "../../causal_conv1d/op_kernel/causal_conv1d_update.h"  // NsCausalConv1d::RunCausalConv1dUpdate (UPDATE/decode subclass)

using namespace AscendC;
using namespace NsCausalConv1d;

namespace {

// Task 2: z/b/a split copy. Note: the conv object's TPipe is not reused -- the CausalConv1dUpdate object is
// locally constructed inside RunCausalConv1dUpdate and destroyed when the call returns (TPipe lifetimes are
// scope-isolated), so this task builds its own TPipe/TBuf.
template <typename T>
__aicore__ inline void FusedCopyZba(GM_ADDR x, GM_ADDR ba, GM_ADDR z, GM_ADDR b, GM_ADDR a,
                                    const __gm__ CausalConv1dTilingData *tilingData)
{
    const int64_t batch = tilingData->batch;
    const int64_t dim = tilingData->dim;                // = qkvWidth, i.e. the column offset of z within a row
    const int64_t xRowStride = tilingData->xRowStride;  // qkvz physical row stride (may include pack padding)
    const int64_t nv = tilingData->numVHeads;
    // zWidth is carried explicitly in tiling (= nv * head_v_dim); it cannot be derived as xRowStride - dim
    // because row-stride view inputs include pack padding in the stride, which would leak into z.
    // When contiguous, xRowStride == dim + zWidth.
    const int64_t zWidth = tilingData->zWidth;
    const int64_t baRowStride = tilingData->baRowStride;  // ba physical row stride (== 2*nv when contiguous)
    if (batch <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());

    // z/b/a share one TPipe and one pair of MTE2/MTE3 events (sequential within the task, no concurrent reuse).
    TPipe copyPipe;
    TEventID mte2ToMte3Event = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    TEventID mte3ToMte2Event = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();

    if (zWidth > 0) {
        GlobalTensor<T> xGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        GlobalTensor<T> zGm;
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z));

        // z: zWidth elements per row. The host has verified that qkvWidth / xRowStride are multiples of 16
        // (i.e. 32B-aligned row starts and row bytes a multiple of 32 for bf16/fp16) and that a single row
        // fits the single-block DataCopy limit, so plain DataCopy is used (same as the DataCopy inside conv).
        TBuf<QuePosition::VECIN> zBuf;
        copyPipe.InitBuffer(zBuf, static_cast<uint32_t>(zWidth * sizeof(T)));
        LocalTensor<T> zLocal = zBuf.Get<T>();

        for (int64_t t = blockIdx; t < batch; t += blockNum) {
            DataCopy(zLocal, xGm[t * xRowStride + dim], static_cast<int32_t>(zWidth));
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            DataCopy(zGm[t * zWidth], zLocal, static_cast<int32_t>(zWidth));
            // wait for this MTE3 write-out before zLocal is reused by the next row
            // (rows are serialized; at most ceil(batch/blockNum) rows per core, negligible cost)
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        }
    }

    if (nv > 0) {
        // b/a: nv elements per row (8B for bf16), below 32B alignment, so plain DataCopy is not usable;
        // bf16 GM scalar access (GlobalTensor GetValue/SetValue) risks wrong values -- do not use it.
        // Use the DataCopyPad path: GM->UB with padding, UB->GM exact (the padding is discarded automatically).
        GlobalTensor<T> baGm;
        baGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ba));
        GlobalTensor<T> bGm;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b));
        GlobalTensor<T> aGm;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a));

        const uint32_t rowBytes = static_cast<uint32_t>(nv) * static_cast<uint32_t>(sizeof(T));
        const uint32_t bufBytes = (rowBytes + 31U) / 32U * 32U;
        TBuf<QuePosition::VECIN> bBuf;
        TBuf<QuePosition::VECIN> aBuf;
        copyPipe.InitBuffer(bBuf, bufBytes);
        copyPipe.InitBuffer(aBuf, bufBytes);
        LocalTensor<T> bLocal = bBuf.Get<T>();
        LocalTensor<T> aLocal = aBuf.Get<T>();

        DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        for (int64_t t = blockIdx; t < batch; t += blockNum) {
            const int64_t baRow = t * baRowStride;  // ba physical row stride (== 2*nv when contiguous)
            const int64_t outRow = t * nv;
            DataCopyPad(bLocal, baGm[baRow], copyParams, padParams);            // b = ba[:, :nv]
            DataCopyPad(aLocal, baGm[baRow + nv], copyParams, padParams);       // a = ba[:, nv:2nv]
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            DataCopyPad(bGm[outRow], bLocal, copyParams);
            DataCopyPad(aGm[outRow], aLocal, copyParams);
            // wait for this MTE3 write-out before bLocal/aLocal are reused by the next row
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        }
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
}

}  // namespace

extern "C" __global__ __aicore__ void fused_qkvzba_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR ba,
                                                          GM_ADDR bias, GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
                                                          GM_ADDR hasInitialState, GM_ADDR numAcceptedTokens,
                                                          GM_ADDR y, GM_ADDR z, GM_ADDR b, GM_ADDR a,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = AscendC::GetUserWorkspace(workspace);
    }

    auto tilingData = reinterpret_cast<__gm__ CausalConv1dTilingData *>(tiling);
    auto dtypeKey = static_cast<uint32_t>(tilingData->dtypeKey);

    // The host fixes run_mode=1 (UPDATE/decode); this dispatches by dtypeKey exactly like the UPDATE branch
    // of the causal_conv1d entry.
    if (dtypeKey == 0) {
        // Task 1 (conv): identical to the original causal_conv1d UPDATE path (same device class); the object's
        // scope is this call and its TPipe is destroyed on return, isolated from the copy task's TPipe below.
        RunCausalConv1dUpdate<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState,
                                          numAcceptedTokens, y, userWorkspace, tilingData);
        // Task 2 (z/b/a copy): data-independent from the conv task
        FusedCopyZba<bfloat16_t>(x, ba, z, b, a, tilingData);
    } else {
        RunCausalConv1dUpdate<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState,
                                    numAcceptedTokens, y, userWorkspace, tilingData);
        FusedCopyZba<half>(x, ba, z, b, a, tilingData);
    }
}
