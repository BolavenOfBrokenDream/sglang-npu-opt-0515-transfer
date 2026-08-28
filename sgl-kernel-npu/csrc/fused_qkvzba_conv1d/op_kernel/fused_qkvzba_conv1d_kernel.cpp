// fused_qkvzba_conv1d kernel 入口（GDN decode：split + causal_conv1d 融合）
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
 * \brief fused_qkvzba_conv1d kernel entry：任务 1 = causal_conv1d UPDATE(decode)，任务 2 = z/b/a 切分拷贝
 *
 * 逐 bit 等价性依据：conv 部分复用 NsCausalConv1d::RunCausalConv1dUpdate（同一 device 类、同一 tiling 结构），
 * 唯一差别是 x GM 读的行距由 dim 放宽为 tilingData->xRowStride（= qkvz 物理行距 stride(0)，连续输入时
 * == size(1)），conv 只消费每行前 dim=qkvWidth 列，与老链路 split 出的 mixed_qkv 内容一致；y 写回、
 * conv_states 读写行距仍为 dim，未改。
 * z/b/a 拷贝与老链路 Triton split kernel 同为纯搬运：z 行 = qkvz 行偏移 qkvWidth 起 zWidth 元素
 * （zWidth 由 tiling 显式携带，ba 行距由 baRowStride 携带——支持打包 GEMM 的行距视图输入，
 * 行距含 pad 时不影响 z 宽与 b/a 内容）；b/a 行 = mixed_ba 行的前/后 numVHeads 元素。
 * 两个任务读写完全不同的 GM 区域，相互独立、无需同步。
 */

// 相对路径包含（契约）：csrc 树里存在两个同名 causal_conv1d_update.h
// （causal_conv1d_update/op_kernel/ 属旧独立算子、无 NsCausalConv1d::RunCausalConv1dUpdate；
//  causal_conv1d/op_kernel/ 才是目标头）。workspace_kernel 的 -I 顺序会把
// 旧算子目录排在前面，裸 include 必命中错误头；相对包含从本文件目录解析、绕过 -I 二义性。
#include "../../causal_conv1d/op_kernel/causal_conv1d_update.h"  // NsCausalConv1d::RunCausalConv1dUpdate（UPDATE/decode 子类）

using namespace AscendC;
using namespace NsCausalConv1d;

namespace {

// 任务 2：z/b/a 切分拷贝。注意：不复用 conv 对象的 TPipe——CausalConv1dUpdate 对象已在 RunCausalConv1dUpdate
// 内局部构造并随调用返回析构（TPipe 生命周期以作用域隔离），此处自建 TPipe/TBuf。
template <typename T>
__aicore__ inline void FusedCopyZba(GM_ADDR x, GM_ADDR ba, GM_ADDR z, GM_ADDR b, GM_ADDR a,
                                    const __gm__ CausalConv1dTilingData *tilingData)
{
    const int64_t batch = tilingData->batch;
    const int64_t dim = tilingData->dim;                // = qkvWidth，即 z 在行内的列偏移
    const int64_t xRowStride = tilingData->xRowStride;  // qkvz 物理行距（可为打包 pad 后的 stride(0)）
    const int64_t nv = tilingData->numVHeads;
    // zWidth 由 tiling 显式携带（= nv * head_v_dim）。不能由 xRowStride - dim 反推——
    // 行距视图输入下行距含打包 pad，反推会把 pad 算进 z。连续输入时 xRowStride == dim + zWidth。
    const int64_t zWidth = tilingData->zWidth;
    const int64_t baRowStride = tilingData->baRowStride;  // ba 物理行距（连续时 = 2*nv）
    if (batch <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());

    // z/b/a 共用一个 TPipe 与同一对 MTE2/MTE3 事件（任务内顺序执行，无并发复用）。
    TPipe copyPipe;
    TEventID mte2ToMte3Event = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    TEventID mte3ToMte2Event = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();

    if (zWidth > 0) {
        GlobalTensor<T> xGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        GlobalTensor<T> zGm;
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z));

        // z：每行 zWidth 元素。host 已校验 qkvWidth / xRowStride 为 16 的倍数（bf16/fp16 下即行起始 32B 对齐、
        // 行字节数为 32 的倍数）且单行不超过单块 DataCopy 上限，故用普通 DataCopy（与 conv 内 DataCopy 同款）。
        TBuf<QuePosition::VECIN> zBuf;
        copyPipe.InitBuffer(zBuf, static_cast<uint32_t>(zWidth * sizeof(T)));
        LocalTensor<T> zLocal = zBuf.Get<T>();

        for (int64_t t = blockIdx; t < batch; t += blockNum) {
            DataCopy(zLocal, xGm[t * xRowStride + dim], static_cast<int32_t>(zWidth));
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            DataCopy(zGm[t * zWidth], zLocal, static_cast<int32_t>(zWidth));
            // 下一行复用 zLocal 前等待本次 MTE3 写出完成（行间串行，行数 ≤ ceil(batch/blockNum)，代价可忽略）
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        }
    }

    if (nv > 0) {
        // b/a：每行 nv 元素（bf16 下 8B），不足 32B 对齐，普通 DataCopy 不可用；
        // bf16 GM 标量读写（GlobalTensor GetValue/SetValue）有错值风险、勿用。
        // 走与 z 同族的 DataCopyPad 路径——GM→UB 带 pad 搬入、UB→GM 精确搬出（假数据自动丢弃）。
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
            const int64_t baRow = t * baRowStride;  // ba 物理行距（连续时 = 2*nv）
            const int64_t outRow = t * nv;
            DataCopyPad(bLocal, baGm[baRow], copyParams, padParams);            // b = ba[:, :nv]
            DataCopyPad(aLocal, baGm[baRow + nv], copyParams, padParams);       // a = ba[:, nv:2nv]
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event);
            DataCopyPad(bGm[outRow], bLocal, copyParams);
            DataCopyPad(aGm[outRow], aLocal, copyParams);
            // 下一行复用 bLocal/aLocal 前等待本次 MTE3 写出完成
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

    // 本算子 host 固定 run_mode=1(UPDATE/decode)，直接照抄 causal_conv1d 入口的 UPDATE 分支写法（dtypeKey 分派）
    if (dtypeKey == 0) {
        // 任务 1（conv）：与原 causal_conv1d UPDATE 路径完全一致（同一 device 类）；对象作用域限于本次调用，
        // 返回后 TPipe 随之析构，与下面的拷贝任务 TPipe 隔离。
        RunCausalConv1dUpdate<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState,
                                          numAcceptedTokens, y, userWorkspace, tilingData);
        // 任务 2（z/b/a 拷贝）：与 conv 任务数据相互独立
        FusedCopyZba<bfloat16_t>(x, ba, z, b, a, tilingData);
    } else {
        RunCausalConv1dUpdate<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState,
                                    numAcceptedTokens, y, userWorkspace, tilingData);
        FusedCopyZba<half>(x, ba, z, b, a, tilingData);
    }
}
