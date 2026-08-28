/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_data.h
 */

#ifndef CUSTOM_CAUSAL_CONV1D_TILING_DATA_H_
#define CUSTOM_CAUSAL_CONV1D_TILING_DATA_H_

#include <cstdint>

enum FnExecutionPlan : int64_t {
    FN_EXECUTION_PLAN_INVALID = 0,
    FN_EXECUTION_PLAN_CUTBS = 1,
    FN_EXECUTION_PLAN_CUTBSD = 2,
};

inline constexpr int64_t ResolveFnExecutionPlan(int64_t baseDimCnt)
{
    return (baseDimCnt <= 0)   ? FN_EXECUTION_PLAN_INVALID
           : (baseDimCnt <= 1) ? FN_EXECUTION_PLAN_CUTBS
                               : FN_EXECUTION_PLAN_CUTBSD;
}

struct CausalConv1dTilingData {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t inputMode;

    int64_t width;

    int64_t stateLen;
    int64_t numCacheLines;
    int64_t batch;
    int64_t activationMode;
    int64_t padSlotId;
    int64_t hasBias;
    int64_t baseDim;
    int64_t baseDimCnt;
    int64_t hasNumAcceptedTokens;
    int64_t hasCacheIndices;
    int64_t hasInitialStateMode;
    int64_t tokenBlockSize;
    int64_t tokenBlockCnt;
    int64_t hasExplicitTokenSeqRanges;
    int64_t explicitTokenSeqRangeCount;
    int64_t tokenTileStartSeq[128];
    int64_t tokenTileEndSeq[128];
    int64_t hasInitStateWorkspace;

    int64_t dtypeKey;
    int64_t runModeKey;
    int64_t widthKey;
    int64_t fnPlanKey;

    int64_t xRowStride;  // x 物理行距（元素数）；host 侧旧算子恒填=dim
    int64_t numVHeads;   // 供 fused_qkvzba_conv1d 的 b/a 拷贝用；旧算子恒 0（memset），不读取
    // fused_qkvzba_conv1d 支持行距视图输入（打包 GEMM 的非连续输出直读）：
    // zWidth 显式携带、不由 xRowStride-dim 反推；baRowStride 替代硬编码 2*numVHeads。
    // 旧算子恒 0（memset），不读取。
    int64_t zWidth;
    int64_t baRowStride;
};
#endif  // CUSTOM_CAUSAL_CONV1D_TILING_DATA_H_
