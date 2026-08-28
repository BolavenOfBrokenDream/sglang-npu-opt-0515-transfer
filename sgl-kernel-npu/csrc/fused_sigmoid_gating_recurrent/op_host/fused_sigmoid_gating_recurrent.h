// fused_sigmoid_gating_recurrent（GDN decode recurrent 的 AscendC 版）host 声明
/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_sigmoid_gating_recurrent.h
 * \brief fused_sigmoid_gating_recurrent host-side function declaration
 */

#ifndef CUSTOM_FUSED_SIGMOID_GATING_RECURRENT_HOST_H_
#define CUSTOM_FUSED_SIGMOID_GATING_RECURRENT_HOST_H_

#include <ATen/ATen.h>
#include "defines.h"

namespace sglang {
namespace npu_kernel {

// GDN decode recurrent（sigmoid gating + delta rule update）AscendC AIV 版。
// 生产 Triton kernel（sgl_kernel_npu fla fused_sigmoid_gating_recurrent strided
// 形态）的 drop-in 替代；仅 decode（T==N，每序列 1 token，varlen 必须
// 给 cu_seqlens）。initial_state_source = ssm state pool [slots, HV, K, V]，原地更新。
// q/k/v 允许末维连续的 strided 视图（行 stride 以元素为单位显式传入，连续时
// q_row_stride == H*K / v_row_stride == HV*V）。返回 o，形状与 v 相同（[1, T, HV, V]）。
HOST_API at::Tensor fused_sigmoid_gating_recurrent_impl(
    const at::Tensor &A_log, const at::Tensor &a, const at::Tensor &dt_bias, double softplus_beta,
    double softplus_threshold, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &b,
    const at::Tensor &initial_state_source, const at::Tensor &initial_state_indices, double scale,
    const at::Tensor &cu_seqlens, bool use_qk_l2norm, int64_t q_row_stride, int64_t k_row_stride,
    int64_t v_row_stride);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CUSTOM_FUSED_SIGMOID_GATING_RECURRENT_HOST_H_
