// fused_qkvzba_conv1d (GDN decode: fused split + causal_conv1d) host declaration
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
 * \file fused_qkvzba_conv1d.h
 * \brief fused_qkvzba_conv1d host-side function declaration
 */

#ifndef CUSTOM_FUSED_QKVZBA_CONV1D_HOST_H_
#define CUSTOM_FUSED_QKVZBA_CONV1D_HOST_H_

#include <ATen/ATen.h>
#include <tuple>
#include "defines.h"

namespace sglang {
namespace npu_kernel {

// Returns (y, z, b, a): y=[B, qkvWidth] (conv output), z=[B, num_v_heads, head_v_dim], b/a=[B, num_v_heads]
HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_qkvzba_conv1d_impl(
    const at::Tensor &qkvz, const at::Tensor &weight, const at::Tensor &conv_states, const at::Tensor &mixed_ba,
    int64_t num_k_heads, int64_t num_v_heads, int64_t head_k_dim, int64_t head_v_dim, const at::Tensor &bias,
    const at::Tensor &query_start_loc, const at::Tensor &cache_indices, int64_t activation_mode, int64_t pad_slot_id);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CUSTOM_FUSED_QKVZBA_CONV1D_HOST_H_
