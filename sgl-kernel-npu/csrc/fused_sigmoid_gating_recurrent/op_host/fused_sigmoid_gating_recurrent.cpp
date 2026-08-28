// fused_sigmoid_gating_recurrent（GDN decode recurrent 的 AscendC 版）host 实现
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
 * \file fused_sigmoid_gating_recurrent.cpp
 * \brief fused_sigmoid_gating_recurrent host-side implementation
 *
 * 语义 = 生产 Triton kernel（sgl_kernel_npu fla fused_sigmoid_gating_recurrent
 * strided 形态）的 drop-in 替代：sigmoid gating + recurrent
 * delta rule update，仅 decode（每序列 1 token，T==N，varlen 必给 cu_seqlens）。
 * 无 tiling/workspace——按 recurrent_gated_delta_rule host 同款简式（EXEC_KERNEL_CMD
 * 直发，kernel 参数全按值/指针烘焙；graph capture 期 at::empty 输出进图私有池，
 * replay 复用——与生产 Triton wrapper 的 q.new_empty 同模式）。
 *
 * 特化锁死项（kernel 按 K=V=128 写死；越限在这里直接报错，wrapper 侧已先行回退
 * stock Triton——本 TORCH_CHECK 是双保险）：K==V==128、HV<=8、HV%H==0、N<=256、
 * q/k/v 仅 bf16、a/b 仅 bf16、A_log/dt_bias 仅 fp32、pool bf16 或 fp32。
 */

#include <algorithm>
#include <cmath>
#include "acl/acl.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_fused_sigmoid_gating_recurrent_bf16.h"
#include "aclrtlaunch_fused_sigmoid_gating_recurrent_fp32.h"
#include "defines.h"
#include "torch_helper.h"
#include "fused_sigmoid_gating_recurrent.h"

namespace sglang {
namespace npu_kernel {

constexpr int64_t FGR_HEAD_DIM = 128;  // kernel 特化的 K/V
constexpr int64_t FGR_MAX_HV = 8;      // kernel gating [8] pad 上限
constexpr int64_t FGR_MAX_N = 256;     // kernel cu/idx UB 驻留上限

HOST_API at::Tensor fused_sigmoid_gating_recurrent_impl(
    const at::Tensor &A_log, const at::Tensor &a, const at::Tensor &dt_bias, double softplus_beta,
    double softplus_threshold, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &b,
    const at::Tensor &initial_state_source, const at::Tensor &initial_state_indices, double scale,
    const at::Tensor &cu_seqlens, bool use_qk_l2norm, int64_t q_row_stride, int64_t k_row_stride,
    int64_t v_row_stride)
{
    TORCH_CHECK(q.defined() && k.defined() && v.defined(), "q/k/v must be defined");
    TORCH_CHECK(a.defined() && b.defined(), "a/b must be defined");
    TORCH_CHECK(A_log.defined() && dt_bias.defined(), "A_log/dt_bias must be defined");
    TORCH_CHECK(initial_state_source.defined() && initial_state_indices.defined() && cu_seqlens.defined(),
                "initial_state_source/initial_state_indices/cu_seqlens must be defined "
                "(decode-only op: cu_seqlens is required)");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D [1, T, H, K] / [1, T, HV, V]");
    const int64_t T = q.size(1);
    const int64_t H = q.size(2);
    const int64_t K = q.size(3);
    const int64_t HV = v.size(2);
    const int64_t V = v.size(3);
    const int64_t N = cu_seqlens.size(0) - 1;

    TORCH_CHECK(q.size(0) == 1 && v.size(0) == 1, "decode requires flattened batch dim B == 1");
    TORCH_CHECK(k.size(1) == T && v.size(1) == T, "q/k/v token dim mismatch");
    TORCH_CHECK(k.size(2) == H && k.size(3) == K, "q/k head dims mismatch");
    TORCH_CHECK(T == N && N >= 1,
                "decode-only: total tokens must equal sequence count (one token per sequence), got T=", T, " N=", N);
    TORCH_CHECK(N <= FGR_MAX_N, "N exceeds kernel UB residency cap (256): ", N);
    TORCH_CHECK(K == FGR_HEAD_DIM && V == FGR_HEAD_DIM, "kernel specialized for K == V == 128, got K=", K, " V=", V);
    TORCH_CHECK(H >= 1 && HV >= 1 && HV <= FGR_MAX_HV && HV % H == 0,
                "require 1 <= H <= HV <= 8 and HV % H == 0, got H=", H, " HV=", HV);

    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "q/k/v must be bf16");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16, "a/b must be bf16");
    TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat,
                "A_log/dt_bias must be fp32");
    const at::ScalarType poolDtype = initial_state_source.scalar_type();
    TORCH_CHECK(poolDtype == at::kBFloat16 || poolDtype == at::kFloat, "ssm pool must be bf16 or fp32");

    // q/k/v 允许 strided 视图，仅要求末维连续且头维按 K/V 紧凑
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1, "q/k/v last dim must be contiguous");
    TORCH_CHECK(q.stride(2) == K && k.stride(2) == K, "q/k head dim must be packed (stride(2) == K)");
    TORCH_CHECK(v.stride(2) == V, "v head dim must be packed (stride(2) == V)");
    TORCH_CHECK(q_row_stride > 0 && k_row_stride > 0 && v_row_stride > 0, "row strides must be positive");

    TORCH_CHECK(a.dim() == 2 && a.size(0) == T && a.size(1) == HV && a.is_contiguous(),
                "a must be contiguous [T, HV]");
    TORCH_CHECK(b.dim() == 2 && b.size(0) == T && b.size(1) == HV && b.is_contiguous(),
                "b must be contiguous [T, HV]");
    TORCH_CHECK(A_log.numel() == HV && A_log.is_contiguous(), "A_log must be contiguous [HV]");
    TORCH_CHECK(dt_bias.numel() == HV && dt_bias.is_contiguous(), "dt_bias must be contiguous [HV]");

    TORCH_CHECK(initial_state_source.dim() == 4 && initial_state_source.size(1) == HV &&
                    initial_state_source.size(2) == K && initial_state_source.size(3) == V &&
                    initial_state_source.is_contiguous(),
                "ssm pool must be contiguous [slots, HV, K, V]（非连续 pool 若拷贝副本会静默丢更新，硬断言）");
    TORCH_CHECK(initial_state_indices.numel() >= N, "initial_state_indices must have at least N elements");
    TORCH_CHECK(cu_seqlens.numel() == N + 1, "cu_seqlens must have N+1 elements");

    // int64 索引条件性转 int32（生产为 int32；不转则是静默拷贝风险点，宁可显式）
    at::Tensor idxI32 = initial_state_indices.scalar_type() == at::kInt
                            ? initial_state_indices
                            : initial_state_indices.to(at::kInt);
    at::Tensor cuI32 = cu_seqlens.scalar_type() == at::kInt ? cu_seqlens : cu_seqlens.to(at::kInt);
    TORCH_CHECK(idxI32.is_contiguous() && cuI32.is_contiguous(), "indices/cu_seqlens must be contiguous");

    TORCH_CHECK(scale > 0, "scale must be positive");
    TORCH_CHECK(softplus_beta != 0, "softplus_beta must be non-zero");

    // 输出与 Triton wrapper 同形同 dtype（o = q.new_empty(N,HV,V).view(v.shape)：
    // 连续 [1, T, HV, V]，T==N 元素一一对应）
    at::Tensor o = at::empty(v.sizes(), q.options());

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t totalItems = N * HV;
    uint32_t blockDim = static_cast<uint32_t>(
        std::min<int64_t>(totalItems, static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv())));
    if (blockDim == 0) {
        blockDim = 1;
    }

    int devidx = q.device().index();
    c10_npu::set_device(devidx);

    const float scaleF = static_cast<float>(scale);
    const float spbF = static_cast<float>(softplus_beta);
    const float invSpbF = 1.0f / spbF;  // 与 Triton kernel 内 1.0/softplus_beta 同为 fp32 除法
    const float thrF = static_cast<float>(softplus_threshold);
    const uint32_t useL2 = use_qk_l2norm ? 1U : 0U;
    // EXEC_KERNEL_CMD 的 ConvertTypes(Ts&...) 只收左值，标量全部先落具名局部变量
    // （直接传 static_cast 右值会编译失败）
    const uint32_t nU32 = static_cast<uint32_t>(N);
    const uint32_t hU32 = static_cast<uint32_t>(H);
    const uint32_t hvU32 = static_cast<uint32_t>(HV);
    const uint32_t qRowStrideU32 = static_cast<uint32_t>(q_row_stride);
    const uint32_t kRowStrideU32 = static_cast<uint32_t>(k_row_stride);
    const uint32_t vRowStrideU32 = static_cast<uint32_t>(v_row_stride);

    if (poolDtype == at::kBFloat16) {
        EXEC_KERNEL_CMD(fused_sigmoid_gating_recurrent_bf16, blockDim, A_log, a, dt_bias, q, k, v, b, o,
                        initial_state_source, idxI32, cuI32, nU32, hU32, hvU32, qRowStrideU32, kRowStrideU32,
                        vRowStrideU32, scaleF, spbF, invSpbF, thrF, useL2);
    } else {
        EXEC_KERNEL_CMD(fused_sigmoid_gating_recurrent_fp32, blockDim, A_log, a, dt_bias, q, k, v, b, o,
                        initial_state_source, idxI32, cuI32, nU32, hU32, hvU32, qRowStrideU32, kRowStrideU32,
                        vRowStrideU32, scaleF, spbF, invSpbF, thrF, useL2);
    }

    return o;
}

}  // namespace npu_kernel
}  // namespace sglang
