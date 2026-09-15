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
 * \file fused_tail.h
 * \brief fused_tail (fin+add+AR+norm layer-tail fusion, tp_ascendc_fusion_v4.1)
 *        host-side function declarations. The three ops' semantics: see the
 *        header comment of op_kernel/fused_tail_kernel_lib.h and this
 *        directory's REGISTRATION.md.
 */

#ifndef CUSTOM_FUSED_TAIL_HOST_H_
#define CUSTOM_FUSED_TAIL_HOST_H_

#include <ATen/ATen.h>
#include "defines.h"

namespace sglang {
namespace npu_kernel {

// fin(+skip1) + spin AIV AR + residual add + gemma rmsnorm single kernel.
// addr_tab int64[>=18]: [0..7]=per-rank data symmem VA, [8..15]=flag VA,
// [17]=cell area VA. [v4.1] eri (int32[M*K], dummy when use_eri=0) and
// norm_w (bf16[H]) are now passed as tensor parameters (v4's tiling VA
// scalar scheme made the tiling hash contain VAs, with the copy chain
// entering the graph per layer); scales accepts fp32 or bf16 (bf16 widened
// exactly in-kernel).
HOST_API void fused_fin_ar_norm_impl(
    const at::Tensor &x, const at::Tensor &scales, const at::Tensor &skip1, const at::Tensor &residual,
    at::Tensor &add_out, at::Tensor &norm_out, const at::Tensor &addr_tab,
    const at::Tensor &eri, const at::Tensor &norm_w,
    int64_t m, int64_t h, int64_t k, int64_t ncores, int64_t rank, int64_t world,
    int64_t has_skip1, int64_t use_eri, double eps, int64_t cycle_limit_us,
    int64_t slot_stride, int64_t ring_stride, int64_t max_tiles,
    int64_t counter_offset, int64_t dfx_offset);

// fin(+skip1) local front stage (first half of the stock HCCL AR route; no
// symmem).
// [v4.1] eri is now a tensor parameter (dummy when use_eri=0); scales
// accepts fp32/bf16.
HOST_API void fused_fin_add_impl(
    const at::Tensor &x, const at::Tensor &scales, const at::Tensor &skip1, at::Tensor &out,
    const at::Tensor &eri, int64_t m, int64_t h, int64_t k, int64_t ncores,
    int64_t has_skip1, int64_t use_eri);

// In-kernel MTE3 zeroing of a local GM region (dedicated to flag symmem
// reset; torch zero_() forbidden, see the kernel file's header comment).
HOST_API void fused_tail_zero_impl(at::Tensor &x, int64_t nbytes, int64_t ncores);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CUSTOM_FUSED_TAIL_HOST_H_
