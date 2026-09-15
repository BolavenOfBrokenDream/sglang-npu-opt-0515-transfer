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
 * \file vendored_gmm.h
 * \brief vendored GMM (plan B: offsets/block-schedule-table front kernel +
 *        scan-skipping main kernel) host API declarations.
 *        Evaluation closed out at gmm1_vendored_gmm v1.2; since
 *        tp_ascendc_fusion_v3 it serves the production decode GMM1 (w13)
 *        path (sglang-side switch SGLANG_NPU_VGMM1, wired in moe_methods.py
 *        NPUUnquantMoEMethod.apply). WARNING: blocks gmm_vendored_v2 (sched
 *        multi-core rewrite measured as a regression and abandoned).
 *
 *   vgmm1_sched: offsets + block-schedule-table front kernel (single AIV
 *     core, DataCopy write-back).
 *     group_list int64[E] counts (production group_list_type=1 semantics) ->
 *     block_table int32[max_blocks*8] (entry layout: see
 *     vendored_gmm_tiling_data.h), row_offsets int32[E] (exclusive cumsum,
 *     debug use), total_blocks int32[8] ([0] valid).
 *     total_m = total row count of x (= sum of counts), used only to clamp
 *     the baseM/baseN upper bounds, identical to what vgmm1_main does
 *     internally (same VgmmComputeBaseTile(totalM)), guaranteeing both
 *     kernels block identically; base_m/base_n > 0 override explicitly
 *     (for driver cross-checking).
 *
 *   vgmm1_main: vendored GMM main kernel (AIC, MatmulImpl = verbatim replica
 *     of the stock GMM_FLOAT ND transB CUBE_ONLY path, with the 256-group
 *     scan loop replaced by direct table reads).
 *     x bf16 [totalM, K] ND contiguous; w bf16 [E, N, K] ND contiguous
 *     (production storage layout, equivalent to stock being called with
 *     weight=[w.transpose(1,2)]); returns y bf16 [totalM, N].
 *
 *   vgmm1_query_tile: pure host query (launches no kernel), returns CPU
 *     int32[6] = [baseM, baseN, baseK, stepKa, stepKb, aicNum] for the
 *     driver's table reference model.
 */

#ifndef VENDORED_GMM_H
#define VENDORED_GMM_H

#include <tuple>

#include <ATen/ATen.h>

#include "defines.h"

namespace sglang {
namespace npu_kernel {

HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor> vgmm1_sched_impl(
    const at::Tensor &group_list, int64_t total_m, int64_t n, int64_t k, int64_t max_blocks, int64_t base_m,
    int64_t base_n);

// C1 (tp_ascendc_fusion_v3.2, sched absorbs the cumsum): input changed to
// v22 rank_hist's partials int32[partNum, partStride]; the kernel
// column-reduces counts in-kernel, then builds the table exactly like
// vgmm1_sched_pre; returns (block_table, row_offsets, total_blocks, counts
// int64[E]) — the first three are bit-identical to vgmm1_sched, counts is
// bit-identical to v22 partials_cumsum (integer column reduction).
HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> vgmm1_sched_partial_impl(
    const at::Tensor &partials, int64_t num_experts, int64_t total_m, int64_t n, int64_t k, int64_t max_blocks,
    int64_t base_m, int64_t base_n);

HOST_API at::Tensor vgmm1_main_impl(const at::Tensor &x, const at::Tensor &w, const at::Tensor &block_table,
                                    const at::Tensor &total_blocks);

HOST_API at::Tensor vgmm1_query_tile_impl(int64_t m, int64_t k, int64_t n);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // VENDORED_GMM_H
