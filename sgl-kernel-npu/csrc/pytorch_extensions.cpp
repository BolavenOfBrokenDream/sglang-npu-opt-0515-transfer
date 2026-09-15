// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "version.h"

#include "torch_helper.h"
#include "sgl_kenel_npu_ops.h"
#include "causal_conv1d_update/op_host/causal_conv1d_update.h"
#include "causal_conv1d/op_host/causal_conv1d.h"
#include "fused_qkvzba_conv1d/op_host/fused_qkvzba_conv1d.h"
#include "fused_sigmoid_gating_recurrent/op_host/fused_sigmoid_gating_recurrent.h"
// VGMM: GMM1 vendored GMM v1.2 (production decode w13 path, SGLANG_NPU_VGMM1)
#include "vendored_gmm/op_host/vendored_gmm.h"
// FUSED_TAIL: MoE layer-tail fin+add+AR+norm fusion
#include "fused_tail/op_host/fused_tail.h"

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sgl_kernel_npu_print_version() -> ()", []() { printf("%s\n", LIB_VERSION_FULL); });
    m.def("sgl_kernel_npu_version() -> str", []() { return std::string("") + LIB_VERSION; });

    m.def("helloworld(Tensor x, Tensor y) -> Tensor");

    m.def(
        "alloc_extend(Tensor pre_lens, Tensor seq_lens, Tensor last_loc, Tensor free_pages, int page_size, "
        "Tensor(a!) out_indices, Tensor(b!) values) -> ()");

    m.def(
        "cache_loc_assign(Tensor req_indices, Tensor token_pool, Tensor start_offset, Tensor end_offset, Tensor "
        "out_cache_loc, int max_step) -> Tensor");

    m.def(
        "cache_loc_update(Tensor req_indices, Tensor token_pool, Tensor start_offset, Tensor end_offset, Tensor "
        "out_cache_loc, int max_step) -> Tensor");

    m.def(
        "assign_cache_op(Tensor! out, Tensor src, Tensor dst_start_idx, Tensor dst_end_idx, Tensor src_start_idx, "
        "Tensor src_end_idx) -> bool");

    m.def(
        "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
        "Tensor tree_mask, Tensor positions, Tensor retrive_index, Tensor retrive_next_token, "
        "Tensor retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode)->()");

    m.def(
        "mla_preprocess(Tensor hiddenState, Tensor gamma0, Tensor beta0, Tensor wdqkv, "
        "Tensor descale0, Tensor gamma1, Tensor beta1, Tensor wuq, "
        "Tensor descale1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk,"
        "Tensor kv_cache, Tensor kv_cache_rope, Tensor slotmapping, "
        "Tensor quant_scale0, Tensor quant_offset0, Tensor bias0, "
        "Tensor quant_scale1, Tensor quant_offset1, Tensor bias1, *, "
        "Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, "
        "str? cache_mode=None, str? quant_mode=None, "
        "Tensor(a!) q_out0, Tensor(b!) kv_cache_out0, Tensor(c!) q_out1, Tensor(d!) kv_cache_out1) "
        "-> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");

    m.def(
        "batch_matmul_transpose(Tensor tensor_a, Tensor tensor_b, Tensor(a!) tensor_c, "
        "str? format_mode=None, str? quant_mode=None) -> ()");

    m.def(
        "transfer_kv_dim_exchange(Tensor device_k, Tensor host_k, "
        "Tensor device_v, Tensor host_v, "
        "Tensor device_indices, Tensor host_indices, int page_size, int direct, int flags) -> ()");

    m.def(
        "bgmv_expand(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");

    m.def(
        "bgmv_shrink(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y,"
        "            float scale) -> ()");

    m.def(
        "sgmv_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");

    m.def(
        "sgmv_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y,"
        " float scale) -> ()");

    m.def(
        "sgemmv_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! lora_ranks,"
        "              Tensor! sliceOffsets, Tensor! y) -> Tensor");

    m.def(
        "sgemmv_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! lora_ranks,"
        "              Tensor! lora_scales, Tensor! y) -> ()");

    m.def(
        "recurrent_gated_delta_rule(Tensor mix_qkv, Tensor(a!) recurrent_state, Tensor beta, "
        "float scale, Tensor actual_seq_lengths, Tensor ssm_state_indices, "
        "int nk, int nv, "
        "Tensor(b!)? intermediate_state=None, Tensor? cache_indices=None, "
        "Tensor? num_accepted_tokens=None, Tensor? g=None, Tensor? gk=None) -> Tensor");

    m.def(
        "sgemmc_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! lora_ranks,"
        "              Tensor! sliceOffsets, Tensor! y) -> Tensor");

    m.def(
        "sgemmc_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! lora_ranks,"
        "              Tensor! lora_scales, Tensor! y, int slice_count) -> ()");

    m.def(
        "mega_chunk_gdn(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "Tensor mask_lower, Tensor mask_full, Tensor minus_identity, Tensor cu_seqlens, "
        "Tensor(a!) out, Tensor(b!) g_sum, Tensor(c!) g_t, Tensor(d!) beta_t, "
        "Tensor(e!) A, Tensor(f!) A_inv_f32, Tensor(g!) A_inv, Tensor(h!) w, "
        "Tensor(i!) u, Tensor(j!) s, Tensor(k!) v_new, Tensor(l!) final_state, "
        "Tensor initial_state, bool has_initial_state, "
        "Tensor(m!) kkt_workspace, Tensor(n!) wy_workspace_a1, "
        "Tensor(o!) wy_workspace_a2, Tensor(p!) h_workspace, "
        "Tensor(q!) o_workspace_qk, Tensor(r!) o_workspace_qs, "
        "Tensor(s!) o_workspace_gated, int block_dim, int batch_size, "
        "int seq_len, int total_tokens, int num_matrices) -> ()");

#ifdef BUILD_CATLASS_MODULE
    m.def("catlass_matmul_basic(Tensor tensor_a, Tensor tensor_b, Tensor(a!) tensor_c, str? format_mode=None) -> ()");

    m.def("softfp8_w8a16_matmul(Tensor mat1, Tensor mat2, Tensor scale, str c) -> Tensor");

    m.def("softfp8_w8a16_grouped_matmul(Tensor mat1, Tensor mat2, Tensor scale, Tensor groupList, str c) -> Tensor");
#endif

    m.def(
        "lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor? actual_seq_lengths_query=None, "
        "Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, "
        "str? layout_query=None, str? layout_key=None, "
        "int? sparse_count=None, int? sparse_mode=None) -> Tensor");

    m.def("apply_token_bitmask(Tensor logits, Tensor bitmask, Tensor? indices=None) -> Tensor");
    m.def("triangular_inverse(Tensor x) -> Tensor");

    m.def(
        "causal_conv1d_update(Tensor x, Tensor weight, Tensor(a!) conv_state, "
        "Tensor conv_state_indices, Tensor? bias=None, Tensor? num_accepted_tokens=None, "
        "Tensor? query_start_loc=None, bool activation_mode=False, int pad_slot_id=-1) -> Tensor");

    m.def(
        "causal_conv1d(Tensor x, Tensor weight, Tensor conv_states, Tensor? bias=None, "
        "Tensor? query_start_loc=None, Tensor? cache_indices=None, Tensor? has_initial_state=None, "
        "Tensor? num_accepted_tokens=None, int activation_mode=0, int pad_slot_id=-1, "
        "int run_mode=0) -> Tensor");

    // GDN decode split + causal_conv1d in a single kernel
    m.def(
        "fused_qkvzba_conv1d(Tensor qkvz, Tensor weight, Tensor conv_states, Tensor mixed_ba, "
        "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim, Tensor? bias=None, "
        "Tensor? query_start_loc=None, Tensor? cache_indices=None, int activation_mode=0, "
        "int pad_slot_id=-1) -> (Tensor, Tensor, Tensor, Tensor)");

    // AscendC AIV version of the GDN decode recurrent (sigmoid gating + delta
    // rule update), decode only (T==N, one token per sequence);
    // initial_state_source is the ssm state pool and is updated in place;
    // q/k/v may be strided views with a contiguous last dim (row strides are
    // passed explicitly).
    m.def(
        "fused_sigmoid_gating_recurrent(Tensor A_log, Tensor a, Tensor dt_bias, "
        "float softplus_beta, float softplus_threshold, "
        "Tensor q, Tensor k, Tensor v, Tensor b, "
        "Tensor(a!) initial_state_source, Tensor initial_state_indices, "
        "float scale, Tensor cu_seqlens, bool use_qk_l2norm, "
        "int q_row_stride, int k_row_stride, int v_row_stride) -> Tensor");

    // MoE layer-tail fin+add+AR+norm fusion (three ops):
    // fused_fin_ar_norm = spin AIV AR single-kernel variant (addr_tab
    // int64[>=18] carries symmem/cell VAs; v4.1: eri/norm_w are tensor
    // parameters — v4's tiling VAs put addresses in the tiling hash and
    // dragged a per-layer copy chain into the graph; scales accepts
    // fp32/bf16);
    // fused_fin_add = fin(+skip1) local front stage (stock HCCL AR variant,
    // no symmem); fused_tail_zero = flag symmem MTE3 clearing (init/reset).
    m.def(
        "fused_fin_ar_norm(Tensor x, Tensor scales, Tensor skip1, Tensor residual, "
        "Tensor(a!) add_out, Tensor(b!) norm_out, Tensor addr_tab, "
        "Tensor eri, Tensor norm_w, "
        "int m, int h, int k, int ncores, int rank, int world, "
        "int has_skip1, int use_eri, float eps, int cycle_limit_us, "
        "int slot_stride, int ring_stride, int max_tiles, "
        "int counter_offset, int dfx_offset) -> ()");
    m.def(
        "fused_fin_add(Tensor x, Tensor scales, Tensor skip1, Tensor(a!) out, "
        "Tensor eri, int m, int h, int k, int ncores, int has_skip1, int use_eri) -> ()");
    m.def("fused_tail_zero(Tensor(a!) x, int nbytes, int ncores) -> ()");

    // VGMM: GMM1 vendored GMM v1.2 (gmm1_vendored_gmm; production decode w13
    // path, switch SGLANG_NPU_VGMM1)
    m.def(
        "vgmm1_sched(Tensor group_list, int total_m, int n, int k, int max_blocks, int base_m=-1, "
        "int base_n=-1) -> (Tensor, Tensor, Tensor)");
    m.def("vgmm1_main(Tensor x, Tensor w, Tensor block_table, Tensor total_blocks) -> Tensor");
    m.def("vgmm1_query_tile(int m, int k, int n) -> Tensor");
    // C1: sched absorbs the cumsum — input = v22 rank_hist partials
    // int32[partNum, partStride], output = (block_table, row_offsets,
    // total_blocks, counts int64[E]); the Triton partials_cumsum node is
    // retired from the chain.
    m.def(
        "vgmm1_sched_partial(Tensor partials, int num_experts, int total_m, int n, int k, int max_blocks, "
        "int base_m=-1, int base_n=-1) -> (Tensor, Tensor, Tensor, Tensor)");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("helloworld", TORCH_FN(sglang::npu_kernel::helloworld));

    m.impl("cache_loc_assign", TORCH_FN(sglang::npu_kernel::cache_loc_assign));

    m.impl("cache_loc_update", TORCH_FN(sglang::npu_kernel::cache_loc_update));

    m.impl("assign_cache_op", TORCH_FN(sglang::npu_kernel::assign_cache_op));

    m.impl("alloc_extend", TORCH_FN(sglang::npu_kernel::alloc_extend));

    m.impl("build_tree_kernel_efficient", TORCH_FN(sglang::npu_kernel::build_tree_efficient));

    m.impl("mla_preprocess", TORCH_FN(sglang::npu_kernel::mla_preprocess));

    m.impl("batch_matmul_transpose", TORCH_FN(sglang::npu_kernel::batch_matmul_transpose));

    m.impl("transfer_kv_dim_exchange", TORCH_FN(sglang::npu_kernel::transfer_kv_dim_exchange));

    m.impl("bgmv_expand", TORCH_FN(sglang::npu_kernel::bgmv_expand));

    m.impl("bgmv_shrink", TORCH_FN(sglang::npu_kernel::bgmv_shrink));

    m.impl("sgmv_expand", TORCH_FN(sglang::npu_kernel::sgmv_expand));

    m.impl("sgmv_shrink", TORCH_FN(sglang::npu_kernel::sgmv_shrink));

    m.impl("sgemmv_expand", TORCH_FN(sglang::npu_kernel::sgemmv_expand));

    m.impl("sgemmv_shrink", TORCH_FN(sglang::npu_kernel::sgemmv_shrink));

    m.impl("recurrent_gated_delta_rule", TORCH_FN(sglang::npu_kernel::recurrent_gated_delta_rule));

    m.impl("sgemmc_expand", TORCH_FN(sglang::npu_kernel::sgemmc_expand));

    m.impl("sgemmc_shrink", TORCH_FN(sglang::npu_kernel::sgemmc_shrink));

    m.impl("mega_chunk_gdn", TORCH_FN(sglang::npu_kernel::mega_chunk_gdn));

#ifdef BUILD_CATLASS_MODULE
    m.impl("catlass_matmul_basic", TORCH_FN(sglang::npu_kernel::catlass_matmul_basic));

    m.impl("softfp8_w8a16_matmul", TORCH_FN(sglang::npu_kernel::softfp8_w8a16_matmul));

    m.impl("softfp8_w8a16_grouped_matmul", TORCH_FN(sglang::npu_kernel::softfp8_w8a16_grouped_matmul));
#endif

    m.impl("lightning_indexer", TORCH_FN(sglang::npu_kernel::lightning_indexer));

    m.impl("triangular_inverse", TORCH_FN(sglang::npu_kernel::tri_inv_col_sweep));

    m.impl("apply_token_bitmask", [](at::Tensor logits, at::Tensor bitmask, const c10::optional<at::Tensor> &indices) {
        auto indices_or_empty = indices.has_value() ? *indices : at::empty({0}, logits.options().dtype(at::kInt));
        return sglang::npu_kernel::apply_token_bitmask(logits, bitmask, indices_or_empty);
    });

    m.impl("causal_conv1d_update",
           [](const at::Tensor &x, const at::Tensor &weight, const at::Tensor &conv_state,
              const at::Tensor &conv_state_indices, const c10::optional<at::Tensor> &bias,
              const c10::optional<at::Tensor> &num_accepted_tokens, const c10::optional<at::Tensor> &query_start_loc,
              bool activation_mode, int64_t pad_slot_id) {
               // Handle optional parameters - convert None to empty tensors
               auto bias_or_empty = bias.has_value() ? *bias : at::empty({0}, x.options());
               auto num_accepted_or_empty =
                   num_accepted_tokens.has_value() ? *num_accepted_tokens : at::empty({0}, x.options().dtype(at::kInt));
               auto query_loc_or_empty =
                   query_start_loc.has_value() ? *query_start_loc : at::empty({0}, x.options().dtype(at::kInt));

               return sglang::npu_kernel::causal_conv1d_update_impl(x, weight, conv_state, conv_state_indices,
                                                                    bias_or_empty, num_accepted_or_empty,
                                                                    query_loc_or_empty, activation_mode, pad_slot_id);
           });

    m.impl("causal_conv1d", [](const at::Tensor &x, const at::Tensor &weight, const at::Tensor &conv_states,
                               const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &query_start_loc,
                               const c10::optional<at::Tensor> &cache_indices,
                               const c10::optional<at::Tensor> &has_initial_state,
                               const c10::optional<at::Tensor> &num_accepted_tokens, int64_t activation_mode,
                               int64_t pad_slot_id, int64_t run_mode) {
        // Handle optional parameters - convert None to empty tensors
        auto bias_or_empty = bias.has_value() ? *bias : at::empty({0}, x.options());
        auto query_start_loc_or_empty =
            query_start_loc.has_value() ? *query_start_loc : at::empty({0}, x.options().dtype(at::kLong));
        auto cache_indices_or_empty =
            cache_indices.has_value() ? *cache_indices : at::empty({0}, x.options().dtype(at::kLong));
        auto has_initial_state_or_empty =
            has_initial_state.has_value() ? *has_initial_state : at::empty({0}, x.options().dtype(at::kLong));
        auto num_accepted_tokens_or_empty =
            num_accepted_tokens.has_value() ? *num_accepted_tokens : at::empty({0}, x.options().dtype(at::kLong));

        return sglang::npu_kernel::causal_conv1d_impl(
            x, weight, bias_or_empty, conv_states, query_start_loc_or_empty, cache_indices_or_empty,
            has_initial_state_or_empty, num_accepted_tokens_or_empty, activation_mode, pad_slot_id, run_mode);
    });

    m.impl("fused_qkvzba_conv1d",
           [](const at::Tensor &qkvz, const at::Tensor &weight, const at::Tensor &conv_states,
              const at::Tensor &mixed_ba, int64_t num_k_heads, int64_t num_v_heads, int64_t head_k_dim,
              int64_t head_v_dim, const c10::optional<at::Tensor> &bias,
              const c10::optional<at::Tensor> &query_start_loc, const c10::optional<at::Tensor> &cache_indices,
              int64_t activation_mode, int64_t pad_slot_id) {
               // Handle optional parameters - convert None to empty tensors
               auto bias_or_empty = bias.has_value() ? *bias : at::empty({0}, qkvz.options());
               auto query_start_loc_or_empty =
                   query_start_loc.has_value() ? *query_start_loc : at::empty({0}, qkvz.options().dtype(at::kLong));
               auto cache_indices_or_empty =
                   cache_indices.has_value() ? *cache_indices : at::empty({0}, qkvz.options().dtype(at::kLong));

               return sglang::npu_kernel::fused_qkvzba_conv1d_impl(
                   qkvz, weight, conv_states, mixed_ba, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                   bias_or_empty, query_start_loc_or_empty, cache_indices_or_empty, activation_mode, pad_slot_id);
           });

    m.impl("fused_sigmoid_gating_recurrent", TORCH_FN(sglang::npu_kernel::fused_sigmoid_gating_recurrent_impl));

    // MoE layer-tail fin+add+AR+norm fusion
    m.impl("fused_fin_ar_norm", TORCH_FN(sglang::npu_kernel::fused_fin_ar_norm_impl));
    m.impl("fused_fin_add", TORCH_FN(sglang::npu_kernel::fused_fin_add_impl));
    m.impl("fused_tail_zero", TORCH_FN(sglang::npu_kernel::fused_tail_zero_impl));

    // VGMM: GMM1 vendored GMM v1.2 (production decode w13 path)
    m.impl("vgmm1_sched", TORCH_FN(sglang::npu_kernel::vgmm1_sched_impl));
    m.impl("vgmm1_main", TORCH_FN(sglang::npu_kernel::vgmm1_main_impl));
    // C1: sched absorbs the cumsum, consuming v22 rank_hist partials
    m.impl("vgmm1_sched_partial", TORCH_FN(sglang::npu_kernel::vgmm1_sched_partial_impl));
}
}  // namespace

namespace {
// vgmm1_query_tile takes three ints and no Tensor: the dispatcher cannot
// compute a backend key and a PrivateUse1 kernel would never route (it would
// report "no fallback function is registered"). It is a pure host query with
// no autograd needs, so register it under the CompositeExplicitAutograd
// catchall (same default landing spot as an inline m.def lambda — reachable
// even without Tensor arguments).
TORCH_LIBRARY_IMPL(npu, CompositeExplicitAutograd, m)
{
    m.impl("vgmm1_query_tile", TORCH_FN(sglang::npu_kernel::vgmm1_query_tile_impl));
}
}
}  // namespace
