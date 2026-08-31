# Decode-optimized GDN recurrent state update for Ascend NPU.
#
# Based on sgl-kernel-npu PR #740's
# ``fla/fused_sigmoid_gating_recurrent_decode_optimized.py``, with strided
# q/k/v support added (``q/k/v_row_stride`` addressing + contiguous-free
# wrapper). The PR original addresses the packed layout ``(bos*H+i_h)*K`` and
# its wrapper calls ``.contiguous()`` on non-contiguous inputs — with fused
# split producing strided views, that costs 3 extra copies per layer.
#
# Optimizations (from PR #740): 1-D grid capped at the AIV vector core count,
# per-program loop over (sequence, value-head) tiles, BV=64 blocking over V
# with gating computed once per (seq, v-head), num_warps=4, and a lightweight
# contiguity check instead of the generic input_guard.
#
# Limitations: decode only (T=1 per sequence; the wrapper asserts
# total_tokens == N). NK>1 is unsupported. The target_verify extras
# (disable_state_update etc.) are unsupported; do not use for verify.

from typing import Optional

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties


def _maybe_contiguous(x):
    """Return ``x`` unchanged when already contiguous, else a contiguous copy."""
    return x if x.is_contiguous() else x.contiguous()


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "N", "NHV"])
def _fused_sigmoid_gating_delta_rule_update_decode_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    scale,
    T,
    N,
    NHV,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BHV: tl.constexpr,
    OVERSUB: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """
    Decode-optimized fused gating + recurrent delta rule update.

    Grid: (num_programs,) where num_programs = min(N*NHV, OVERSUB*num_vectorcore).
    Each program loops over its assigned (sequence, value-head) tiles with
    stride num_programs, processing the value dimension in BV=64 blocks and
    reusing the gating computation across those blocks.

    q/k/v are addressed by token row stride: for contiguous inputs
    q_row_stride == H*K / v_row_stride == HV*V, identical to the PR original.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_tiles = N * NHV

    o_k = tl.arange(0, BK)
    o_base_v = tl.arange(0, BV)
    mask_k = o_k < K

    NV = tl.cdiv(V, BV)

    for g_idx in tl.range(pid, total_tiles, num_programs):
        i_n = g_idx // NHV
        i_nhv = g_idx % NHV

        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            t_len = (eos - bos).to(tl.int32)
        else:
            bos = i_n * T
            t_len = T

        for i_bhv in tl.static_range(0, BHV):
            i_hv = i_nhv * BHV + i_bhv
            i_h = i_hv // (HV // H)

            # Gating: compute once per (sequence, value-head).
            b_A_log = tl.load(A_log + i_hv).to(tl.float32)
            b_a = tl.load(a + bos * HV + i_hv).to(tl.float32)
            b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
            b_b = tl.load(b + bos * HV + i_hv).to(tl.float32)

            x = b_a + b_dt_bias
            beta_x = softplus_beta * x
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x
            b_decay = tl.exp(b_g)
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))

            # q/k are shared across the value dimension; set pointers once.
            p_q = q + bos * q_row_stride + i_h * K + o_k
            p_k = k + bos * k_row_stride + i_h * K + o_k

            for i_v in tl.range(0, NV):
                o_v = i_v * BV + o_base_v
                mask_v = o_v < V
                mask_h = mask_k[:, None] & mask_v[None, :]

                # v uses v_row_stride likewise; o is a contiguous buffer
                # owned by this wrapper and keeps the PR-original addressing.
                p_v = v + bos * v_row_stride + i_hv * V + o_v
                p_o = o + (bos * HV + i_hv) * V + o_v

                if USE_INITIAL_STATE:
                    idx = tl.load(h0_indices + i_n)
                    p_h0 = (
                        h0_source
                        + idx * HV * K * V
                        + i_hv * K * V
                        + o_k[:, None] * V
                        + o_v[None, :]
                    )

                for i in tl.range(0, t_len):
                    b_q = tl.load(p_q + i * q_row_stride, mask=mask_k).to(tl.float32)
                    b_k = tl.load(p_k + i * k_row_stride, mask=mask_k).to(tl.float32)
                    b_v = tl.load(p_v + i * v_row_stride, mask=mask_v).to(tl.float32)

                    if USE_INITIAL_STATE:
                        if idx >= 0:
                            b_h = tl.load(p_h0 + i * HV * K * V, mask=mask_h).to(
                                tl.float32
                            )
                        else:
                            b_h = tl.zeros([BK, BV], dtype=tl.float32)
                    else:
                        b_h = tl.zeros([BK, BV], dtype=tl.float32)

                    if USE_QK_L2NORM_IN_KERNEL:
                        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
                        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

                    b_q = b_q * scale

                    # Recurrent delta rule update.
                    b_h *= b_decay
                    b_v -= tl.sum(b_h * b_k[:, None], 0)
                    b_v *= b_beta
                    b_h += b_k[:, None] * b_v[None, :]
                    b_o = tl.sum(b_h * b_q[:, None], 0)

                    if USE_INITIAL_STATE:
                        if idx >= 0:
                            tl.store(
                                p_h0 + i * HV * K * V,
                                b_h.to(h0_source.dtype.element_ty),
                                mask=mask_h,
                            )

                    tl.store(p_o + i * HV * V, b_o.to(o.dtype.element_ty), mask=mask_v)


def fused_sigmoid_gating_delta_rule_update_decode_npu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Decode-optimized recurrent delta rule update for Ascend NPU.

    Drop-in replacement for the generic
    ``fused_sigmoid_gating_delta_rule_update_npu`` on the decode path, with
    the same algorithm. It launches with num_warps=4 for better small-batch
    throughput on the AIV vector cores.

    q/k/v may be non-contiguous views (only last-dim stride==1 is required);
    the kernel addresses them by the passed token row strides, eliminating 3
    per-layer q/k/v copies on the decode path. Contiguous inputs address
    identically to the PR original.

    Args / Returns: same as ``fused_sigmoid_gating_delta_rule_update_npu``
    (return shape matches v; the PR original returns (N, HV, V) and this
    version ``view``s back to v's shape — under decode the token count == N,
    so the elements correspond one-to-one).
    """
    # q/k/v stay as views; only last-dim stride==1 is asserted, row strides go
    # to the kernel.
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    # The pool must be contiguous: copying a non-contiguous pool via
    # _maybe_contiguous would write state updates back into the copy
    # (silently losing them), hence the hard assert.
    assert initial_state_source.is_contiguous()
    A_log = _maybe_contiguous(A_log)
    a = _maybe_contiguous(a)
    dt_bias = _maybe_contiguous(dt_bias)
    b = _maybe_contiguous(b)
    if initial_state_indices is not None:
        initial_state_indices = _maybe_contiguous(initial_state_indices)
    if cu_seqlens is not None:
        cu_seqlens = _maybe_contiguous(cu_seqlens)

    with torch.npu.device(q.device.index):
        B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Decode-only guard: o is addressed by token index (bos) but has only N
    # rows, valid only with exactly 1 token per sequence — varlen requires
    # total_tokens == N (production decode has query_start_loc step 1),
    # non-varlen requires T == 1. verify/prefill must not use this kernel.
    if cu_seqlens is not None:
        assert T == N, (
            f"decode-optimized kernel requires one token per sequence, "
            f"got total_tokens={T}, sequences={N}"
        )
    else:
        assert T == 1, (
            f"decode-optimized kernel requires T == 1 without cu_seqlens, got T={T}"
        )

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported in the decode-optimized kernel"

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, "scale must be positive"

    # Each program handles exactly one value head to keep the UB working set
    # small and avoid cross-head synchronization.
    BHV = 1
    NHV = HV

    o = q.new_empty(N, HV, V)

    # 1-D grid sized to the AIV vector core count; num_warps=4 was tuned for
    # small-batch decode. Oversubscription does not help because the loop
    # trip count is already small.
    num_aicore, num_vectorcore = get_device_properties()
    OVERSUB = 1
    num_programs = min(N * NHV, num_vectorcore * OVERSUB)
    num_programs = max(1, num_programs)
    grid = (num_programs,)

    num_warps = 4
    num_stages = 1

    _fused_sigmoid_gating_delta_rule_update_decode_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        q_row_stride=q.stride(1),
        k_row_stride=k.stride(1),
        v_row_stride=v.stride(1),
        scale=scale,
        T=T,
        N=N,
        NHV=NHV,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BHV=BHV,
        OVERSUB=OVERSUB,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
        multibuffer=False,
    )
    # Match the generic wrapper's return shape (*v.shape); under decode the
    # token count == N, so the elements correspond one-to-one.
    return o.view(v.shape)
