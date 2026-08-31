# Switch/guard module for Qwen3.5 NPU decode TP-path AscendC fusion.
#
# Two ops (registered by sgl-kernel-npu csrc as torch.ops.npu.*):
#   op1 fused_qkvzba_conv1d            GDN decode: qkvzba split + causal_conv1d(UPDATE),
#                                      single kernel, returns (y, z, b, a); conv_states updated in place.
#   op2 fused_sigmoid_gating_recurrent GDN decode: sigmoid gating + delta rule update,
#                                      single AIV_ONLY kernel; drop-in replacement for the
#                                      production Triton kernel; ssm pool updated in place.
#
# fused_norm_qkv_proj_scatter / fused_sigmoid_mul_mm are not built in this version
# (dropped); their guard functions are kept as always-disabled stubs so that an old
# qwen3_5.py importing them still works.
#
# Switches:
#   SGLANG_NPU_TP_ASCENDC_FUSION          master switch (default "0"), op1 only
#   SGLANG_NPU_TP_ASCENDC_FUSION_QKVZBA   op1 switch: follows master when unset, explicit "0" disables op1 alone
#   SGLANG_NPU_GDN_RECURRENT_ASCENDC      op2 switch (default "0"); evaluated at gdn_triton.py
#                                         import time, must be set before graph capture / server start
#   SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG=1  log guard misses once per key
# Guard decisions are baked into the graph at capture time; changing env afterwards has no effect.

import logging
import os

import torch

logger = logging.getLogger(__name__)

_TP_FUSION_ENV = "SGLANG_NPU_TP_ASCENDC_FUSION"
_TP_FUSION_QKVZBA_ENV = "SGLANG_NPU_TP_ASCENDC_FUSION_QKVZBA"
_GDN_RECURRENT_ASCENDC_ENV = "SGLANG_NPU_GDN_RECURRENT_ASCENDC"
_TP_DEBUG_ENV = "SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG"
_tp_debug_logged = set()

# Host-side hard constraints of the ops (mirror the csrc host TORCH_CHECKs; see each REGISTRATION.md):
_TP_C0_ALIGN = 16  # bf16 C0 alignment (op1 row width must be a multiple of 16)
_RECURRENT_HEAD_DIM = 128  # fused_sigmoid_gating_recurrent kernel specializes K/V to 128
_RECURRENT_MAX_HV = 8  # gating [8] pad limit of the same kernel
_RECURRENT_MAX_N = 256  # cu/idx UB residency limit of the same kernel

_TP_OP_CACHE = {}


def tp_fusion_enabled() -> bool:
    """Master switch for TP-path AscendC fusion (default off; "1" enables, op1 only)."""
    return os.environ.get(_TP_FUSION_ENV, "0") == "1"


def tp_fusion_qkvzba_enabled() -> bool:
    """op1 (fused_qkvzba_conv1d) switch: follows the master switch when unset; explicit "0" disables op1 alone."""
    return os.environ.get(
        _TP_FUSION_QKVZBA_ENV, os.environ.get(_TP_FUSION_ENV, "0")
    ) == "1"


def gdn_recurrent_ascendc_enabled() -> bool:
    """op2 (fused_sigmoid_gating_recurrent, AscendC recurrent) switch: default "0", enabled only by explicit "1"."""
    return os.environ.get(_GDN_RECURRENT_ASCENDC_ENV, "0") == "1"


def tp_debug_log(key, msg):
    """When the debug switch is on, log once per key (for locating guard misses)."""
    if os.environ.get(_TP_DEBUG_ENV, "0") != "1":
        return
    if key in _tp_debug_logged:
        return
    _tp_debug_logged.add(key)
    logger.warning("[tp_ascendc_fusion] %s", msg)


def tp_op_available(name: str) -> bool:
    """Whether the op is registered on torch.ops.npu (i.e. sgl-kernel-npu was built with these ops).

    Registration happens when the sgl_kernel_npu extension loads (process start), so
    the result is process-static and cached.
    """
    got = _TP_OP_CACHE.get(name)
    if got is None:
        got = hasattr(torch.ops.npu, name)
        _TP_OP_CACHE[name] = got
    return got


# ---------------------------------------------------------------------------
# op1 (GDN decode): fused_qkvzba_conv1d
#   = fused_qkvzba_split_reshape_cat_contiguous + causal_conv1d(run_mode=1)
# ---------------------------------------------------------------------------
def tp_fused_qkvzba_conv1d_shape_supported(
    num_k_heads_tp, num_v_heads_tp, head_k_dim, head_v_dim, conv_kernel_size
) -> bool:
    """Shape guard baked at init (Qwen3_5GatedDeltaNet level; head counts are per-rank after TP split).

    Mirrors the op host TORCH_CHECKs: num_v % num_k == 0 with ratio in {1,2,4} (same
    as the replaced qwen3_5.py split-branch condition), conv width in [2,4], qkvWidth
    and qkvz row width 16-aligned (32B DataCopy alignment of the z copy), z row
    bytes <= 65535 (single-block DataCopy limit).
    """
    ok = (
        tp_fusion_qkvzba_enabled()
        and tp_op_available("fused_qkvzba_conv1d")
        and num_k_heads_tp > 0
        and num_v_heads_tp > 0
        and head_k_dim > 0
        and head_v_dim > 0
        and num_v_heads_tp % num_k_heads_tp == 0
        and (num_v_heads_tp // num_k_heads_tp) in (1, 2, 4)
        and 2 <= conv_kernel_size <= 4
    )
    if ok:
        qkv_width = 2 * num_k_heads_tp * head_k_dim + num_v_heads_tp * head_v_dim
        x_row_stride = qkv_width + num_v_heads_tp * head_v_dim
        ok = (
            qkv_width % _TP_C0_ALIGN == 0
            and x_row_stride % _TP_C0_ALIGN == 0
            and num_v_heads_tp * head_v_dim * 2 <= 65535
        )
    if not ok:
        tp_debug_log(
            ("qkvzba_shape",),
            "fused_qkvzba_conv1d init shape guard missed (falling back to stock split+causal_conv1d): "
            f"nk_tp={num_k_heads_tp} nv_tp={num_v_heads_tp} dk={head_k_dim} "
            f"dv={head_v_dim} width={conv_kernel_size} "
            f"op_registered={tp_op_available('fused_qkvzba_conv1d')} "
            f"{_TP_FUSION_QKVZBA_ENV}={os.environ.get(_TP_FUSION_QKVZBA_ENV, '<follows master>')!r} "
            f"{_TP_FUSION_ENV}={os.environ.get(_TP_FUSION_ENV, '0')!r}",
        )
    return ok


def tp_fused_qkvzba_conv1d_inputs_ok(
    qkvz, mixed_ba, num_k_heads_tp, num_v_heads_tp, head_k_dim, head_v_dim
) -> bool:
    """Lightweight runtime check on the backend side (tuple-input tensor attributes; the main shape set is baked at init).

    Contiguity contract is a "row-stride view": column-contiguous (stride(1)==1), row
    stride >= logical width, and qkvz row stride a multiple of 16 (mirrors the host
    TORCH_CHECK for 32B alignment of the z copy). Contiguous inputs satisfy this
    naturally; row-stride views (slices of the packed GEMM output) are read directly,
    avoiding two .contiguous() calls. Must be deployed together with the matching
    csrc host version (older hosts still TORCH_CHECK is_contiguous).
    """
    qkv_width = 2 * num_k_heads_tp * head_k_dim + num_v_heads_tp * head_v_dim
    return (
        torch.is_tensor(qkvz)
        and torch.is_tensor(mixed_ba)
        and qkvz.dim() == 2
        and mixed_ba.dim() == 2
        and qkvz.dtype == torch.bfloat16
        and mixed_ba.dtype == torch.bfloat16
        and qkvz.stride(1) == 1
        and qkvz.stride(0) >= qkvz.shape[1]
        and qkvz.stride(0) % _TP_C0_ALIGN == 0
        and mixed_ba.stride(1) == 1
        and mixed_ba.stride(0) >= mixed_ba.shape[1]
        and qkvz.shape[1] == qkv_width + num_v_heads_tp * head_v_dim
        and mixed_ba.shape[0] == qkvz.shape[0]
        and mixed_ba.shape[1] == 2 * num_v_heads_tp
    )


# ---------------------------------------------------------------------------
# op2 (GDN decode): fused_sigmoid_gating_recurrent (AscendC recurrent)
#
# Not bit-exact with the production Triton kernel (K-dim reduction order +
# instruction-level Exp/Ln/Div differences); acceptance follows bounded error +
# long-run drift + e2e A/B (see the op package README "precision gate" section).
# ---------------------------------------------------------------------------
def _ascendc_recurrent_supported(
    q, k, v, a, b, initial_state_source, initial_state_indices, cu_seqlens,
    A_log, dt_bias,
) -> bool:
    """Runtime guard (mirrors host TORCH_CHECKs; on miss the wrapper falls back to stock Triton).

    Reads tensor attributes only (shape/stride/dtype/contiguity), never device data,
    so it is graph-capture safe. fp32 dtype of A_log/dt_bias is not enforced here —
    the wrapper widens losslessly (see call site); only numel is checked to avoid a
    host TORCH_CHECK failure.
    """
    ok = (
        tp_op_available("fused_sigmoid_gating_recurrent")
        and torch.is_tensor(q)
        and q.dim() == 4
        and q.size(0) == 1
        and cu_seqlens is not None
        and torch.is_tensor(cu_seqlens)
        and q.size(1) == cu_seqlens.numel() - 1  # T == N: 1 token per sequence
        and 1 <= q.size(1) <= _RECURRENT_MAX_N
        and q.size(3) == _RECURRENT_HEAD_DIM
        and v.size(3) == _RECURRENT_HEAD_DIM
        and q.size(2) >= 1
        and v.size(2) >= 1
        and v.size(2) <= _RECURRENT_MAX_HV
        and v.size(2) % q.size(2) == 0
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and initial_state_source.dtype in (torch.bfloat16, torch.float32)
        and q.stride(3) == 1
        and k.stride(3) == 1
        and v.stride(3) == 1
        and q.stride(2) == q.size(3)
        and k.stride(2) == k.size(3)
        and v.stride(2) == v.size(3)
        and a.is_contiguous()
        and b.is_contiguous()
        and initial_state_source.is_contiguous()
        and initial_state_source.dim() == 4
        and initial_state_source.size(1) == v.size(2)
        and initial_state_source.size(2) == _RECURRENT_HEAD_DIM
        and initial_state_source.size(3) == _RECURRENT_HEAD_DIM
        and initial_state_indices is not None
        and initial_state_indices.numel() >= q.size(1)
        and A_log.numel() == v.size(2)  # host: A_log/dt_bias numel == HV
        and dt_bias.numel() == v.size(2)
    )
    if not ok:
        tp_debug_log(
            ("recurrent_shape",),
            "fused_sigmoid_gating_recurrent runtime guard missed (falling back to stock Triton): "
            f"q={tuple(q.shape)}/{q.dtype} v={tuple(v.shape)}/{v.dtype} "
            f"pool={tuple(initial_state_source.shape)}/{initial_state_source.dtype} "
            f"T_vs_N={q.size(1)}/{cu_seqlens.numel() - 1 if torch.is_tensor(cu_seqlens) else None} "
            f"op_registered={tp_op_available('fused_sigmoid_gating_recurrent')}",
        )
    return ok


def fused_sigmoid_gating_delta_rule_update_ascendc(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    """AscendC drop-in replacement for the production Triton wrapper
    (fused_sigmoid_gating_delta_rule_update_npu), same signature and semantics;
    decode only (T==N).

    On guard miss, falls back to the stock Triton wrapper (silent; with
    SGLANG_NPU_TP_ASCENDC_FUSION_DEBUG=1 the reason is logged once).
    """
    if not _ascendc_recurrent_supported(
        q, k, v, a, b, initial_state_source, initial_state_indices, cu_seqlens,
        A_log, dt_bias,
    ):
        from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update_npu,
        )

        return fused_sigmoid_gating_delta_rule_update_npu(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
    # scale default matches the Triton wrapper (python-side K**-0.5, same double->float path)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    # A_log/dt_bias: the host only accepts fp32, but they may be bf16 after checkpoint
    # load. Widen losslessly here (bit-exact with the in-kernel .to(tl.float32) of the
    # Triton kernel), mirroring the Triton wrapper's .contiguous().
    if A_log.dtype != torch.float32:
        A_log = A_log.float()
    if dt_bias.dtype != torch.float32:
        dt_bias = dt_bias.float()
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    return torch.ops.npu.fused_sigmoid_gating_recurrent(
        A_log,
        a,
        dt_bias,
        float(softplus_beta),
        float(softplus_threshold),
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
        float(scale),
        cu_seqlens,
        bool(use_qk_l2norm_in_kernel),
        q.stride(1),
        k.stride(1),
        v.stride(1),
    )


# ---------------------------------------------------------------------------
# Legacy stubs (always disabled): fused_norm_qkv_proj_scatter / fused_sigmoid_mul_mm
# These two ops were dropped and are not built in this version. If an old qwen3_5.py
# imports the 5 functions below, the stubs keep it working and never activate
# (callers fall back to the original path).
# ---------------------------------------------------------------------------
def tp_fusion_nqps_enabled() -> bool:
    """fused_norm_qkv_proj_scatter switch stub: always False (op not built)."""
    return False


def tp_fusion_sigmm_enabled() -> bool:
    """fused_sigmoid_mul_mm switch stub: always False (op not built)."""
    return False


def tp_norm_qkv_scatter_shape_supported(layer) -> bool:
    """fused_norm_qkv_proj_scatter init guard stub: always False."""
    return False


def tp_norm_qkv_scatter_context(
    layer, hidden_states, residual, forward_batch, captured_last_layer_outputs
):
    """fused_norm_qkv_proj_scatter runtime guard stub: always None (caller falls back to the original path)."""
    return None


def tp_sigmoid_mul_mm_shape_supported(o_proj) -> bool:
    """fused_sigmoid_mul_mm init guard stub: always False."""
    return False


def tp_sigmoid_mul_mm_runtime_ok(attn_output, gate) -> bool:
    """fused_sigmoid_mul_mm runtime guard stub: always False."""
    return False
