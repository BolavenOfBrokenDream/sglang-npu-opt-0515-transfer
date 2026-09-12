# MoE layer-tail chain (fin+add+AR+norm) fusion for Qwen3.5-35B-A3B NPU decode,
# 0515-baseline port of tp_ascendc_fusion_v4. Switch/guard/spin-context module.
#
# Fused chain = the four layer-tail nodes of the production qwen2_moe.py
# dual-stream path:
#   npu_moe_finalize_routing (inside UnquantizedFusedMoEMethod.forward_npu)
#   -> add (shared-expert output, dual-stream only)
#   -> tensor_model_parallel_all_reduce
#   -> A3a add_gemma_rms_norm (next layer's input_layernorm, gemma w+1)
# Two variants (operator package: csrc/fused_tail/REGISTRATION.md):
#   spin = torch.ops.npu.fused_fin_ar_norm single kernel (fin+skip1 + spin AIV
#          AR + add + norm; needs the symmem context _SpinContext initialized
#          at the end of load_weights).
#   hccl = torch.ops.npu.fused_fin_add (fin+skip1 local front stage)
#          + stock tensor_model_parallel_all_reduce + stock A3a norm.
#          No symmem, no spin surface.
# M preference (probe r24 graph-mode verdict): the spin AR+norm fusion delta is
# positive only for small M (M<=32ish); the fin+add node fusion is always
# positive under HCCL AR. Production bs=32 per engine -> default spin; switch
# to hccl explicitly for the bs64 tier.
#
# 0515 anchor design (differs from the v4 dispatcher anchor): on this baseline
# the stock finalize lives INSIDE UnquantizedFusedMoEMethod.forward_npu (the
# dispatcher's combine is a pass-through), so the deferred pieces
# (xexp/eri/scales_fp32) are produced by forward_npu(..., deferred=True)
# instead of a dispatcher method. The gate therefore requires
# a2a_backend.is_none() + the quant method's fused_tail_deferred_supported
# marker (see unquant.py) instead of v4's AscendTP dispatcher checks.
#
# Switches (decided at capture and baked into the graph; changing env after
# capture has no effect):
#   SGLANG_NPU_MOE_TAIL_FUSION       master switch (default "0"); effective
#                                    only with multi stream
#                                    (SGLANG_NPU_USE_MULTI_STREAM=1) — the
#                                    add in the chain is dual-stream-only.
#   SGLANG_NPU_MOE_TAIL_FUSION_AR    variant: "spin" (default) / "hccl".
#   SGLANG_NPU_MOE_TAIL_FUSION_DEBUG=1  log guard misses / init results once per key
#   SGLANG_NPU_FUSED_TAIL_PROBE=1    debug probe: before launch, host-side
#                                    per-call eri range / pieces contract check
#                                    with dump; on violation dump .pt and raise
#                                    loudly (kernel not launched). Eager-only
#                                    forensics — skipped automatically during
#                                    graph capture (D2H sync would break
#                                    recording). Two D2H syncs per call, never
#                                    enable in production.
#   SGLANG_NPU_FUSED_TAIL_PROBE_PATH probe dump dir (default ./fused_tail_probe)
#
# Deployment: this file goes to
#   <sglang>/python/sglang/srt/hardware_backend/npu/tp_fused_tail_npu.py
# Version coupling: qwen2_moe.py (deferred wiring) / qwen3_5.py (residual
# stash + init hook) / layernorm.py (ctx consumption) / unquant.py (deferred
# pieces) / csrc fused_tail must be deployed together; the spin variant also
# needs the symmem env (libshmem + SHMEM_UID_SESSION_ID with a distinct port
# per engine, or SHMEM_UID_SOCK_IFNAME for auto port selection).

import logging
import os

import torch

logger = logging.getLogger(__name__)

_FT_ENV = "SGLANG_NPU_MOE_TAIL_FUSION"
_FT_AR_ENV = "SGLANG_NPU_MOE_TAIL_FUSION_AR"
_FT_DEBUG_ENV = "SGLANG_NPU_MOE_TAIL_FUSION_DEBUG"
_ft_debug_logged = set()

# Constants shared with csrc host/kernel (single source of truth in csrc;
# bilateral consistency enforced by host TORCH_CHECKs and the build-stamp
# comparison at init):
_FT_NCORES = 16            # launch cores (probe default tier; bs<=64 H2048: <=2 tiles/core)
_FT_ZERO_NCORES = 48       # clearing kernel cores (spread over all cores)
_FT_CYCLE_LIMIT_US = 50000  # spin hard limit (two in-kernel checkpoints, wall bound 2x=100ms)
_FT_TILE_BYTES = 8192
_FT_RING = 4
_FT_FLAG_ITEM_BYTES = 32
_FT_MAX_CORES = 48
_FT_CELL_BYTES = 128
_FT_COUNTER_OFFSET = 0
_FT_DFX_OFFSET = _FT_MAX_CORES * _FT_CELL_BYTES  # 48*128
_FT_MCELL_BYTES = 2 * _FT_MAX_CORES * _FT_CELL_BYTES
_FT_EXPECT_BUILD_REV = 20260922  # bump on any kernel change (csrc FUSED_TAIL_BUILD_REV)

_FT_OP_CACHE = {}
_mode_cache = None

_spin_ctx = None       # _SpinContext singleton (spin variant only)
_spin_disabled = False  # init failed / env unsatisfied -> globally disable (silent stock fallback)


def _env_str(name, default):
    try:
        from sglang.srt.environ import envs

        return str(getattr(envs, name).get())
    except Exception:
        return os.environ.get(name, default)


def _env_bool(name, default=False):
    # EnvBool fields parse to python bool; EnvStr/EnvInt fields may not.
    try:
        from sglang.srt.environ import envs

        return bool(getattr(envs, name).get())
    except Exception:
        return os.environ.get(name, "1" if default else "0") == "1"


def _ft_debug_log(key, msg):
    if not _env_bool(_FT_DEBUG_ENV):
        return
    if key in _ft_debug_logged:
        return
    _ft_debug_logged.add(key)
    logger.warning("[tp_fused_tail] %s", msg)


# ---------------------------------------------------------------------------
# Debug probe (SGLANG_NPU_FUSED_TAIL_PROBE, eager forensics only, see header)
# ---------------------------------------------------------------------------
_FT_PROBE_ENV = "SGLANG_NPU_FUSED_TAIL_PROBE"
_FT_PROBE_PATH_ENV = "SGLANG_NPU_FUSED_TAIL_PROBE_PATH"
_ft_probe_seq = 0


def _ft_probe_on() -> bool:
    return os.environ.get(_FT_PROBE_ENV, "0") == "1"


def _ft_capturing() -> bool:
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


def _ft_probe_rank():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return os.getpid()


def _ft_probe_write(text):
    base = os.environ.get(_FT_PROBE_PATH_ENV, "./fused_tail_probe")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"probe_rank{_ft_probe_rank()}.log")
    with open(path, "a") as f:
        f.write(text + "\n")


def _ft_probe_pieces(tag, xp, eri, scales, m, h, k, extra=""):
    """Pre-launch eri range check (direct OOB condition) with per-call dump;
    on violation dump the full eri .pt and raise. Other fields (xexp_rows vs
    m*k, eri_numel) are recorded but not enforced."""
    global _ft_probe_seq
    _ft_probe_seq += 1
    eri_min = int(eri.min().item())
    eri_max = int(eri.max().item())
    rows = int(xp.shape[0])
    ok = eri_min >= 0 and eri_max < rows
    line = (
        f"seq={_ft_probe_seq} tag={tag} m={m} h={h} k={k} xexp_rows={rows} "
        f"eri_numel={eri.numel()} eri_min={eri_min} eri_max={eri_max} "
        f"xp=0x{xp.data_ptr():x} eri=0x{eri.data_ptr():x} "
        f"scales=0x{scales.data_ptr():x} {extra} ok={int(ok)}"
    )
    _ft_probe_write(line)
    if not ok:
        base = os.environ.get(_FT_PROBE_PATH_ENV, "./fused_tail_probe")
        torch.save(
            eri.cpu(),
            os.path.join(
                base, f"eri_violation_seq{_ft_probe_seq}_rank{_ft_probe_rank()}.pt"
            ),
        )
        raise RuntimeError(f"fused_tail probe: eri range violation ({line})")


def _ft_probe_gate(tag, xp, eri, scales, m, h, k, extra=""):
    if _ft_probe_on() and not _ft_capturing():
        _ft_probe_pieces(tag, xp, eri, scales, m, h, k, extra)


def ft_op_available(name: str) -> bool:
    """Whether the op is registered on torch.ops.npu (i.e. sgl-kernel-npu was
    built with fused_tail). Registration happens at extension load (process
    start), so the result is process-static and cached."""
    got = _FT_OP_CACHE.get(name)
    if got is None:
        got = hasattr(torch.ops.npu, name)
        _FT_OP_CACHE[name] = got
    return got


def _multi_stream_on() -> bool:
    return _env_bool("SGLANG_NPU_USE_MULTI_STREAM")


def _native_gemma_norm_on() -> bool:
    """SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM=1 bypasses the ctx consumption
    point (forward_npu) — with the upstream finalize already skipped, an
    unconsumed ctx yields an undefined tensor, so the fusion is disabled under
    that env."""
    return _env_bool("SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM")


def fused_tail_ar_mode():
    """Fusion variant: None (no fusion) / "spin" / "hccl". Cached per process
    (baked at graph capture; UTs must call _reset_caches_for_test() after
    changing env).

    Gates (all required for non-None):
      - SGLANG_NPU_MOE_TAIL_FUSION=1 (master switch)
      - multi stream on (SGLANG_NPU_USE_MULTI_STREAM=1 — the add in the chain
        is dual-stream-only)
      - ops registered (wheel built with fused_tail; day-0 gate, old wheel
        silently falls back)
      - native gemma norm fallback env not set (would bypass the consumption
        point)
    """
    global _mode_cache
    if _mode_cache is not None:
        return _mode_cache
    mode = None
    if (
        _env_bool(_FT_ENV)
        and _multi_stream_on()
        and ft_op_available("fused_fin_ar_norm")
        and ft_op_available("fused_fin_add")
        and not _native_gemma_norm_on()
    ):
        ar = _env_str(_FT_AR_ENV, "spin").strip().lower()
        if ar in ("spin", "hccl"):
            mode = ar
        else:
            logger.warning(
                "[tp_fused_tail] %s=%r invalid (expect spin/hccl), fusion disabled",
                _FT_AR_ENV,
                ar,
            )
    _mode_cache = mode
    if mode is None:
        _ft_debug_log(
            ("mode",),
            f"fused tail not enabled: {_FT_ENV}={_env_str(_FT_ENV, '0')!r} "
            f"multi_stream={_multi_stream_on()} "
            f"op_registered={ft_op_available('fused_fin_ar_norm')}/"
            f"{ft_op_available('fused_fin_add')} "
            f"native_gemma={_native_gemma_norm_on()}",
        )
    return mode


def fused_tail_spin_stash_enabled() -> bool:
    """Whether qwen3_5 decoder layers should stash residual for the MoE block
    (spin variant only; hit on every forward, kept cheap — mode is cached)."""
    return fused_tail_ar_mode() == "spin" and not _spin_disabled


def _reset_caches_for_test():
    """UT only: clear mode/op caches and the spin singleton (env untouched)."""
    global _mode_cache, _spin_ctx, _spin_disabled
    _mode_cache = None
    _FT_OP_CACHE.clear()
    _spin_ctx = None
    _spin_disabled = False


# ---------------------------------------------------------------------------
# spin context (symmem + cell area + warmup + deployment self-check), spin only
# ---------------------------------------------------------------------------
def _sm_empty(sm, numel, dtype, device):
    last = None
    for a, kw in (((numel,), dict(dtype=dtype, device=device)),
                  ((numel, 1), dict(dtype=dtype, device=device))):
        try:
            return sm.empty(*a, **kw)
        except Exception as exc:  # noqa: PERF203
            last = exc
    raise last


def _sm_rendezvous(sm, buf):
    import torch.distributed as dist

    last = None
    for kw in (dict(group=dist.group.WORLD), dict(group_name=None), dict()):
        try:
            return sm.rendezvous(buf, **kw)
        except Exception as exc:
            last = exc
    raise last


def _sm_get_buffer(h, r, numel, dtype):
    last = None
    for kw in (dict(sizes=(numel,), dtype=dtype, storage_offset=0),
               dict(sizes=(numel,), dtype=dtype),
               dict(shape=(numel,), dtype=dtype)):
        try:
            return h.get_buffer(r, **kw)
        except Exception as exc:
            last = exc
    raise last


class _SpinContext:
    """data/flag symmem blocks + cell area (plain GM) + addr_tab + init-time
    warmup/self-check.

    Layout (same source as the csrc kernel lib header; host derives with the
    same formulas for consistency checking):
      slot(g,r) = dataVA + g*ring_stride + r*slot_stride (g in [0,4) generations)
      line(g,t) = flagVA + g*flag_ring_stride + t*world*32 (phase A only)
      mcell = [counter 48*128 | dfx 48*128] (plain GM, torch zeroing is safe)
    All layers share one context: per-core counters tick in lockstep for ring
    generation rotation, zero resets during replay. symmem rendezvous over the
    WORLD group — spin requires the TP group == WORLD (enforced by the gate),
    otherwise the peer VA table and AR semantics would be misplaced.
    """

    def __init__(self, hidden_size: int, mmax: int):
        import torch.distributed as dist

        if hidden_size not in (2048, 4096):
            raise RuntimeError(f"hidden_size={hidden_size} unsupported (2048/4096 only)")
        if mmax is None or mmax < 2:
            raise RuntimeError(f"mmax={mmax} invalid")
        mmax = int(mmax) & ~1  # H2048 tile alignment requires even m; round capacity down
        if dist.get_world_size() > 8:
            raise RuntimeError("world > 8 unsupported (addr_tab capacity)")

        from torch.distributed import _symmetric_memory as sm

        if hasattr(sm, "enable_symm_mem_for_group"):
            sm.enable_symm_mem_for_group(dist.group.WORLD.group_name)

        self.ok = False
        self.device = torch.device(f"npu:{torch.npu.current_device()}")
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.h = int(hidden_size)
        self.mmax = mmax
        self.slot_stride = mmax * self.h * 2
        self.ring_stride = self.world * self.slot_stride
        self.max_tiles = (mmax * self.h * 2) // _FT_TILE_BYTES
        data_bytes = _FT_RING * self.ring_stride
        self.flag_ring_stride = self.max_tiles * self.world * _FT_FLAG_ITEM_BYTES
        flag_bytes = _FT_RING * self.flag_ring_stride
        self.flag_bytes = flag_bytes
        self.mcell = torch.zeros(_FT_MCELL_BYTES, dtype=torch.uint8, device=self.device)

        self.data = _sm_empty(sm, data_bytes, torch.uint8, self.device)
        self.flag = _sm_empty(sm, flag_bytes, torch.uint8, self.device)
        hd = _sm_rendezvous(sm, self.data)
        hf = _sm_rendezvous(sm, self.flag)
        # Keep peer tensors referenced against GC (post-rendezvous early-free
        # behavior unverified, probe discipline).
        self._keep = [self.data, self.flag, hd, hf]
        peer_data = [_sm_get_buffer(hd, r, data_bytes, torch.uint8) for r in range(self.world)]
        peer_flag = [_sm_get_buffer(hf, r, flag_bytes, torch.uint8) for r in range(self.world)]
        self._keep.extend(peer_data)
        self._keep.extend(peer_flag)
        addr_tab = torch.zeros(18, dtype=torch.int64, device=self.device)
        for r in range(self.world):
            addr_tab[r] = peer_data[r].data_ptr()
            addr_tab[8 + r] = peer_flag[r].data_ptr()
        for r in range(self.world, 8):  # pad unused ranks with local VAs (kernel reads by world)
            addr_tab[r] = self.data.data_ptr()
            addr_tab[8 + r] = self.flag.data_ptr()
        addr_tab[16] = 0  # reserved
        addr_tab[17] = self.mcell.data_ptr()
        self.addr_tab = addr_tab

        # flag clearing must go through the kernel's MTE3 (torch zero_() is an
        # AIV vector write that leaves dirty L2 lines on symmem; a dirty zero
        # line evicted later can erase a peer flag — probe first-round
        # spin-timeout root cause).
        self._zero_flag()
        torch.npu.synchronize()
        dist.barrier()
        self._warmup_and_selfcheck()
        self.ok = True
        logger.info(
            "[tp_fused_tail] spin context initialized: world=%d rank=%d h=%d mmax=%d "
            "data=%dB flag=%dB",
            self.world, self.rank, self.h, self.mmax, data_bytes, flag_bytes,
        )

    def _zero_flag(self):
        torch.ops.npu.fused_tail_zero(self.flag, self.flag_bytes, _FT_ZERO_NCORES)

    def _warmup_and_selfcheck(self):
        """Cross-card write-path warmup + deployment self-check (init time,
        outside capture).

        The first cross-card MTE3 write can drop a 512B chunk; from the second
        on the path is reliable (probe ut2) — what is cold is the local->peer
        first-write path itself. Use 8 dummy launches (M=mmax, two full ring
        rotations covering all data/flag areas) for end-to-end warmup; a cold
        loss can only surface as a dummy-launch spin timeout (writes DFX and
        exits without hanging), then flag/mcell are fully cleared for a clean
        production start. Self-check: cross-rank MAX-reduce of the DFX area
        must be all-zero (otherwise disable spin) + counter cell slot[1] build
        stamp == expected (stale-wheel discipline).
        """
        import torch.distributed as dist

        M, H, K = self.mmax, self.h, 8
        dev = self.device
        xp = torch.zeros(M * K, H, dtype=torch.bfloat16, device=dev)
        scales = torch.zeros(M, K, dtype=torch.float32, device=dev)
        skip1 = torch.zeros(M, H, dtype=torch.bfloat16, device=dev)
        res = torch.zeros(M, H, dtype=torch.bfloat16, device=dev)
        add_out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        norm_out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        eri = torch.zeros(M * K, dtype=torch.int32, device=dev)  # all-0 = legal rows
        norm_w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
        dist.barrier()  # align before burst (weight-load times differ; prevents dummy spin waits)
        for _ in range(2 * _FT_RING):
            torch.ops.npu.fused_fin_ar_norm(
                xp, scales, skip1, res, add_out, norm_out, self.addr_tab,
                int(eri.data_ptr()), int(norm_w.data_ptr()),
                M, H, K, _FT_NCORES, self.rank, self.world,
                1, 1, 1e-6, _FT_CYCLE_LIMIT_US,
                self.slot_stride, self.ring_stride, self.max_tiles,
                _FT_COUNTER_OFFSET, _FT_DFX_OFFSET,
            )
        torch.npu.synchronize()
        dist.barrier()

        # build-stamp self-check (BumpRing writes counter cell slot[1] per launch)
        rev = int(self.mcell[0:_FT_CELL_BYTES].view(torch.int32)[1].item())
        if rev != _FT_EXPECT_BUILD_REV:
            raise RuntimeError(
                f"kernel build stamp mismatch: got {rev}, expect {_FT_EXPECT_BUILD_REV}"
                " (wheel not rebuilt with this package or .so not swapped)"
            )
        # DFX cross-rank MAX-reduce: any core timeout => disable spin
        dfx = self.mcell[_FT_DFX_OFFSET:_FT_DFX_OFFSET + _FT_MAX_CORES * _FT_CELL_BYTES]
        n_bad = int((dfx.view(torch.int32) != 0).sum().item())
        t = torch.tensor([n_bad], dtype=torch.int32, device=dev)
        dist.all_reduce(t)
        if int(t.item()) != 0:
            raise RuntimeError(f"warmup dummy launch hit spin timeouts (DFX nonzero count={int(t.item())})")

        # Clean start: clear flag/mcell + double barrier (a timed-out launch
        # may leave unconsumed flags; counter back to 0 so the ring starts at 0)
        self._zero_flag()
        self.mcell.zero_()
        torch.npu.synchronize()
        dist.barrier()


def maybe_init_fused_tail_spin(hidden_size=None, mmax=None):
    """Idempotent spin-context init (collective + warmup, must be outside graph
    capture). Called at the end of the model's load_weights; UTs may call it
    explicitly. Failure = log + globally disable spin (fused_tail_gate silently
    falls back to the stock chain).
    mmax defaults to the server args' decode-graph max bs (the 0515 args split
    it into cuda_graph_max_bs_decode; larger-M forms such as MTP fall back via
    the runtime m>mmax gate)."""
    global _spin_ctx, _spin_disabled
    if fused_tail_ar_mode() != "spin" or _spin_disabled or _spin_ctx is not None:
        return
    if mmax is None:
        try:
            from sglang.srt.server_args import get_global_server_args

            args = get_global_server_args()
            mmax = int(
                getattr(args, "cuda_graph_max_bs", None)
                or getattr(args, "cuda_graph_max_bs_decode", None)
                or 64
            )
        except Exception:
            mmax = 64
    try:
        _spin_ctx = _SpinContext(hidden_size=hidden_size, mmax=mmax)
        _ft_debug_log(("spin_init",), "spin context init OK")
    except Exception as exc:
        logger.warning(
            "[tp_fused_tail] spin context init failed (fusion disabled, stock "
            "fallback): %s (SHMEM_UID_SESSION_ID=%s; aclshmemx_get_uniqueid -1 "
            "two typical causes: (1) co-located engines share one fixed port — "
            "each engine needs its own port; (2) port <1024 rejected by "
            "MIN_PORT (:0 unusable). Zero-orchestration alternative = unset "
            "SESSION_ID and set SHMEM_UID_SOCK_IFNAME=<nic prefix> for auto "
            "nic+random free port)",
            exc,
            os.environ.get("SHMEM_UID_SESSION_ID", "<unset>"),
        )
        _spin_ctx = None
        _spin_disabled = True


def _require_spin_ctx(m_now, h_now=2048):
    """Return the initialized spin context; lazily build it under eager if
    missing (UT/custom models without the load_weights hook). Building during
    capture is a loud failure (collectives cannot be recorded). An existing
    context with a different h is a loud failure (capacity/layout mismatch)."""
    if _spin_disabled:
        raise RuntimeError("spin context disabled (init failed); fused launch must not be reached")
    if _spin_ctx is not None:
        return _spin_ctx
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode():
        raise RuntimeError(
            "spin context not initialized during graph capture — the "
            "load_weights init hook in qwen3_5.py is missing or did not run "
            "(all three load_weights, CausalLM + the two VL wrapper classes, "
            "need the hook; qwen3_5.py/qwen2_moe.py/tp_fused_tail_npu.py must "
            "be deployed together)"
        )
    maybe_init_fused_tail_spin(hidden_size=int(h_now), mmax=max(64, int(m_now) + 2))
    if _spin_ctx is None:
        raise RuntimeError("spin context lazy init failed")
    return _spin_ctx


# ---------------------------------------------------------------------------
# Gate and launch (called from the qwen2_moe.py wiring points)
# ---------------------------------------------------------------------------
def fused_tail_gate(moe_block, hidden_states) -> bool:
    """Pre-decision for the deferred path (skip stock finalize, take the fused
    chain).

    Once True there is no way back (the finalize is skipped), so this gate
    must cover every prerequisite of finish — it only reads shapes/dtypes/
    module attributes/env, never device data, capture-safe. Any failed check
    -> False -> caller takes the stock chain (with stock finalize).
    """
    mode = fused_tail_ar_mode()
    if mode is None:
        return False
    if mode == "spin" and _spin_disabled:
        return False
    try:
        # Chain-structure prerequisites: shared expert present (the add's
        # operand = skip1), not shared-expert fusion (that route has no add
        # node), TP>1 (AR is meaningful), NPU local MoE path only —
        # a2a_backend.is_none() excludes deepep/fuseep/megamoe/flashinfer/
        # mooncake/mori/nixl in one shot.
        if moe_block.shared_expert is None:
            return False
        if getattr(moe_block, "enable_shared_expert_fusion", False):
            return False
        if moe_block.tp_size <= 1:
            return False
        from sglang.srt.layers.moe import get_moe_a2a_backend

        if not get_moe_a2a_backend().is_none():
            return False
        # 0515 anchor: the deferred pieces are produced by
        # UnquantizedFusedMoEMethod.forward_npu(deferred=True); the marker is
        # the same-batch deployment check (absent on quantized methods and on
        # an unpatched unquant.py).
        quant_method = getattr(moe_block.experts, "quant_method", None)
        if not getattr(quant_method, "fused_tail_deferred_supported", False):
            return False
        # The kernel consumes bf16 xexp; the stock chain's original_dtype must
        # be bf16 (quantized paths are excluded by the marker anyway).
        if hidden_states.dtype != torch.bfloat16:
            return False
        # Shape gates: M>0, two H tiers, M even for H2048 (tile alignment),
        # M <= host hard limit 4096 (the hccl variant's only capacity gate;
        # prefill-sized M falls back — spin additionally has the mmax gate so
        # it never activates outside decode).
        m = hidden_states.shape[0]
        h = hidden_states.shape[-1]
        if m < 1 or m > 4096 or h not in (2048, 4096):
            return False
        if h == 2048 and (m % 2) != 0:
            return False
        if (m * h * 2) % _FT_TILE_BYTES != 0:
            return False
        if mode == "spin":
            if _spin_ctx is not None and m > _spin_ctx.mmax:
                return False
            # spin's symmem rendezvouses over WORLD — the TP group must be
            # WORLD (production = single engine TP8, no DP/PP), otherwise the
            # peer VA table and AR semantics would be misplaced.
            import torch.distributed as dist

            if dist.get_world_size() != moe_block.tp_size:
                return False
        return True
    except Exception as exc:
        _ft_debug_log(("gate_exc",), f"gate check raised (stock fallback): {exc!r}")
        return False


def fused_tail_finish(moe_block, pieces, shared_output, residual, num_tokens, hidden_dim):
    """Launch the fused chain (gate passed; broken prerequisites fail loudly —
    the finalize is already skipped, silently falling back would yield an
    undefined tensor).

    pieces = the deferred return of UnquantizedFusedMoEMethod.forward_npu
    (xexp/eri/scales_fp32); shared_output = skip1. Returns
    [num_tokens, hidden_dim]: the spin variant returns the norm_out carrier
    with _sglang_fused_tail_ctx attached (the actual kernel launch is deferred
    to the next GemmaRMSNorm.forward_npu — the norm weight is only visible
    there); the hccl variant returns the AR-completed hidden (the stock A3a
    norm consumes it as usual).
    """
    mode = fused_tail_ar_mode()
    xp, eri, scales = pieces.xexp, pieces.eri, pieces.scales_fp32
    if shared_output is None:
        raise RuntimeError("fused_tail: shared_output is None (gate required shared_expert present)")
    m, h = shared_output.shape[0], shared_output.shape[-1]
    k = scales.shape[-1]
    if scales.dtype != torch.float32:
        scales = scales.float()  # topk output dtype drift fallback (bf16->fp32 lossless)
    if xp.dtype != torch.bfloat16 or eri.dtype != torch.int32:
        raise RuntimeError(
            f"fused_tail: bad pieces dtype xexp={xp.dtype} eri={eri.dtype} (need bf16/int32)"
        )
    _ft_probe_gate(
        mode, xp, eri, scales, m, h, k,
        extra=f"skip1=0x{shared_output.data_ptr():x}",
    )

    if mode == "hccl":
        contrib = torch.empty(m, h, dtype=torch.bfloat16, device=shared_output.device)
        torch.ops.npu.fused_fin_add(
            xp, scales, shared_output, contrib,
            int(eri.data_ptr()), m, h, k, _FT_NCORES, 1, 1,
        )
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(contrib)
        return out.view(num_tokens, hidden_dim)

    # ---- spin variant: allocate outputs, attach ctx, defer the launch to
    #      GemmaRMSNorm.forward_npu ----
    if residual is None:
        raise RuntimeError(
            "fused_tail(spin): residual missing — the qwen3_5.py decoder-layer "
            "residual stash hunk is not in effect (qwen3_5.py/qwen2_moe.py "
            "must be deployed together)"
        )
    ctx = _require_spin_ctx(m, h)
    if m > ctx.mmax or h != ctx.h:
        raise RuntimeError(
            f"fused_tail(spin): shape over capacity m={m}/h={h} vs mmax={ctx.mmax}/h={ctx.h}"
            " (the gate should have caught this — ctx init vs runtime shape mismatch)"
        )
    add_out = torch.empty(m, h, dtype=torch.bfloat16, device=shared_output.device)
    norm_out = torch.empty(m, h, dtype=torch.bfloat16, device=shared_output.device)
    fctx = _FusedTailCtx(
        xexp=xp, scales=scales, skip1=shared_output, residual=residual,
        eri=eri, add_out=add_out, norm_out=norm_out, spin_ctx=ctx,
        m=m, h=h, k=k,
    )
    out = norm_out.view(num_tokens, hidden_dim)
    out._sglang_fused_tail_ctx = fctx  # input references kept alive via ctx until launch
    return out


class _FusedTailCtx:
    """Deferred-launch context of the spin variant (attached to the norm_out
    carrier tensor's attribute, consumed exactly once by
    GemmaRMSNorm.forward_npu). Holds all input references alive."""

    __slots__ = ("xexp", "scales", "skip1", "residual", "eri",
                 "add_out", "norm_out", "spin_ctx", "m", "h", "k")

    def __init__(self, xexp, scales, skip1, residual, eri, add_out, norm_out,
                 spin_ctx, m, h, k):
        self.xexp = xexp
        self.scales = scales
        self.skip1 = skip1
        self.residual = residual
        self.eri = eri
        self.add_out = add_out
        self.norm_out = norm_out
        self.spin_ctx = spin_ctx
        self.m = m
        self.h = h
        self.k = k


# ---------------------------------------------------------------------------
# ctx consumption (called from the GemmaRMSNorm.forward_npu hunk in layernorm.py)
# ---------------------------------------------------------------------------
def maybe_consume_fused_tail_ctx(x, norm_module):
    """If x carries a fused-tail ctx, launch the spin fusion kernel and return
    (norm_out, add_out); otherwise return None (caller takes the stock norm).

    The norm weight is only visible here (next layer's input_layernorm / final
    norm), so the launch is deferred to this point; x itself is the norm_out
    carrier (the kernel writes it in place). The ctx is consumed once (deleted
    before launch, preventing double consumption)."""
    fctx = getattr(x, "_sglang_fused_tail_ctx", None)
    if fctx is None:
        return None
    del x._sglang_fused_tail_ctx
    w = norm_module.weight
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.numel() != fctx.h:
        raise RuntimeError(
            f"fused_tail(spin): bad norm weight dtype={w.dtype} "
            f"contiguous={w.is_contiguous()} numel={w.numel()} (need bf16 contiguous [{fctx.h}]; "
            "upstream finalize already skipped, no fallback — disable SGLANG_NPU_MOE_TAIL_FUSION)"
        )
    ctx = fctx.spin_ctx
    if _ft_probe_on() and not _ft_capturing():
        _ft_probe_write(
            f"spin_launch m={fctx.m} h={fctx.h} k={fctx.k} "
            f"w=0x{w.data_ptr():x} xexp=0x{fctx.xexp.data_ptr():x} "
            f"eri=0x{fctx.eri.data_ptr():x} skip1=0x{fctx.skip1.data_ptr():x} "
            f"residual=0x{fctx.residual.data_ptr():x} "
            f"add_out=0x{fctx.add_out.data_ptr():x} norm_out=0x{fctx.norm_out.data_ptr():x}"
        )
    torch.ops.npu.fused_fin_ar_norm(
        fctx.xexp, fctx.scales, fctx.skip1, fctx.residual,
        fctx.add_out, fctx.norm_out, ctx.addr_tab,
        int(fctx.eri.data_ptr()), int(w.data_ptr()),
        fctx.m, fctx.h, fctx.k, _FT_NCORES, ctx.rank, ctx.world,
        1, 1, float(norm_module.variance_epsilon), _FT_CYCLE_LIMIT_US,
        ctx.slot_stride, ctx.ring_stride, ctx.max_tiles,
        _FT_COUNTER_OFFSET, _FT_DFX_OFFSET,
    )
    return x, fctx.add_out
