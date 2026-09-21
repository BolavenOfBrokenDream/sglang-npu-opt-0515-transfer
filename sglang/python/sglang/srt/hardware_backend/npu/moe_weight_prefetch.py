"""MoE expert weight L2 prefetch for NPU decode (aclgraph replay).

Design:
- Two independent launch points, both self-targeted (the layer's own MoE
  weights), GDN layers only, capture-only:
  - "gdn": layer head, right after prepare_attn — i.e. after the layer-input
    allreduce (stock chain) or the fin_add_ar_norm consumed at input_layernorm
    (spin tail fusion). Default OPS gmm1 (w13); the window is the whole GDN
    attention segment.
  - "moe": MoE entry, right after prepare_mlp (post-GDN allreduce + norm).
    Default OPS gmm2 (w2). Only meaningful on the k9 single-stream stack:
    with the shared expert folded into the routed GMM the MoE head has no
    side-stream activity and the segment up to vgmm_sched runs at low
    bandwidth, a good spot for a CMO. Requires SGLANG_MOE_FRONT_FUSION=1 and
    k9 active, otherwise the point is dropped with a warning.
- Each point fires ONE CMO prefetch op per layer of a single OPS tensor; the
  size is the point's budget (MiB) clamped by L2_CAP_RATIO * L2 size
  (default 0.7) and the tensor size. FA layers are never targets (their FIA
  KV stream self-evicts and steals bandwidth).
- Best-effort: the main stream does not wait for the CMO stream before GMM
  (cache warm-up has no RAW hazard); drained at step end (prevents cross-step
  backlog, and capture legality requires the side stream to return to the
  main stream via event).
- Launched only during graph capture (eager / prefill never activate); CMO is
  an SDMA task and occupies no AIV/AIC cores. Only layer objects registered
  for the target model trigger launches (`_moe_prefetch_emit` marker); layers
  of the MTP draft model never launch even if they reuse this forward.

Note: torch_npu is always lazily imported inside functions, so this module is
safe to import on CUDA and other platforms.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from typing import Dict, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# ACL_DEV_ATTR_L2_CACHE_SIZE = RT_DEV_ATTR_L2_CACHE_SIZE = 302
_L2_CACHE_ATTR = 302

_VALID_OPS = ("gmm1", "gmm2")
_VALID_POINTS = ("gdn", "moe")


# ---------------------------------------------------------------------------
# L2 capacity query (ctypes, no compilation needed)
# ---------------------------------------------------------------------------
def _dlopen_first(candidates):
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def _try_get_info(lib, func_name) -> Optional[int]:
    fn = getattr(lib, func_name, None)
    if fn is None:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    val = ctypes.c_int64(-1)
    try:
        rc = fn(0, _L2_CACHE_ATTR, ctypes.byref(val))
    except Exception:
        return None
    if rc == 0 and val.value > 0:
        return int(val.value)
    return None


def _query_l2_size_bytes() -> Optional[int]:
    """Return the L2 cache size in bytes; None on query failure (caller disables prefetch)."""
    home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    acl_paths = ["libascendcl.so", "libascendcl.so.1"]
    rt_paths = ["libruntime.so"]
    if home:
        acl_paths.append(os.path.join(home, "lib64", "libascendcl.so"))
        rt_paths.append(os.path.join(home, "lib64", "libruntime.so"))
    acl_paths += sorted(
        glob.glob("/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so*")
    )
    rt_paths += sorted(
        glob.glob("/usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so*")
    )

    lib = _dlopen_first(acl_paths)
    if lib is not None:
        for name in ("aclrtGetDeviceInfo", "aclrtGetDeviceAttr"):
            size = _try_get_info(lib, name)
            if size is not None:
                return size
    lib = _dlopen_first(rt_paths)
    if lib is not None:
        size = _try_get_info(lib, "rtsDeviceGetInfo")
        if size is not None:
            return size
    return None


def _is_capture_mode() -> bool:
    """Lazy import to avoid a module-level dependency on the model_executor chain."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    return get_is_capture_mode()


class _PointCfg:
    """One launch point: a single OPS tensor prefetched with a single CMO op."""

    __slots__ = ("name", "ops", "budget_bytes", "size_bytes")

    def __init__(self, name: str, ops: str, budget_bytes: int):
        self.name = name
        self.ops = ops  # "gmm1" (w13) / "gmm2" (w2)
        self.budget_bytes = budget_bytes
        self.size_bytes = 0  # resolved once at first layer registration


# ---------------------------------------------------------------------------
# Manager (process-level singleton)
# ---------------------------------------------------------------------------
class _MoeWeightPrefetchManager:
    def __init__(self):
        self.configured = False
        self.enabled = False
        self.points: Dict[str, _PointCfg] = {}
        self.l2_size: Optional[int] = None
        self.cap_bytes = 0
        self.stream = None
        # layer_id -> {"w13": Tensor, "w2": Tensor, "prefetch_target": bool}
        self.layers: Dict[int, dict] = {}
        self._emitted_in_pass = False

    # ---- Configuration (parsed once at first registration) ----
    def configure(self):
        if self.configured:
            return
        self.configured = True

        if not envs.SGLANG_NPU_MOE_PREFETCH.get():
            return  # enabled stays False; everything below is a no-op

        raw = str(envs.SGLANG_NPU_MOE_PREFETCH_LAUNCH_POINT.get()).lower()
        points = [t.strip() for t in raw.split(",") if t.strip()]
        unknown = [t for t in points if t not in _VALID_POINTS]
        if unknown:
            logger.warning(
                f"[MOE_PREFETCH] unknown launch points {unknown} ignored, "
                f"valid: {_VALID_POINTS}"
            )

        def _ops(env_val, point):
            ops = str(env_val).strip().lower()
            if ops not in _VALID_OPS:
                logger.warning(
                    f"[MOE_PREFETCH] invalid {point} ops {ops!r} (one ops per "
                    f"point, valid: {_VALID_OPS}); {point} point disabled"
                )
                return None
            return ops

        def _budget(env_val, point):
            mib = int(env_val)
            if mib <= 0:
                logger.warning(
                    f"[MOE_PREFETCH] invalid {point} budget {mib}MiB, fallback to 32"
                )
                mib = 32
            return mib << 20

        cfgs: Dict[str, _PointCfg] = {}
        if "gdn" in points:
            ops = _ops(envs.SGLANG_NPU_MOE_PREFETCH_GDN_OPS.get(), "gdn")
            if ops is not None:
                cfgs["gdn"] = _PointCfg(
                    "gdn",
                    ops,
                    _budget(envs.SGLANG_NPU_MOE_PREFETCH_GDN_BUDGET_MIB.get(), "gdn"),
                )
        if "moe" in points:
            ops = _ops(envs.SGLANG_NPU_MOE_PREFETCH_MOE_OPS.get(), "moe")
            if ops is not None:
                cfgs["moe"] = _PointCfg(
                    "moe",
                    ops,
                    _budget(envs.SGLANG_NPU_MOE_PREFETCH_MOE_BUDGET_MIB.get(), "moe"),
                )

        # The moe point only exists on the k9 single-stream stack (front
        # fusion routing + shared expert folded into the routed GMM).
        if "moe" in cfgs:
            ff = bool(envs.SGLANG_MOE_FRONT_FUSION.get())
            try:
                from sglang.srt.hardware_backend.npu.tp_fused_tail_npu import (
                    k9_mainstream_enabled,
                )

                k9 = k9_mainstream_enabled()
            except Exception:
                k9 = False
            if not (ff and k9):
                logger.warning(
                    f"[MOE_PREFETCH] 'moe' launch point requires "
                    f"SGLANG_MOE_FRONT_FUSION=1 and k9 active "
                    f"(front_fusion={ff}, k9={k9}); dropped"
                )
                del cfgs["moe"]

        if not cfgs:
            logger.warning("[MOE_PREFETCH] no valid launch point; prefetch disabled")
            return

        self.l2_size = _query_l2_size_bytes()
        if self.l2_size is None:
            logger.warning(
                "[MOE_PREFETCH] failed to query L2 cache size "
                "(aclrtGetDeviceInfo(302)); prefetch disabled"
            )
            return
        ratio = float(envs.SGLANG_NPU_MOE_PREFETCH_L2_CAP_RATIO.get())
        if not (0.0 < ratio <= 1.0):
            logger.warning(
                f"[MOE_PREFETCH] invalid L2_CAP_RATIO={ratio}, fallback to 0.7"
            )
            ratio = 0.7
        self.cap_bytes = int(ratio * self.l2_size)

        self.points = cfgs
        self.enabled = True
        logger.info(
            f"[MOE_PREFETCH] configured: "
            f"points={ {k: (v.ops, v.budget_bytes >> 20) for k, v in cfgs.items()} } "
            f"l2={self.l2_size >> 20}MiB cap={self.cap_bytes >> 20}MiB"
        )

    # ---- Per-point size resolution (once at first MoE layer registration; layers are homogeneous) ----
    def _resolve_sizes(self, w13_bytes: int, w2_bytes: int):
        for cfg in self.points.values():
            if cfg.size_bytes > 0:
                continue
            weight_bytes = w13_bytes if cfg.ops == "gmm1" else w2_bytes
            size = min(cfg.budget_bytes, self.cap_bytes, weight_bytes)
            if size < cfg.budget_bytes:
                logger.warning(
                    f"[MOE_PREFETCH] {cfg.name} budget "
                    f"{cfg.budget_bytes >> 20}MiB clamped to {size >> 20}MiB "
                    f"(cap {self.cap_bytes >> 20}MiB / tensor {weight_bytes >> 20}MiB)"
                )
            cfg.size_bytes = size

    # ---- Model registration (called during model init, outside capture) ----
    def register_model(self, model):
        self.configure()
        if not self.enabled:
            return

        if self.stream is None:
            import torch_npu  # noqa: F401

            # Dedicated CMO stream: created outside capture (driver call), reused as a singleton.
            self.stream = torch.npu.Stream()

        block_types = None
        config = getattr(model, "config", None)
        if config is not None:
            try:
                block_types = config.layers_block_type
            except Exception:
                block_types = None

        n_reg = 0
        layers = getattr(model, "layers", None)
        if layers is None:
            return
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            w13 = getattr(experts, "w13_weight", None)
            w2 = getattr(experts, "w2_weight", None)
            layer_id = getattr(layer, "layer_id", None)
            if w13 is None or w2 is None or layer_id is None:
                continue  # dense MLP / PPMissingLayer / non-MoE layer

            self._resolve_sizes(
                w13.numel() * w13.element_size(),
                w2.numel() * w2.element_size(),
            )
            # Prefetch target: GDN layers only (an FA layer's FIA KV stream
            # self-evicts and steals bandwidth); w13 and w2 share the target set.
            is_gdn = (
                block_types is not None
                and 0 <= layer_id < len(block_types)
                and block_types[layer_id] == "linear_attention"
            )
            self.layers[layer_id] = {
                "w13": w13,
                "w2": w2,
                "prefetch_target": bool(is_gdn),
            }
            # The launch gate is marked per layer object: unregistered layers (e.g.
            # MTP draft model layers, whose layer_id may collide with the registry)
            # never launch.
            layer._moe_prefetch_emit = True
            n_reg += 1

        if n_reg:
            n_target = sum(1 for v in self.layers.values() if v["prefetch_target"])
            logger.info(
                f"[MOE_PREFETCH] registered {n_reg} MoE layers, "
                f"{n_target} GDN prefetch targets"
            )

    # ---- Launch (capture only; best-effort, no wait) ----
    def _emit_for_point(self, point: str, layer, anchor: torch.Tensor):
        cfg = self.points.get(point)
        if cfg is None or cfg.size_bytes <= 0:
            return
        if not self.enabled or not _is_capture_mode():
            return
        if not getattr(layer, "_moe_prefetch_emit", False):
            return  # unregistered layers (MTP draft etc.) never launch
        target = self.layers.get(layer.layer_id)
        if target is None or not target["prefetch_target"]:
            return
        import torch_npu

        weight = target["w13"] if cfg.ops == "gmm1" else target["w2"]
        cur = torch.npu.current_stream()
        self.stream.wait_stream(cur)  # fork edge: the penalty is paid by the prefetch stream
        with torch.npu.stream(self.stream):
            torch_npu.npu_prefetch(weight, anchor, cfg.size_bytes, 0)
        self._emitted_in_pass = True

    def emit(self, layer, anchor: torch.Tensor):
        """gdn point: layer head (after prepare_attn), self-targeted single CMO."""
        self._emit_for_point("gdn", layer, anchor)

    def emit_moe_entry(self, layer, anchor: torch.Tensor):
        """moe point: MoE entry (after prepare_mlp), self-targeted single CMO."""
        self._emit_for_point("moe", layer, anchor)

    # ---- Step-end drain (called after the model forward layer loop) ----
    def drain(self):
        if not self.enabled or not self._emitted_in_pass or not _is_capture_mode():
            return
        torch.npu.current_stream().wait_stream(self.stream)
        self._emitted_in_pass = False


_MANAGER = _MoeWeightPrefetchManager()


# ---------------------------------------------------------------------------
# Public interface (called by qwen3_5.py)
# ---------------------------------------------------------------------------
def moe_prefetch_register_model(model):
    _MANAGER.register_model(model)


def moe_prefetch_emit(layer, anchor: torch.Tensor):
    _MANAGER.emit(layer, anchor)


def moe_prefetch_emit_moe_entry(layer, anchor: torch.Tensor):
    _MANAGER.emit_moe_entry(layer, anchor)


def moe_prefetch_step_drain():
    _MANAGER.drain()
