"""MoE expert weight L2 prefetch for NPU decode (aclgraph replay).

Design:
- Self-targeted single launch point: after this layer's prepare_attn, before
  attention starts; the layer's own weight prefetch is launched there. EP
  inter-layer communication (reduceScatter + allGather) completes inside the next
  layer's prepare_attn, and TP's layer-end all_reduce inside the previous layer's
  postprocess_layer — so this point sits after all inter-layer communication on
  both paths and does not contend with it for bandwidth; self-targeting also gives
  layer 0 an attention window.
- GMM1 (w13) full prefetch: launches w13(j), GDN layers only; on by default.
- GMM2 (w2) full prefetch: merged into the same launch point, w2(j) chunks appended
  after w13(j) (consumption order: w2 is consumed by GMM2(j), one GMM1+SwiGLU later
  than w13); off by default. Merged-launch gate: w13+w2 <= 0.8*L2.
- Best-effort: the main stream does not wait for the CMO stream before GMM (cache
  warm-up has no RAW hazard); drained at step end (prevents cross-step backlog, and
  capture legality requires the side stream to return to the main stream via event).
- Launched only during graph capture (eager / prefill never activate); CMO is an
  SDMA task and occupies no AIV/AIC cores.
- Only layer objects registered for the target model trigger launches
  (`_moe_prefetch_emit` marker); layers of the MTP draft model never launch even if
  they reuse this forward from the same file.

Note: torch_npu is always lazily imported inside functions, so this module is safe
to import on CUDA and other platforms.
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
_VALID_MODES = ("auto", "full", "active")


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


# ---------------------------------------------------------------------------
# Manager (process-level singleton)
# ---------------------------------------------------------------------------
class _MoeWeightPrefetchManager:
    def __init__(self):
        self.configured = False
        self.enabled = False
        self.prefetch_gmm1 = False
        self.prefetch_gmm2 = False
        self.mode = "full"
        self.chunk_bytes = 16 << 20
        self.l2_size: Optional[int] = None
        self.budget_bytes = 0
        self.stream = None
        # layer_id -> {"w13": Tensor, "w2": Tensor, "prefetch_target": bool}
        self.layers: Dict[int, dict] = {}
        self._capacity_checked = False
        self._emitted_in_pass = False

    # ---- Configuration (parsed once at first registration) ----
    def configure(self):
        if self.configured:
            return
        self.configured = True

        if not envs.SGLANG_NPU_MOE_PREFETCH.get():
            return  # enabled stays False; everything below is a no-op

        ops_raw = str(envs.SGLANG_NPU_MOE_PREFETCH_OPS.get()).lower()
        ops = [t.strip() for t in ops_raw.split(",") if t.strip()]
        unknown = [t for t in ops if t not in _VALID_OPS]
        if unknown:
            logger.warning(
                f"[MOE_PREFETCH] unknown ops {unknown} ignored, valid: {_VALID_OPS}"
            )
        self.prefetch_gmm1 = "gmm1" in ops
        self.prefetch_gmm2 = "gmm2" in ops

        mode = str(envs.SGLANG_NPU_MOE_PREFETCH_MODE.get()).lower()
        if mode not in _VALID_MODES:
            logger.warning(
                f"[MOE_PREFETCH] unknown mode {mode!r}, fallback to 'auto'"
            )
            mode = "auto"
        if mode == "active":
            # In-graph npu_prefetch offset/max_size are capture-time constants and
            # nodes execute unconditionally, so runtime active indices cannot be fed
            # into the baked CMO params; 'active' would need a copy-kernel +
            # write-allocate design. Fall back to 'full'.
            logger.warning(
                "[MOE_PREFETCH] mode='active' is not supported "
                "(in-graph CMO params are capture-time constants); "
                "falling back to 'full'"
            )
            mode = "full"
        self.mode = "full" if mode == "auto" else mode  # auto == full

        chunk_mib = int(envs.SGLANG_NPU_MOE_PREFETCH_CHUNK_MIB.get())
        if chunk_mib <= 0:
            logger.warning(
                f"[MOE_PREFETCH] invalid CHUNK_MIB={chunk_mib}, fallback to 16"
            )
            chunk_mib = 16
        self.chunk_bytes = chunk_mib << 20

        self.l2_size = _query_l2_size_bytes()
        if self.l2_size is None:
            logger.warning(
                "[MOE_PREFETCH] failed to query L2 cache size "
                "(aclrtGetDeviceInfo(302)); prefetch disabled"
            )
            return

        budget_mib = int(envs.SGLANG_NPU_MOE_PREFETCH_BUDGET_MIB.get())
        self.budget_bytes = (
            (budget_mib << 20) if budget_mib > 0 else int(0.8 * self.l2_size)
        )

        self.enabled = self.prefetch_gmm1 or self.prefetch_gmm2
        logger.info(
            f"[MOE_PREFETCH] configured: enabled={self.enabled} "
            f"gmm1={self.prefetch_gmm1} gmm2={self.prefetch_gmm2} "
            f"mode={self.mode} chunk={chunk_mib}MiB "
            f"l2={self.l2_size >> 20}MiB budget={self.budget_bytes >> 20}MiB"
        )

    # ---- Capacity check (checked once at first MoE layer registration; layers are homogeneous) ----
    def _check_capacity(self, w13_bytes: int, w2_bytes: int):
        if self._capacity_checked:
            return
        self._capacity_checked = True
        mib = 1 << 20
        if self.prefetch_gmm1 and w13_bytes > self.budget_bytes:
            logger.warning(
                f"[MOE_PREFETCH] w13 size {w13_bytes / mib:.1f}MiB > budget "
                f"{self.budget_bytes / mib:.1f}MiB (0.8*L2 or BUDGET_MIB); "
                f"gmm1 prefetch disabled for this config "
                f"(e.g. TP4 w13=256MiB is a known unsupported case)"
            )
            self.prefetch_gmm1 = False
        if self.prefetch_gmm2:
            # Merged-launch gate: w13 (if gmm1 is on) + w2 stay resident in the same
            # cross-layer window; gmm2 is allowed only if the total fits the budget.
            resident = (w13_bytes if self.prefetch_gmm1 else 0) + w2_bytes
            if resident > self.budget_bytes:
                logger.warning(
                    f"[MOE_PREFETCH] resident w13+w2 size {resident / mib:.1f}MiB "
                    f"> budget {self.budget_bytes / mib:.1f}MiB; "
                    f"merged gmm2 prefetch disabled for this config "
                    f"(w13 and w2 launch together, so both must fit; "
                    f"TP8 w13+w2=192MiB is a known unsupported case)"
                )
                self.prefetch_gmm2 = False
        self.enabled = self.prefetch_gmm1 or self.prefetch_gmm2
        if self.enabled:
            logger.info(
                f"[MOE_PREFETCH] capacity check passed: w13={w13_bytes / mib:.1f}MiB "
                f"(gmm1={self.prefetch_gmm1}) w2={w2_bytes / mib:.1f}MiB "
                f"(gmm2={self.prefetch_gmm2})"
            )

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

            self._check_capacity(
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
    def _emit_chunked(self, weights, anchor: torch.Tensor):
        import torch_npu

        cur = torch.npu.current_stream()
        self.stream.wait_stream(cur)  # fork edge: the penalty is paid by the prefetch stream
        with torch.npu.stream(self.stream):
            for weight in weights:
                nbytes = weight.numel() * weight.element_size()
                offset = 0
                while offset < nbytes:
                    size = min(self.chunk_bytes, nbytes - offset)
                    torch_npu.npu_prefetch(weight, anchor, size, offset)
                    offset += size
        self._emitted_in_pass = True

    def emit(self, layer, anchor: torch.Tensor):
        """Launch this layer's w13(+w2) prefetch before its attention starts (after all inter-layer communication).

        Launch point = after prepare_attn in decoder layer forward. EP inter-layer
        communication (reduceScatter + allGather) completes inside the next layer's
        prepare_attn and TP's layer-end AR inside the previous layer's
        postprocess_layer, so this point is after all inter-layer communication on
        both paths. Chunk order: w13 first (consumed by GMM1), w2 after (consumed by
        GMM2, with one extra GMM1+SwiGLU of slack).
        """
        if not self.enabled or not _is_capture_mode():
            return
        if not getattr(layer, "_moe_prefetch_emit", False):
            return  # unregistered layers (MTP draft etc.) never launch
        target = self.layers.get(layer.layer_id)
        if target is None or not target["prefetch_target"]:
            return
        weights = []
        if self.prefetch_gmm1:
            weights.append(target["w13"])
        if self.prefetch_gmm2:
            weights.append(target["w2"])
        if not weights:
            return
        self._emit_chunked(weights, anchor)

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


def moe_prefetch_step_drain():
    _MANAGER.drain()
