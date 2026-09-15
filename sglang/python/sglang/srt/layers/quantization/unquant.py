from __future__ import annotations

import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp, copy_or_rebind_param
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DispatchOutput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
# MoE 前段融合包（moe_front_fusion/v1，合入自 daikang 分支）总开关：
# v2.2 自写 init_routing（0 容差逐位验收），仅作用于 BF16 无量化路径。
_moe_front_fusion = envs.SGLANG_MOE_FRONT_FUSION.get()
# GMM2（w2 down_proj）走 sgl_kernel_npu.moe.persistent_gmm 的 Triton
# persistent kernel（默认 stock，开启：SGLANG_GMM2_TRITON=1）。仅作用于
# BF16 无量化无 bias 路径。
_gmm2_triton = envs.SGLANG_GMM2_TRITON.get()
# VGMM1: GMM1 (w13 gate_up_proj) via the vendored GMM v1.2 ops registered by
# the rebuilt sgl_kernel_npu wheel — vgmm1_sched / vgmm1_sched_partial (AIV
# single-core front op: counts/partials -> block-schedule table) + vgmm1_main
# (AIC main kernel, scan-free table lookup). Unquant BF16 no-bias w13 path
# only. Default off: SGLANG_NPU_VGMM1=1.
_vgmm1 = envs.SGLANG_NPU_VGMM1.get()
# The on-machine verified domain is small decode M; larger M (prefill) is
# unverified and falls back to stock. Default 1024 covers decode bs<=128 x
# topk8 — raise only after re-running large-M T0/T1/T2.
_vgmm1_max_m = envs.SGLANG_NPU_VGMM1_MAX_M.get()
# Diagnostics: log guard misses and the counts-vs-table consistency account
# (skipped during graph capture — device sync is illegal there).
_vgmm1_debug = envs.SGLANG_NPU_VGMM1_DEBUG.get()
# Host-side staged-verification knob read per call by the csrc host; the
# wrapper only needs to recognize 6 (host sentinel tier over-returns y, which
# is sliced back to [total_m, n] here). All other tiers are wrapper-agnostic.
_vgmm1_stage = os.environ.get("VGMM_STAGE", "")


def _vgmm1_is_capturing() -> bool:
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


# vgmm1_query_tile is a pure host query (no kernel launch, CPU tensor out);
# cache baseM/baseN per (m, k, n) — decode graph shapes are fixed, computed
# once at capture, zero device sync, capture-safe.
_vgmm1_tile_cache: Dict[tuple, tuple] = {}


def _vgmm1_tile(m: int, k: int, n: int):
    key = (m, k, n)
    tile = _vgmm1_tile_cache.get(key)
    if tile is None:
        qt = torch.ops.npu.vgmm1_query_tile(m, k, n)
        tile = (int(qt[0]), int(qt[1]))
        _vgmm1_tile_cache[key] = tile
    return tile

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast


class Bf16GemmBackend(Enum):
    AUTO = "auto"
    CUTEDSL = "cutedsl"

    def is_auto(self) -> bool:
        return self == Bf16GemmBackend.AUTO

    def is_cutedsl(self) -> bool:
        return self == Bf16GemmBackend.CUTEDSL


_BF16_GEMM_BACKEND: Optional[Bf16GemmBackend] = None
_cutedsl_bf16_gemm = None
_use_cutedsl_bf16_gemm = None


def initialize_bf16_gemm_config(server_args: ServerArgs) -> None:
    global _BF16_GEMM_BACKEND, _cutedsl_bf16_gemm, _use_cutedsl_bf16_gemm

    backend = Bf16GemmBackend(server_args.bf16_gemm_backend)

    if backend.is_cutedsl():
        from sglang.srt.utils import is_sm100_supported

        if not is_sm100_supported():
            raise ValueError(
                "--bf16-gemm-backend cutedsl requires SM100/SM103 (Blackwell)"
            )

        from sglang.jit_kernel.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm,
        )

        _cutedsl_bf16_gemm = cutedsl_bf16_gemm
        _use_cutedsl_bf16_gemm = use_cutedsl_bf16_gemm

    _BF16_GEMM_BACKEND = backend


def get_bf16_gemm_backend() -> Bf16GemmBackend:
    global _BF16_GEMM_BACKEND
    if _BF16_GEMM_BACKEND is None:
        _BF16_GEMM_BACKEND = Bf16GemmBackend.AUTO
    return _BF16_GEMM_BACKEND


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        elif (
            get_bf16_gemm_backend().is_cutedsl()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and layer.weight.dtype == torch.bfloat16
            and (bias is None or bias.dtype == torch.bfloat16)
            and _use_cutedsl_bf16_gemm(
                x.numel() // x.shape[-1],
                layer.weight.shape[0],
                layer.weight.shape[1],
            )
        ):
            x_shapes = x.shape
            output = _cutedsl_bf16_gemm(x.view(-1, x_shapes[-1]), layer.weight, bias)
            return output.view(*x_shapes[:-1], -1)

        return F.linear(x, layer.weight, bias)


class FusedTailPieces(NamedTuple):
    """Pre-finalize MoE pieces for the FUSED_TAIL deferred path (see
    hardware_backend/npu/tp_fused_tail_npu.py): xexp = post-GMM2 expert
    outputs in expanded order [M*K, H] bf16; eri = flat-slot (t*top_k+k) ->
    expanded-row index int32 [M*K]; scales_fp32 = pre-cast topk weights
    (fp32 on the NPU topk contract)."""

    xexp: torch.Tensor
    eri: torch.Tensor
    scales_fp32: torch.Tensor


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    # Same-batch deployment marker for the FUSED_TAIL gate: this method can
    # produce deferred pre-finalize pieces via forward_npu(deferred=True).
    fused_tail_deferred_supported = True

    def __init__(
        self,
        use_triton_kernels: bool = False,
        use_flashinfer_trtllm_moe: bool = False,
        use_deep_gemm: bool = False,
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self.use_deep_gemm = use_deep_gemm
        self._cache_permute_indices = dict({})

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _should_use_aiter_moe = (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and self._aiter_ck_moe_supported(layer)
        )
        if _should_use_aiter_moe:
            copy_or_rebind_param(
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()
            copy_or_rebind_param(
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
            if hasattr(layer, "w13_weight_bias"):
                layer.w13_weight_bias = Parameter(
                    layer.w13_weight_bias.float(), requires_grad=False
                )
            if hasattr(layer, "w2_weight_bias"):
                layer.w2_weight_bias = Parameter(
                    layer.w2_weight_bias.float(), requires_grad=False
                )

        if (
            self.use_deep_gemm
            and layer.w13_weight.dtype == torch.bfloat16
            and get_moe_a2a_backend().is_deepep()
            and get_deepep_mode().enable_low_latency()
            and not _is_npu
            and not _is_hip
            and hasattr(layer, "dispatcher")
        ):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                    is_gated_act_gemm=layer.moe_runner_config.is_gated,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )
        if _is_npu and _gmm2_triton:
            # Triton GMM2 kernel 以裸指针按 [E, N, K] ND 连续存储读取 w2，
            # 必须保证 ND 连续：FRACTAL_NZ 分形重排会让裸指针读取出错
            # （本分支 unquant NPU 路径本身不做 format cast，此处仅兜底）。
            layer.w2_weight.data = layer.w2_weight.data.contiguous()
        # if _is_npu:
        #     for weight_name in ["w13_weight", "w2_weight"]:
        #         weight = getattr(layer, weight_name)
        #         weight.data = npu_format_cast(weight)

        return

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        self,
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        weight_name: str,
    ) -> None:
        """Restore canonical BF16 MoE load shapes before hot weight copy.

        The flashinfer TRT-LLM BF16 postprocess reshapes expert weights into
        block layout. During weight update, checkpoint tensors are in
        canonical layout and need a temporary shape restore for copy.
        """
        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
            return

        expected_shape = None
        if weight_name.endswith(".experts.w13_weight"):
            w13_rows = (
                2 * layer.intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else layer.intermediate_size_per_partition
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)
        elif weight_name.endswith(".experts.w2_weight"):
            expected_shape = (
                layer.num_local_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:
            return

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        if param.data.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."
            )

        param.data = param.data.reshape(expected_shape)

    def _aiter_ck_moe_supported(self, layer) -> bool:
        # aiter CK fused-MoE requires intermediate_size_per_partition to be 128-aligned
        # (GemmSpec=Default; otherwise CK raises "not support this GEMM problem").
        return layer.intermediate_size_per_partition % 128 == 0

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()
                else MoeRunnerBackend.FLASHINFER_TRTLLM
            )
        elif self.use_flashinfer_cutlass:
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

            backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        elif self.use_deep_gemm:
            backend = MoeRunnerBackend.DEEP_GEMM
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

        # aiter CK fused-MoE only supports 128-aligned shapes; otherwise use triton.
        self._aiter_runner: Optional[MoeRunner] = None
        if (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and get_moe_a2a_backend().supports_aiter()
        ):
            if self._aiter_ck_moe_supported(layer):
                self._aiter_runner = MoeRunner(
                    MoeRunnerBackend.AITER, moe_runner_config
                )
            elif get_moe_runner_backend().is_aiter():
                raise ValueError(
                    "moe_runner_backend=aiter is not supported for "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "use --moe-runner-backend triton."
                )
            else:
                logger.warning_once(
                    "aiter CK fused-MoE does not support "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "using triton MoE runner."
                )

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        x = dispatch_output.hidden_states

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.runner.runner_backend.is_deep_gemm():
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight
            from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

            # Only use_fp8=False when SGLANG_DEEPEP_BF16_DISPATCH is true,
            # otherwise use_fp8=True for FP8 dispatch path
            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=use_fp8,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="bf16",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                output_dtype=x.dtype,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=not layer.should_fuse_routed_scaling_factor_in_topk,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            if self._aiter_runner is not None:
                from sglang.srt.layers.moe.moe_runner.aiter import (
                    AiterMoeQuantInfo,
                )

                quant_info = AiterMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    expert_mask=layer.dispatcher.expert_mask_gpu,
                )
                return self._aiter_runner.run(dispatch_output, quant_info)

            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                getattr(layer, "w13_weight_bias", None),
                getattr(layer, "w2_weight_bias", None),
                layer.moe_runner_config.gemm1_alpha,
                layer.moe_runner_config.gemm1_clamp_limit,
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
            "relu2",  # Nemotron-H (NemotronHForCausalLM) uses squared-ReLU.
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            if moe_runner_config.apply_router_weight_on_input:
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights)
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
        deferred: bool = False,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep(dispatch_output):
            if deferred:
                raise RuntimeError(
                    "fused_tail: the deferred path is incompatible with deepep dispatch"
                )
            return self._forward_npu_deepep(layer, dispatch_output)

        # x.shape = [B*S, H]
        x = dispatch_output.hidden_states
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        # FUSED_TAIL: keep the pre-cast scales reference for the deferred path
        # (fp32 on the NPU topk contract; the stock finalize consumes the bf16
        # cast below, so the fused side is strictly no worse).
        topk_weights_raw = topk_weights
        original_dtype = x.dtype
        num_tokens = x.shape[0]
        # v4.1: the bf16 cast is dead on the deferred path (the pieces carry
        # the raw reference and the fused kernels widen scales in-kernel), so
        # issue it only where the stock finalize actually consumes it.
        if not deferred:
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set

        m = num_tokens * top_k
        h = x.shape[1]
        expert_offsets = None
        # C1 table triple: when the front arm below takes the v22_partials
        # path, vgmm1_sched_partial emits the block-schedule table right here
        # and the w13 vgmm1 branch skips its vgmm1_sched node (the Triton
        # partials_cumsum node leaves the chain entirely).
        vgmm1_block_table = None
        vgmm1_total_blocks = None
        vgmm1_table_n = None
        if (
            _moe_front_fusion
            and m <= 512
            and m % 8 == 0
            and (h & (h - 1)) == 0
            and x.dtype == torch.bfloat16
        ):
            # v2.2 自写 init routing（moe_front_fusion/v1，合入自 daikang 分支，
            # 六轮单测 0 容差逐位验收）：语义与 npu_moe_init_routing_v2(type=1)
            # 逐位一致，并原生输出 exclusive offsets（int32 [E]）供 persistent
            # GMM2 offsets= 直用。形态门外回退 stock v2。
            c1 = (
                _vgmm1
                and not self.with_bias
                and m <= _vgmm1_max_m
                # day-0 gate: a wheel without vgmm1_sched_partial falls back
                # to the v22 five-tuple arm (the w13 branch then builds the
                # table via vgmm1_sched — behavior identical to pre-C1).
                and hasattr(torch.ops.npu, "vgmm1_sched_partial")
            )
            if c1:
                # C1: rank_hist + gather two launches (cumsum node retired),
                # partials handed to the AscendC vgmm1_sched_partial which does
                # the column reduction (counts/excl) and the table build in one
                # pass. max_blocks upper bound = m + E (same convention as the
                # w13 branch's total_m + num_experts).
                from sgl_kernel_npu.moe.moe_front_routing import (
                    moe_init_routing_v22_partials,
                )

                _n13 = layer.w13_weight.shape[1]
                _k13 = layer.w13_weight.shape[2]
                _base_m, _base_n = _vgmm1_tile(m, _k13, _n13)
                hidden_states, expanded_row_idx, partials = (
                    moe_init_routing_v22_partials(x, topk_ids, num_experts, top_k)
                )
                (
                    vgmm1_block_table,
                    excl,
                    vgmm1_total_blocks,
                    expert_tokens,
                ) = torch.ops.npu.vgmm1_sched_partial(
                    partials, num_experts, m, _n13, _k13,
                    m + num_experts, _base_m, _base_n,
                )
                vgmm1_table_n = _n13
                expert_offsets = excl
            else:
                from sgl_kernel_npu.moe.moe_front_routing import moe_init_routing_v22

                hidden_states, expanded_row_idx, expert_tokens, excl, _incl = (
                    moe_init_routing_v22(x, topk_ids, num_experts, top_k)
                )
                expert_offsets = excl
        else:
            hidden_states, expanded_row_idx, expert_tokens, _ = (
                torch.ops.npu.npu_moe_init_routing_v2(
                    x,
                    topk_ids,
                    active_num=m,
                    expert_num=num_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, num_experts],
                    quant_mode=-1,
                )
            )
            expert_tokens = expert_tokens.to(torch.int64)
        w13_bias = [layer.w13_weight_bias] if self.with_bias else None
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None

        # gmm1: gate_up_proj
        _w13 = layer.w13_weight
        _total_m = hidden_states.shape[0]
        # Guards only read tensor attributes/static shapes — graph-capture
        # safe; a miss falls back to stock silently (an NZ/non-contiguous w13
        # is structurally rejected by is_contiguous()).
        if (
            _vgmm1
            and w13_bias is None
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and _w13.dtype == torch.bfloat16
            and _w13.dim() == 3
            and _w13.is_contiguous()
            and _w13.shape[2] == hidden_states.shape[1]
            and expert_tokens.dtype == torch.int64
            and expert_tokens.is_contiguous()
            and 0 < _total_m <= _vgmm1_max_m
        ):
            # One-shot forensics print (host metadata, no device sync,
            # capture-safe): contract = w13 [E, N, K] ND contiguous (fmt=0);
            # fmt=29 (FRACTAL_NZ) means the layout is broken — check this line
            # first when debugging.
            try:
                import torch_npu as _torch_npu

                _w13_fmt = _torch_npu.get_npu_format(_w13)
            except Exception:
                _w13_fmt = "unknown"
            logger.warning_once(
                f"[VGMM1] w13 fmt={_w13_fmt}(0=ND,29=NZ) "
                f"shape={tuple(_w13.shape)} contig={_w13.is_contiguous()} "
                f"dtype={_w13.dtype} | x shape={tuple(hidden_states.shape)} "
                f"dtype={hidden_states.dtype} contig={hidden_states.is_contiguous()} "
                f"| counts shape={tuple(expert_tokens.shape)} "
                f"dtype={expert_tokens.dtype} | total_m={_total_m}"
            )
            _e, _n, _k = _w13.shape
            # baseM/baseN: same cached host query as the front arm (same key,
            # same inputs) so the table build and the main kernel's read agree.
            _base_m, _base_n = _vgmm1_tile(_total_m, _k, _n)
            if (
                vgmm1_block_table is not None
                and vgmm1_total_blocks is not None
                and vgmm1_table_n == _n
            ):
                # C1: the front arm's vgmm1_sched_partial already emitted the
                # table (bitwise-identical to vgmm1_sched, UT-exhaustive
                # checked) — the sched node is skipped entirely. A table_n
                # mismatch refuses this branch and rebuilds via sched: a
                # derivation error costs performance, never correctness.
                _block_table, _total_blocks = (
                    vgmm1_block_table,
                    vgmm1_total_blocks,
                )
            else:
                # max_blocks upper bound = total_m + E (per non-empty group
                # blocks <= ceil(m/baseM) <= m/baseM + 1, total <=
                # total_m/baseM + E).
                _block_table, _row_offsets, _total_blocks = (
                    torch.ops.npu.vgmm1_sched(
                        expert_tokens,
                        _total_m,
                        _n,
                        _k,
                        _total_m + _e,
                        _base_m,
                        _base_n,
                    )
                )
            if _vgmm1_debug and not _vgmm1_is_capturing():
                # Diagnostic sync point (reached only in warmup/eager; skipped
                # under capture). The sched table's row ranges are clamped to
                # sum(counts), so max_write_end > total_m implies counts/table
                # vs x row-count inconsistency; == total_m means the table is
                # clean and any OOB lives in the main kernel itself.
                _sum = int(expert_tokens.sum())
                _tb = int(_total_blocks[0])
                _t = _block_table.view(-1, 8)[:_tb].to(torch.int64)
                _max_end = (
                    int((_t[:, 1] + _t[:, 2] * _base_m + _t[:, 4]).max())
                    if _tb > 0
                    else 0
                )
                logger.warning(
                    f"[VGMM1_DBG] total_m={_total_m} sum={_sum} "
                    f"min={int(expert_tokens.min())} "
                    f"max={int(expert_tokens.max())} "
                    f"numel={expert_tokens.numel()} blocks={_tb} "
                    f"max_write_end={_max_end} "
                    f"base_m={_base_m} base_n={_base_n}"
                )
            _y = torch.ops.npu.vgmm1_main(
                hidden_states, _w13, _block_table, _total_blocks
            )
            if _vgmm1_stage == "6":
                # Host sentinel tier over-returns guard rows; slice back.
                _y = _y[:_total_m]
            hidden_states = _y
        else:
            if _vgmm1 and w13_bias is None:
                logger.warning_once(
                    f"[VGMM1] guard miss, fallback to stock npu_grouped_matmul: "
                    f"x(dtype={hidden_states.dtype},contig={hidden_states.is_contiguous()}) "
                    f"w13(shape={tuple(_w13.shape)},dtype={_w13.dtype},"
                    f"contig={_w13.is_contiguous()}) "
                    f"counts(dtype={expert_tokens.dtype}) total_m={_total_m} "
                    f"max_m={_vgmm1_max_m}"
                )
            hidden_states = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w13_weight.transpose(1, 2)],
                bias=w13_bias,
                split_item=2,
                group_list_type=1,
                group_type=0,
                group_list=expert_tokens,
                output_dtype=original_dtype,
            )[0]

        # act_fn:
        if self.moe_runner_config.activation == "npu_swiglu_oai":
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

            hidden_states = swiglu_oai(layer, hidden_states)
        elif self.moe_runner_config.activation == "silu":
            if self.moe_runner_config.gemm1_clamp_limit is not None:
                from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

                hidden_states, _ = swiglu_quant(
                    hidden_states,
                    group_list=expert_tokens,
                    group_list_type=1,
                    need_quant=False,
                    do_limit=True,
                    limit=self.moe_runner_config.gemm1_clamp_limit,
                )
            else:
                hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        if _gmm2_triton and w2_bias is None:
            # Triton persistent GMM2（gmm/v1，合入自 daikang 分支）：w2 为
            # [E, N, K] ND 连续存储（跳过 FRACTAL_NZ cast），expert_tokens 为
            # per-expert counts（group_list_type=1，int64）。offsets 缺省时由
            # 内置前置 kernel 推导；SGLANG_MOE_FRONT_FUSION=1 时 v2.2 init
            # 原生 excl 直喂，省 _gmm2_offsets_kernel ~4.4µs/层。
            from sgl_kernel_npu.moe.persistent_gmm import persistent_grouped_matmul

            hidden_states = persistent_grouped_matmul(
                hidden_states,
                layer.w2_weight,
                expert_tokens,
                offsets=expert_offsets,
            )
        else:
            hidden_states = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w2_weight.transpose(1, 2)],
                bias=w2_bias,
                split_item=2,
                group_list_type=1,
                group_type=0,
                group_list=expert_tokens,
                output_dtype=original_dtype,
            )[0]

        if deferred:
            # FUSED_TAIL (MoE layer-tail fusion): skip the stock finalize and
            # hand the pre-finalize pieces up to qwen2_moe's fused chain
            # (fin+add+AR+norm). expanded_row_idx is the flat-slot
            # (t*top_k+k) -> expanded-row index for both init-routing arms
            # above (v22 is bitwise-identical to stock v2), exactly the fused
            # kernel's eri contract.
            if expanded_row_idx.dtype != torch.int32:
                expanded_row_idx = expanded_row_idx.to(torch.int32)
            return FusedTailPieces(
                xexp=hidden_states,
                eri=expanded_row_idx.contiguous(),
                scales_fp32=topk_weights_raw,
            )

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def _forward_npu_deepep(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            hidden_states, _, _, _, num_recv_tokens_per_expert = dispatch_output
            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )
            combine_cls = DeepEPNormalCombineInput
        else:
            hidden_states, _, _, _, group_list, _ = dispatch_output
            group_list = group_list.to(torch.int64)
            combine_cls = DeepEPLLCombineInput

        hidden_states = npu_fused_moe_without_routing_weights_bf16(
            layer, hidden_states, group_list_type, group_list, output_dtype
        )
        return combine_cls(
            hidden_states=hidden_states,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    def forward_musa(self, *args, **kwargs) -> CombineInput:
        return self.forward_cuda(*args, **kwargs)

    forward_native = forward_cpu
