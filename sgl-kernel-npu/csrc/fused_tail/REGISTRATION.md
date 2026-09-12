# REGISTRATION — fused_tail three ops (MoE layer-tail fin+add+AR+norm fusion)

Registration: the three `torch.ops.npu.*` ops ship inside `libsgl_kernel_npu.so`
and are loaded by `sgl_kernel_npu/__init__.py` via `torch.ops.load_library`.

## 1. Op list

| op | kernel (one kernel per file) | purpose |
|---|---|---|
| `torch.ops.npu.fused_fin_ar_norm` | `op_kernel/fused_fin_ar_norm_kernel.cpp` (AIV_ONLY) | fin(+skip1) + spin AIV AR + add + gemma rmsnorm single kernel (spin variant) |
| `torch.ops.npu.fused_fin_add` | `op_kernel/fused_fin_add_kernel.cpp` (AIV_ONLY) | fin(+skip1) local front stage (first half of the stock HCCL AR variant; no symmem) |
| `torch.ops.npu.fused_tail_zero` | `op_kernel/fused_tail_zero_kernel.cpp` (AIV_ONLY) | MTE3 clearing of the flag symmem (init/reset; torch zero_() forbidden) |

Schemas (FUSED_TAIL section of `csrc/pytorch_extensions.cpp`):

```
fused_fin_ar_norm(Tensor x, Tensor scales, Tensor skip1, Tensor residual,
    Tensor(a!) add_out, Tensor(b!) norm_out, Tensor addr_tab,
    int eri_va, int norm_w_va, int m, int h, int k, int ncores, int rank, int world,
    int has_skip1, int use_eri, float eps, int cycle_limit_us,
    int slot_stride, int ring_stride, int max_tiles,
    int counter_offset, int dfx_offset) -> ()
fused_fin_add(Tensor x, Tensor scales, Tensor skip1, Tensor(a!) out,
    int eri_va, int m, int h, int k, int ncores, int has_skip1, int use_eri) -> ()
fused_tail_zero(Tensor(a!) x, int nbytes, int ncores) -> ()
```

## 2. Kernel entry signature convention (auto_gen "last two params = workspace/tiling")

```
fused_fin_ar_norm(x, scales, skip1, residual, add_out, norm_out, addr_tab, dummy_workspace, tiling)
fused_fin_add(x, scales, skip1, out, dummy_workspace, tiling)
fused_tail_zero(x, dummy_workspace, tiling)
```

- The workspace slot always carries a dummy: that slot's argument is
  deterministically corrupted on arrival at the kernel (unresolved), so no
  valid parameter is ever placed there; the host passes addr_tab/out/x
  duplicates which the kernel ignores.
- Build: dedicated `ascendc_library(fused_tail_kernel ...)` with
  `-DHAVE_WORKSPACE -DHAVE_TILING` and `--cce-auto-sync=on` (matches the probe
  validation baseline; all sync is handwritten).
- Stubs: `op_host/stub/aclrtlaunch_{fused_fin_ar_norm,fused_fin_add,fused_tail_zero}.h`
  handwritten fallbacks when auto_gen does not emit them; signatures match the
  kernel entries parameter by parameter.

## 3. Host TORCH_CHECK list (python wrapper guards must mirror these)

- Common: h in {2048,4096}; m in [1,4096]; k in [1,16]; ncores in [1,48];
  has_skip1/use_eri in {0,1}; use_eri=1 implies eri_va != 0; x contiguous
  bf16 [M*K,H]; scales contiguous fp32 [M,K]; has_skip1=1 implies skip1
  contiguous bf16 [M,H].
- fused_fin_ar_norm: addr_tab contiguous int64[>=18]; world in [1,8];
  rank in [0,world); residual/add_out/norm_out contiguous bf16 [M,H];
  norm_w_va != 0; m*h*2 % 8192 == 0 (m even for H2048); tiles <= max_tiles;
  ring_stride == world*slot_stride; payload <= slot_stride (prevents pushing
  out of bounds into the peer's symmem); counter_offset == 0,
  dfx_offset == 48*128.
- fused_fin_add: out contiguous bf16 [M,H]; no tile alignment requirement
  (row-granular split).
- fused_tail_zero: x contiguous with numel >= nbytes; nbytes a positive
  multiple of 32.

## 4. Tiling cache

Whole-struct FNV-1a hash + static device buffer (1024 slots) for graph-capture
pointer stability; entries' pinned staging is kept alive statically (the H2D
copy is recorded into the graph, so the source must stay immutable). Flag
layout derivatives (flag_ring_stride/flag_phase_stride) are derived host-side
with the same formulas as the python wrapper (single source of truth).

## 5. Day-0 checks

```bash
nm -D $SGLK_PKG/lib/libsgl_kernel_npu.so | grep "U aclrtlaunch_fused_fi" \
  && echo "missing: unresolved launch symbols, rebuild all"
strings $SGLK_PKG/lib/libsgl_kernel_npu.so | grep -c "ascend.meta.fused_fi"   # must be 3
python - <<'EOF'
import torch, sgl_kernel_npu
for name in ("fused_fin_ar_norm", "fused_fin_add", "fused_tail_zero"):
    assert hasattr(torch.ops.npu, name), f"{name} not registered"
print("fused_tail ops registered OK")
EOF
```

Kernel build stamp (spin init self-check): csrc `FUSED_TAIL_BUILD_REV` and the
wrapper's `_FT_EXPECT_BUILD_REV` share one value = 20260922; bump both on any
kernel change.
