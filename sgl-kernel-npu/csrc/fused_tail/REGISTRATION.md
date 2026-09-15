# REGISTRATION — fused_tail three-op registration/checklist (tp_ascendc_fusion_v4.1)

Registration: the three `torch.ops.npu.*` ops ship inside
`libsgl_kernel_npu.so` and are loaded by `sgl_kernel_npu/__init__.py` via
`torch.ops.load_library`.

**v4.1 changes** (relative to v4; eliminates the [cast + mem x2] small ops in
front of every fused op in graph-mode profiling):

1. **eri/norm_w changed from int VA scalars to tensor parameters**
   (`fused_fin_ar_norm`'s `int eri_va, int norm_w_va` -> `Tensor eri,
   Tensor norm_w`, positions unchanged; `fused_fin_add`'s `int eri_va` ->
   `Tensor eri`). The tiling struct no longer contains any VA (the old
   fields are renamed `reserved_va0/1`, kept always-0 to preserve offsets),
   and the **tiling hash degenerates to shape level** — 40 layers share 1
   entry, so the per-layer in-graph tiling H2D+D2D copy pair (mem x2) drops
   to at most 1 pair per bs tier. With use_eri=0, eri is a dummy tensor
   (kernel does not read it, host does not validate its contents, but the
   schema requires defined).
2. **scales accepts fp32 or bf16** (bf16 is widened to fp32 exactly
   in-kernel, bit-equivalent to a host-side `.float()`; new tiling field
   `scales_bf16`, included in the hash) — the wrapper's `scales.float()`
   fallback is deleted (cast small-op eliminated).
3. Build stamp `FUSED_TAIL_BUILD_REV`: 20260922 -> **20260924** (wrapper
   `_FT_EXPECT_BUILD_REV` shares the same value on both sides; bump both on
   any kernel change).
4. `fused_tail_zero` signature unchanged.

## 1. Op list

| op | kernel (one kernel per file) | purpose |
|---|---|---|
| `torch.ops.npu.fused_fin_ar_norm` | `op_kernel/fused_fin_ar_norm_kernel.cpp` (AIV_ONLY) | fin(+skip1) + spin AIV AR + add + gemma rmsnorm single kernel (spin variant) |
| `torch.ops.npu.fused_fin_add` | `op_kernel/fused_fin_add_kernel.cpp` (AIV_ONLY) | fin(+skip1) local front stage (first half of the HCCL variant; no symmem) |
| `torch.ops.npu.fused_tail_zero` | `op_kernel/fused_tail_zero_kernel.cpp` (AIV_ONLY) | MTE3 clearing of the flag symmem (init/reset; torch zero_() forbidden) |

Schemas (FUSED_TAIL section of `csrc/pytorch_extensions.cpp`):

```
fused_fin_ar_norm(Tensor x, Tensor scales, Tensor skip1, Tensor residual,
    Tensor(a!) add_out, Tensor(b!) norm_out, Tensor addr_tab,
    Tensor eri, Tensor norm_w, int m, int h, int k, int ncores, int rank, int world,
    int has_skip1, int use_eri, float eps, int cycle_limit_us,
    int slot_stride, int ring_stride, int max_tiles,
    int counter_offset, int dfx_offset) -> ()
fused_fin_add(Tensor x, Tensor scales, Tensor skip1, Tensor(a!) out,
    Tensor eri, int m, int h, int k, int ncores, int has_skip1, int use_eri) -> ()
fused_tail_zero(Tensor(a!) x, int nbytes, int ncores) -> ()
```

## 2. Kernel entry signature convention (auto_gen "last two params = workspace/tiling")

```
fused_fin_ar_norm(x, scales, skip1, residual, add_out, norm_out, addr_tab, eri, norm_w, dummy_workspace, tiling)
fused_fin_add(x, scales, skip1, out, eri, dummy_workspace, tiling)
fused_tail_zero(x, dummy_workspace, tiling)
```

- **The workspace slot always carries a dummy**: probes proved that this
  slot's argument is deterministically corrupted on arrival at the kernel
  (the aux +out_offset unresolved case, mechanism not located) — no valid
  parameter is ever placed there; the host passes addr_tab/out/x duplicate
  values in the dummy slot (kernel ignores them).
- Build: dedicated `ascendc_library(fused_tail_kernel ...)` with
  `-DHAVE_WORKSPACE -DHAVE_TILING` and `--cce-auto-sync=on` (matches the
  probe validation baseline; all sync is handwritten, correctness does not
  rely on auto-sync).
- Stubs: `op_host/stub/aclrtlaunch_{fused_fin_ar_norm,fused_fin_add,
  fused_tail_zero}.h` handwritten fallbacks (copied into place when auto_gen
  does not emit them; signatures match the kernel entries parameter by
  parameter).

## 3. Host TORCH_CHECK list (python wrapper guards must mirror these)

- Common: **all tensor parameters must be defined** (pass dummies for unused
  slots); h in {2048,4096}; m in [1,4096]; k in [1,16]; ncores in [1,48];
  has_skip1/use_eri in {0,1}; **use_eri=1 implies eri contiguous int32
  [M*K]** (with use_eri=0 the eri contents/dtype are not validated — the
  caller passes a dummy tensor, the kernel does not read it); x contiguous
  bf16 [M*K,H]; **scales contiguous fp32 or bf16 [M,K]** (bf16 widened
  exactly in-kernel, tiling.scales_bf16); has_skip1=1 implies skip1
  contiguous bf16 [M,H].
- fused_fin_ar_norm: addr_tab contiguous int64[>=18]; world in [1,8];
  rank in [0,world); residual/add_out/norm_out contiguous bf16 [M,H];
  **norm_w contiguous bf16 [H]**; **m*h*2 % 8192 == 0** (m even for H2048);
  tiles <= max_tiles; ring_stride == world*slot_stride; **payload <=
  slot_stride** (prevents push overruns writing into the peer's symmem);
  counter_offset==0, dfx_offset==48*128.
- fused_fin_add: out contiguous bf16 [M,H]; no tile alignment requirement
  (row-granular core split).
- fused_tail_zero: x contiguous with numel >= nbytes; nbytes a positive
  multiple of 32.

## 4. Tiling cache

Hash (whole-struct FNV-1a) + static device buffer (1024 slots) reuse —
graph-capture pointer stability; flag-layout derivatives
(flag_ring_stride/flag_phase_stride) are derived host-side with the same
formulas as the python wrapper (single source of truth; the counter/dfx
offset check is the two-sided consistency guard).

[v4.1] The tiling struct no longer contains any VA: v4's eri_va/norm_w_va
fields are renamed reserved_va0/1 and stay always-0 (struct offsets
preserved); the hash **degenerates to shape level** (entries shared across
identical m/h/k/ncores/has_skip1/use_eri/scales_bf16, 40 layers 1 entry).
New `scales_bf16` field (1 = bf16 scales fed directly). **The statically
kept-alive pinned staging (g_ftPinnedKeepAlive) is still a required safety
net, not dead code**: entries created inside capture drop from "per (bs
tier x layer)" to "at most 1 per bs tier", but entries for new shapes not
warmed up in eager can still be born during capture, and once their H2D
copy chain is recorded into the graph, replay re-reads the pinned source
address. The >1024 fallback path is reachable only in eager (the capture-
time entry count is bounded by the number of bs tiers << 1024).

## 5. Day-0 check commands

```bash
# Build artifacts (op registration success != kernel linked into the binary;
# undefined U entries are the real missing signal):
nm -D $SGLK_PKG/lib/libsgl_kernel_npu.so | grep "U aclrtlaunch_fused_fi" \
  && echo "truly missing: unresolved launch symbols, rebuild everything"
strings $SGLK_PKG/lib/libsgl_kernel_npu.so | grep -c "ascend.meta.fused_fi"   # must be 3
# Runtime load:
python -c "import torch, sgl_kernel_npu; \
[assertion for assertion in ()]; \
print([hasattr(torch.ops.npu, n) for n in ('fused_fin_ar_norm','fused_fin_add','fused_tail_zero')])"
# Kernel build stamp (spin init self-check; csrc FUSED_TAIL_BUILD_REV and the
# wrapper's _FT_EXPECT_BUILD_REV share one value = 20260924; bump both on any
# kernel change).
```

## 6. Build-error playbook

- `aclrtlaunch_fused_*.h` not found: auto_gen did not emit them -> the three
  `op_host/stub/` files are already on the include path
  (`fused_tail/op_host/stub`), so this should normally not fire; if it still
  does, check that the CMakeLists include section contains stub/.
- Kernel compile UB overflow: the three kernels' budgets are in each file's
  header comment (worst H4096 spin ~155KB < 170KB); on error, first re-check
  that account against newly added fields (eri 64B).
- `RegisterAscendBinary aiv ret 107000`: multiple kernels in one file strikes
  again — this directory keeps one kernel per file; if this appears, check
  whether someone merged multiple entries into the same .cpp.
