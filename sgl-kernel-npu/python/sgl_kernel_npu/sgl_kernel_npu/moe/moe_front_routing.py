# -*- coding: utf-8 -*-
"""MoE 前段 v2.2 自写 init_routing（Triton / triton-ascend，CANN 9.0.0）。

服务器落点：sgl_kernel_npu 包内 ``sgl_kernel_npu/moe/moe_front_routing.py``
（namespace 子包，无需 __init__.py；集成方以
``from sgl_kernel_npu.moe.moe_front_routing import moe_init_routing_v22`` 导入）。

语义契约：与 ``torch.ops.npu.npu_moe_init_routing_v2``（expert_tokens_num_type=1）
**逐位一致**（RL 精度硬约束，0 容差验收）：
  - expanded_x：x 行按「expert id 稳定计数排序」的置换（同 expert 内保持
    flat slot 原序，flat slot 为 t-major：f = t*top_k + k）；
  - eri[f] = flat slot f 对应的 expanded 行号（= stock expanded_row_idx 逐位）；
  - counts = per-expert 计数（int64，stock type=1 形态，喂 stock GMM
    ``group_list_type=1``）。
顺带原生输出两个前缀和（免任何前置 kernel）：
  - excl（int32）：exclusive 前缀和，直喂 persistent_gmm 的 ``offsets=``
    （省 _gmm2_offsets_kernel，~4.4µs/层）；
  - incl（int64）：inclusive 前缀和，stock GMM ``group_list_type=0`` 形态（备用）。

formulation（v2.2，code/moe_front_fusion 六轮单测定案；
结果 src/moe_front_fusion_result{,_2.._6}.csv，结论见
docs/practice/npu-moe-forward-region-probe.md）：

  - rank（eri）：(e,slot) 字典序打包单比较，key = e*PM + slot，「j 排在 i 前」
    ⟺ key_j < key_i；比较矩阵转置 [PM, BM]、归约走 axis=0（本机实证快方向）；
    每 program 顺带产**部分直方图**（[BM,PE] axis=0 一次归约），跨 program
    并行取代串行扫描。
  - counts/excl/incl：第二个 kernel 对 partials [NP,PE] 做 axis=0 二次归约 +
    1D tl.cumsum（生产验证原语）直出，i64 直写省 cast；无 atomic、无零初始化。
  - gather（expanded_x）：独立 kernel 行块向量化 [RPP,H]（RPP=8 为 32KB），
    **不与 rank 合并**——五轮实证融合版 18.2µs 败（归约链后串接大 tile 访存
    长链），拆分版部件和仅 6.2µs。
  - 精度验收：九分布（uniform/random/zipf/skew/first/last 含极端）× 多 seed
    与 stock 逐位一致 + finalize 闭环 0 diff + 双跑确定（两形态各一轮全绿）。
  - 性能（graph-loop，T=32/topk=8/E=256/H=2048，µs/层）：全程 10.9 vs
    stock 17.0~18.2；链路口径 chain_renorm_v22 24.0 vs stock 链 37.4
    （同 run -13.4）。

形态门（wrapper 内自适应；超出形态由集成方回退 stock）：
  M = T*top_k ≤ 512 且 M % 8 == 0，H 为 2 的幂（gather 行块要求），
  x 为 bf16。bm/rpp 按 M 整除自适应（32/16/8 与 8/4/2/1）。
"""

import torch
import triton
import triton.language as tl

try:
    from sgl_kernel_npu.fla.utils import input_guard
except ImportError:  # 独立调试（未安装 sgl_kernel_npu）时退化为恒等装饰器
    def input_guard(fn):
        return fn


def _pow2_ceil(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@triton.jit
def _rank_hist_kernel(
    ids_ptr, eri_ptr, part_ptr,
    M: tl.constexpr, PM: tl.constexpr, PE: tl.constexpr, BM: tl.constexpr,
):
    """eri + 部分直方图（grid = M//BM）。

    lt[j, i] = key_j < key_i（[PM, BM]，axis=0 归约——本机实证快方向）；
    part[pid] = 本 program BM 个 slot 的 expert 直方图（[BM, PE] axis=0）。
    padding slot 的 key=32768*PM+... 大于一切真实 key，不改变真实名次。
    """
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    jj = tl.arange(0, PM)
    e_all = tl.load(ids_ptr + jj, mask=jj < M, other=32768).to(tl.int32)
    e_rows = tl.load(ids_ptr + rows, mask=rows < M, other=-1).to(tl.int32)
    key_all = e_all * PM + jj
    key_rows = e_rows * PM + rows
    lt = key_all[:, None] < key_rows[None, :]       # [PM, BM]
    rank = tl.sum(lt.to(tl.int16), axis=0)          # [BM]
    tl.store(eri_ptr + rows, rank.to(tl.int32), mask=rows < M)
    rngE = tl.arange(0, PE)
    eq = e_rows[:, None] == rngE[None, :]           # [BM, PE]（-1 不命中任何 expert）
    part = tl.sum(eq.to(tl.int16), axis=0).to(tl.int32)
    tl.store(part_ptr + pid * PE + rngE, part)


@triton.jit
def _partials_cumsum_kernel(
    part_ptr, counts_ptr, excl_ptr, incl_ptr,
    NP: tl.constexpr, NPP: tl.constexpr, PE: tl.constexpr, E: tl.constexpr,
    NPB: tl.constexpr,
):
    """partials [NP, PE] -> counts(i64) / excl(i32) / incl(i64)，单 program。

    全白名单原语：axis=0 二次归约 + 1D tl.cumsum（GDN 生产路径同款）。
    NP 方向按 NPB≤16 行分块累加：K9 闸内最大 M=504 时 bm=8 ⇒ NP=63、
    NPP=64，[64,512] i32 整载 = 128KB 超 UB 预算，ConvertLinalgRToBinary /
    BiShengHIR 编译失败（k9_v22_routing_probe T=56 实证）；分块后单 tile
    ≤ [16,512] i32 = 32KB（已验证域）。整数加法满足结合律，分块累加与
    整载逐位一致；NPP≤16 时 NPB=NPP 单趟循环，IR 与旧版等价。
    """
    rngE = tl.arange(0, PE)
    counts = tl.zeros((PE,), dtype=tl.int32)
    for nb in range(0, NPP, NPB):
        np_ = nb + tl.arange(0, NPB)
        part = tl.load(part_ptr + np_[:, None] * PE + rngE[None, :],
                       mask=(np_ < NP)[:, None], other=0)
        counts += tl.sum(part, axis=0)              # [PE] i32
    incl = tl.cumsum(counts, axis=0)
    excl = incl - counts
    e_valid = rngE < E
    tl.store(counts_ptr + rngE, counts.to(tl.int64), mask=e_valid)
    tl.store(excl_ptr + rngE, excl, mask=e_valid)
    tl.store(incl_ptr + rngE, incl.to(tl.int64), mask=e_valid)


@triton.jit
def _routing_gather_kernel(
    x_ptr, eri_ptr, out_ptr,
    TOPK: tl.constexpr, H: tl.constexpr, RPP: tl.constexpr,
):
    """行块置换：out[eri[f]] = x[f // TOPK]，每 program 处理 RPP 个 flat slot。

    [RPP, H] 二维 tile 一次装载/写出（RPP=8 为 32KB），行内 4KB 连续。
    grid = (M // RPP,)（wrapper 保证 M % RPP == 0，故不带 mask）。
    """
    pid = tl.program_id(0)
    fs = pid * RPP + tl.arange(0, RPP)
    dest = tl.load(eri_ptr + fs)  # [RPP] int32
    tok = fs // TOPK
    hh = tl.arange(0, H)
    rows = tl.load(x_ptr + tok[:, None] * H + hh[None, :])
    tl.store(out_ptr + dest[:, None] * H + hh[None, :], rows)


def _pick_tile(m: int, candidates) -> int:
    for c in candidates:
        if m % c == 0:
            return c
    return candidates[-1]


@input_guard
def moe_init_routing_v22(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    top_k: int,
):
    """v2.2 自写 init_routing（rank_hist + partials_cumsum + gather 三 launch）。

    形态门（调用方负责判定，超出门请回退 stock）：M = T*top_k ≤ 512 且
    M % 8 == 0；H 为 2 的幂；x 为 bf16 contiguous。

    参数：
      x:         [T, H] bf16，decode 隐藏状态。
      topk_ids:  [T, top_k] int32 contiguous（来自 gating 算子）。
      num_experts / top_k：路由配置。

    返回 (expanded_x, eri, counts_i64, excl_i32, incl_i64)：
      expanded_x [M, H] bf16 / eri [M] int32 —— 与 stock 逐位一致（0 容差）；
      counts_i64 [E] —— stock type=1 形态（GMM group_list_type=1）；
      excl_i32   [E] —— persistent_gmm 的 offsets= 直用（省 4.4µs 前置 kernel）；
      incl_i64   [E] —— stock GMM group_list_type=0 形态（备用）。
    """
    T, H = x.shape
    M = T * top_k
    ids_flat = topk_ids.reshape(-1).to(torch.int32)  # 同 dtype 时 .to 不拷贝
    dev = x.device
    PM = _pow2_ceil(M)
    PE = _pow2_ceil(num_experts)
    bm = _pick_tile(M, (32, 16, 8))
    rpp = _pick_tile(M, (8, 4, 2, 1))
    npart = M // bm
    eri = torch.empty(M, dtype=torch.int32, device=dev)
    partials = torch.empty((npart, PE), dtype=torch.int32, device=dev)
    expanded = torch.empty((M, H), dtype=x.dtype, device=dev)
    counts = torch.empty(num_experts, dtype=torch.int64, device=dev)
    excl = torch.empty(num_experts, dtype=torch.int32, device=dev)
    incl = torch.empty(num_experts, dtype=torch.int64, device=dev)
    _rank_hist_kernel[(npart,)](ids_flat, eri, partials,
                                M=M, PM=PM, PE=PE, BM=bm, num_stages=1)
    _routing_gather_kernel[(M // rpp,)](x, eri, expanded,
                                        TOPK=top_k, H=H, RPP=rpp)
    _partials_cumsum_kernel[(1,)](partials, counts, excl, incl,
                                  NP=npart, NPP=_pow2_ceil(npart), PE=PE,
                                  E=num_experts,
                                  NPB=min(_pow2_ceil(npart), 16),
                                  num_stages=1)
    return expanded, eri, counts, excl, incl


@input_guard
def moe_init_routing_v22_partials(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    top_k: int,
):
    """C1 variant: rank_hist + gather **two launches**, skipping the
    partials_cumsum kernel and returning partials instead — the column
    reduction (counts/excl) and the vgmm1 block-schedule table are both
    produced downstream by ``torch.ops.npu.vgmm1_sched_partial`` (single-core
    AscendC), so the cumsum node leaves the chain entirely.

    Called only under the joint gate SGLANG_MOE_FRONT_FUSION=1 &
    SGLANG_NPU_VGMM1=1; every other path keeps ``moe_init_routing_v22``
    (five-tuple, byte-identical to FF v1).

    Shape gate identical to v22: M = T*top_k <= 512 and M % 8 == 0; H a power
    of two; x bf16.

    Returns (expanded_x, eri, partials):
      expanded_x [M, H] bf16 / eri [M] int32 — bitwise-identical to stock
        (zero tolerance);
      partials [npart, PE] int32 — rank_hist's partial histogram
        (npart = M//bm, bm the largest of {32,16,8} dividing M;
        PE = pow2_ceil(num_experts)), fed directly into vgmm1_sched_partial
        (column reduction = counts, exact i32 adds, bitwise-equal to v22).
    """
    T, H = x.shape
    M = T * top_k
    ids_flat = topk_ids.reshape(-1).to(torch.int32)  # no copy when dtype matches
    dev = x.device
    PM = _pow2_ceil(M)
    PE = _pow2_ceil(num_experts)
    bm = _pick_tile(M, (32, 16, 8))
    rpp = _pick_tile(M, (8, 4, 2, 1))
    npart = M // bm
    eri = torch.empty(M, dtype=torch.int32, device=dev)
    partials = torch.empty((npart, PE), dtype=torch.int32, device=dev)
    expanded = torch.empty((M, H), dtype=x.dtype, device=dev)
    _rank_hist_kernel[(npart,)](ids_flat, eri, partials,
                                M=M, PM=PM, PE=PE, BM=bm, num_stages=1)
    _routing_gather_kernel[(M // rpp,)](x, eri, expanded,
                                        TOPK=top_k, H=H, RPP=rpp)
    return expanded, eri, partials
