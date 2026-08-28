# -*- coding: utf-8 -*-
"""Persistent grouped matmul（Triton / triton-ascend），面向 EP1 decode 的 skinny GMM2。

服务器落点：sgl_kernel_npu 包内 ``sgl_kernel_npu/moe/persistent_gmm.py``
（namespace 子包，无需 __init__.py；集成方以
``from sgl_kernel_npu.moe.persistent_gmm import persistent_grouped_matmul`` 导入）。

迁移说明（gmm/v1，基线 sglang main@bd3f6a793 + sgl_kernel_npu 3.2.1）：
  面向 8 卡 16 die / 4 engine（MoE-TP、EP1）decode：E=256, N=2048，
  K=64(TP8)/128(TP4)，平均 ~1 rows/expert，stock GMM2 实测 ~65us
  （mac_ratio 6.9%，scalar/fixpipe 接近饱和，AIV 全闲）。瓶颈结构：
  逐 group 调度与组间流水线排空，而非 FLOPs。

v1.1 修复（2026-08-10，bisect 实证见 src/bisect_result.txt）：
  v1 的 masked-sum in-kernel offsets（row0 = sum(where(e_rng < e, c_vec, 0))，
  继承自 old_env v1.2.1，且当年修复后无 rollout 复测记录）在本机工具链
  （CANN 9.0.0 / triton-ascend，默认 --enable-auto-multi-buffer）下被误编译：
  persistent 数据依赖控制流中的 scalar-masked reduction 计算出错误的 row0，
  输出整体错位（stages=3 rel_diff≈1.3；stages=1 直接 1.7e38 垃圾值），
  rollout 乱码。二分证实：同一 kernel 改用外部 offsets 标量直读后与 stock
  GMM 逐位一致（max_abs=0.000000，tl.trans / num_stages / base_n 均无辜）。
  本版 offsets 改为「前置 offsets kernel + 标量直读」：
    - _gmm2_offsets_kernel：单 program 前缀和；替代 torch.cumsum 链
      （~58us/层，且其 torch 算子对 graph capture 不友好）。
    - _persistent_gmm_kernel：row0 = tl.load(offsets_ptr + e) 标量直读，
      与 bisect 中逐位正确的 ext_off 路径完全一致。
  若前置 kernel 在本机仍验证不过，回退方案：wrapper 用
  ``offsets = torch.cumsum(group_list, 0) - group_list``（2 个小算子，
  见 persistent_grouped_matmul 注释），kernel 无需改动。

v1.2 修复（2026-08-11，rollout 实证：offsets kernel 37us 吞掉全部收益）：
  v1.1 的两级分组扫描用了 [G,G,G] 三维 tl.where+tl.sum(axis=2)，在本机
  triton-ascend 上未被向量化、scalarize 成逐元素循环（4096 次标量乘加 ×
  单 program），实测 ~37us/层，吞掉 persistent kernel 省下的 ~24us/层。
  （rollout 实测：persistent kernel 本体 41us vs stock 65us 正确且稳定。）

v1.3 修复（2026-08-11）：v1.2 的 Hillis-Steele（全局 scratch 就地
  store->移位 load）在 ConvertTritonIRToLinalgIR 阶段 PassManager 编译失败。
  改用 tl.cumsum 一维扫描——该原语在本工具链已被生产验证
  （sgl_kernel_npu.fla.cumsum 的 chunk_local_cumsum_scalar_kernel，GDN 路径），
  扫描前先转 int32（int64 reduction 风险的老教训）。目标 <=5-7us/层。

v1.4 调优（2026-08-11，code/moe_whole_process 探针实测）：
  num_progs 40→24。Ascend910_9382 为 24 AIC/die，静态 round-robin 下
  p40 有 16 核背 2 个 program 形成慢波、p32 更灾难（8 核双份）；
  uniform counts（全活跃）单测 64.3→54.1us（1.03→1.24TB/s，达
  read_only_sum 实测上限 1.33TB/s 的 ~93%，基本贴墙）。
  同次探针证伪两个假设：
    - 权重无条件装载（varU，装载移出 if m_i>0）按实际读取字节折算速率
      与条件装载完全相同（~1.1TB/s）——瓶颈不是 if 阻断跨迭代流水；
    - num_stages 2/3/4 计时无差异。
  HBM 可达带宽标定：read 1328 / copy r+w 1275 GB/s（与旧环境
  1311/1214 一致）；GMM1 生产 profile 61us（implied 2.2TB/s）与该上限
  矛盾，冷态 floor 应 ~101us，口径待澄清（L2 热或 shape 混入）。

kernel 设计（继承 v1.2.1 结论）：

  - persistent grid：num_progs 个常驻 program，静态 round-robin 领取 (expert, n_tile)
    任务；任务成本由权重字节主导且各任务均匀（与 M_i 无关），静态调度即够，
    无需 atomic 队列。
  - 空 expert（count=0）整任务跳过，死 expert 权重零流量。
  - 权重 tile 在任务内对所有 m_tile 复用；BASE_M=16 + 内层 m_tile 循环覆盖热点 expert。
  - fp32 累加（tl.dot 默认），与 torch.ops.npu.npu_grouped_matmul 数值语义一致。
  - layout="kn" 路径注意：本机工具链 base_n=512 编译期 cbuf overflow，
    如确需 kn 布局先试 base_n<=256（nk 为生产路径，不受影响）。

输入契约：
  x:          [total_M, K] bf16，行按 expert 分组连续（npu_moe_init_routing_v2 输出序）
  w:          [E, N, K] bf16（layout="nk"，生产存储布局）或 [E, K, N] 连续（layout="kn"）。
              注意：w 必须是 ND 连续存储；集成方在 _gmm2_triton 开启时已跳过
              w2 的 FRACTAL_NZ cast，NZ 重排存储会让裸指针读取出错。
  group_list: [E] per-expert counts（count 模式，int32/int64，device）
"""

import torch
import triton
import triton.language as tl

try:
    from sgl_kernel_npu.fla.utils import input_guard
except ImportError:  # 独立调试（未安装 sgl_kernel_npu）时退化为恒等装饰器
    def input_guard(fn):
        return fn


@triton.jit
def _gmm2_offsets_kernel(
    counts_ptr, offsets_ptr,
    E: tl.constexpr, P: tl.constexpr,
):
    """exclusive prefix offsets[e] = sum(counts[<e])，int32 [E]，单 program。

    tl.cumsum 一维扫描（P = >=E 的最小 2 的幂）。formulation 选择依据
    （本机 CANN 9.0.0 / triton-ascend 实证）：
      - tl.cumsum 在本工具链可用：sgl_kernel_npu.fla.cumsum 的
        chunk_local_cumsum_scalar_kernel（GDN 生产路径）即用 tl.cumsum；
      - 先做 int32 转换再扫描：规避 triton-ascend 的 int64 reduction 风险
        （v1.2.1 起的老教训）；
      - 禁止 formulation：persistent 动态控制流内 scalar-masked reduction
        （v1，误编译数值错）；三维 where/sum tile（v1.1，scalarize ~37us）；
        全局 scratch 就地 store->移位 load 的 Hillis-Steele（v1.2，
        ConvertTritonIRToLinalgIR PassManager 编译失败）。
    """
    rng = tl.arange(0, P)
    e_mask = rng < E
    c = tl.load(counts_ptr + rng, mask=e_mask, other=0).to(tl.int32)
    incl = tl.cumsum(c, axis=0)
    tl.store(offsets_ptr + rng, incl - c, mask=e_mask)


@triton.jit
def _persistent_gmm_kernel(
    x_ptr, w_ptr, counts_ptr, offsets_ptr, out_ptr,
    num_progs,
    E: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BASE_M: tl.constexpr, BASE_N: tl.constexpr, W_KN: tl.constexpr,
):
    pid = tl.program_id(0)
    N_TILES: tl.constexpr = N // BASE_N
    TOTAL_TASKS: tl.constexpr = E * N_TILES
    k_rng = tl.arange(0, K)
    bm_rng = tl.arange(0, BASE_M)
    bn_rng = tl.arange(0, BASE_N)

    for t in range(pid, TOTAL_TASKS, num_progs):
        e = t // N_TILES
        j = t % N_TILES
        m_i = tl.load(counts_ptr + e).to(tl.int32)
        if m_i > 0:
            # bisect 逐位验证过的形态：外部 offsets + 标量直读。
            # 严禁改回 kernel 内 masked-sum 推导（本机工具链误编译，rollout 乱码）。
            row0 = tl.load(offsets_ptr + e)
            if W_KN:
                # [E,K,N] 连续：直接读 [K, BASE_N] tile（k 行各 2*BASE_N 字节连续突发）
                w_t = tl.load(
                    w_ptr + e.to(tl.int64) * N * K
                    + k_rng[:, None] * N + (j * BASE_N + bn_rng)[None, :]
                )
            else:
                # [E,N,K] 连续（生产存储布局）：读 [BASE_N, K] 后寄存器转置
                w_tile = tl.load(
                    w_ptr + e.to(tl.int64) * N * K
                    + (j * BASE_N + bn_rng)[:, None] * K + k_rng[None, :]
                )
                w_t = tl.trans(w_tile)
            for m0 in range(0, m_i, BASE_M):
                rows = row0 + m0 + bm_rng
                m_mask = (m0 + bm_rng) < m_i
                x_tile = tl.load(
                    x_ptr + rows[:, None] * K + k_rng[None, :],
                    mask=m_mask[:, None], other=0.0,
                )
                acc = tl.dot(x_tile, w_t)  # fp32 [BASE_M, BASE_N]
                tl.store(
                    out_ptr + rows[:, None] * N + (j * BASE_N + bn_rng)[None, :],
                    acc.to(tl.bfloat16),
                    mask=m_mask[:, None],
                )


def _pow2_ceil(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@input_guard
def persistent_grouped_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    group_list: torch.Tensor,
    base_n: int = 512,
    num_progs: int = 24,
    num_stages: int = 3,
    layout: str = "nk",
    out: torch.Tensor = None,
    offsets: torch.Tensor = None,
) -> torch.Tensor:
    """GMM2 专用 persistent grouped matmul，返回 [total_M, N] bf16。

    base_n:     N 方向 tile（{256,512,1024} 之一，须整除 N）
    num_progs:  常驻 program 数。默认 24：Ascend910_9382 为 24 AIC/die，
                静态 round-robin 下与核数相等才负载均衡（v1.4 探针实测：
                p40 慢 ~19%，p32 最灾难；扫参范围 {20,24,32,40,48}）。
    num_stages: triton 流水级数（探针实测 2/3/4 无差异，保持默认 3）
    layout:     "nk" = w 为 [E,N,K]（生产存储布局）；"kn" = w 为 [E,K,N] 连续
    offsets:    可选外部 exclusive offsets（int32 [E]）；缺省时由本函数内置的
                offsets 前置 kernel 推导（无 torch 计算，graph capture 友好）。
                回退形态：offsets = torch.cumsum(group_list, 0) - group_list
                （to(int32) 后传入），kernel 无需任何改动。
    """
    E, N, K = w.shape
    total_M = x.shape[0]
    if out is None:
        out = torch.empty((total_M, N), dtype=torch.bfloat16, device=x.device)
    if offsets is None:
        offsets = torch.empty(E, dtype=torch.int32, device=x.device)
        P = _pow2_ceil(E)
        _gmm2_offsets_kernel[(1,)](
            group_list, offsets, E=E, P=P
        )
    _persistent_gmm_kernel[(num_progs,)](
        x_ptr=x,
        w_ptr=w,
        counts_ptr=group_list,
        offsets_ptr=offsets,
        out_ptr=out,
        num_progs=num_progs,
        E=E,
        K=K,
        N=N,
        BASE_M=16,
        BASE_N=base_n,
        W_KN=(layout == "kn"),
        num_stages=num_stages,
    )
    return out
