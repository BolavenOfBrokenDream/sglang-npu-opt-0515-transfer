// fused_sigmoid_gating_recurrent
// GDN decode recurrent（sigmoid gating + delta rule update）的 AscendC AIV 版 kernel。
//
// 语义基准 = 生产 Triton kernel（sgl_kernel_npu fla fused_sigmoid_gating_recurrent
// strided 形态）逐语句同算术。仅 decode：每序列恰好 1 token（host 锁死 T==N）、
// K==V==128（host 锁死）、HV<=8（gating 小向量按 [8] pad）。pool 布局 [slot, HV, K, V]，
// 每 (slot, hv) 32KB(bf16) 全连续块——单发 DataCopyPad 整头搬入/写回。
//
// 精度设计（与 Triton 基准逐条对齐）：
//   1. softplus 阈值分支原样：softplus(x) = (beta*x <= thr) ? log(1+exp(beta*x))/beta : x
//      —— 两条支路都算（与 tl.where 同语义，exp 上溢 lane 由标量选择丢弃）；
//   2. decay = exp(g)、g = -exp(A_log)*softplus、beta = 1/(1+exp(-b)) 逐字公式
//      （Div 链，不用 AscendC Sigmoid 现成件）；
//   3. L2norm 保留除式 q/(sqrt(Σq²)+1e-6)：ReduceSum + 标量 sqrt（先 sqrt 再 +1e-6，
//      与 Triton 同序）+ Duplicate 满宽 + 向量 Div——⚠️ 勿换向量 Rsqrt/Reciprocal
//      （二者 float 精度不满足双万分之一，CANN 9.0.0 文档明写）；
//   4. h += k⊗v' 用 Mul+Add 两条指令（不用 MulAddDst/FMA 收缩，匹配 Triton 的乘、加
//      两次舍入）；
//   5. 两处 K 维归约 = K 维两块（64 行/块）× 块内 5 pass 相邻配对树 + 块间顺序累加；
//   6. 写出 CAST_RINT（fp32→bf16 RNE，同 Triton .to(bf16)）；bf16→fp32 CAST_NONE 精确；
//   7. 全程 fp32 中间值（state 进 UB 即 cast fp32，算完 cast 回 pool dtype）。
//
// 已知不可先验 bitwise 的两处：① K 维归约加法顺序；② Exp/Ln 多项式实现 vs
// triton-ascend lowering（Div 在 A3 默认 INTRINSIC 最大 1 ulp，同归此清单）。
//
// 并行/流水：grid = min(N*HV, AIV 核数)（host 侧 GetCoreNumAiv），核内按
// idx = blockIdx, blockIdx+blockDim, ... 跨 (seq, v-head) 工作项循环；输入双缓冲
// 预取（item i+1 的 MTE2 与 item i 的 V 计算重叠）。跨管事件全显式：
//   MTE2_V[p]  DMA-in(i) 就绪 -> V 读取（cast 段）
//   V_MTE2[p]  V 读完 qkvBuf_[p]（cast 段结束）-> 下下次 DMA-in 可重写
//   V_MTE3[p]  写出段 cast 完成 -> DMA-out 可读 stateBuf_/oBuf_
//   MTE3_V[p]  DMA-out(i-2) 完成 -> V 可重写 stateBuf_[p]/oBuf_[p]（ComputeItem 起始、
//              任何 V 写之前；fp32 臂以 stateBuf_ 为工作砖就地改写，必须最早）
//   MTE3_MTE2[p] DMA-out(i-2) 完成 -> DMA-in(i) 可重写 stateBuf_[p]/oBuf_[p]（对 stateBuf_
//              与 V_MTE3 链传递覆盖 V 侧写完；qkvBuf_ 无 MTE3 经手，故单列 V_MTE2；
//              必须 parity 双事件——单事件版尾部 Set(CO M-2)->Set(CO M-1) 无中间 Wait，
//              核内 >=2 工作项时首 launch 即同 ID 连续 Set 的卡死 UB，见约束之三）
// 首两轮迭代用 iterCnt_ 条件跳过对应等待（对应缓冲尚无前序访问，免等在从未置位
// 的 flag 上会挂死）。KERNEL_TASK_TYPE 钉 AIV_ONLY；kernel 按完整手工同步纪律编写
// （PipeBarrier/SetFlag 全显式），tp_fusion_recurrent_kernel 库钉 --cce-auto-sync=off
// 构建（auto-sync 只覆盖 TQue 规范编程模型、本 kernel 用不上，=on 仅多编译器自插
// 冗余事件）。
//
// ⚠️ 结构约束（勿回退合并）：bf16/fp32 两个入口必须 **一 kernel 一文件**。
// KERNEL_TASK_TYPE_DEFAULT 与「单文件多 kernel」不共存——同文件仅首个 kernel
// 注册成功，其余 RegisterAscendBinary <type> ret 107000、launch 查不到 kernel
// 挂死。故入口拆为 fused_sigmoid_gating_recurrent_{bf16,fp32}_kernel.cpp，
// 类与参数宏集中在本头。
//
// ⚠️ 结构约束之二（勿回退 Fetch）：全部事件必须 AllocEventID（末尾 ReleaseEvents
// 配对）。FetchEventID 不占用 ID（CANN 9.0.0 文档原话「此接口不会申请 TEventID，
// 仅提供可用的 TEventID」），同一 HardEvent 调两次返回**同一** TEventID——本 kernel
// 的 parity 双缓冲事件（MTE2_V/V_MTE2/V_MTE3/MTE3_V 各 ×2）会全部塌缩成单条硬件
// flag，核内 >=2 工作项时塌缩 ID 上出现同 ID 连续两次 SetFlag 无中间 Wait =
// 文档明写的卡死 UB → kernel 挂死、host sync 永不返回。
//
// ⚠️ 结构约束之三（勿带未消费 SetFlag 退出 kernel）：官方文档明写「SetFlag/WaitFlag
// 必须成对出现」；树内生产 kernel 的收尾纪律全部是排空（recurrent_gated_delta_rule
// 的 SEvent.release()=wait() 循环、causal_conv1d.h 末尾显式 WaitFlag 排空、
// fused_qkvzba_conv1d 每 Set 紧跟 Wait）。未消费的 SetFlag 硬件 flag 跨 launch
// 残留：同核下一次 launch 的 AllocEventID 确定性发出同一批 ID，首个对残留通道的
// SetFlag 即构成「同 ID 连续两次 SetFlag 无中间 Wait」= 文档明写卡死 UB——与形状
// 无关、进程内第 2 次 launch 必挂。收尾 DrainEvents 按 iterCnt_ 条件 Wait 掉全部
// 遗留 flag；MTE3_MTE2 拆 parity 双事件（每 ID Set/Wait 严格交替）。
//
// UB 预算 bf16 ~155KB / fp32 ~160KB（192KB 上限内）。

#ifndef __FUSED_SIGMOID_GATING_RECURRENT_KERNEL_LIB_H_
#define __FUSED_SIGMOID_GATING_RECURRENT_KERNEL_LIB_H_

#include "kernel_operator.h"

// event_t / GetTPipePtr() 在 CANN 9.0.0 是全局命名空间符号（cce_aicore_intrinsics.h /
// kernel_tpipe.h；树内 causal_conv1d.h 现货同用法），不要加 AscendC:: 限定。

using namespace AscendC;

namespace fused_sigmoid_gating_recurrent {

constexpr uint32_t FGR_K = 128;       // head_k_dim（host TORCH_CHECK 锁死）
constexpr uint32_t FGR_V = 128;       // head_v_dim（host TORCH_CHECK 锁死）
constexpr uint32_t FGR_V_SLAB = 64;   // V 维 slab（prod buffer UB 预算）
constexpr uint32_t FGR_K_SLAB = 64;   // K 维归约 slab（prod buffer UB 预算）
constexpr uint32_t FGR_MAX_HV = 8;    // gating 小向量按 [8] pad（host 锁死 HV<=8）
constexpr uint32_t FGR_STATE = FGR_K * FGR_V;        // 16384，每 (slot,head) state 元素数
constexpr uint32_t FGR_STATE_HALF = FGR_STATE / 2;   // 8192 = 单指令 255 repeat × 64 上限内
constexpr uint32_t FGR_MAX_N = 256;   // cu_seqlens/cache_indices UB 驻留上限（生产 bs<=128）
constexpr float FGR_L2_EPS = 1e-6f;   // 与 Triton kernel 的 +1e-6 逐字一致（sqrt 之后加）

// UB 内 qkv parity 缓冲的元素偏移（bf16：q/k/v 各 128 元素，a/b 各 16 元素 pad 槽）
constexpr uint32_t OFF_Q = 0;
constexpr uint32_t OFF_K = FGR_K;                    // 128
constexpr uint32_t OFF_V = FGR_K + FGR_K;            // 256
constexpr uint32_t OFF_A = FGR_K + FGR_K + FGR_V;    // 384（bf16 16 元素 = 32B 槽）
constexpr uint32_t OFF_B = OFF_A + 16;               // 400
constexpr uint32_t QKV_PARITY_ELEMS = OFF_B + 16;    // 416 元素 = 832B

// gating 群 [8] 槽位（元素偏移，单位 float）
enum GateSlot {
    GS_DT_BIAS = 0,   // dt_bias pad 到 [8]
    GS_EXP_ALOG = 8,  // exp(A_log) pad 到 [8]（prologue 算好）
    GS_A_F32 = 16,    // 当前项 a 行（cast 后）
    GS_B_F32 = 24,    // 当前项 b 行（cast 后）
    GS_XA = 32,       // x = a + dt_bias
    GS_BETAX = 40,    // beta*x
    GS_E1 = 48,       // exp(beta*x)+1
    GS_SPV = 56,      // softplus 支路值 (1/beta)*log(1+exp(beta*x))
    GS_BETAV = 64,    // sigmoid(b) 全 lane
    GS_G = 72,        // 标量 g 的 [8] 槽（lane0 有效）
    GS_DECAY = 80,    // exp(g)（lane0 有效）
    GS_ONES = 88,     // 全 1（beta Div 的分子）
    GS_TMP = 96,      // 临时（exp(-b) 链）
    GATE_ELEMS = 104  // 总元素（104 * 4B = 416B，32B 对齐）
};

template <typename T>  // ssm pool dtype：bfloat16_t（生产）/ float（精度回归臂）
class FusedSigmoidGatingRecurrent {
public:
    __aicore__ inline FusedSigmoidGatingRecurrent(uint32_t n, uint32_t h, uint32_t hv, uint32_t qRowStride,
                                                  uint32_t kRowStride, uint32_t vRowStride, float scale,
                                                  float softplusBeta, float invSoftplusBeta, float softplusThreshold,
                                                  uint32_t useQkL2norm)
    {
        n_ = n;
        h_ = h;
        hv_ = hv;
        hvPerH_ = (h > 0) ? (hv / h) : 1;  // host 锁死 hv%h==0
        qRowStride_ = qRowStride;
        kRowStride_ = kRowStride;
        vRowStride_ = vRowStride;
        scale_ = scale;
        spb_ = softplusBeta;
        invSpb_ = invSoftplusBeta;
        thr_ = softplusThreshold;
        useL2_ = useQkL2norm;
    }

    __aicore__ inline void Init(GM_ADDR A_log, GM_ADDR a, GM_ADDR dt_bias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR b,
                                GM_ADDR o, GM_ADDR pool, GM_ADDR cache_indices, GM_ADDR cu_seqlens, TPipe *pipe)
    {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockDim_ = GetBlockNum();
        iterCnt_ = 0;

        aLogGm_.SetGlobalBuffer((__gm__ float *)A_log);
        aGm_.SetGlobalBuffer((__gm__ bfloat16_t *)a);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)dt_bias);
        qGm_.SetGlobalBuffer((__gm__ bfloat16_t *)q);
        kGm_.SetGlobalBuffer((__gm__ bfloat16_t *)k);
        vGm_.SetGlobalBuffer((__gm__ bfloat16_t *)v);
        bGm_.SetGlobalBuffer((__gm__ bfloat16_t *)b);
        oGm_.SetGlobalBuffer((__gm__ bfloat16_t *)o);
        poolGm_.SetGlobalBuffer((__gm__ T *)pool);
        idxGm_.SetGlobalBuffer((__gm__ int32_t *)cache_indices);
        cuGm_.SetGlobalBuffer((__gm__ int32_t *)cu_seqlens);

        // ---- UB 分配（bf16 臂 ~155KB / fp32 臂 ~160KB，192KB 上限内）----
        pipe_->InitBuffer(stateBuf_[0], FGR_STATE * sizeof(T));
        pipe_->InitBuffer(stateBuf_[1], FGR_STATE * sizeof(T));
        if (IsBf16()) {
            // fp32 工作砖（bf16 臂：state 进 UB 即 cast 到此；fp32 臂直接以 stateBuf 为工作砖）
            pipe_->InitBuffer(workBuf_, FGR_STATE * sizeof(float));
        }
        pipe_->InitBuffer(prodBuf_, FGR_K_SLAB * FGR_V_SLAB * sizeof(float));
        pipe_->InitBuffer(brcbBuf_, FGR_K * 8 * sizeof(float));
        pipe_->InitBuffer(qkvBuf_[0], QKV_PARITY_ELEMS * sizeof(bfloat16_t));
        pipe_->InitBuffer(qkvBuf_[1], QKV_PARITY_ELEMS * sizeof(bfloat16_t));
        pipe_->InitBuffer(oBuf_[0], FGR_V * sizeof(bfloat16_t));
        pipe_->InitBuffer(oBuf_[1], FGR_V * sizeof(bfloat16_t));
        pipe_->InitBuffer(qF32Buf_, FGR_K * sizeof(float));
        pipe_->InitBuffer(kF32Buf_, FGR_K * sizeof(float));
        pipe_->InitBuffer(vF32Buf_, FGR_V * sizeof(float));
        pipe_->InitBuffer(normTmpBuf_, FGR_K * sizeof(float));
        pipe_->InitBuffer(reduceWorkBuf_, FGR_K * sizeof(float));  // ReduceSum work 独立分配
        pipe_->InitBuffer(sumBuf_, 2 * 8 * sizeof(float));         // [0]=q 平方和，[8]=k 平方和
        pipe_->InitBuffer(denomBuf_, FGR_K * sizeof(float));
        pipe_->InitBuffer(deltaBuf_, FGR_V * sizeof(float));
        pipe_->InitBuffer(oF32Buf_, FGR_V * sizeof(float));
        pipe_->InitBuffer(cuBuf_, (FGR_MAX_N + 8) * sizeof(int32_t));  // 32B 对齐
        pipe_->InitBuffer(idxBuf_, FGR_MAX_N * sizeof(int32_t));
        pipe_->InitBuffer(gateBuf_, GATE_ELEMS * sizeof(float));

        // ---- 事件（TPipe 场景禁止自选 ID；必须 AllocEventID——FetchEventID 不占坑，
        // 同一 HardEvent 重复 Fetch 返回同一 TEventID，parity 双缓冲事件会塌缩成单条
        // 硬件 flag；同 ID 连续两次 SetFlag（中间无 Wait）命中 CANN 文档明写的
        // 「程序卡死等未定义行为」）----
        evtVS_ = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
        evtSV_ = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
        evtMte2S_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        evtMte2V_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        evtMte2V_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        evtVMte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        evtVMte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        evtVMte3_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        evtVMte3_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        evtMte3V_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        evtMte3V_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        evtMte3Mte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        evtMte3Mte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t totalItems = n_ * hv_;
        if (blockIdx_ >= totalItems) {
            return;  // 多启动的核无工作项（blockDim=min(N*HV, 核数) 时不触发，防御）
        }
        Prologue();
        // 软流水（iteration 处理工作项 idx，预取 idx+blockDim；等待条件见文件头事件表，
        // iterCnt_ 条件跳过首轮/次轮不存在的前序访问）：
        CopyIn(blockIdx_, 0);
        uint32_t parity = 0;
        for (uint32_t idx = blockIdx_; idx < totalItems; idx += blockDim_) {
            const uint32_t next = idx + blockDim_;
            const uint32_t p = parity;
            ComputeItem(idx, p);
            if (next < totalItems) {
                CopyIn(next, 1 - p);
            }
            CopyOut(idx, p);
            parity = 1 - parity;
            iterCnt_++;
        }
        DrainEvents();
        ReleaseEvents();
    }

    // 收尾排空（「SetFlag/WaitFlag 必须成对出现」的官方约束 + 树内 RGDR
    // SEvent.release()/causal_conv1d 末尾 WaitFlag 同款纪律）：末两轮迭代遗留的
    // SetFlag 在本 launch 内无人消费，必须退出前逐一 Wait——否则已置位硬件 flag
    // 跨 launch 残留，下一次 launch 对同 ID 的首个 SetFlag 即「同 ID 连续两次
    // SetFlag 无中间 Wait」卡死 UB（与形状无关、进程内第 2 次 launch 必挂）。
    // 条件与 CopyIn/ComputeItem 的跳过条件互补：iterCnt_==M 时，parities (M-1)&1
    // 与 M&1 的 V_MTE2/MTE3_V/MTE3_MTE2 恰为已置位未消费的全部 flag。
    __aicore__ inline void DrainEvents()
    {
        if (iterCnt_ >= 1) {
            const uint32_t lastP = (iterCnt_ - 1) & 1;
            WaitFlag<HardEvent::V_MTE2>(evtVMte2_[lastP]);      // ComputeItem(M-1) 置位
            WaitFlag<HardEvent::MTE3_V>(evtMte3V_[lastP]);      // CopyOut(M-1) 置位
            WaitFlag<HardEvent::MTE3_MTE2>(evtMte3Mte2_[lastP]);  // CopyOut(M-1) 置位
            if (iterCnt_ >= 2) {
                const uint32_t prevP = iterCnt_ & 1;
                WaitFlag<HardEvent::V_MTE2>(evtVMte2_[prevP]);      // ComputeItem(M-2) 置位
                WaitFlag<HardEvent::MTE3_V>(evtMte3V_[prevP]);      // CopyOut(M-2) 置位
                WaitFlag<HardEvent::MTE3_MTE2>(evtMte3Mte2_[prevP]);  // CopyOut(M-2) 置位
            }
        }
    }

    // 与 AllocEventID 配对（causal_conv1d.h 同款镜像序；早退核无硬件副作用，
    // TPipe 随 kernel 退出析构，池记账仅影响本次 launch 内）
    __aicore__ inline void ReleaseEvents()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(evtVS_);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(evtSV_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(evtMte2S_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evtMte2V_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evtMte2V_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evtVMte2_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evtVMte2_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evtVMte3_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evtVMte3_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(evtMte3V_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(evtMte3V_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evtMte3Mte2_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evtMte3Mte2_[1]);
    }

private:
    __aicore__ inline bool IsBf16() const
    {
        return IsSameType<T, bfloat16_t>::value;
    }

    __aicore__ inline LocalTensor<float> Gate(uint32_t slot)
    {
        return gateBuf_.Get<float>()[slot];
    }

    __aicore__ inline void Prologue()
    {
        // cu_seqlens / cache_indices / A_log / dt_bias 一次性进 UB（后续 GetValue 零 GM 读）
        LocalTensor<int32_t> cuLocal = cuBuf_.Get<int32_t>();
        LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
        DataCopyPadParams padParams;
        DataCopyParams cuParams{1, static_cast<uint16_t>((n_ + 1) * sizeof(int32_t)), 0, 0};
        DataCopyPad(cuLocal, cuGm_, cuParams, padParams);
        DataCopyParams idxParams{1, static_cast<uint16_t>(n_ * sizeof(int32_t)), 0, 0};
        DataCopyPad(idxLocal, idxGm_, idxParams, padParams);

        LocalTensor<float> dtPad = Gate(GS_DT_BIAS);
        LocalTensor<float> expAlogPad = Gate(GS_EXP_ALOG);
        DataCopyParams hvParams{1, static_cast<uint16_t>(hv_ * sizeof(float)), 0, 0};
        DataCopyPad(dtPad, dtBiasGm_, hvParams, padParams);
        DataCopyPad(expAlogPad, aLogGm_, hvParams, padParams);

        LocalTensor<float> ones = Gate(GS_ONES);
        Duplicate(ones, 1.0f, FGR_MAX_HV);  // beta Div 的全 1 分子
        // gating 输入 MTE2->V 后就绪
        SetFlag<HardEvent::MTE2_V>(evtMte2V_[0]);
        WaitFlag<HardEvent::MTE2_V>(evtMte2V_[0]);
        PipeBarrier<PIPE_V>();
        Exp(expAlogPad, expAlogPad, FGR_MAX_HV);  // exp(A_log)，pad lane 得 1.0，无害不读
        PipeBarrier<PIPE_V>();
        // cu/idx 的 MTE2->S（全 kernel 仅这一次跨管同步，之后 UB GetValue 直读）
        SetFlag<HardEvent::MTE2_S>(evtMte2S_);
        WaitFlag<HardEvent::MTE2_S>(evtMte2S_);
    }

    // 工作项 idx = i_n * HV + i_hv 的全部输入 DMA-in 到 parity 缓冲
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t p)
    {
        // 缓冲复用等待（只挡 MTE2 管；iterCnt_<1 时 parity 缓冲尚无前序访问，跳过）：
        // stateBuf_[p]/oBuf_[p] 的前序读者是 DMA-out（MTE3）；qkvBuf_[p] 的前序读者是
        // cast 段（V，仅 DMA-in 复用前需等）
        if (iterCnt_ >= 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(evtMte3Mte2_[p]);
            WaitFlag<HardEvent::V_MTE2>(evtVMte2_[p]);
        }
        const uint32_t iN = idx / hv_;
        const uint32_t iHv = idx - iN * hv_;
        LocalTensor<int32_t> cuLocal = cuBuf_.Get<int32_t>();
        LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
        const int32_t bos = cuLocal.GetValue(iN);
        const int32_t slot = idxLocal.GetValue(iN);
        const uint32_t iH = iHv / hvPerH_;

        LocalTensor<bfloat16_t> qkvLocal = qkvBuf_[p].Get<bfloat16_t>();
        DataCopyPadParams padParams;
        // q/k 行（strided 视图按显式行距寻址；连续时行距 == H*K，寻址恒等）
        DataCopyParams rowQKParams{1, static_cast<uint16_t>(FGR_K * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(qkvLocal[OFF_Q], qGm_[static_cast<uint64_t>(bos) * qRowStride_ + iH * FGR_K], rowQKParams,
                    padParams);
        DataCopyPad(qkvLocal[OFF_K], kGm_[static_cast<uint64_t>(bos) * kRowStride_ + iH * FGR_K], rowQKParams,
                    padParams);
        // v 行
        DataCopyParams rowVParams{1, static_cast<uint16_t>(FGR_V * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(qkvLocal[OFF_V], vGm_[static_cast<uint64_t>(bos) * vRowStride_ + iHv * FGR_V], rowVParams,
                    padParams);
        // a/b 行（[HV] bf16，8B/16B 小行走 pad 通路）
        DataCopyParams rowABParams{1, static_cast<uint16_t>(hv_ * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(qkvLocal[OFF_A], aGm_[static_cast<uint64_t>(bos) * hv_], rowABParams, padParams);
        DataCopyPad(qkvLocal[OFF_B], bGm_[static_cast<uint64_t>(bos) * hv_], rowABParams, padParams);

        // state 32KB(bf16)/64KB(fp32) 全连续块；slot<0 时不搬（Compute 内零初始化，
        // 与 Triton 的 idx>=0 守卫同语义）
        if (slot >= 0) {
            LocalTensor<T> stateLocal = stateBuf_[p].Get<T>();
            DataCopyParams stateParams{FGR_K, static_cast<uint16_t>(FGR_V * sizeof(T)), 0, 0};
            DataCopyPad(stateLocal, poolGm_[(static_cast<uint64_t>(slot) * hv_ + iHv) * FGR_STATE], stateParams,
                        padParams);
        }
        SetFlag<HardEvent::MTE2_V>(evtMte2V_[p]);
    }

    // L2 归一化的 V 侧归约（q/k 共用）；标量 sqrt+1e-6 与除式在 ComputeItem 的
    // 标量段/V 段 B 完成（除式逐字对齐 Triton b_x/(sqrt(Σx²)+1e-6)）
    __aicore__ inline void L2Norm(LocalTensor<float> xF32, uint32_t sumSlot)
    {
        LocalTensor<float> normTmp = normTmpBuf_.Get<float>();
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        LocalTensor<float> reduceWork = reduceWorkBuf_.Get<float>();
        Mul(normTmp, xF32, xF32, FGR_K);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(sumLocal[sumSlot], normTmp, reduceWork, FGR_K);
        PipeBarrier<PIPE_V>();
    }

    // softplus 双支路 + beta 的 V 侧链（全 [8] 小向量，每 lane 一个 head；pad lane 无害不读）
    __aicore__ inline void GatingVecPart()
    {
        LocalTensor<float> aF32 = Gate(GS_A_F32);
        LocalTensor<float> bF32 = Gate(GS_B_F32);
        LocalTensor<float> dtPad = Gate(GS_DT_BIAS);
        LocalTensor<float> xa = Gate(GS_XA);
        LocalTensor<float> betax = Gate(GS_BETAX);
        LocalTensor<float> e1 = Gate(GS_E1);
        LocalTensor<float> spv = Gate(GS_SPV);
        LocalTensor<float> betav = Gate(GS_BETAV);
        LocalTensor<float> ones = Gate(GS_ONES);
        LocalTensor<float> tmp = Gate(GS_TMP);

        Add(xa, aF32, dtPad, FGR_MAX_HV);      // x = a + dt_bias
        PipeBarrier<PIPE_V>();
        Muls(betax, xa, spb_, FGR_MAX_HV);     // beta*x
        PipeBarrier<PIPE_V>();
        Exp(e1, betax, FGR_MAX_HV);            // exp(beta*x)（bx>thr 的 lane 可上溢，
        PipeBarrier<PIPE_V>();                 // 与 tl.where 双支路同语义，标量选择丢弃）
        Adds(e1, e1, 1.0f, FGR_MAX_HV);        // 1+exp
        PipeBarrier<PIPE_V>();
        Ln(spv, e1, FGR_MAX_HV);               // log(1+exp)
        PipeBarrier<PIPE_V>();
        Muls(spv, spv, invSpb_, FGR_MAX_HV);   // (1/beta)*log(1+exp)
        PipeBarrier<PIPE_V>();
        // beta = 1/(1+exp(-b))：逐字公式（不用 AscendC Sigmoid 现成件）
        Muls(tmp, bF32, -1.0f, FGR_MAX_HV);    // -b（乘 -1.0 精确翻符号，无舍入差）
        PipeBarrier<PIPE_V>();
        Exp(tmp, tmp, FGR_MAX_HV);
        PipeBarrier<PIPE_V>();
        Adds(tmp, tmp, 1.0f, FGR_MAX_HV);
        PipeBarrier<PIPE_V>();
        Div(betav, ones, tmp, FGR_MAX_HV);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeItem(uint32_t idx, uint32_t p)
    {
        const uint32_t iN = idx / hv_;
        const uint32_t iHv = idx - iN * hv_;
        LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
        const int32_t slot = idxLocal.GetValue(iN);

        LocalTensor<bfloat16_t> qkvLocal = qkvBuf_[p].Get<bfloat16_t>();
        LocalTensor<float> qF32 = qF32Buf_.Get<float>();
        LocalTensor<float> kF32 = kF32Buf_.Get<float>();
        LocalTensor<float> vF32 = vF32Buf_.Get<float>();
        LocalTensor<float> work = IsBf16() ? workBuf_.Get<float>() : stateBuf_[p].Get<float>();

        WaitFlag<HardEvent::MTE2_V>(evtMte2V_[p]);  // 当前项 q/k/v/a/b/state 就绪
        // out 缓冲/工作砖重写等待：DMA-out(i-2)（MTE3）完成后 V 才可写 stateBuf_[p]/
        // oBuf_[p]——fp32 臂以 stateBuf_[p] 为工作砖、V 段 C 起就地改写，故等待必须
        // 放在任何 V 写之前（bf16 臂的首笔 V 写远在写出段，提前等待无额外停顿）。
        // iterCnt_<2 时该 parity 尚无前序 DMA-out，跳过。
        if (iterCnt_ >= 2) {
            WaitFlag<HardEvent::MTE3_V>(evtMte3V_[p]);
        }
        // ---- V 段 A：cast + L2 归约 + gating 双支路 ----
        Cast(qF32, qkvLocal[OFF_Q], RoundMode::CAST_NONE, FGR_K);
        Cast(kF32, qkvLocal[OFF_K], RoundMode::CAST_NONE, FGR_K);
        Cast(vF32, qkvLocal[OFF_V], RoundMode::CAST_NONE, FGR_V);
        Cast(Gate(GS_A_F32), qkvLocal[OFF_A], RoundMode::CAST_NONE, FGR_MAX_HV);
        Cast(Gate(GS_B_F32), qkvLocal[OFF_B], RoundMode::CAST_NONE, FGR_MAX_HV);
        PipeBarrier<PIPE_V>();
        if (IsBf16()) {
            // state 进工作砖（bf16->fp32 精确）；slot<0 零初始化（Triton idx>=0 守卫同语义）
            if (slot >= 0) {
                Cast(work, stateBuf_[p].Get<T>(), RoundMode::CAST_NONE, FGR_STATE_HALF);
                Cast(work[FGR_STATE_HALF], stateBuf_[p].Get<T>()[FGR_STATE_HALF], RoundMode::CAST_NONE,
                     FGR_STATE_HALF);
            } else {
                Duplicate(work, 0.0f, FGR_STATE_HALF);
                Duplicate(work[FGR_STATE_HALF], 0.0f, FGR_STATE_HALF);
            }
            PipeBarrier<PIPE_V>();
        } else if (slot < 0) {
            Duplicate(work, 0.0f, FGR_STATE_HALF);
            Duplicate(work[FGR_STATE_HALF], 0.0f, FGR_STATE_HALF);
            PipeBarrier<PIPE_V>();
        }
        // qkvBuf_[p] 的 V 侧读取到此为止（stateBuf_ 的 V 侧最后一笔在写出段，
        // 经 V_MTE3->MTE3->MTE3_MTE2 链传递覆盖，无需单列事件）
        SetFlag<HardEvent::V_MTE2>(evtVMte2_[p]);
        if (useL2_ != 0) {
            L2Norm(qF32, 0);  // sumBuf_[0]
            L2Norm(kF32, 8);  // sumBuf_[8]
        }
        GatingVecPart();
        // ---- 标量段（一次 V_S + 一次 S_V 内完成全部标量）----
        SetFlag<HardEvent::V_S>(evtVS_);
        WaitFlag<HardEvent::V_S>(evtVS_);
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        float denomQ = 1.0f;
        float denomK = 1.0f;
        if (useL2_ != 0) {
            // 与 Triton 同序：先 sqrt 再 +1e-6（标量 sqrt；
            // 勿换向量 Rsqrt/Reciprocal——float 精度不满足双万分之一）
            denomQ = sqrt(sumLocal.GetValue(0)) + FGR_L2_EPS;
            denomK = sqrt(sumLocal.GetValue(8)) + FGR_L2_EPS;
        }
        const float xHv = Gate(GS_XA).GetValue(iHv);
        const float bxHv = Gate(GS_BETAX).GetValue(iHv);
        const float spHv = Gate(GS_SPV).GetValue(iHv);
        const float betaS = Gate(GS_BETAV).GetValue(iHv);
        const float expAlogS = Gate(GS_EXP_ALOG).GetValue(iHv);
        // softplus 阈值分支（与 tl.where 逐 lane 同语义；本 kernel 每 lane 恰是一头）
        const float softplusS = (bxHv <= thr_) ? spHv : xHv;
        const float gS = (-expAlogS) * softplusS;  // g = -exp(A_log)*softplus（负号精确）
        SetFlag<HardEvent::S_V>(evtSV_);
        WaitFlag<HardEvent::S_V>(evtSV_);
        // ---- V 段 B：除式归一化 + decay 的 exp ----
        LocalTensor<float> denom = denomBuf_.Get<float>();
        if (useL2_ != 0) {
            Duplicate(denom, denomQ, FGR_K);
            PipeBarrier<PIPE_V>();
            Div(qF32, qF32, denom, FGR_K);  // 真除式（A3 Div 默认 INTRINSIC 最大 1 ulp，
            PipeBarrier<PIPE_V>();          // 归入指令级差异清单，误差预算路径兜底）
            Duplicate(denom, denomK, FGR_K);
            PipeBarrier<PIPE_V>();
            Div(kF32, kF32, denom, FGR_K);
            PipeBarrier<PIPE_V>();
        }
        Muls(qF32, qF32, scale_, FGR_K);  // scale 在 norm 之后（Triton 同序）
        PipeBarrier<PIPE_V>();
        Duplicate(Gate(GS_G), gS, FGR_MAX_HV);
        PipeBarrier<PIPE_V>();
        Exp(Gate(GS_DECAY), Gate(GS_G), FGR_MAX_HV);  // decay = exp(g)（lane0 有效）
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(evtVS_);
        WaitFlag<HardEvent::V_S>(evtVS_);
        const float decayS = Gate(GS_DECAY).GetValue(0);
        SetFlag<HardEvent::S_V>(evtSV_);  // 与前一段标量组同款收尾：decayS 进 V 段 C
        WaitFlag<HardEvent::S_V>(evtSV_); // 的 Muls 标量操作数前补齐 S->V 跨管同步
        // ---- V 段 C：delta rule 主链 ----
        // h *= decay（Triton：b_h *= b_decay，先于 delta）
        Muls(work, work, decayS, FGR_STATE_HALF);
        Muls(work[FGR_STATE_HALF], work[FGR_STATE_HALF], decayS, FGR_STATE_HALF);
        PipeBarrier<PIPE_V>();
        // k 广播砖（Brcb：每元素扩成 8 份；配合 src1BlkStride=0 实现整行 × k[k]）
        LocalTensor<float> brcb = brcbBuf_.Get<float>();
        Brcb(brcb, kF32, FGR_K / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        LocalTensor<float> prod = prodBuf_.Get<float>();
        LocalTensor<float> delta = deltaBuf_.Get<float>();
        // delta[v] = Σ_k h[k,v]·k[k]（K 两块：块内树形 5 pass 留 2 行，合并后块间顺序累加）
        for (uint32_t s = 0; s < FGR_V / FGR_V_SLAB; s++) {
            ReduceKSlab(prod, work, brcb, s, 0);
            Add(delta[s * FGR_V_SLAB], prod, prod[FGR_V_SLAB], FGR_V_SLAB);  // 块0 两行合并落槽
            PipeBarrier<PIPE_V>();
            ReduceKSlab(prod, work, brcb, s, 1);
            Add(prod, prod, prod[FGR_V_SLAB], FGR_V_SLAB);  // 块1 两行合并到 prod 行0
            PipeBarrier<PIPE_V>();
            Add(delta[s * FGR_V_SLAB], delta[s * FGR_V_SLAB], prod, FGR_V_SLAB);  // 块间累加
            PipeBarrier<PIPE_V>();
        }
        // v' = (v - delta) * beta（Sub 后 Muls 两次舍入，与 Triton 同）
        Sub(vF32, vF32, delta, FGR_V);
        PipeBarrier<PIPE_V>();
        Muls(vF32, vF32, betaS, FGR_V);
        PipeBarrier<PIPE_V>();
        // h += k ⊗ v'（Mul+Add 两条指令、两次舍入；不用 MulAddDst/FMA 收缩）
        for (uint32_t s = 0; s < FGR_V / FGR_V_SLAB; s++) {
            for (uint32_t ks = 0; ks < FGR_K / FGR_K_SLAB; ks++) {
                // prod[r, :] = v'[s 槽] * k[ks*64+r]（src0RepStride=0：整 slab 复用同一 v' 槽）
                Mul<float>(prod, vF32[s * FGR_V_SLAB], brcb[ks * FGR_K_SLAB * 8], FGR_V_SLAB,
                           static_cast<uint8_t>(FGR_K_SLAB), {1, 1, 0, 8, 0, 1});
                PipeBarrier<PIPE_V>();
                const uint32_t wOff = ks * FGR_K_SLAB * FGR_V + s * FGR_V_SLAB;
                Add<float>(work[wOff], work[wOff], prod, FGR_V_SLAB, static_cast<uint8_t>(FGR_K_SLAB),
                           {1, 1, 1, 16, 16, 8});
                PipeBarrier<PIPE_V>();
            }
        }
        // q 广播砖（覆写 k 广播砖：此后不再用 k）
        Brcb(brcb, qF32, FGR_K / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        // o[v] = Σ_k h_new[k,v]·q[k]（与 delta 同构；用 rank-1 更新后的 h）
        LocalTensor<float> oF32 = oF32Buf_.Get<float>();
        for (uint32_t s = 0; s < FGR_V / FGR_V_SLAB; s++) {
            ReduceKSlab(prod, work, brcb, s, 0);
            Add(oF32[s * FGR_V_SLAB], prod, prod[FGR_V_SLAB], FGR_V_SLAB);
            PipeBarrier<PIPE_V>();
            ReduceKSlab(prod, work, brcb, s, 1);
            Add(prod, prod, prod[FGR_V_SLAB], FGR_V_SLAB);
            PipeBarrier<PIPE_V>();
            Add(oF32[s * FGR_V_SLAB], oF32[s * FGR_V_SLAB], prod, FGR_V_SLAB);
            PipeBarrier<PIPE_V>();
        }
        // ---- 写出段 ----
        LocalTensor<bfloat16_t> oLocal = oBuf_[p].Get<bfloat16_t>();
        Cast(oLocal, oF32, RoundMode::CAST_RINT, FGR_V);  // RNE，同 Triton .to(bf16)
        if (IsBf16()) {
            Cast(stateBuf_[p].Get<T>(), work, RoundMode::CAST_RINT, FGR_STATE_HALF);
            Cast(stateBuf_[p].Get<T>()[FGR_STATE_HALF], work[FGR_STATE_HALF], RoundMode::CAST_RINT, FGR_STATE_HALF);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evtVMte3_[p]);
    }

    // prod[r, :] = work[(ks*64+r)*128 + s 槽] * brcbVec[ks*64+r]，随后 5 pass 相邻配对
    // 树形归约（就地、读地址恒不小于写地址，无自覆盖），留下 2 行于 prod[0..63] 与
    // prod[64..127]（调用方一次 Add 合并/落槽）。
    __aicore__ inline void ReduceKSlab(LocalTensor<float> prod, LocalTensor<float> work, LocalTensor<float> brcb,
                                       uint32_t vSlab, uint32_t kSlab)
    {
        const uint32_t wOff = kSlab * FGR_K_SLAB * FGR_V + vSlab * FGR_V_SLAB;
        // src1BlkStride=0：repeat 内 8 个 block 同读 brcb 的一个 8 元素块（= 标量×8）；
        // src1RepStride=1：repeat 间逐块推进（k/q 逐行）
        Mul<float>(prod, work[wOff], brcb[kSlab * FGR_K_SLAB * 8], FGR_V_SLAB, static_cast<uint8_t>(FGR_K_SLAB),
                   {1, 1, 0, 8, 16, 1});
        PipeBarrier<PIPE_V>();
        // 5 pass 树形：64→32→16→8→4→2 行（末次两行合并由调用方按落槽需求做）
        for (uint32_t repeats = FGR_K_SLAB / 2; repeats >= 2; repeats = repeats / 2) {
            Add<float>(prod, prod, prod[FGR_V_SLAB], FGR_V_SLAB, static_cast<uint8_t>(repeats), {1, 1, 1, 8, 16, 16});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t p)
    {
        const uint32_t iN = idx / hv_;
        const uint32_t iHv = idx - iN * hv_;
        LocalTensor<int32_t> cuLocal = cuBuf_.Get<int32_t>();
        LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
        const int32_t bos = cuLocal.GetValue(iN);
        const int32_t slot = idxLocal.GetValue(iN);

        WaitFlag<HardEvent::V_MTE3>(evtVMte3_[p]);  // 写出段 cast 完成
        // o 行写出（与 Triton 同寻址：(bos*HV+i_hv)*V）
        DataCopyParams oParams{1, static_cast<uint16_t>(FGR_V * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(oGm_[(static_cast<uint64_t>(bos) * hv_ + iHv) * FGR_V], oBuf_[p].Get<bfloat16_t>(), oParams);
        // state 写回（slot<0 不写，Triton 同守卫）
        if (slot >= 0) {
            DataCopyParams stateParams{FGR_K, static_cast<uint16_t>(FGR_V * sizeof(T)), 0, 0};
            DataCopyPad(poolGm_[(static_cast<uint64_t>(slot) * hv_ + iHv) * FGR_STATE], stateBuf_[p].Get<T>(),
                        stateParams);
        }
        // 两道置位：stateBuf_/oBuf_ 的下轮重写门（MTE3_V 给 V 侧、MTE3_MTE2 给 MTE2 侧；
        // 均按 parity——MTE3_MTE2 若用单事件，末两轮 Set 无中间 Wait = 卡死 UB）
        SetFlag<HardEvent::MTE3_V>(evtMte3V_[p]);
        SetFlag<HardEvent::MTE3_MTE2>(evtMte3Mte2_[p]);
    }

private:
    GlobalTensor<float> aLogGm_;
    GlobalTensor<bfloat16_t> aGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<bfloat16_t> qGm_;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<bfloat16_t> bGm_;
    GlobalTensor<bfloat16_t> oGm_;
    GlobalTensor<T> poolGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<int32_t> cuGm_;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> stateBuf_[2];
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> prodBuf_;
    TBuf<TPosition::VECCALC> brcbBuf_;
    TBuf<TPosition::VECCALC> qkvBuf_[2];
    TBuf<TPosition::VECCALC> oBuf_[2];
    TBuf<TPosition::VECCALC> qF32Buf_;
    TBuf<TPosition::VECCALC> kF32Buf_;
    TBuf<TPosition::VECCALC> vF32Buf_;
    TBuf<TPosition::VECCALC> normTmpBuf_;
    TBuf<TPosition::VECCALC> reduceWorkBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> denomBuf_;
    TBuf<TPosition::VECCALC> deltaBuf_;
    TBuf<TPosition::VECCALC> oF32Buf_;
    TBuf<TPosition::VECCALC> cuBuf_;
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> gateBuf_;

    TEventID evtVS_;
    TEventID evtSV_;
    TEventID evtMte2S_;
    TEventID evtMte2V_[2];
    TEventID evtVMte2_[2];
    TEventID evtVMte3_[2];
    TEventID evtMte3V_[2];
    TEventID evtMte3Mte2_[2];

    uint32_t n_;
    uint32_t h_;
    uint32_t hv_;
    uint32_t hvPerH_;
    uint32_t qRowStride_;
    uint32_t kRowStride_;
    uint32_t vRowStride_;
    float scale_;
    float spb_;
    float invSpb_;
    float thr_;
    uint32_t useL2_;
    uint32_t blockIdx_;
    uint32_t blockDim_;
    uint32_t iterCnt_;
};

}  // namespace fused_sigmoid_gating_recurrent

using fused_sigmoid_gating_recurrent::FusedSigmoidGatingRecurrent;

#define FGR_KERNEL_ARGS                                                                                               \
    GM_ADDR A_log, GM_ADDR a, GM_ADDR dt_bias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR b, GM_ADDR o, GM_ADDR pool,   \
            GM_ADDR cache_indices, GM_ADDR cu_seqlens, uint32_t n, uint32_t h, uint32_t hv, uint32_t qRowStride,      \
            uint32_t kRowStride, uint32_t vRowStride, float scale, float softplusBeta, float invSoftplusBeta,         \
            float softplusThreshold, uint32_t useQkL2norm

#endif  // __FUSED_SIGMOID_GATING_RECURRENT_KERNEL_LIB_H_
