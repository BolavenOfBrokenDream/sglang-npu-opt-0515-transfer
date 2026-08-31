// fused_sigmoid_gating_recurrent
// AscendC AIV kernel for GDN decode recurrent (sigmoid gating + delta rule update).
//
// Semantics match the production Triton kernel (sgl_kernel_npu fla
// fused_sigmoid_gating_recurrent, strided form) statement by statement. Decode
// only: exactly 1 token per sequence (host enforces T == N), K == V == 128 (host
// enforced), HV <= 8 (gating vectors padded to [8]). Pool layout [slot, HV, K, V];
// each (slot, hv) state is one contiguous 32KB (bf16) block — single DataCopyPad
// per head in/out.
//
// Precision design (aligned item by item with the Triton reference):
//   1. softplus threshold branch verbatim: softplus(x) = (beta*x <= thr) ?
//      log(1+exp(beta*x))/beta : x — both branches computed (same semantics as
//      tl.where; the overflowing exp lane is discarded by the scalar select);
//   2. decay = exp(g), g = -exp(A_log)*softplus, beta = 1/(1+exp(-b)) verbatim
//      (Div chains; no AscendC Sigmoid shortcut);
//   3. L2norm keeps the division form q/(sqrt(sum(q^2))+1e-6): ReduceSum + scalar
//      sqrt (sqrt before +1e-6, same order as Triton) + full-width Duplicate +
//      vector Div — do NOT switch to vector Rsqrt/Reciprocal (their float
//      precision fails the 2e-4 requirement, per CANN 9.0.0 docs);
//   4. h += k x v' uses separate Mul+Add (no MulAddDst/FMA contraction, matching
//      Triton's two roundings);
//   5. both K-dim reductions = 2 K slabs (64 rows each) x 5-pass pairwise tree
//      within a slab + sequential accumulation across slabs;
//   6. store with CAST_RINT (fp32->bf16 RNE, same as Triton .to(bf16));
//      bf16->fp32 CAST_NONE is exact;
//   7. all intermediates fp32 (state cast to fp32 on UB entry, cast back to pool
//      dtype on store).
//
// Two points are not a-priori bitwise: (1) K-dim reduction addition order;
// (2) Exp/Ln polynomial implementations vs triton-ascend lowering (A3 Div in
// default INTRINSIC mode is at most 1 ulp, same bucket).
//
// Parallelism/pipelining: grid = min(N*HV, AIV core count) (host-side
// GetCoreNumAiv); each core loops over work items idx = blockIdx,
// blockIdx+blockDim, ...; double-buffered input prefetch (item i+1's MTE2
// overlaps item i's V compute). All cross-pipe events explicit:
//   MTE2_V[p]  DMA-in(i) ready -> V reads (cast stage)
//   V_MTE2[p]  V done reading qkvBuf_[p] (cast stage ends) -> next-next DMA-in may overwrite
//   V_MTE3[p]  store-stage cast done -> DMA-out may read stateBuf_/oBuf_
//   MTE3_V[p]  DMA-out(i-2) done -> V may rewrite stateBuf_[p]/oBuf_[p] (start of
//              ComputeItem, before any V write; the fp32 arm rewrites stateBuf_ in
//              place as its work tile, so it must be earliest)
//   MTE3_MTE2[p] DMA-out(i-2) done -> DMA-in(i) may rewrite stateBuf_[p]/oBuf_[p]
//              (V-side writes to stateBuf_ are covered transitively via the V_MTE3
//              chain; qkvBuf_ is never touched by MTE3, hence the separate V_MTE2;
//              must be a parity pair — a single event would tail with
//              Set(M-2)->Set(M-1) and no Wait in between, i.e. same-ID consecutive
//              SetFlag with >=2 work items per core = documented hang UB, see
//              constraint 3)
// The first two iterations use iterCnt_ conditions to skip waits whose buffers
// have no prior accessor (waiting on a never-set flag hangs). KERNEL_TASK_TYPE
// pinned AIV_ONLY; the kernel uses fully manual sync (explicit
// PipeBarrier/SetFlag) and the tp_fusion_recurrent_kernel library is built with
// --cce-auto-sync=off (auto-sync only covers the TQue programming model, unused
// here; =on would only insert redundant compiler events).
//
// Structural constraint 1 (do not merge back): the bf16/fp32 entries must be one
// kernel per file. KERNEL_TASK_TYPE_DEFAULT does not coexist with multiple
// kernels in one file — only the first registers; the rest fail with
// RegisterAscendBinary ret 107000 and the launch hangs on a missing kernel. Hence
// fused_sigmoid_gating_recurrent_{bf16,fp32}_kernel.cpp, with the class and arg
// macro centralized here.
//
// Structural constraint 2 (do not switch back to Fetch): all events must use
// AllocEventID (paired with ReleaseEvents at the end). FetchEventID does not
// occupy an ID (CANN 9.0.0 docs: "this interface does not allocate a TEventID,
// it only provides an available one"); two calls for the same HardEvent return
// the SAME TEventID — the parity double-buffer events here (MTE2_V/V_MTE2/V_MTE3/
// MTE3_V x2 each) would all collapse onto single hardware flags, and with >=2
// work items per core a collapsed ID sees two consecutive SetFlag with no Wait
// in between = documented hang UB: kernel hangs, host sync never returns.
//
// Structural constraint 3 (do not exit the kernel with unconsumed SetFlag):
// "SetFlag/WaitFlag must appear in pairs" per the official docs; all production
// kernels in the tree drain at exit (recurrent_gated_delta_rule's
// SEvent.release() wait loop, causal_conv1d.h's explicit trailing WaitFlag,
// fused_qkvzba_conv1d's Set immediately followed by Wait). An unconsumed SetFlag
// leaves the hardware flag set across launches: the next launch on the same core
// deterministically gets the same IDs from AllocEventID, and the first SetFlag on
// a leftover channel is "two consecutive SetFlag on the same ID with no Wait" =
// documented hang UB — shape-independent, guaranteed on the 2nd launch in the
// process. DrainEvents waits out all leftover flags with iterCnt_ conditions;
// MTE3_MTE2 is split into a parity pair (strict Set/Wait alternation per ID).
//
// UB budget: bf16 ~155KB / fp32 ~160KB (within the 192KB limit).

#ifndef __FUSED_SIGMOID_GATING_RECURRENT_KERNEL_LIB_H_
#define __FUSED_SIGMOID_GATING_RECURRENT_KERNEL_LIB_H_

#include "kernel_operator.h"

// event_t / GetTPipePtr() are global-namespace symbols in CANN 9.0.0
// (cce_aicore_intrinsics.h / kernel_tpipe.h; causal_conv1d.h in-tree does the
// same) — do not add the AscendC:: qualifier.

using namespace AscendC;

namespace fused_sigmoid_gating_recurrent {

constexpr uint32_t FGR_K = 128;       // head_k_dim (host TORCH_CHECK enforced)
constexpr uint32_t FGR_V = 128;       // head_v_dim (host TORCH_CHECK enforced)
constexpr uint32_t FGR_V_SLAB = 64;   // V-dim slab (prod buffer UB budget)
constexpr uint32_t FGR_K_SLAB = 64;   // K-dim reduction slab (prod buffer UB budget)
constexpr uint32_t FGR_MAX_HV = 8;    // gating vectors padded to [8] (host enforces HV<=8)
constexpr uint32_t FGR_STATE = FGR_K * FGR_V;        // 16384 state elements per (slot, head)
constexpr uint32_t FGR_STATE_HALF = FGR_STATE / 2;   // 8192 = within the 255-repeat x 64 single-instruction cap
constexpr uint32_t FGR_MAX_N = 256;   // cu_seqlens/cache_indices UB residency cap (production bs<=128)
constexpr float FGR_L2_EPS = 1e-6f;   // verbatim with the Triton kernel's +1e-6 (added after sqrt)

// element offsets of the qkv parity buffer in UB (bf16: q/k/v 128 elements each, a/b 16-element pad slots each)
constexpr uint32_t OFF_Q = 0;
constexpr uint32_t OFF_K = FGR_K;                    // 128
constexpr uint32_t OFF_V = FGR_K + FGR_K;            // 256
constexpr uint32_t OFF_A = FGR_K + FGR_K + FGR_V;    // 384 (bf16 16 elements = 32B slot)
constexpr uint32_t OFF_B = OFF_A + 16;               // 400
constexpr uint32_t QKV_PARITY_ELEMS = OFF_B + 16;    // 416 elements = 832B

// gating group [8] slots (element offsets, in float)
enum GateSlot {
    GS_DT_BIAS = 0,   // dt_bias padded to [8]
    GS_EXP_ALOG = 8,  // exp(A_log) padded to [8] (computed in the prologue)
    GS_A_F32 = 16,    // current item's a row (after cast)
    GS_B_F32 = 24,    // current item's b row (after cast)
    GS_XA = 32,       // x = a + dt_bias
    GS_BETAX = 40,    // beta*x
    GS_E1 = 48,       // exp(beta*x)+1
    GS_SPV = 56,      // softplus branch value (1/beta)*log(1+exp(beta*x))
    GS_BETAV = 64,    // sigmoid(b), all lanes
    GS_G = 72,        // [8] slot of scalar g (lane0 valid)
    GS_DECAY = 80,    // exp(g) (lane0 valid)
    GS_ONES = 88,     // all ones (numerator of the beta Div)
    GS_TMP = 96,      // scratch (exp(-b) chain)
    GATE_ELEMS = 104  // total elements (104 * 4B = 416B, 32B aligned)
};

template <typename T>  // ssm pool dtype: bfloat16_t (production) / float (accuracy-regression arm)
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
        hvPerH_ = (h > 0) ? (hv / h) : 1;  // host enforces hv%h==0
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

        // ---- UB allocation (bf16 arm ~155KB / fp32 arm ~160KB, within the 192KB limit) ----
        pipe_->InitBuffer(stateBuf_[0], FGR_STATE * sizeof(T));
        pipe_->InitBuffer(stateBuf_[1], FGR_STATE * sizeof(T));
        if (IsBf16()) {
            // fp32 work tile (bf16 arm: state is cast here on UB entry; fp32 arm uses stateBuf directly as the work tile)
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
        pipe_->InitBuffer(reduceWorkBuf_, FGR_K * sizeof(float));  // ReduceSum workspace allocated separately
        pipe_->InitBuffer(sumBuf_, 2 * 8 * sizeof(float));         // [0]=q sum of squares, [8]=k sum of squares
        pipe_->InitBuffer(denomBuf_, FGR_K * sizeof(float));
        pipe_->InitBuffer(deltaBuf_, FGR_V * sizeof(float));
        pipe_->InitBuffer(oF32Buf_, FGR_V * sizeof(float));
        pipe_->InitBuffer(cuBuf_, (FGR_MAX_N + 8) * sizeof(int32_t));  // 32B aligned
        pipe_->InitBuffer(idxBuf_, FGR_MAX_N * sizeof(int32_t));
        pipe_->InitBuffer(gateBuf_, GATE_ELEMS * sizeof(float));

        // ---- events (TPipe forbids self-chosen IDs; must AllocEventID —
        // FetchEventID does not occupy an ID: repeated Fetch on the same
        // HardEvent returns the same TEventID, so the parity double-buffer
        // events would collapse onto a single hardware flag; two consecutive
        // SetFlag on the same ID with no Wait in between hits the
        // CANN-documented hang/UB) ----
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
            return;  // extra cores have no work items (not triggered when blockDim=min(N*HV, cores); defensive)
        }
        Prologue();
        // software pipeline (iteration handles work item idx, prefetches idx+blockDim; wait
        // conditions per the event table in the file header, iterCnt_ conditions skip
        // nonexistent prior accesses in the first two rounds):
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

    // Exit drain (official "SetFlag/WaitFlag must appear in pairs" constraint, same
    // discipline as in-tree RGDR SEvent.release() / causal_conv1d trailing WaitFlag):
    // the SetFlags of the last two iterations have no consumer within this launch and
    // must each be waited before exit — otherwise set hardware flags persist across
    // launches, and the next launch's first SetFlag on the same ID is "two consecutive
    // SetFlag on the same ID with no Wait in between" hang UB (shape-independent,
    // guaranteed on the 2nd launch in the process).
    // Conditions are complementary to the skip conditions in CopyIn/ComputeItem: with
    // iterCnt_ == M, the V_MTE2/MTE3_V/MTE3_MTE2 of parities (M-1)&1 and M&1 are
    // exactly the set-but-unconsumed flags.
    __aicore__ inline void DrainEvents()
    {
        if (iterCnt_ >= 1) {
            const uint32_t lastP = (iterCnt_ - 1) & 1;
            WaitFlag<HardEvent::V_MTE2>(evtVMte2_[lastP]);      // set by ComputeItem(M-1)
            WaitFlag<HardEvent::MTE3_V>(evtMte3V_[lastP]);      // set by CopyOut(M-1)
            WaitFlag<HardEvent::MTE3_MTE2>(evtMte3Mte2_[lastP]);  // set by CopyOut(M-1)
            if (iterCnt_ >= 2) {
                const uint32_t prevP = iterCnt_ & 1;
                WaitFlag<HardEvent::V_MTE2>(evtVMte2_[prevP]);      // set by ComputeItem(M-2)
                WaitFlag<HardEvent::MTE3_V>(evtMte3V_[prevP]);      // set by CopyOut(M-2)
                WaitFlag<HardEvent::MTE3_MTE2>(evtMte3Mte2_[prevP]);  // set by CopyOut(M-2)
            }
        }
    }

    // paired with AllocEventID (mirrored order, same as causal_conv1d.h; early-exit cores
    // have no hardware side effects, TPipe is destroyed on kernel exit, pool bookkeeping
    // only matters within this launch)
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
        // cu_seqlens / cache_indices / A_log / dt_bias enter UB once (subsequent GetValue does zero GM reads)
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
        Duplicate(ones, 1.0f, FGR_MAX_HV);  // all-ones numerator for the beta Div
        // gating inputs ready after MTE2->V
        SetFlag<HardEvent::MTE2_V>(evtMte2V_[0]);
        WaitFlag<HardEvent::MTE2_V>(evtMte2V_[0]);
        PipeBarrier<PIPE_V>();
        Exp(expAlogPad, expAlogPad, FGR_MAX_HV);  // exp(A_log); pad lanes get 1.0, harmless, never read
        PipeBarrier<PIPE_V>();
        // MTE2->S for cu/idx (the only cross-pipe sync of this kind in the kernel; afterwards UB GetValue reads directly)
        SetFlag<HardEvent::MTE2_S>(evtMte2S_);
        WaitFlag<HardEvent::MTE2_S>(evtMte2S_);
    }

    // DMA-in all inputs of work item idx = i_n * HV + i_hv into the parity buffer
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t p)
    {
        // buffer-reuse waits (MTE2 pipe only; skipped while iterCnt_<1 as the parity
        // buffers have no prior accessor): prior reader of stateBuf_[p]/oBuf_[p] is
        // DMA-out (MTE3); prior reader of qkvBuf_[p] is the cast stage (V, only needs
        // waiting before DMA-in reuse)
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
        // q/k rows (strided views addressed by explicit row stride; contiguous case stride == H*K, identical addressing)
        DataCopyParams rowQKParams{1, static_cast<uint16_t>(FGR_K * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(qkvLocal[OFF_Q], qGm_[static_cast<uint64_t>(bos) * qRowStride_ + iH * FGR_K], rowQKParams,
                    padParams);
        DataCopyPad(qkvLocal[OFF_K], kGm_[static_cast<uint64_t>(bos) * kRowStride_ + iH * FGR_K], rowQKParams,
                    padParams);
        // v row
        DataCopyParams rowVParams{1, static_cast<uint16_t>(FGR_V * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(qkvLocal[OFF_V], vGm_[static_cast<uint64_t>(bos) * vRowStride_ + iHv * FGR_V], rowVParams,
                    padParams);
        // a/b rows ([HV] bf16, 8B/16B small transfers go through the pad path)
        DataCopyParams rowABParams{1, static_cast<uint16_t>(hv_ * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(qkvLocal[OFF_A], aGm_[static_cast<uint64_t>(bos) * hv_], rowABParams, padParams);
        DataCopyPad(qkvLocal[OFF_B], bGm_[static_cast<uint64_t>(bos) * hv_], rowABParams, padParams);

        // state 32KB(bf16)/64KB(fp32) contiguous block; skipped when slot<0 (zero-initialized
        // in Compute, same semantics as Triton's idx>=0 guard)
        if (slot >= 0) {
            LocalTensor<T> stateLocal = stateBuf_[p].Get<T>();
            DataCopyParams stateParams{FGR_K, static_cast<uint16_t>(FGR_V * sizeof(T)), 0, 0};
            DataCopyPad(stateLocal, poolGm_[(static_cast<uint64_t>(slot) * hv_ + iHv) * FGR_STATE], stateParams,
                        padParams);
        }
        SetFlag<HardEvent::MTE2_V>(evtMte2V_[p]);
    }

    // V-side reduction for L2 normalization (shared by q/k); scalar sqrt+1e-6 and the
    // division are done in ComputeItem's scalar stage / V stage B (division verbatim
    // with Triton b_x/(sqrt(sum(x^2))+1e-6))
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

    // V-side chain of the softplus dual branch + beta (all [8] small vectors, one head per lane; pad lanes harmless, never read)
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
        Exp(e1, betax, FGR_MAX_HV);            // exp(beta*x) (lanes with bx>thr may overflow;
        PipeBarrier<PIPE_V>();                 // same two-branch semantics as tl.where, discarded by the scalar select)
        Adds(e1, e1, 1.0f, FGR_MAX_HV);        // 1+exp
        PipeBarrier<PIPE_V>();
        Ln(spv, e1, FGR_MAX_HV);               // log(1+exp)
        PipeBarrier<PIPE_V>();
        Muls(spv, spv, invSpb_, FGR_MAX_HV);   // (1/beta)*log(1+exp)
        PipeBarrier<PIPE_V>();
        // beta = 1/(1+exp(-b)): verbatim formula (no AscendC Sigmoid shortcut)
        Muls(tmp, bF32, -1.0f, FGR_MAX_HV);    // -b (multiply by -1.0 flips the sign exactly, no rounding)
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

        WaitFlag<HardEvent::MTE2_V>(evtMte2V_[p]);  // current item's q/k/v/a/b/state ready
        // out-buffer/work-tile rewrite wait: V may write stateBuf_[p]/oBuf_[p] only
        // after DMA-out(i-2) (MTE3) completes — the fp32 arm uses stateBuf_[p] as its
        // work tile and rewrites it in place from V stage C, so the wait must precede
        // any V write (the bf16 arm's first V write is much later in the store stage;
        // waiting early costs nothing). Skipped while iterCnt_<2 (no prior DMA-out on
        // this parity).
        if (iterCnt_ >= 2) {
            WaitFlag<HardEvent::MTE3_V>(evtMte3V_[p]);
        }
        // ---- V stage A: cast + L2 reduction + gating dual branch ----
        Cast(qF32, qkvLocal[OFF_Q], RoundMode::CAST_NONE, FGR_K);
        Cast(kF32, qkvLocal[OFF_K], RoundMode::CAST_NONE, FGR_K);
        Cast(vF32, qkvLocal[OFF_V], RoundMode::CAST_NONE, FGR_V);
        Cast(Gate(GS_A_F32), qkvLocal[OFF_A], RoundMode::CAST_NONE, FGR_MAX_HV);
        Cast(Gate(GS_B_F32), qkvLocal[OFF_B], RoundMode::CAST_NONE, FGR_MAX_HV);
        PipeBarrier<PIPE_V>();
        if (IsBf16()) {
            // state into the work tile (bf16->fp32 exact); zero-init when slot<0 (same semantics as Triton's idx>=0 guard)
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
        // V-side reads of qkvBuf_[p] end here (stateBuf_'s last V-side write is in the
        // store stage, covered transitively via the V_MTE3->MTE3->MTE3_MTE2 chain, no
        // separate event needed)
        SetFlag<HardEvent::V_MTE2>(evtVMte2_[p]);
        if (useL2_ != 0) {
            L2Norm(qF32, 0);  // sumBuf_[0]
            L2Norm(kF32, 8);  // sumBuf_[8]
        }
        GatingVecPart();
        // ---- scalar stage (all scalars within one V_S + one S_V) ----
        SetFlag<HardEvent::V_S>(evtVS_);
        WaitFlag<HardEvent::V_S>(evtVS_);
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        float denomQ = 1.0f;
        float denomK = 1.0f;
        if (useL2_ != 0) {
            // same order as Triton: sqrt first, then +1e-6 (scalar sqrt;
            // do NOT switch to vector Rsqrt/Reciprocal — float precision fails 2e-4)
            denomQ = sqrt(sumLocal.GetValue(0)) + FGR_L2_EPS;
            denomK = sqrt(sumLocal.GetValue(8)) + FGR_L2_EPS;
        }
        const float xHv = Gate(GS_XA).GetValue(iHv);
        const float bxHv = Gate(GS_BETAX).GetValue(iHv);
        const float spHv = Gate(GS_SPV).GetValue(iHv);
        const float betaS = Gate(GS_BETAV).GetValue(iHv);
        const float expAlogS = Gate(GS_EXP_ALOG).GetValue(iHv);
        // softplus threshold branch (same per-lane semantics as tl.where; each lane here is exactly one head)
        const float softplusS = (bxHv <= thr_) ? spHv : xHv;
        const float gS = (-expAlogS) * softplusS;  // g = -exp(A_log)*softplus (sign flip exact)
        SetFlag<HardEvent::S_V>(evtSV_);
        WaitFlag<HardEvent::S_V>(evtSV_);
        // ---- V stage B: division normalization + decay exp ----
        LocalTensor<float> denom = denomBuf_.Get<float>();
        if (useL2_ != 0) {
            Duplicate(denom, denomQ, FGR_K);
            PipeBarrier<PIPE_V>();
            Div(qF32, qF32, denom, FGR_K);  // true division (A3 Div in default INTRINSIC mode
            PipeBarrier<PIPE_V>();          // is at most 1 ulp, counted in the instruction-level difference list)
            Duplicate(denom, denomK, FGR_K);
            PipeBarrier<PIPE_V>();
            Div(kF32, kF32, denom, FGR_K);
            PipeBarrier<PIPE_V>();
        }
        Muls(qF32, qF32, scale_, FGR_K);  // scale after norm (Triton order)
        PipeBarrier<PIPE_V>();
        Duplicate(Gate(GS_G), gS, FGR_MAX_HV);
        PipeBarrier<PIPE_V>();
        Exp(Gate(GS_DECAY), Gate(GS_G), FGR_MAX_HV);  // decay = exp(g) (lane0 valid)
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(evtVS_);
        WaitFlag<HardEvent::V_S>(evtVS_);
        const float decayS = Gate(GS_DECAY).GetValue(0);
        SetFlag<HardEvent::S_V>(evtSV_);  // S->V sync so decayS is visible as the
        WaitFlag<HardEvent::S_V>(evtSV_); // scalar operand of the Muls in V stage C
        // ---- V stage C: delta rule main chain ----
        // h *= decay (Triton: b_h *= b_decay, before delta)
        Muls(work, work, decayS, FGR_STATE_HALF);
        Muls(work[FGR_STATE_HALF], work[FGR_STATE_HALF], decayS, FGR_STATE_HALF);
        PipeBarrier<PIPE_V>();
        // k broadcast tile (Brcb: each element widened 8x; with src1BlkStride=0 gives full row x k[k])
        LocalTensor<float> brcb = brcbBuf_.Get<float>();
        Brcb(brcb, kF32, FGR_K / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        LocalTensor<float> prod = prodBuf_.Get<float>();
        LocalTensor<float> delta = deltaBuf_.Get<float>();
        // delta[v] = sum_k h[k,v]*k[k] (2 K slabs: in-slab 5-pass tree leaves 2 rows, merged then accumulated across slabs)
        for (uint32_t s = 0; s < FGR_V / FGR_V_SLAB; s++) {
            ReduceKSlab(prod, work, brcb, s, 0);
            Add(delta[s * FGR_V_SLAB], prod, prod[FGR_V_SLAB], FGR_V_SLAB);  // merge slab-0's two rows into the slot
            PipeBarrier<PIPE_V>();
            ReduceKSlab(prod, work, brcb, s, 1);
            Add(prod, prod, prod[FGR_V_SLAB], FGR_V_SLAB);  // merge slab-1's two rows into prod row 0
            PipeBarrier<PIPE_V>();
            Add(delta[s * FGR_V_SLAB], delta[s * FGR_V_SLAB], prod, FGR_V_SLAB);  // cross-slab accumulation
            PipeBarrier<PIPE_V>();
        }
        // v' = (v - delta) * beta (Sub then Muls, two roundings, same as Triton)
        Sub(vF32, vF32, delta, FGR_V);
        PipeBarrier<PIPE_V>();
        Muls(vF32, vF32, betaS, FGR_V);
        PipeBarrier<PIPE_V>();
        // h += k x v' (separate Mul+Add, two roundings; no MulAddDst/FMA contraction)
        for (uint32_t s = 0; s < FGR_V / FGR_V_SLAB; s++) {
            for (uint32_t ks = 0; ks < FGR_K / FGR_K_SLAB; ks++) {
                // prod[r, :] = v'[s slot] * k[ks*64+r] (src0RepStride=0: the whole slab reuses the same v' slot)
                Mul<float>(prod, vF32[s * FGR_V_SLAB], brcb[ks * FGR_K_SLAB * 8], FGR_V_SLAB,
                           static_cast<uint8_t>(FGR_K_SLAB), {1, 1, 0, 8, 0, 1});
                PipeBarrier<PIPE_V>();
                const uint32_t wOff = ks * FGR_K_SLAB * FGR_V + s * FGR_V_SLAB;
                Add<float>(work[wOff], work[wOff], prod, FGR_V_SLAB, static_cast<uint8_t>(FGR_K_SLAB),
                           {1, 1, 1, 16, 16, 8});
                PipeBarrier<PIPE_V>();
            }
        }
        // q broadcast tile (overwrites the k broadcast tile: k unused from here on)
        Brcb(brcb, qF32, FGR_K / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        // o[v] = sum_k h_new[k,v]*q[k] (same shape as delta; uses the rank-1-updated h)
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
        // ---- store stage ----
        LocalTensor<bfloat16_t> oLocal = oBuf_[p].Get<bfloat16_t>();
        Cast(oLocal, oF32, RoundMode::CAST_RINT, FGR_V);  // RNE, same as Triton .to(bf16)
        if (IsBf16()) {
            Cast(stateBuf_[p].Get<T>(), work, RoundMode::CAST_RINT, FGR_STATE_HALF);
            Cast(stateBuf_[p].Get<T>()[FGR_STATE_HALF], work[FGR_STATE_HALF], RoundMode::CAST_RINT, FGR_STATE_HALF);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evtVMte3_[p]);
    }

    // prod[r, :] = work[(ks*64+r)*128 + s slot] * brcbVec[ks*64+r], then a 5-pass
    // pairwise tree reduction (in place, read addresses never below write addresses,
    // no self-overwrite) leaving 2 rows at prod[0..63] and prod[64..127] (the caller
    // merges/places them with one Add).
    __aicore__ inline void ReduceKSlab(LocalTensor<float> prod, LocalTensor<float> work, LocalTensor<float> brcb,
                                       uint32_t vSlab, uint32_t kSlab)
    {
        const uint32_t wOff = kSlab * FGR_K_SLAB * FGR_V + vSlab * FGR_V_SLAB;
        // src1BlkStride=0: all 8 blocks within a repeat read the same 8-element brcb
        // block (= scalar x8); src1RepStride=1: advance block by block across repeats
        // (k/q row by row)
        Mul<float>(prod, work[wOff], brcb[kSlab * FGR_K_SLAB * 8], FGR_V_SLAB, static_cast<uint8_t>(FGR_K_SLAB),
                   {1, 1, 0, 8, 16, 1});
        PipeBarrier<PIPE_V>();
        // 5-pass tree: 64->32->16->8->4->2 rows (the final two-row merge is done by the caller per slot placement)
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

        WaitFlag<HardEvent::V_MTE3>(evtVMte3_[p]);  // store-stage cast done
        // o row store (same addressing as Triton: (bos*HV+i_hv)*V)
        DataCopyParams oParams{1, static_cast<uint16_t>(FGR_V * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(oGm_[(static_cast<uint64_t>(bos) * hv_ + iHv) * FGR_V], oBuf_[p].Get<bfloat16_t>(), oParams);
        // state writeback (not written when slot<0, same guard as Triton)
        if (slot >= 0) {
            DataCopyParams stateParams{FGR_K, static_cast<uint16_t>(FGR_V * sizeof(T)), 0, 0};
            DataCopyPad(poolGm_[(static_cast<uint64_t>(slot) * hv_ + iHv) * FGR_STATE], stateBuf_[p].Get<T>(),
                        stateParams);
        }
        // two sets: next-round rewrite gates for stateBuf_/oBuf_ (MTE3_V for the V side,
        // MTE3_MTE2 for the MTE2 side; both per parity — a single MTE3_MTE2 event would
        // tail with two Sets and no Wait in between = hang UB)
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
