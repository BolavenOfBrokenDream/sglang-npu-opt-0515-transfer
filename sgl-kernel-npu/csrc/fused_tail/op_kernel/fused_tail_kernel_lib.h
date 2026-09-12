/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/*!
 * \file fused_tail_kernel_lib.h
 * \brief fused_tail device-kernel shared implementation — all common machinery
 *        of the fin(+skip1)+AR+add+gemma rmsnorm layer-tail fusion.
 *
 * Productionized from the tp_oneshot_ar_probe P5/P8 kernels; mechanism follows
 * deepep production idioms:
 *   - Cross-card data plane: GM->UB->MTE3 DataCopyPad->peer window GM (DMA
 *     reads/writes HBM directly, cross-card visibility is intrinsic; no dcci
 *     on the data/flag path).
 *   - Flag protocol [pinned v2]: phase-A has no self item (spin target R-1),
 *     remote flag items are written twice, consumed lines are cleared by pure
 *     MTE3 (a dcci clean would race with in-flight peer flag writes); item
 *     stride pinned to 32B.
 *   - Data-before-flag program order: all push-phase data is issued on MTE3,
 *     then PipeBarrier<PIPE_ALL>, then flags.
 *   - Reduction: fp32 accumulation + fixed rank order 0..R-1 (r==rank uses the
 *     UB local contribution) + a single final CAST_RINT to bf16, so two runs
 *     are bit-identical.
 *   - Timeout: GetSystemCycle + CYCLES_PER_US=50, two checkpoints (1x limit
 *     records forensics, 2x limit writes DFX and exits without hanging;
 *     wall-clock bound 2x cycle_limit).
 *   - Scalar paths (address table/counters/DFX): gm_store/gm_load/gm_dcci;
 *     the cell area lives in plain GM, 128B/cell one full line per core, and
 *     BumpRing re-reads after writing with retry.
 *   - SyncFunc: FetchEventID + SetFlag/WaitFlag paired and consumed
 *     immediately; the kernel exits with no leftover SetFlag.
 *
 * Buffer layout (all VAs arrive via the addr_tab int64 tensor and are read
 * in-kernel with gm_load, so they are graph-capture safe and never baked into
 * tiling):
 *
 *   data symmem (one block per rank, byte offsets):
 *     slot(g, r)  = dataVA + g*ring_stride + r*slot_stride   g in [0,RING=4)
 *       tile t    = slot(g,r) + t*tile_bytes
 *
 *   flag symmem (one block per rank, byte offsets; phase A only = RING gens):
 *     line(g,t)   = flagVA + g*flag_ring_stride + t*world*32
 *       item i    = line + i*32 (32B = 8 fp32, [0]=1.0f marks arrival)
 *
 *   cell area (plain GM tensor, VA via addr_tab[17]):
 *     [0,48*128)        = counter cells (128B exclusive line per core;
 *                         slot[0]=counter, slot[1]=build stamp
 *                         FUSED_TAIL_BUILD_REV, slot[8]=readback-exhausted mark)
 *     [48*128,+48*128)  = DFX cells (per-core timeout/late forensics)
 *
 * Kernel organization: combine-style with zero intra-kernel cross-core sync —
 * each core runs an independent pipeline, no SyncAll, no schedule-mode
 * dependency. tile = 8KB bf16.
 *
 * Build discipline: one kernel per file +
 * KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY) (multiple kernels in one file
 * with explicit task type: only the first registers successfully);
 * --cce-auto-sync=on (matches the probe validation baseline; all sync is
 * handwritten and correctness does not rely on auto-sync).
 *
 * Kernel entry signature convention (auto_gen launch wrapper "last two params
 * = workspace/tiling"): the workspace slot carries a dummy parameter — that
 * slot's argument is deterministically corrupted on arrival (unresolved), so
 * no valid parameter is ever placed there.
 */

#ifndef FUSED_TAIL_KERNEL_LIB_H
#define FUSED_TAIL_KERNEL_LIB_H

#include "kernel_operator.h"
#include "common_tiling_kernel.h"

#include "../op_host/fused_tail_tiling_data.h"

namespace sglang {
namespace npu_kernel {
namespace fused_tail {

using T = bfloat16_t;
using namespace AscendC;

constexpr uint32_t FT_RING = 4;               // buffer generations (tolerates RING-1=3 gens of cross-rank skew)
constexpr uint32_t FT_FLAG_ITEM_BYTES = 32U;  // flag item stride (deepep 32B idiom, pinned)
constexpr uint32_t FT_FLOATS_PER_FLAG = 8U;   // 32B/4B
constexpr uint64_t FT_CYCLES_PER_US = 50UL;   // GetSystemCycle -> us (deepep value)
constexpr uint32_t FT_MAX_CORES = 48U;        // AIV core limit (cell area capacity)
constexpr uint32_t FT_MAX_RANK = 8U;          // R limit (addr_tab capacity)
constexpr uint32_t FT_FLAG_LINE_BYTES = FT_MAX_RANK * FT_FLAG_ITEM_BYTES;  // 256B
constexpr int32_t FT_DFX_MAGIC = 0xDEAD0000;  // timeout cell magic (| core low 16 bits)
constexpr uint32_t FT_CELL_BYTES = 128U;      // one full 128B line per cell
// Build stamp: BumpRing writes it to counter cell slot[1] every launch and the
// wrapper reads it back at init (deployment self-check). Must be bumped on any
// kernel change.
constexpr int32_t FUSED_TAIL_BUILD_REV = 20260922;  // tiling pinned keep-alive fix (host side)
// Explicit scratch for the multi-level Sum reduction (the 4-arg form is safer
// than relying on the 3-arg overload's internal stack buffer); requirement =
// ceil(ceil(n/64)/8)*32B, h=4096 needs 256B.
constexpr uint32_t FT_SUM_TMP_BYTES = 512U;
// fin front-stage eri row staging limit (K<=16 x 4B = 64B).
constexpr uint32_t FT_FINIDX_MAX_BYTES = 64U;

// ---------------------------------------------------------------------------
// Scalar primitives: gm_store/gm_load/gm_dcci (deepep hccl_shmem.hpp idiom).
// Address-table/counter/DFX scalar paths only; cross-card data and flags move
// by DMA (MTE2/MTE3), never dcci.
// ---------------------------------------------------------------------------
template <typename S>
__aicore__ inline void gm_store(__gm__ S *addr, S val)
{
    *((__gm__ S *)addr) = val;
}

template <typename S>
__aicore__ inline S gm_load(__gm__ S *cache)
{
    return *((__gm__ S *)cache);
}

template <typename S>
__aicore__ inline void gm_dcci(__gm__ S *addr)
{
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

// ---------------------------------------------------------------------------
// SyncFunc: deepep idiom. Set/Wait paired and consumed immediately, no residue
// across calls (structural immunity to the "leftover SetFlag at kernel exit"
// hang).
// ---------------------------------------------------------------------------
template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// ---------------------------------------------------------------------------
// TailArBase: common pieces for tiling/address table/ring counter/flag IO/
// spin/clearing/DFX (used by the spin-AR kernel fused_fin_ar_norm only;
// fused_fin_add/fused_tail_zero never touch these).
// ---------------------------------------------------------------------------
class TailArBase {
public:
    __aicore__ inline void InitBase(GM_ADDR addr_tab, const FusedTailTilingData *td, TPipe *pipe)
    {
        td_ = td;
        core_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        tab_ = reinterpret_cast<__gm__ int64_t *>(addr_tab);
        flagSelfVA_ = FlagVA(td_->rank);

        // flag staging: [0]=1.0f, rest 0 (the spin side sums items, each
        // contributing 1.0).
        pipe->InitBuffer(flagStgBuf_, FT_FLAG_ITEM_BYTES);
        pipe->InitBuffer(flagLineBuf_, FT_FLAG_LINE_BYTES);
        pipe->InitBuffer(flagSumBuf_, FT_FLAG_ITEM_BYTES);
        pipe->InitBuffer(flagZeroBuf_, FT_FLAG_LINE_BYTES);
        pipe->InitBuffer(spinSumTmpBuf_, FT_SUM_TMP_BYTES);  // explicit Sum scratch for spin
        flagStg_ = flagStgBuf_.Get<float>();
        flagLine_ = flagLineBuf_.Get<float>();
        flagSum_ = flagSumBuf_.Get<float>();
        flagZero_ = flagZeroBuf_.Get<float>();
        spinSumTmp_ = spinSumTmpBuf_.Get<uint8_t>();
        Duplicate(flagStg_, 0.0f, FT_FLOATS_PER_FLAG);
        Duplicate(flagZero_, 0.0f, FT_FLAG_LINE_BYTES / sizeof(float));
        SyncFunc<HardEvent::V_S>();
        flagStg_.SetValue(0, 1.0f);
        // SetValue is a scalar (S) write later used as an MTE3 DataCopy source;
        // the S_MTE3 ordering is guaranteed inside WriteFlagItem.
    }

    // ---- address table: addr_tab[0..7]=data VA, [8..15]=flag VA; r==rank is
    //      the local VA ----
    __aicore__ inline uint64_t DataVA(uint32_t r) const
    {
        return static_cast<uint64_t>(gm_load(tab_ + r));
    }

    __aicore__ inline uint64_t FlagVA(uint32_t r) const
    {
        return static_cast<uint64_t>(gm_load(tab_ + 8U + r));
    }

    // ---- cell area base (plain GM tensor, VA via addr_tab[17]; capture safe —
    //      contents are gm_load'ed at runtime, same channel as [0..15]) ----
    __aicore__ inline uint64_t CellVA() const
    {
        return static_cast<uint64_t>(gm_load(tab_ + 17U));
    }

    // ---- ring counter: per-core private cell (128B exclusive line), must be
    //      called by every core on every launch (including early-exit cores) to
    //      stay in lockstep. slot[0]=counter; slot[1]=build stamp;
    //      slot[8]=readback-exhausted mark. Write-then-readback with retry
    //      (<=4); a strictly +1 counter sequence is a correctness prerequisite
    //      of the ring protocol. ----
    __aicore__ inline uint32_t BumpRing()
    {
        __gm__ int32_t *cell = reinterpret_cast<__gm__ int32_t *>(CellVA() + td_->counter_offset +
                                                                  core_ * FT_CELL_BYTES);
        int32_t v = gm_load(cell);
        uint32_t att = 0;
        for (; att < 4U; ++att) {
            gm_store(cell, v + 1);
            gm_store(cell + 1, FUSED_TAIL_BUILD_REV);
            gm_dcci(reinterpret_cast<__gm__ uint8_t *>(cell));
            __asm__ __volatile__("" ::: "memory");
            if (gm_load(cell) == v + 1) {
                break;
            }
        }
        if (att >= 4U) {  // readback exhausted (rare): leave evidence in slot[8], continue with local v
            gm_store(cell + 8, FT_DFX_MAGIC | static_cast<int32_t>(0x0BADU));
            gm_dcci(reinterpret_cast<__gm__ uint8_t *>(cell));
        }
        launchIdx_ = static_cast<uint32_t>(v);
        return (static_cast<uint32_t>(v)) & (FT_RING - 1U);  // this launch uses the pre-increment generation
    }

    // ---- flag line address (local flag area; item stride pinned 32B) ----
    __aicore__ inline uint64_t FlagLineAddr(uint64_t flagVA, uint32_t phase, uint32_t ring, uint32_t tile) const
    {
        return flagVA + static_cast<uint64_t>(phase) * td_->flag_phase_stride +
               static_cast<uint64_t>(ring) * td_->flag_ring_stride +
               static_cast<uint64_t>(tile) * td_->world * FT_FLAG_ITEM_BYTES;
    }

    // ---- write one flag item (32B DMA, [0]=1.0f); target flagVA comes from
    //      the caller (peer = push acknowledgement) ----
    __aicore__ inline void WriteFlagItem(uint64_t flagVA, uint32_t phase, uint32_t ring, uint32_t tile, uint32_t item)
    {
        GlobalTensor<float> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(FlagLineAddr(flagVA, phase, ring, tile) +
                                                            item * FT_FLAG_ITEM_BYTES));
        // flagStg_ content was written by scalar init; S_MTE3 makes MTE3 see it
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy<float>(dst, flagStg_, FT_FLOATS_PER_FLAG);
    }

    // ---- phase-A spin target item count: v2 protocol = R-1 (no self item) ----
    __aicore__ inline float SpinTargetA() const
    {
        return static_cast<float>(static_cast<int32_t>(td_->world - 1U));
    }

    // ---- clear a local flag line after consumption (pure MTE3, no dcci — a
    //      dcci clean would write a stale local DCache line back to HBM and
    //      race with in-flight peer flag writes) ----
    __aicore__ inline void ClearFlagLine(uint32_t phase, uint32_t ring, uint32_t tile)
    {
        GlobalTensor<float> dst;
        uint64_t lineAddr = FlagLineAddr(flagSelfVA_, phase, ring, tile);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lineAddr));
        uint32_t lineFloats = td_->world * (FT_FLAG_ITEM_BYTES / sizeof(float));
        SyncFunc<HardEvent::V_MTE3>();  // flagZero_ was written by V(Duplicate)
        DataCopy<float>(dst, flagZero_, lineFloats);
    }

    // ---- arrival bitmask of a flag line: bit i = item i's [0] >= 0.5f (DFX
    //      forensics; caller must hold a valid line in flagLine_ with MTE2_V
    //      and V_S already synced) ----
    __aicore__ inline uint32_t ArriveMask() const
    {
        uint32_t mask = 0U;
        uint32_t stride = FT_FLAG_ITEM_BYTES / sizeof(float);
        for (uint32_t i = 0; i < td_->world; ++i) {
            if (flagLine_(i * stride) >= 0.5f) {
                mask |= (1U << i);
            }
        }
        return mask;
    }

    // ---- spin on a local flag line: DataCopy line into UB + Sum == target±0.5
    //      (deepep combine WaitDispatch idiom; dst/src kept separate to avoid
    //      Sum source/dest overlap). Two-checkpoint forensics: at 1x
    //      cycle_limit record (s1,m1) and keep waiting; only at 2x declare
    //      timeout — distinguishing "late" (peer write landing/visibility
    //      delay) from "lost" (never arrives); lines completed within
    //      (limit, 2x limit) write DFX with phase|0x100 as the late mark.
    //      Returns true = timeout (DFX cell written; caller must exit without
    //      further spinning). ----
    __aicore__ inline bool SpinFlagLine(uint32_t phase, uint32_t ring, uint32_t tile, float target)
    {
        GlobalTensor<float> line;
        line.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(FlagLineAddr(flagSelfVA_, phase, ring, tile)));
        uint32_t n = td_->world * (FT_FLAG_ITEM_BYTES / sizeof(float));
        SumParams sumParams{1U, n, n};
        float minTarget = target - 0.5f;
        float maxTarget = target + 0.5f;
        float localState = 0.0f;
        uint64_t start = static_cast<uint64_t>(AscendC::GetSystemCycle());
        uint64_t limit = static_cast<uint64_t>(td_->cycle_limit);
        bool marked = false;
        float sum1 = 0.0f;
        uint32_t mask1 = 0U;
        while ((localState < minTarget) || (localState > maxTarget)) {
            SyncFunc<HardEvent::S_MTE2>();
            DataCopy<float>(flagLine_, line, n);
            SyncFunc<HardEvent::MTE2_V>();
            Sum(flagSum_, flagLine_, spinSumTmp_, sumParams);
            SyncFunc<HardEvent::V_S>();
            localState = flagSum_(0);
            uint64_t now = static_cast<uint64_t>(AscendC::GetSystemCycle());
            if (!marked && (now - start) > limit) {
                sum1 = localState;
                mask1 = ArriveMask();
                marked = true;
            }
            if ((now - start) > 2U * limit) {
                WriteDfx(tile, phase, localState, ArriveMask(), sum1, mask1, 1U);
                return true;
            }
        }
        if (marked) {
            // Late success: completed within (limit, 2x limit) — record DFX
            // with the late mark (normal return).
            WriteDfx(tile, phase | 0x100U, localState, ArriveMask(), sum1, mask1, 1U);
        }
        return false;
    }

    // ---- timeout/late DFX: slot[0]=magic|core low 16 bits, [1]=tile |
    //      launch index<<16, [2]=phase (|0x100=late mark), [3]=Sum at exit,
    //      [4]=arrival mask at exit, [5]=Sum at 1x limit, [6]=mask at 1x
    //      limit, [7]=marked. Reading: s~0 all lost / s~target-1 missing items
    //      (m shows which) / s>target foreign writes mixed in; s1->s evolution
    //      = late vs truly lost. ----
    __aicore__ inline void WriteDfx(uint32_t tile, uint32_t phase, float lastSum, uint32_t mask = 0U,
                                    float sum1 = 0.0f, uint32_t mask1 = 0U, uint32_t marked = 0U)
    {
        __gm__ int32_t *cell = reinterpret_cast<__gm__ int32_t *>(CellVA() + td_->dfx_offset +
                                                                  core_ * FT_CELL_BYTES);
        union { float f; int32_t i; } us, u1;
        us.f = lastSum;
        u1.f = sum1;
        gm_store(cell, FT_DFX_MAGIC | static_cast<int32_t>(core_ & 0xFFFFU));
        gm_store(cell + 1, static_cast<int32_t>((tile & 0xFFFFU) | (launchIdx_ << 16)));
        gm_store(cell + 2, static_cast<int32_t>(phase));
        gm_store(cell + 3, us.i);
        gm_store(cell + 4, static_cast<int32_t>(mask));
        gm_store(cell + 5, u1.i);
        gm_store(cell + 6, static_cast<int32_t>(mask1));
        gm_store(cell + 7, static_cast<int32_t>(marked));
        gm_dcci(reinterpret_cast<__gm__ uint8_t *>(cell));
    }

    // ---- slot address (target card dataVA; tile granularity) ----
    __aicore__ inline uint64_t SlotAddr(uint64_t dataVA, uint32_t ring, uint32_t slot, uint32_t tile) const
    {
        return dataVA + static_cast<uint64_t>(ring) * td_->ring_stride +
               static_cast<uint64_t>(slot) * td_->slot_stride +
               static_cast<uint64_t>(tile) * td_->tile_bytes;
    }

    // ---- push one tile (bf16 data already in UB) to slot[rank][tile] of all
    //      peers. Caller must guarantee srcUb's last write is synced with MTE3:
    //      V sources need SyncFunc<V_MTE3>, MTE2 sources SyncFunc<MTE2_MTE3>. ----
    __aicore__ inline void PushTileData(const LocalTensor<T> &srcUb, uint32_t ring, uint32_t tile)
    {
        DataCopyExtParams extParams{1U, td_->tile_bytes, 0U, 0U, 0U};
        for (uint32_t q = 0; q < td_->world; ++q) {
            if (q == td_->rank) {
                continue;
            }
            GlobalTensor<T> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(SlotAddr(DataVA(q), ring, td_->rank, tile)));
            DataCopyPad(dst, srcUb, extParams);
        }
    }

    // ---- send flags: PipeBarrier<PIPE_ALL> (all push-phase data issued on
    //      MTE3 before flags) then write peer flags (v2 protocol: no self
    //      item, spin target R-1; remote items written twice — single-write
    //      loss probability p becomes ~p^2, a probabilistic mitigation). ----
    __aicore__ inline void SendTileFlags(uint32_t ring, uint32_t tile)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t q = 0; q < td_->world; ++q) {
            if (q == td_->rank) {
                continue;
            }
            WriteFlagItem(FlagVA(q), 0U, ring, tile, td_->rank);
            WriteFlagItem(FlagVA(q), 0U, ring, tile, td_->rank);
        }
    }

    // ---- reduce one tile: fixed rank order 0..R-1, r==rank uses the UB local
    //      contribution, others read local slot[r][tile]; fp32 accumulation +
    //      a single CAST_RINT. Result written to arUb (bf16 tile = AR output
    //      point). ----
    __aicore__ inline void ReduceTileUbContrib(const LocalTensor<T> &ownContribUb, uint32_t ring, uint32_t tile,
                                               LocalTensor<T> &arUb)
    {
        uint32_t tileElems = td_->tile_bytes / sizeof(T);
        DataCopyExtParams extParams{1U, td_->tile_bytes, 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
        Duplicate(accF_, 0.0f, tileElems);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < td_->world; ++r) {
            if (r == td_->rank) {
                Cast(tmpF_, ownContribUb, RoundMode::CAST_NONE, tileElems);
            } else {
                GlobalTensor<T> src;
                src.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(SlotAddr(DataVA(td_->rank), ring, r, tile)));
                LocalTensor<T> st = slotQueue_.AllocTensor<T>();
                DataCopyPad(st, src, extParams, padParams);
                slotQueue_.EnQue(st);
                st = slotQueue_.DeQue<T>();
                Cast(tmpF_, st, RoundMode::CAST_NONE, tileElems);
                slotQueue_.FreeTensor(st);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(accF_, accF_, tmpF_, tileElems);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Cast(arUb, accF_, RoundMode::CAST_RINT, tileElems);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---- reduction buffers ----
    __aicore__ inline void InitReduceBufs(TPipe *pipe, uint32_t tileBytes)
    {
        pipe->InitBuffer(slotQueue_, 1, tileBytes);
        pipe->InitBuffer(accBuf_, tileBytes * 2U);  // fp32 accumulator (bf16 -> fp32 doubles bytes)
        pipe->InitBuffer(tmpBuf_, tileBytes * 2U);  // cast temporary
        accF_ = accBuf_.Get<float>();
        tmpF_ = tmpBuf_.Get<float>();
    }

    const FusedTailTilingData *td_;
    uint32_t core_;
    __gm__ int64_t *tab_;
    uint64_t flagSelfVA_;
    uint32_t launchIdx_ = 0U;  // pre-increment BumpRing count = this process's launch index (DFX)
    TBuf<> flagStgBuf_;
    TBuf<> flagLineBuf_;
    TBuf<> flagSumBuf_;
    TBuf<> flagZeroBuf_;
    TBuf<> spinSumTmpBuf_;
    LocalTensor<float> flagStg_;
    LocalTensor<float> flagLine_;
    LocalTensor<float> flagSum_;
    LocalTensor<float> flagZero_;
    LocalTensor<uint8_t> spinSumTmp_;
    // reduction buffers
    TQue<TPosition::VECIN, 1> slotQueue_;
    TBuf<> accBuf_;
    TBuf<> tmpBuf_;
    LocalTensor<float> accF_;
    LocalTensor<float> tmpF_;
};

// ---------------------------------------------------------------------------
// TailArNormBase (fused_fin_ar_norm tail stage): TailArBase + "AR-result bf16
// row -> +residual -> add_out -> gemma rmsnorm -> norm_out". The math replicates
// A3a add_gemma_rms_norm point for point:
//   add  = bf16(ar) + bf16(res): up-cast to fp32, Add, CAST_RINT to bf16 -> add_out
//   xf   = fp32(add_out); var = Sum(xf^2)/H; out = xf*rsqrt(var+eps)*(w_fp32+1)
//        -> CAST_RINT -> norm_out ((w+1) is the gemma convention)
// The only difference vs stock is the cross-rank summation order (fixed rank
// order here), so it is non-bitwise but in the same error band.
// ---------------------------------------------------------------------------
class TailArNormBase : public TailArBase {
public:
    // res/norm w/add_out/norm_out GM + tail-stage buffer init; w is preloaded
    // as fp32 (w+1).
    __aicore__ inline void InitNorm(GM_ADDR res, GM_ADDR w, GM_ADDR add_out, GM_ADDR norm_out, TPipe *pipe)
    {
        uint32_t h = td_->h;
        uint32_t rowBytes = h * sizeof(T);
        resGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(res));
        addOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(add_out));
        normOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(norm_out));
        GlobalTensor<T> wGm;
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(w));

        pipe->InitBuffer(arBuf_, td_->tile_bytes);        // ar bf16 tile
        pipe->InitBuffer(resQueue_, 1, rowBytes);
        pipe->InitBuffer(addFBuf_, rowBytes * 2U);        // add/xf fp32 (row granularity)
        pipe->InitBuffer(sqBuf_, rowBytes * 2U);          // xf^2
        pipe->InitBuffer(wQueue_, 1, rowBytes);
        pipe->InitBuffer(wFBuf_, rowBytes * 2U);          // (w+1) fp32
        pipe->InitBuffer(outQueue_, 1, rowBytes);
        pipe->InitBuffer(varBuf_, FT_FLAG_ITEM_BYTES);
        pipe->InitBuffer(sumTmpBuf_, FT_SUM_TMP_BYTES);  // explicit Sum scratch
        arT_ = arBuf_.Get<T>();
        addF_ = addFBuf_.Get<float>();
        sqF_ = sqBuf_.Get<float>();
        wF_ = wFBuf_.Get<float>();
        varUb_ = varBuf_.Get<float>();
        sumTmp_ = sumTmpBuf_.Get<uint8_t>();
        invH_ = 1.0f / static_cast<float>(static_cast<int32_t>(h));  // no direct uint->float on aicore; go via int32

        DataCopyExtParams extRow{1U, rowBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
        LocalTensor<T> wt = wQueue_.AllocTensor<T>();
        DataCopyPad(wt, wGm, extRow, padParams);
        wQueue_.EnQue(wt);
        wt = wQueue_.DeQue<T>();
        Cast(wF_, wt, RoundMode::CAST_NONE, h);
        wQueue_.FreeTensor(wt);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(wF_, wF_, 1.0f, h);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // AR-result bf16 row (in UB, row slice of arT_) -> write add_out/norm_out rows
    __aicore__ inline void AddNormRow(const LocalTensor<T> &arRowUb, uint32_t rowIdx)
    {
        uint32_t h = td_->h;
        DataCopyExtParams extRow{1U, static_cast<uint32_t>(h * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
        uint64_t rowOff = static_cast<uint64_t>(rowIdx) * h;

        LocalTensor<T> rt = resQueue_.AllocTensor<T>();
        DataCopyPad(rt, resGm_[rowOff], extRow, padParams);
        resQueue_.EnQue(rt);
        rt = resQueue_.DeQue<T>();
        Cast(tmpF_, arRowUb, RoundMode::CAST_NONE, h);   // ar row up to fp32 (tmpF_ reuse)
        AscendC::PipeBarrier<PIPE_V>();
        Cast(addF_, rt, RoundMode::CAST_NONE, h);
        resQueue_.FreeTensor(rt);
        AscendC::PipeBarrier<PIPE_V>();
        Add(addF_, addF_, tmpF_, h);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<T> ao = outQueue_.AllocTensor<T>();
        Cast(ao, addF_, RoundMode::CAST_RINT, h);        // add_out (= stock add node point)
        AscendC::PipeBarrier<PIPE_V>();
        Cast(addF_, ao, RoundMode::CAST_NONE, h);        // xf = fp32(add_out) (A3a same point)
        outQueue_.EnQue(ao);
        ao = outQueue_.DeQue<T>();
        DataCopyPad(addOutGm_[rowOff], ao, extRow);
        outQueue_.FreeTensor(ao);
        AscendC::PipeBarrier<PIPE_V>();
        // var = Sum(xf^2)/h + eps -> rsqrt (4-arg overload + explicit scratch)
        Mul(sqF_, addF_, addF_, h);
        AscendC::PipeBarrier<PIPE_V>();
        SumParams sumParams{1U, h, h};
        Sum(varUb_, sqF_, sumTmp_, sumParams);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(varUb_, varUb_, invH_, FT_FLOATS_PER_FLAG);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(varUb_, varUb_, td_->eps, FT_FLOATS_PER_FLAG);
        AscendC::PipeBarrier<PIPE_V>();
        Rsqrt(varUb_, varUb_, FT_FLOATS_PER_FLAG);
        AscendC::PipeBarrier<PIPE_V>();
        SyncFunc<HardEvent::V_S>();
        float rs = varUb_(0);
        // out = xf * rs * (w+1) -> CAST_RINT
        Muls(addF_, addF_, rs, h);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(addF_, addF_, wF_, h);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<T> no = outQueue_.AllocTensor<T>();
        Cast(no, addF_, RoundMode::CAST_RINT, h);
        outQueue_.EnQue(no);
        no = outQueue_.DeQue<T>();
        DataCopyPad(normOutGm_[rowOff], no, extRow);
        outQueue_.FreeTensor(no);
    }

    GlobalTensor<T> resGm_;
    GlobalTensor<T> addOutGm_;
    GlobalTensor<T> normOutGm_;
    TBuf<> arBuf_;
    TQue<TPosition::VECIN, 1> resQueue_;
    TBuf<> addFBuf_;
    TBuf<> sqBuf_;
    TQue<TPosition::VECIN, 1> wQueue_;
    TBuf<> wFBuf_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    TBuf<> varBuf_;
    TBuf<> sumTmpBuf_;
    LocalTensor<T> arT_;
    LocalTensor<float> addF_;
    LocalTensor<float> sqF_;
    LocalTensor<float> wF_;
    LocalTensor<float> varUb_;
    LocalTensor<uint8_t> sumTmp_;
    float invH_;
};

// ---------------------------------------------------------------------------
// TailFinDbFront (used by fused_fin_add): double-buffered finalize(+skip1)
// front stage (numerically bit-identical to the serial front stage of
// fused_fin_ar_norm — same per-row ascending-k Cast/Muls/Add sequence and the
// same single CAST_RINT-to-bf16 point):
//   1. xexp queue depth 2 + in-row prefetch pipeline (kk+1's DataCopyPad
//      overlaps kk's vector compute);
//   2. scales/eri loaded per row with one 32B DataCopyPad batch into UB
//      (S_MTE2 / MTE2_S event pair);
//   3. skip1 load co-issued with the xexp k=0 load.
// use_eri=1: x=xp unpermuted [M*K,H], row address = eri[m*K+k]*h (eri int32
// [M*K], VA via tiling.eri_va).
// eventID budget (TQue: each buffer slot costs one MTE2_V + one V_MTE2
// eventID, pool = 8/class): VECIN slots = xQueue 2 + skip1Queue 1 = 3, ample.
// ---------------------------------------------------------------------------
class TailFinDbFront {
public:
    __aicore__ inline void Init(GM_ADDR xexp, GM_ADDR scales, GM_ADDR skip1,
                                const FusedTailTilingData *td, TPipe *pipe)
    {
        td_ = td;
        uint32_t h = td->h;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xexp));
        scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales));
        skip1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(skip1));
        if (td->use_eri != 0U) {
            eriGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(td->eri_va));
        }
        pipe->InitBuffer(xQueue_, 2, h * sizeof(T));
        pipe->InitBuffer(skip1Queue_, 1, h * sizeof(T));
        pipe->InitBuffer(scalesBuf_, FT_FINIDX_MAX_BYTES);
        pipe->InitBuffer(eriBuf_, FT_FINIDX_MAX_BYTES);
        pipe->InitBuffer(finAccBuf_, h * sizeof(float));
        finAccF_ = finAccBuf_.Get<float>();
        scalesF_ = scalesBuf_.Get<float>();
        eriI_ = eriBuf_.Get<int32_t>();
    }

    // Compute one row: finAccF_ = fp32(skip1[m]) (has_skip1) or 0; ascending k:
    // finAccF_ += scales[m,k]*fp32(row(m,k)). The caller then does a single
    // CAST_RINT to bf16. tmpF = row-granular fp32 temporary (caller-owned).
    __aicore__ inline void ComputeRow(uint32_t mIdx, LocalTensor<float> &tmpF)
    {
        uint32_t K = td_->k;
        uint32_t h = td_->h;
        DataCopyExtParams extRow{1U, static_cast<uint32_t>(h * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
        LoadIdxRow(mIdx);
        // skip1 co-issued with xexp k=0
        LocalTensor<T> s1;
        if (td_->has_skip1 != 0U) {
            s1 = skip1Queue_.AllocTensor<T>();
            DataCopyPad(s1, skip1Gm_[static_cast<uint64_t>(mIdx) * h], extRow, padParams);
            skip1Queue_.EnQue(s1);
        }
        IssueX(0U, mIdx, extRow, padParams);
        if (td_->has_skip1 != 0U) {
            s1 = skip1Queue_.DeQue<T>();
            Cast(finAccF_, s1, RoundMode::CAST_NONE, h);
            skip1Queue_.FreeTensor(s1);
        } else {
            Duplicate(finAccF_, 0.0f, h);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t kk = 0; kk < K; ++kk) {
            if (kk + 1U < K) {  // prefetch next row, overlapping the vector compute below
                IssueX(kk + 1U, mIdx, extRow, padParams);
            }
            float s = scalesF_(kk);
            LocalTensor<T> xr = xQueue_.DeQue<T>();
            Cast(tmpF, xr, RoundMode::CAST_NONE, h);
            xQueue_.FreeTensor(xr);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(tmpF, tmpF, s, h);
            AscendC::PipeBarrier<PIPE_V>();
            Add(finAccF_, finAccF_, tmpF, h);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline LocalTensor<float> &Acc() { return finAccF_; }

private:
    // scales/eri per-row batch load: K*4B (K=8 -> 32B) in one DataCopyPad;
    // S_MTE2 = previous row's scalar read done, allow overwrite; MTE2_S = this
    // row's scalar read waits for the load.
    __aicore__ inline void LoadIdxRow(uint32_t mIdx)
    {
        uint32_t K = td_->k;
        DataCopyExtParams extIdx{1U, K * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> padF{false, 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> padI{false, 0U, 0U, 0U};
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(scalesF_, scalesGm_[static_cast<uint64_t>(mIdx) * K], extIdx, padF);
        if (td_->use_eri != 0U) {
            DataCopyPad(eriI_, eriGm_[static_cast<uint64_t>(mIdx) * K], extIdx, padI);
        }
        SyncFunc<HardEvent::MTE2_S>();
    }

    __aicore__ inline void IssueX(uint32_t kk, uint32_t mIdx,
                                  const DataCopyExtParams &extRow,
                                  const DataCopyPadExtParams<T> &padParams)
    {
        uint64_t srcRow = static_cast<uint64_t>(mIdx) * td_->k + kk;
        if (td_->use_eri != 0U) {  // eri indirect read: expanded row of xp
            srcRow = static_cast<uint64_t>(static_cast<int64_t>(eriI_(kk)));
        }
        LocalTensor<T> xb = xQueue_.AllocTensor<T>();
        DataCopyPad(xb, xGm_[srcRow * td_->h], extRow, padParams);
        xQueue_.EnQue(xb);
    }

    const FusedTailTilingData *td_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> scalesGm_;
    GlobalTensor<T> skip1Gm_;
    GlobalTensor<int32_t> eriGm_;
    TQue<TPosition::VECIN, 2> xQueue_;  // double buffer
    TQue<TPosition::VECIN, 1> skip1Queue_;
    TBuf<> scalesBuf_;
    TBuf<> eriBuf_;
    TBuf<> finAccBuf_;
    LocalTensor<float> finAccF_;
    LocalTensor<float> scalesF_;
    LocalTensor<int32_t> eriI_;
};

}  // namespace fused_tail
}  // namespace npu_kernel
}  // namespace sglang

#endif  // FUSED_TAIL_KERNEL_LIB_H
