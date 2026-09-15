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
 * \brief fused_tail (tp_ascendc_fusion_v4.1 production op) device-kernel
 *        shared implementation — all common machinery of the
 *        fin(+skip1)+AR+add+gemma rmsnorm layer-tail fusion.
 *
 * Derived from the evaluation probe tp_oneshot_ar_probe's oar_kernel_lib.h
 * (production-trimmed P5/P8 version; mechanism research and all root causes
 * are recorded in its header comment and the package README §9); the
 * mechanism copies deepep production idioms:
 *   - Cross-card data plane: GM->UB->MTE3 DataCopyPad->peer window GM (DMA
 *     reads/writes HBM directly, cross-card visibility is intrinsic, no dcci
 *     anywhere on the data/flag path).
 *   - Flag protocol [pinned v2]: phase-A has no self item (spin target R-1),
 *     remote flag items are written twice, consumed lines are cleared by pure
 *     MTE3 (deepep combine :1001-1002 semantics, no dcci — a dcci clean would
 *     race with the peer's in-flight flag writes); item stride pinned to 32B.
 *     The probe's v1/v3 arms do not ship to production.
 *   - Data-before-flag program order: all push-phase data is issued on MTE3,
 *     then PipeBarrier<PIPE_ALL>, then flags (same as deepep dispatch :937).
 *   - Reduction: fp32 accumulation + fixed rank order 0..R-1 (r==rank uses
 *     the UB local contribution) + a single final CAST_RINT to bf16 => two
 *     runs are bit-identical (RL semantics).
 *   - Timeout: GetSystemCycle + CYCLES_PER_US=50, two checkpoints (1x limit
 *     records forensics, 2x limit writes DFX then exits without hanging;
 *     wall-clock bound 2x cycle_limit).
 *   - Scalar paths (address table/counters/DFX): gm_store/gm_load/gm_dcci
 *     (deepep hccl_shmem.hpp:32-55 verbatim); the cell area lives in plain
 *     GM (lesson, proven in r7: 32B cells inside symmem share a 128B line
 *     and concurrent store+dcci loses words), 128B/cell one full line per
 *     core + BumpRing read-back-verify retry after writing.
 *   - SyncFunc: FetchEventID + SetFlag/WaitFlag paired and consumed
 *     immediately; the kernel exits with no leftover SetFlag (structural
 *     immunity to the tp_fusion_v2 hang lesson).
 *   - Push forward-edge discipline: a V-written source gets an explicit
 *     SyncFunc<V_MTE3> (P5 all-green precedent); an MTE2 source requires
 *     SyncFunc<MTE2_MTE3> (proven in r8).
 *
 * Buffer layout (all VAs arrive via the addr_tab int64 tensor and are read
 * in-kernel as scalars with gm_load => graph-capture safe, addresses are not
 * baked into tiling):
 *
 *   data symmem (one block per rank, byte offsets; no P3 out area —
 *   two-shot does not ship to production):
 *     slot(g, r)  = dataVA + g*ring_stride + r*slot_stride   g in [0,RING=4) gens
 *       tile t    = slot(g,r) + t*tile_bytes
 *
 *   flag symmem (one block per rank, byte offsets; only the phase-A segment
 *   = RING gens):
 *     line(g,t)   = flagVA + g*flag_ring_stride + t*world*32
 *       item rank i = line + i*32 (32B = 8 fp32, [0]=1.0f marks arrival)
 *
 *   cell area (plain GM tensor, VA via addr_tab[17]):
 *     [0,48*128)        = counter cells (128B exclusive line per core;
 *                         slot[0]=counter, slot[1]=build stamp
 *                         FUSED_TAIL_BUILD_REV, slot[8]=readback-exhausted mark)
 *     [48*128,+48*128)  = DFX cells (per-core timeout/late forensics)
 *
 * Kernel organization: combine-style with zero intra-kernel cross-core
 * sync — each core runs an independent pipeline, no SyncAll => no dependency
 * on schedule mode. tile = 8KB bf16.
 *
 * Build discipline: one kernel per file +
 * KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY) (multiple kernels in one
 * file + explicit task type: only the first registers successfully, proven
 * in two rounds); --cce-auto-sync=on (matches the probe validation baseline;
 * all sync is handwritten and correctness does not rely on it).
 *
 * Kernel entry signature convention (auto_gen launch wrapper "last two
 * params = workspace/tiling"): the workspace slot carries a dummy parameter
 * position — probes proved that this slot's argument is deterministically
 * corrupted on arrival at the kernel (the aux +out_offset unresolved case,
 * mechanism not located), so no valid parameter is ever placed there.
 *
 * [v4.1] eri / norm w are now passed as proper GM_ADDR formal parameters
 * (before the workspace slot), no longer baked into tiling — the tiling hash
 * degenerates to shape level (40 layers share 1 entry), eliminating v4's
 * per-layer in-graph tiling H2D+D2D copy pair (~3us/layer); the addresses
 * are stable for the graph's lifetime (eri is a graph-pool static buffer,
 * norm w a parameter), same safety level as existing tensor parameters like
 * x/scales. scales supports direct bf16 feeding (tiling.scales_bf16): the
 * scalar path widens exactly via bit <<16, the vector path via UB
 * Cast(CAST_NONE), both bit-equivalent to a host-side .float().
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
constexpr uint64_t FT_CYCLES_PER_US = 50UL;   // GetSystemCycle -> us (same value as deepep)
constexpr uint32_t FT_MAX_CORES = 48U;        // AIV core count limit (cell area capacity)
constexpr uint32_t FT_MAX_RANK = 8U;          // R limit (addr_tab capacity)
constexpr uint32_t FT_FLAG_LINE_BYTES = FT_MAX_RANK * FT_FLAG_ITEM_BYTES;  // 256B
constexpr int32_t FT_DFX_MAGIC = 0xDEAD0000;  // timeout cell magic (| core low 16 bits)
constexpr uint32_t FT_CELL_BYTES = 128U;      // cell padded to 128B, one full line per core (r7 lesson)
// Build stamp: BumpRing writes counter cell slot[1] on every launch; the
// wrapper reads it back at init for comparison (the "stale binary illusion"
// has burned multiple rounds of machine time; deployment self-proof).
// Bump this value on any kernel change.
constexpr int32_t FUSED_TAIL_BUILD_REV = 20260924;  // v4.1 (eri/norm_w parameterization + bf16 scales)
// Explicit scratch for Sum's multi-level reduction (the 3-arg overload takes
// a temporary stack buffer via PopStackBuffer internally, which risks UB;
// the 4-arg form is safer, hence kept); requirement = ceil(ceil(n/64)/8)*32B,
// 256B for h=4096
constexpr uint32_t FT_SUM_TMP_BYTES = 512U;
// fin front stage eri row staging limit (K<=16 x 4B = 64B)
constexpr uint32_t FT_FINIDX_MAX_BYTES = 64U;
// fin front stage scales row element limit (K<=16; the UB Cast widening of
// bf16 scales is done on the full 16 elements)
constexpr uint32_t FT_FINIDX_MAX_ELEMS = FT_FINIDX_MAX_BYTES / sizeof(float);  // 16

// ---------------------------------------------------------------------------
// Scalar primitives: gm_store/gm_load/gm_dcci (deepep hccl_shmem.hpp:32-55
// verbatim). Used only for address-table/counter/DFX scalar paths; cross-card
// data and flags travel via DMA (MTE2/MTE3) without dcci.
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
// SyncFunc: same as deepep moe_distribute_v2_base.h:36-41. Set/Wait paired
// and consumed immediately, no residue across calls (structural immunity to
// the tp_fusion_v2 "kernel exits with an unconsumed SetFlag" hang lesson).
// ---------------------------------------------------------------------------
template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// ---------------------------------------------------------------------------
// TailArBase: common pieces for tiling/address table/ring counter/flag
// read-write/spin/clear/DFX (the spin AR kernel = fused_fin_ar_norm only;
// fused_fin_add/fused_tail_zero do not touch this).
// ---------------------------------------------------------------------------
class TailArBase {
public:
    __aicore__ inline void InitBase(GM_ADDR addr_tab, const FusedTailTilingData *td, TPipe *pipe)
    {
        td_ = td;
        core_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        tab_ = reinterpret_cast<__gm__ int64_t *>(addr_tab);
        flagSelfVA_ = FlagVA(td_->rank);

        // flag staging: [0]=1.0f, rest 0 (the spin side's Sum target counts
        // items, each item contributes 1.0)
        pipe->InitBuffer(flagStgBuf_, FT_FLAG_ITEM_BYTES);
        pipe->InitBuffer(flagLineBuf_, FT_FLAG_LINE_BYTES);
        pipe->InitBuffer(flagSumBuf_, FT_FLAG_ITEM_BYTES);
        pipe->InitBuffer(flagZeroBuf_, FT_FLAG_LINE_BYTES);
        pipe->InitBuffer(spinSumTmpBuf_, FT_SUM_TMP_BYTES);  // spin Sum explicit scratch
        flagStg_ = flagStgBuf_.Get<float>();
        flagLine_ = flagLineBuf_.Get<float>();
        flagSum_ = flagSumBuf_.Get<float>();
        flagZero_ = flagZeroBuf_.Get<float>();
        spinSumTmp_ = spinSumTmpBuf_.Get<uint8_t>();
        Duplicate(flagStg_, 0.0f, FT_FLOATS_PER_FLAG);
        Duplicate(flagZero_, 0.0f, FT_FLAG_LINE_BYTES / sizeof(float));
        SyncFunc<HardEvent::V_S>();
        flagStg_.SetValue(0, 1.0f);
        // SetValue is a scalar (S) write and is later used as an MTE3
        // DataCopy source: S_MTE3 sync is guaranteed inside WriteFlagItem
    }

    // ---- address table: addr_tab[0..7]=data VA, [8..15]=flag VA; r==rank is the local VA ----
    __aicore__ inline uint64_t DataVA(uint32_t r) const
    {
        return static_cast<uint64_t>(gm_load(tab_ + r));
    }

    __aicore__ inline uint64_t FlagVA(uint32_t r) const
    {
        return static_cast<uint64_t>(gm_load(tab_ + 8U + r));
    }

    // ---- cell area base (plain GM tensor, VA via addr_tab[17]; capture-safe
    //      — contents read at runtime with gm_load, same channel as [0..15]) ----
    __aicore__ inline uint64_t CellVA() const
    {
        return static_cast<uint64_t>(gm_load(tab_ + 17U));
    }

    // ---- ring counter: per-core private cell (128B exclusive line); must be
    //      called on every launch (including early-exit cores) to keep lockstep.
    //      slot[0]=counter; slot[1]=build stamp; slot[8]=readback-exhausted mark.
    //      Read-back-verify with retry after writing (<=4 attempts; r7 proved
    //      residual single-point loss on small cell writes, retry self-heals;
    //      a strictly +1 counter sequence is a correctness precondition of
    //      the ring protocol) ----
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
        if (att >= 4U) {  // readback exhausted (rare): leave evidence in slot[8], continue with local v (protocol-consistent)
            gm_store(cell + 8, FT_DFX_MAGIC | static_cast<int32_t>(0x0BADU));
            gm_dcci(reinterpret_cast<__gm__ uint8_t *>(cell));
        }
        launchIdx_ = static_cast<uint32_t>(v);
        return (static_cast<uint32_t>(v)) & (FT_RING - 1U);  // this launch uses the generation before increment
    }

    // ---- flag line address (local flag area; item stride pinned to 32B) ----
    __aicore__ inline uint64_t FlagLineAddr(uint64_t flagVA, uint32_t phase, uint32_t ring, uint32_t tile) const
    {
        return flagVA + static_cast<uint64_t>(phase) * td_->flag_phase_stride +
               static_cast<uint64_t>(ring) * td_->flag_ring_stride +
               static_cast<uint64_t>(tile) * td_->world * FT_FLAG_ITEM_BYTES;
    }

    // ---- write a flag item (32B DMA, [0]=1.0f); the target card's flagVA is
    //      given by the caller (peer = push acknowledgment) ----
    __aicore__ inline void WriteFlagItem(uint64_t flagVA, uint32_t phase, uint32_t ring, uint32_t tile, uint32_t item)
    {
        GlobalTensor<float> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(FlagLineAddr(flagVA, phase, ring, tile) +
                                                            item * FT_FLAG_ITEM_BYTES));
        // flagStg_'s content was written by scalar init; S_MTE3 guarantees
        // MTE3 reads the latest value
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy<float>(dst, flagStg_, FT_FLOATS_PER_FLAG);
    }

    // ---- phase-A spin target item count: v2 protocol = R-1 (no self item) ----
    __aicore__ inline float SpinTargetA() const
    {
        return static_cast<float>(static_cast<int32_t>(td_->world - 1U));
    }

    // ---- clear the local flag line after consumption (same pure-MTE3
    //      semantics as deepep combine :1001-1002, no dcci — a dcci clean
    //      would write the local DCache's stale line back to HBM as a whole,
    //      racing with the peer's in-flight flag writes) ----
    __aicore__ inline void ClearFlagLine(uint32_t phase, uint32_t ring, uint32_t tile)
    {
        GlobalTensor<float> dst;
        uint64_t lineAddr = FlagLineAddr(flagSelfVA_, phase, ring, tile);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lineAddr));
        uint32_t lineFloats = td_->world * (FT_FLAG_ITEM_BYTES / sizeof(float));
        SyncFunc<HardEvent::V_MTE3>();  // flagZero_ is written by V(Duplicate)
        DataCopy<float>(dst, flagZero_, lineFloats);
    }

    // ---- flag line arrival bitmask: bit i = item i's [0] >= 0.5f (for DFX
    //      forensics; at the call site flagLine_ must hold valid line
    //      contents (MTE2_V + V_S already synced)) ----
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

    // ---- spin on the local flag line: DataCopy the line into UB + Sum ==
    //      target±0.5 (formula-for-formula same as deepep combine
    //      WaitDispatch :952-1003; dst/src kept separate to avoid Sum
    //      source/destination overlap).
    //      Two-checkpoint forensics: at 1x cycle_limit record (s1,m1) and
    //      keep waiting; only at 2x is it judged a timeout — distinguishes
    //      "late" (peer write landing/visibility delayed) from "truly lost"
    //      (never arrives); a line that completes within (limit, 2x limit)
    //      writes DFX with phase|0x100 as the late mark.
    //      Returns true = timed out (DFX cell written; the caller must exit
    //      immediately and stop spinning). ----
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
            // Late success: completed within (limit, 2x limit) — write DFX
            // with the late mark as evidence (normal return)
            WriteDfx(tile, phase | 0x100U, localState, ArriveMask(), sum1, mask1, 1U);
        }
        return false;
    }

    // ---- timeout/late DFX: slot[0]=magic|core low 16 bits, [1]=tile |
    //      launch index<<16, [2]=phase (|0x100=late mark), [3]=Sum at exit,
    //      [4]=arrival mask at exit, [5]=Sum at 1x limit, [6]=mask at 1x
    //      limit, [7]=marked.
    //      Interpretation: s~0 = all lost / s~target-1 = missing item (m
    //      identifies which) / s>target = foreign write mixed in;
    //      the s1->s evolution = late vs truly lost. ----
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

    // ---- slot address (target card's dataVA; tile granularity) ----
    __aicore__ inline uint64_t SlotAddr(uint64_t dataVA, uint32_t ring, uint32_t slot, uint32_t tile) const
    {
        return dataVA + static_cast<uint64_t>(ring) * td_->ring_stride +
               static_cast<uint64_t>(slot) * td_->slot_stride +
               static_cast<uint64_t>(tile) * td_->tile_bytes;
    }

    // ---- push one tile (bf16 data already in UB) to all peer
    //      slot[rank][tile].
    //      The caller must guarantee srcUb's last write is synced with MTE3:
    //      a V source requires SyncFunc<V_MTE3> (the P5 finB_ precedent), an
    //      MTE2 source requires SyncFunc<MTE2_MTE3>. ----
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

    // ---- send flags: PipeBarrier<PIPE_ALL> (flags sent only after all
    //      push-phase data has been issued on MTE3, same discipline as deepep
    //      dispatch SetStatus :937) -> write peer flags (v2 protocol: no self
    //      item, spin target R-1; remote items written twice — single-write
    //      loss probability p => double-write ~p^2, the discrimination/
    //      mitigation lever for probabilistic write loss, validated probe-wide). ----
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

    // ---- reduce one tile: fixed rank order 0..R-1; r==rank uses the local
    //      contribution in UB, the rest read local slot[r][tile]; fp32
    //      accumulation + single CAST_RINT (same as deepep combine).
    //      Result written to arUb (bf16 tile, = AR output point). ----
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
        pipe->InitBuffer(accBuf_, tileBytes * 2U);  // fp32 accumulator (bf16->fp32 doubles the bytes)
        pipe->InitBuffer(tmpBuf_, tileBytes * 2U);  // cast temporary
        accF_ = accBuf_.Get<float>();
        tmpF_ = tmpBuf_.Get<float>();
    }

    const FusedTailTilingData *td_;
    uint32_t core_;
    __gm__ int64_t *tab_;
    uint64_t flagSelfVA_;
    uint32_t launchIdx_ = 0U;  // BumpRing pre-increment count = this process's launch index (DFX forensics)
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
// TailArNormBase (fused_fin_ar_norm tail): TailArBase + "AR result bf16 row ->
// +residual -> add_out -> gemma rmsnorm -> norm_out". The math replicates A3a
// point by point (full_attention_fusion_npu.py:65-85; production stock
// add_gemma_rms_norm uses the same formula):
//   add  = bf16(ar) + bf16(res): widen to fp32 Add -> CAST_RINT to bf16, write add_out
//   xf   = fp32(add_out); var = Sum(xf^2)/H; out = xf*rsqrt(var+eps)*(w_fp32+1)
//        -> CAST_RINT to norm_out ((w+1) is the gemma convention)
// The only difference from stock = the cross-rank summation order (the
// reduction stage uses fixed rank order) => not bitwise, same error band.
// ---------------------------------------------------------------------------
class TailArNormBase : public TailArBase {
public:
    // res/norm w/add_out/norm_out GM + tail buffer init; w preloaded as fp32 (w+1)
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
        pipe->InitBuffer(addFBuf_, rowBytes * 2U);        // add/xf fp32 (row granular)
        pipe->InitBuffer(sqBuf_, rowBytes * 2U);          // xf^2
        pipe->InitBuffer(wQueue_, 1, rowBytes);
        pipe->InitBuffer(wFBuf_, rowBytes * 2U);          // (w+1) fp32
        pipe->InitBuffer(outQueue_, 1, rowBytes);
        pipe->InitBuffer(varBuf_, FT_FLAG_ITEM_BYTES);
        pipe->InitBuffer(sumTmpBuf_, FT_SUM_TMP_BYTES);  // Sum explicit scratch
        arT_ = arBuf_.Get<T>();
        addF_ = addFBuf_.Get<float>();
        sqF_ = sqBuf_.Get<float>();
        wF_ = wFBuf_.Get<float>();
        varUb_ = varBuf_.Get<float>();
        sumTmp_ = sumTmpBuf_.Get<uint8_t>();
        invH_ = 1.0f / static_cast<float>(static_cast<int32_t>(h));  // aicore forbids direct uint->float conversion, go through int32

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

    // AR result bf16 row (in UB, a row segment of arT_) -> written out as add_out/norm_out rows
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
        Cast(tmpF_, arRowUb, RoundMode::CAST_NONE, h);   // ar row widened to fp32 (tmpF_ reused)
        AscendC::PipeBarrier<PIPE_V>();
        Cast(addF_, rt, RoundMode::CAST_NONE, h);
        resQueue_.FreeTensor(rt);
        AscendC::PipeBarrier<PIPE_V>();
        Add(addF_, addF_, tmpF_, h);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<T> ao = outQueue_.AllocTensor<T>();
        Cast(ao, addF_, RoundMode::CAST_RINT, h);        // add_out (= stock add node point)
        AscendC::PipeBarrier<PIPE_V>();
        Cast(addF_, ao, RoundMode::CAST_NONE, h);        // xf = fp32(add_out) (same point as A3a)
        outQueue_.EnQue(ao);
        ao = outQueue_.DeQue<T>();
        DataCopyPad(addOutGm_[rowOff], ao, extRow);
        outQueue_.FreeTensor(ao);
        AscendC::PipeBarrier<PIPE_V>();
        // var = Sum(xf^2)/h + eps -> rsqrt (4-arg overload + explicit scratch,
        // see the AddNormRow header note)
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
// TailFinDbFront (for fused_fin_add): double-buffered version of the
// finalize(+skip1) front stage (a sub 0/2 trim of probe FinDbFront,
// **numerically bit-identical to the P5 serial front stage** — per-row
// ascending k, same Cast/Muls/Add sequence and single CAST_RINT-to-bf16
// point):
//   (1) xexp queue depth 2 + intra-row prefetch pipeline (kk+1's DataCopyPad
//       overlaps kk's V compute);
//   (2) scales/eri batch-loaded into UB with one 32B DataCopyPad per row
//       (S_MTE2 releases overwrite / MTE2_S waits for the load, same event-pair
//       discipline as deepep cam_moe_*);
//   (3) skip1 and xexp k=0 loads are co-issued.
// use_eri=1: x=xp **unpermuted** [M*K,H], row address = eri[m*K+k]*h (eri
// int32 [M*K], [v4.1] passed via tensor parameter; production =
// AscendTPDispatchOutput.expanded_row_idx).
// [v4.1] scales supports bf16 (tiling.scales_bf16=1): the row load goes
// through bf16 staging + UB Cast(CAST_NONE) exact widening to fp32
// (bit-equivalent to host .float()); the event pair changes from MTE2_S to
// the MTE2_V->Cast->V_S chain.
// eventID budget (r24fix lesson: each TQue buffer slot occupies one MTE2_V
// plus one V_MTE2 eventID, pool = 8/class): this class's VECIN slots =
// xQueue 2 + skip1Queue 1 = 3, ample headroom.
// ---------------------------------------------------------------------------
class TailFinDbFront {
public:
    __aicore__ inline void Init(GM_ADDR xexp, GM_ADDR scales, GM_ADDR skip1, GM_ADDR eri,
                                const FusedTailTilingData *td, TPipe *pipe)
    {
        td_ = td;
        uint32_t h = td->h;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xexp));
        if (td->scales_bf16 != 0U) {
            scalesGmB_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(scales));
        } else {
            scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales));
        }
        skip1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(skip1));
        if (td->use_eri != 0U) {
            eriGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(eri));
        }
        pipe->InitBuffer(xQueue_, 2, h * sizeof(T));
        pipe->InitBuffer(skip1Queue_, 1, h * sizeof(T));
        pipe->InitBuffer(scalesBuf_, FT_FINIDX_MAX_BYTES);
        pipe->InitBuffer(scalesBBuf_, FT_FINIDX_MAX_BYTES);
        pipe->InitBuffer(eriBuf_, FT_FINIDX_MAX_BYTES);
        pipe->InitBuffer(finAccBuf_, h * sizeof(float));
        finAccF_ = finAccBuf_.Get<float>();
        scalesF_ = scalesBuf_.Get<float>();
        scalesB_ = scalesBBuf_.Get<T>();
        eriI_ = eriBuf_.Get<int32_t>();
    }

    // Compute one row: finAccF_ = fp32(skip1[m]) (has_skip1) or 0; ascending k
    // finAccF_ += scales[m,k]*fp32(row(m,k)). The caller then does a single
    // CAST_RINT to bf16.
    // tmpF = row-granular fp32 temporary (caller's own buffer).
    __aicore__ inline void ComputeRow(uint32_t mIdx, LocalTensor<float> &tmpF)
    {
        uint32_t K = td_->k;
        uint32_t h = td_->h;
        DataCopyExtParams extRow{1U, static_cast<uint32_t>(h * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
        LoadIdxRow(mIdx);
        // skip1 and xexp k=0 co-issue
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
            if (kk + 1U < K) {  // prefetch the next row, overlapping the V compute below
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
    // scales/eri row batch load: with fp32, one DataCopyPad of K*4B (K=8 ->
    // 32B); S_MTE2 = previous row's S reads done, overwrite allowed; MTE2_S =
    // this row's S reads wait for the load.
    // [v4.1] bf16 scales: load K*2B into bf16 staging -> MTE2_V -> Cast
    // widening (exact, CAST_NONE) -> V_S (scalar reads wait for the Cast);
    // K<=16 casts the full count in one go (the buffer is sized for 16
    // elements; [K,16) is unused garbage).
    __aicore__ inline void LoadIdxRow(uint32_t mIdx)
    {
        uint32_t K = td_->k;
        SyncFunc<HardEvent::S_MTE2>();
        if (td_->scales_bf16 != 0U) {
            DataCopyExtParams extIdxB{1U, K * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            DataCopyExtParams extIdxI{1U, K * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<T> padB{false, 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> padI{false, 0U, 0U, 0U};
            DataCopyPad(scalesB_, scalesGmB_[static_cast<uint64_t>(mIdx) * K], extIdxB, padB);
            if (td_->use_eri != 0U) {
                DataCopyPad(eriI_, eriGm_[static_cast<uint64_t>(mIdx) * K], extIdxI, padI);
            }
            SyncFunc<HardEvent::MTE2_V>();
            // S_V: the previous row's scalar reads of scalesF_ must finish
            // before this row's Cast overwrites it (the bf16 path's new S->V
            // forward edge; on the fp32 path the overwriter is MTE2, guarded
            // by the row-head S_MTE2)
            SyncFunc<HardEvent::S_V>();
            Cast(scalesF_, scalesB_, RoundMode::CAST_NONE, FT_FINIDX_MAX_ELEMS);
            SyncFunc<HardEvent::V_S>();
            return;
        }
        DataCopyExtParams extIdx{1U, K * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> padF{false, 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> padI{false, 0U, 0U, 0U};
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
        if (td_->use_eri != 0U) {  // eri indirect read: expanded row id into xp (production eri semantics)
            srcRow = static_cast<uint64_t>(static_cast<int64_t>(eriI_(kk)));
        }
        LocalTensor<T> xb = xQueue_.AllocTensor<T>();
        DataCopyPad(xb, xGm_[srcRow * td_->h], extRow, padParams);
        xQueue_.EnQue(xb);
    }

    const FusedTailTilingData *td_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> scalesGm_;
    GlobalTensor<T> scalesGmB_;  // [v4.1] bf16 scales view (enabled when scales_bf16=1)
    GlobalTensor<T> skip1Gm_;
    GlobalTensor<int32_t> eriGm_;
    TQue<TPosition::VECIN, 2> xQueue_;  // double buffered
    TQue<TPosition::VECIN, 1> skip1Queue_;
    TBuf<> scalesBuf_;
    TBuf<> scalesBBuf_;  // [v4.1] bf16 scales row staging (Cast widening source)
    TBuf<> eriBuf_;
    TBuf<> finAccBuf_;
    LocalTensor<float> finAccF_;
    LocalTensor<float> scalesF_;
    LocalTensor<T> scalesB_;
    LocalTensor<int32_t> eriI_;
};

}  // namespace fused_tail
}  // namespace npu_kernel
}  // namespace sglang

#endif  // FUSED_TAIL_KERNEL_LIB_H
