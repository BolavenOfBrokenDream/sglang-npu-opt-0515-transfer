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
 * \file fused_fin_ar_norm_kernel.cpp
 * \brief fused_fin_ar_norm = finalize(+skip1) + spin AIV AR + residual add +
 *        gemma rmsnorm single kernel (tp_ascendc_fusion_v4.1 production op,
 *        35B MoE layer-tail chain x 40 layers; productionized port of probe
 *        oar_p5_fin_ar_add_norm). One kernel per file
 *        (KERNEL_TYPE_AIV_ONLY).
 *
 * finalize front stage (row by row m, same order and same rounding points as
 * stock moe_finalize_routing_v2 bf16 cuth):
 *   acc fp32 = has_skip1 ? Cast(fp32, skip1[m]) : 0
 *   ascending k: acc += scales[m,k] * Cast(fp32, row(m,k))   (Muls scalar + Add)
 *   single CAST_RINT to bf16 = local contribution row (stock finalize output point)
 * Row addressing (new in v4, implementing the probe README §11.4 wiring
 * recommendation route 1 — P5 serial front stage + FinDbFront's eri indirect
 * read transplant; the two mechanisms are orthogonal):
 *   use_eri=1 (production form): row(m,k) = xp[eri[m*K+k]] (xp unpermuted,
 *     production directly uses AscendTPDispatchOutput.expanded_row_idx, zero
 *     permutation, zero pre-arranging kernel; each eri row is batch-loaded
 *     into UB with one 32B DataCopyPad plus the S_MTE2/MTE2_S event pair,
 *     same discipline as FinDbFront::LoadIdxRow)
 *   use_eri=0: row(m,k) = xexp[m*K+k] (already laid out, UT reference arm)
 * eri only changes where the row number comes from; the ascending-k
 * accumulation and all rounding points are unchanged, so use_eri=1 is
 * bit-identical to use_eri=0 when eri is the identity permutation.
 *
 * Then interleaved per tile: fin contribution tile (UB resident) -> push+flag
 * -> spin -> fixed-rank-order reduction (r==rank uses the UB fin tile) ->
 * CAST_RINT ar -> row-granular AddNormRow -> consume+clear.
 *
 * [v4.1] norm w and eri are now passed as proper tensor parameters (v4's
 * tiling VA scalar scheme put VAs in the tiling hash, making the H2D+D2D
 * copy chain enter the graph per layer at ~3us/layer wasted; after
 * parameterization the tiling is shape-level, 40 layers share 1 entry).
 * scales supports direct bf16 feeding (tiling.scales_bf16, scalar path
 * widened exactly via bit <<16, bit-equivalent to host .float()).
 *
 * UB budget (per core, worst H4096: row 8KB, tile 8KB):
 *   finAccF 16 + finB 8 + xexpQueue 8 + skip1Queue 8 + slotQueue 8 + accF 16
 *   + tmpF 16 + arT 8 + resQueue 8 + addF 16 + sqF 16 + wQueue 8 + wF 16
 *   + outQueue 8 + varUb 0.03 + eri 0.06 + flag small buffers ~2.3 ~= 155KB < 170KB.
 * eventID budget (pool 8/class): VECIN slots = xexp 1 + skip1 1 + slot 1
 *   + res 1 + w 1 = 5 (the eri row staging is a TBuf, not a TQue, and takes
 *   no eventID), headroom 3.
 */

#include "fused_tail_kernel_lib.h"

namespace sglang {
namespace npu_kernel {
namespace fused_tail {

class FusedFinArNorm : public TailArNormBase {
public:
    __aicore__ inline void Init(GM_ADDR xexp, GM_ADDR scales, GM_ADDR skip1, GM_ADDR res,
                                GM_ADDR add_out, GM_ADDR norm_out, GM_ADDR addr_tab,
                                GM_ADDR eri, GM_ADDR norm_w,
                                const FusedTailTilingData *td, TPipe *pipe)
    {
        InitBase(addr_tab, td, pipe);
        td_ = td;
        xexpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xexp));
        if (td->scales_bf16 != 0U) {  // [v4.1] bf16 scales fed directly (scalar path widened via bit <<16)
            scalesGmB_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(scales));
        } else {
            scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales));
        }
        skip1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(skip1));
        if (td->use_eri != 0U) {
            eriGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(eri));
        }
        uint32_t h = td->h;
        pipe->InitBuffer(finAccBuf_, h * sizeof(float));
        pipe->InitBuffer(finBBuf_, td->tile_bytes);
        pipe->InitBuffer(xexpQueue_, 1, h * sizeof(T));
        pipe->InitBuffer(skip1Queue_, 1, h * sizeof(T));
        pipe->InitBuffer(eriBuf_, FT_FINIDX_MAX_BYTES);
        finAccF_ = finAccBuf_.Get<float>();
        finB_ = finBBuf_.Get<T>();
        eriI_ = eriBuf_.Get<int32_t>();
        InitReduceBufs(pipe, td->tile_bytes);
        // [v4.1] norm w travels as a tensor parameter (v4's tiling.norm_w_va
        // put a VA in the hash and the copy chain entered the graph per
        // layer; after parameterization it uses the same channel and same
        // safety level as existing parameters like x/scales)
        InitNorm(res, norm_w, add_out, norm_out, pipe);
    }

    __aicore__ inline void Process()
    {
        uint32_t ring = BumpRing();
        uint32_t C = td_->ncores;
        uint32_t K = td_->k;
        uint32_t nTiles = td_->n_tiles;
        uint32_t rowsPerTile = td_->rows_per_tile;
        uint32_t h = td_->h;
        DataCopyExtParams extRow{1U, static_cast<uint32_t>(h * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
        DataCopyExtParams extIdx{1U, K * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> padI{false, 0U, 0U, 0U};
        for (uint32_t t = core_; t < nTiles; t += C) {
            // ---------------- finalize front stage: every row in the tile ----------------
            for (uint32_t i = 0; i < rowsPerTile; ++i) {
                uint32_t mIdx = t * rowsPerTile + i;
                if (td_->use_eri != 0U) {
                    // eri row batch load (same event pair as
                    // FinDbFront::LoadIdxRow: S_MTE2 = previous row's S reads
                    // done, overwrite allowed / MTE2_S = this row's S reads
                    // wait for the load)
                    SyncFunc<HardEvent::S_MTE2>();
                    DataCopyPad(eriI_, eriGm_[static_cast<uint64_t>(mIdx) * K], extIdx, padI);
                    SyncFunc<HardEvent::MTE2_S>();
                }
                if (td_->has_skip1 != 0U) {
                    LocalTensor<T> s1 = skip1Queue_.AllocTensor<T>();
                    DataCopyPad(s1, skip1Gm_[static_cast<uint64_t>(mIdx) * h], extRow, padParams);
                    skip1Queue_.EnQue(s1);
                    s1 = skip1Queue_.DeQue<T>();
                    Cast(finAccF_, s1, RoundMode::CAST_NONE, h);
                    skip1Queue_.FreeTensor(s1);
                } else {
                    Duplicate(finAccF_, 0.0f, h);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t kk = 0; kk < K; ++kk) {
                    // scales read directly as scalar (same as stock
                    // finalize's inline GetValue); [v4.1] with bf16, read the
                    // int16 bits <<16 for exact widening (= host .float())
                    float s = LoadScaleScalar(static_cast<uint64_t>(mIdx) * K + kk);
                    uint64_t srcRow = static_cast<uint64_t>(mIdx) * K + kk;
                    if (td_->use_eri != 0U) {  // eri indirect read: expanded row id into xp
                        srcRow = static_cast<uint64_t>(static_cast<int64_t>(eriI_(kk)));
                    }
                    LocalTensor<T> xr = xexpQueue_.AllocTensor<T>();
                    DataCopyPad(xr, xexpGm_[srcRow * h], extRow, padParams);
                    xexpQueue_.EnQue(xr);
                    xr = xexpQueue_.DeQue<T>();
                    Cast(tmpF_, xr, RoundMode::CAST_NONE, h);
                    xexpQueue_.FreeTensor(xr);
                    AscendC::PipeBarrier<PIPE_V>();
                    Muls(tmpF_, tmpF_, s, h);
                    AscendC::PipeBarrier<PIPE_V>();
                    Add(finAccF_, finAccF_, tmpF_, h);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                // Single CAST_RINT = local contribution row (same order and
                // rounding points as stock finalize)
                Cast(finB_[i * h], finAccF_, RoundMode::CAST_RINT, h);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // ---------------- push + spin reduction + tail ----------------
            SyncFunc<HardEvent::V_MTE3>();  // finB_ is written by V(Cast), used as MTE3 push source
            PushTileData(finB_, ring, t);
            SendTileFlags(ring, t);
            if (SpinFlagLine(0U, ring, t, SpinTargetA())) {
                return;
            }
            ReduceTileUbContrib(finB_, ring, t, arT_);
            for (uint32_t i = 0; i < rowsPerTile; ++i) {
                AddNormRow(arT_[i * h], t * rowsPerTile + i);
            }
            ClearFlagLine(0U, ring, t);
        }
    }

private:
    // scales scalar read: direct fp32 read; with bf16, read the int16 bits
    // <<16 for exact widening (bf16->fp32 is a lossless bit expansion,
    // bit-equivalent to a host-side .float() — eliminates the wrapper cast
    // small-op).
    __aicore__ inline float LoadScaleScalar(uint64_t idx) const
    {
        if (td_->scales_bf16 != 0U) {
            union {
                uint32_t u;
                float f;
            } cvt;
            cvt.u = static_cast<uint32_t>(static_cast<uint16_t>(scalesGmB_.GetValue(idx))) << 16U;
            return cvt.f;
        }
        return scalesGm_.GetValue(idx);
    }

    GlobalTensor<T> xexpGm_;
    GlobalTensor<float> scalesGm_;
    GlobalTensor<int16_t> scalesGmB_;  // [v4.1] bf16 scales bit view (enabled when scales_bf16=1)
    GlobalTensor<T> skip1Gm_;
    GlobalTensor<int32_t> eriGm_;
    TBuf<> finAccBuf_;
    TBuf<> finBBuf_;
    TBuf<> eriBuf_;
    TQue<TPosition::VECIN, 1> xexpQueue_;
    TQue<TPosition::VECIN, 1> skip1Queue_;
    LocalTensor<float> finAccF_;
    LocalTensor<T> finB_;
    LocalTensor<int32_t> eriI_;
};

}  // namespace fused_tail
}  // namespace npu_kernel
}  // namespace sglang

using sglang::npu_kernel::FusedTailTilingData;
using sglang::npu_kernel::fused_tail::FusedFinArNorm;

extern "C" __global__ __aicore__ void fused_fin_ar_norm(GM_ADDR x, GM_ADDR scales, GM_ADDR skip1, GM_ADDR residual,
                                                        GM_ADDR add_out, GM_ADDR norm_out, GM_ADDR addr_tab,
                                                        GM_ADDR eri, GM_ADDR norm_w,
                                                        GM_ADDR dummy_workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FusedTailTilingData tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);
    AscendC::TPipe pipe;
    FusedFinArNorm op;
    // x=xp/xexp [M*K,H] bf16, scales [M,K] fp32 or bf16 (tiling.scales_bf16;
    // bf16 widened exactly in-kernel), skip1 [M,H] (dummy when has_skip1=0),
    // residual [M,H], add_out/norm_out [M,H] bf16; addr_tab int64[18]
    // ([0..7] data VA, [8..15] flag VA, [17] cell area VA); eri int32[M*K],
    // norm_w bf16[H] [v4.1] now proper parameters (located before the
    // workspace slot; with use_eri=0 eri is a dummy and not read);
    // dummy_workspace occupies auto_gen's workspace slot (that slot's
    // argument may be corrupted on arrival at the kernel — unresolved case —
    // so no valid parameter goes into this slot).
    op.Init(x, scales, skip1, residual, add_out, norm_out, addr_tab, eri, norm_w, &tilingData, &pipe);
    op.Process();
}
