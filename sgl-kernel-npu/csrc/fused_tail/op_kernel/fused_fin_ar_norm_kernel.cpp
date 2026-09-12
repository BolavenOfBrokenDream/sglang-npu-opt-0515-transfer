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
 *        gemma rmsnorm single kernel (productionized from probe
 *        oar_p5_fin_ar_add_norm). One kernel per file (KERNEL_TYPE_AIV_ONLY).
 *
 * finalize front stage (per row m, same order and rounding points as stock
 * moe_finalize_routing bf16):
 *   acc fp32 = has_skip1 ? Cast(fp32, skip1[m]) : 0
 *   ascending k: acc += scales[m,k] * Cast(fp32, row(m,k))   (Muls scalar + Add)
 *   single CAST_RINT to bf16 = local contribution row (stock finalize output point)
 * Row addressing:
 *   use_eri=1 (production): row(m,k) = xp[eri[m*K+k]] — xp unpermuted, eri is
 *     the flat-slot -> expanded-row index produced by init routing; eri rows
 *     are batch-loaded into UB with one 32B DataCopyPad per row plus the
 *     S_MTE2/MTE2_S event pair.
 *   use_eri=0: row(m,k) = xexp[m*K+k] (already laid out; UT reference arm).
 * eri only changes where the row number comes from; the ascending-k
 * accumulation and all rounding points are unchanged, so use_eri=1 is
 * bit-identical to use_eri=0 when eri is the identity permutation.
 *
 * Then per tile: fin contribution tile (resident in UB) -> push+flag -> spin
 * -> fixed-rank-order reduction (r==rank uses the UB fin tile) -> CAST_RINT ar
 * -> row-granular AddNormRow -> consume+clear.
 *
 * norm w arrives via tiling.norm_w_va (each layer has a different VA, baked
 * into tiling at capture — parameter addresses never change for the graph's
 * lifetime).
 *
 * UB budget (per core, worst H4096: row 8KB, tile 8KB):
 *   finAccF 16 + finB 8 + xexpQueue 8 + skip1Queue 8 + slotQueue 8 + accF 16
 *   + tmpF 16 + arT 8 + resQueue 8 + addF 16 + sqF 16 + wQueue 8 + wF 16
 *   + outQueue 8 + varUb 0.03 + eri 0.06 + flag small bufs ~2.3 = ~155KB < 170KB.
 * eventID budget (pool 8/class): VECIN slots = xexp 1 + skip1 1 + slot 1
 *   + res 1 + w 1 = 5 (eri row staging is a TBuf, not a TQue), headroom 3.
 */

#include "fused_tail_kernel_lib.h"

namespace sglang {
namespace npu_kernel {
namespace fused_tail {

class FusedFinArNorm : public TailArNormBase {
public:
    __aicore__ inline void Init(GM_ADDR xexp, GM_ADDR scales, GM_ADDR skip1, GM_ADDR res,
                                GM_ADDR add_out, GM_ADDR norm_out, GM_ADDR addr_tab,
                                const FusedTailTilingData *td, TPipe *pipe)
    {
        InitBase(addr_tab, td, pipe);
        td_ = td;
        xexpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xexp));
        scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales));
        skip1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(skip1));
        if (td->use_eri != 0U) {
            eriGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(td->eri_va));
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
        // norm w via tiling.norm_w_va (per-launch scalar form of the probe's
        // addr_tab VA channel)
        InitNorm(res, reinterpret_cast<GM_ADDR>(td->norm_w_va), add_out, norm_out, pipe);
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
            // ---------------- finalize front stage: each row in the tile ----------------
            for (uint32_t i = 0; i < rowsPerTile; ++i) {
                uint32_t mIdx = t * rowsPerTile + i;
                if (td_->use_eri != 0U) {
                    // eri row batch load (S_MTE2: previous row's scalar read
                    // done, allow overwrite; MTE2_S: this row's scalar read
                    // waits for the load)
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
                    // scales scalar direct read (same inline GetValue idiom as
                    // stock finalize)
                    float s = scalesGm_.GetValue(static_cast<uint64_t>(mIdx) * K + kk);
                    uint64_t srcRow = static_cast<uint64_t>(mIdx) * K + kk;
                    if (td_->use_eri != 0U) {  // eri indirect read: expanded row of xp
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
                // single CAST_RINT = local contribution row (stock finalize point)
                Cast(finB_[i * h], finAccF_, RoundMode::CAST_RINT, h);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // ---------------- push + spin reduce + tail stage ----------------
            SyncFunc<HardEvent::V_MTE3>();  // finB_ was written by V(Cast), used as MTE3 push source
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
    GlobalTensor<T> xexpGm_;
    GlobalTensor<float> scalesGm_;
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
                                                        GM_ADDR dummy_workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FusedTailTilingData tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);
    AscendC::TPipe pipe;
    FusedFinArNorm op;
    // x=xp/xexp [M*K,H] bf16, scales [M,K] fp32, skip1 [M,H] (dummy when
    // has_skip1=0), residual [M,H], add_out/norm_out [M,H] bf16; addr_tab
    // int64[18] ([0..7] data VA, [8..15] flag VA, [17] cell area VA);
    // dummy_workspace occupies auto_gen's workspace slot (that slot's argument
    // is corrupted on arrival — no valid parameter goes there).
    op.Init(x, scales, skip1, residual, add_out, norm_out, addr_tab, &tilingData, &pipe);
    op.Process();
}
