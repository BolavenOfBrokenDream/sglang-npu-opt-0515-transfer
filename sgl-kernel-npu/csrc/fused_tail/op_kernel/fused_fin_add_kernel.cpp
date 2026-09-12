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
 * \file fused_fin_add_kernel.cpp
 * \brief fused_fin_add = finalize(+skip1) local front stage (first half of the
 *        "fin+add fusion + stock HCCL AR + A3a norm" variant; productionized
 *        from probe oar_p8_fin_add_db). One kernel per file
 *        (KERNEL_TYPE_AIV_ONLY).
 *
 * Duty: fp32-accumulate the local contribution rows (TailFinDbFront double
 * buffering, numerically bit-identical to fused_fin_ar_norm's serial front
 * stage) -> single CAST_RINT to bf16 (stock finalize output point) -> write
 * out [M,H]. No AR, no norm, no symmem — the caller chains
 * tensor_model_parallel_all_reduce(out) + stock A3a norm to form the
 * "same fin+add fusion, AR over stock HCCL" variant (better than spin for
 * M>=64 per probe r24).
 * Row-granular core split (mIdx = core_, core_+C, ...); no tile/flag/ring
 * concept, no cross-core sync.
 *
 * use_eri=1 (production): x=xp unpermuted, row address = eri[m*K+k]*h (eri VA
 * via tiling.eri_va); use_eri=0: x already laid out (UT reference arm).
 *
 * UB budget (per core, worst H4096: row 8KB): finAccF 16 + xQueue 2x8=16
 *   + skip1Queue 8 + tmpF 16 + outQueue 8 + scales/eri 0.125 = ~64KB << 170KB.
 * eventID budget (pool 8/class): VECIN slots = xQueue 2 + skip1Queue 1 = 3,
 * headroom 5.
 */

#include "fused_tail_kernel_lib.h"

namespace sglang {
namespace npu_kernel {
namespace fused_tail {

class FusedFinAdd {
public:
    __aicore__ inline void Init(GM_ADDR xexp, GM_ADDR scales, GM_ADDR skip1, GM_ADDR out,
                                const FusedTailTilingData *td, TPipe *pipe)
    {
        td_ = td;
        core_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t h = td->h;
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
        pipe->InitBuffer(tmpBuf_, h * sizeof(float));
        pipe->InitBuffer(outQueue_, 1, h * sizeof(T));
        tmpF_ = tmpBuf_.Get<float>();
        fin_.Init(xexp, scales, skip1, td, pipe);
    }

    __aicore__ inline void Process()
    {
        uint32_t h = td_->h;
        DataCopyExtParams extRow{1U, static_cast<uint32_t>(h * sizeof(T)), 0U, 0U, 0U};
        for (uint32_t mIdx = core_; mIdx < td_->m; mIdx += td_->ncores) {
            fin_.ComputeRow(mIdx, tmpF_);
            LocalTensor<T> ob = outQueue_.AllocTensor<T>();
            Cast(ob, fin_.Acc(), RoundMode::CAST_RINT, h);  // stock finalize output point
            outQueue_.EnQue(ob);
            ob = outQueue_.DeQue<T>();
            DataCopyPad(outGm_[static_cast<uint64_t>(mIdx) * h], ob, extRow);
            outQueue_.FreeTensor(ob);
        }
    }

private:
    const FusedTailTilingData *td_;
    uint32_t core_;
    TailFinDbFront fin_;
    GlobalTensor<T> outGm_;
    TBuf<> tmpBuf_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    LocalTensor<float> tmpF_;
};

}  // namespace fused_tail
}  // namespace npu_kernel
}  // namespace sglang

using sglang::npu_kernel::FusedTailTilingData;
using sglang::npu_kernel::fused_tail::FusedFinAdd;

extern "C" __global__ __aicore__ void fused_fin_add(GM_ADDR x, GM_ADDR scales, GM_ADDR skip1, GM_ADDR out,
                                                    GM_ADDR dummy_workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FusedTailTilingData tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);
    AscendC::TPipe pipe;
    FusedFinAdd op;
    // x=xp/xexp [M*K,H] bf16, scales [M,K] fp32, skip1 [M,H] (dummy when
    // has_skip1=0), out=local contribution [M,H] bf16; dummy_workspace
    // occupies auto_gen's workspace slot (that slot's argument is corrupted on
    // arrival — no valid parameter goes there).
    op.Init(x, scales, skip1, out, &tilingData, &pipe);
    op.Process();
}
