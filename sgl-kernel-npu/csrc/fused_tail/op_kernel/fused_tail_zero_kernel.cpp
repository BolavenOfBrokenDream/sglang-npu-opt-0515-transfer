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
 * \file fused_tail_zero_kernel.cpp
 * \brief fused_tail_zero = in-kernel MTE3 zeroing of a local GM region
 *        (tp_ascendc_fusion_v4.1 production op; productionized port of probe
 *        oar_p1_bw sub=6/7 ZeroOwn). One kernel per file
 *        (KERNEL_TYPE_AIV_ONLY).
 *
 * Purpose: clearing the local flag symmem at spin-context init/reset.
 * **torch zero_() must NOT be used** — an AIV vector write leaves dirty L2
 * lines on symmem; peer MTE3 writes land in HBM without invalidating the
 * local L2, and a dirty zero line can be evicted and written back at a
 * random later time, possibly after the peer's flag has landed, erasing it
 * back to 0 (root cause of the probe ut3/ut4 first-round spin timeouts).
 * MTE3 writes HBM directly with zero L2 residue, plus a per-128B-line dcci
 * after writing (guarantees the zeros really land, no L2 residue — spin
 * MTE2 reads go around L2 to HBM).
 */

#include "fused_tail_kernel_lib.h"

namespace sglang {
namespace npu_kernel {
namespace fused_tail {

class FusedTailZero {
public:
    __aicore__ inline void Init(GM_ADDR x, const FusedTailTilingData *td, TPipe *pipe)
    {
        td_ = td;
        core_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        base_ = reinterpret_cast<uint64_t>(x);
        pipe->InitBuffer(zeroBuf_, td->tile_bytes);
        zeroUb_ = zeroBuf_.Get<float>();
        Duplicate(zeroUb_, 0.0f, td->tile_bytes / sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t total = td_->total_bytes;
        uint32_t tileBytes = td_->tile_bytes;
        SyncFunc<HardEvent::V_MTE3>();  // zeroUb_ is written by V(Duplicate), used as MTE3 source
        uint32_t nTiles = (total + tileBytes - 1U) / tileBytes;
        for (uint32_t t = core_; t < nTiles; t += td_->ncores) {
            uint32_t len = (t + 1U < nTiles) ? tileBytes : (total - t * tileBytes);
            DataCopyExtParams extParams{1U, len, 0U, 0U, 0U};  // len%32==0 guaranteed by host
            GlobalTensor<float> dst;
            dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(base_ + static_cast<uint64_t>(t) * tileBytes));
            DataCopyPad(dst, zeroUb_, extParams);
        }
        SyncFunc<HardEvent::MTE3_S>();  // dcci only after all clearing writes complete
        for (uint32_t t = core_; t < nTiles; t += td_->ncores) {
            uint32_t len = (t + 1U < nTiles) ? tileBytes : (total - t * tileBytes);
            uint64_t blk = base_ + static_cast<uint64_t>(t) * tileBytes;
            for (uint32_t off = 0; off < len; off += 128U) {
                gm_dcci(reinterpret_cast<__gm__ uint8_t *>(blk + off));
            }
        }
    }

private:
    const FusedTailTilingData *td_;
    uint32_t core_;
    uint64_t base_;
    TBuf<> zeroBuf_;
    LocalTensor<float> zeroUb_;
};

}  // namespace fused_tail
}  // namespace npu_kernel
}  // namespace sglang

using sglang::npu_kernel::FusedTailTilingData;
using sglang::npu_kernel::fused_tail::FusedTailZero;

extern "C" __global__ __aicore__ void fused_tail_zero(GM_ADDR x, GM_ADDR dummy_workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FusedTailTilingData tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);
    AscendC::TPipe pipe;
    FusedTailZero op;
    // x = GM region to clear (uint8-view tensor, local flag symmem);
    // total_bytes arrives via tiling (multiple of 32 guaranteed by host);
    // dummy_workspace occupies auto_gen's workspace slot (that slot's
    // argument may be corrupted on arrival at the kernel — unresolved case —
    // so no valid parameter goes into this slot).
    op.Init(x, &tilingData, &pipe);
    op.Process();
}
