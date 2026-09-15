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
 * \file vgmm_main_kernel.cpp
 * \brief vgmm1_main_aic: vendored GMM main kernel (the "scan-skipping" body
 *        of plan B).
 *
 * Approach = take the stock GroupedMatmul GMM_FLOAT / ND / transB /
 * CUBE_ONLY path (ops-transformer-9.0.0 grouped_matmul.cpp:519-537 ->
 * GMM_CUBE_IMP -> GMMProcess + GMMCompute in grouped_matmul.h) verbatim,
 * with a single change: remove GMMProcess::Process's 256-group scan loop
 * (per-core O(E) scalar bookkeeping, measured 59ns x 256 ~= 15us) and
 * replace it with direct block-table reads: each core only loops over its
 * own blocks (b = coreIdx; b < totalBlocks; b += coreNum), with 7 scalar
 * table reads per block replacing the per-group
 * GetSplitValueFromGroupList / UpdateMnConfig / MNBlockIdxCompute.
 * The MM body (MatmulImpl MDL + IterateAll writing GM directly) is a
 * verbatim replica of stock GMMCompute::MMCompute's single-tensor / bf16 /
 * ND / transB / no-bias branch (grouped_matmul.h:518-562), including the
 * weight L2 bypass hint when blockDimM==1 (:508-512).
 *
 * Block-to-core dispatch identical to stock: global block ids are numbered
 * in group-major order, block b belongs to core b%coreNum (stock's
 * count % coreNum rolling semantics is exactly global block id modulo).
 *
 * One kernel per file (registration iron rule);
 * KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY) same as stock
 * grouped_matmul.cpp:362.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common_tiling_kernel.h"
#include "../op_host/vendored_gmm_tiling_data.h"

using namespace AscendC;
using namespace matmul;  // = stock grouped_matmul.cpp:41. CANN 9.0.0 headers
                         // have no AscendC::matmul nested namespace; matmul is
                         // a global alias from kernel_kfc.h:443
                         // (namespace matmul = AscendC;), MatmulImpl itself
                         // lives under AscendC

namespace sglang {
namespace npu_kernel {
namespace vgmm {

// = stock matmulCFGUnitFlag non-int8 branch (grouped_matmul_utils.h:166-167),
// positional initialization replicated verbatim (MatmulConfig field order is
// ABI of the same-version CANN header; semantics = MDL template +
// enUnitFlag). Do not switch to designated initializers without first
// checking the field order of the local CANN headers.
constexpr MatmulConfig VGMM_MM_CFG{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                   0, 0, 0, 0, true};

template <bool trans = false>
using XType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, trans>;
template <bool trans = false>
using WType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, trans>;
using YType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;     // MM_DTYPE_Y = DTYPE_Y
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;  // DTYPE_BIAS, unused with hasBias=false

// Same as grouped_matmul_utils.h:192
template <class AT_, class BT_, class CT_, class BiasT_, const auto &MM_CFG>
struct MMImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
};

}  // namespace vgmm
}  // namespace npu_kernel
}  // namespace sglang

extern "C" __global__ __aicore__ void vgmm1_main_aic(GM_ADDR x, GM_ADDR weight, GM_ADDR block_table,
                                                     GM_ADDR total_blocks, GM_ADDR y, GM_ADDR workspace,
                                                     GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if ASCEND_IS_AIV {
        return;  // same guard as stock GMM_CUBE_IMP
    }
    (void)workspace;
    AscendCUtils::SetOverflow(1);  // same as stock grouped_matmul.cpp:361

    sglang::npu_kernel::VgmmTilingData tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);  // stack-copy equivalent of GET_TILING_DATA_MEMBER

    // VGMM_STAGE staged power-down debugging (delivered by host via
    // baseParams.reserved[0], for localizing 507015):
    //   1=CopyTiling only; 2=+table reads (total_blocks + all entries of this
    //   core); 3=+mm.Init;
    //   4=+per-block addressing (SetOrgShape/SetSingleShape/SetTensorA/B) but
    //   skip IterateAll; 0=full
    //   (5=pure host-side y over-allocation guard, kernel unaware and runs
    //   full as usual, see vendored_gmm.cpp)
    const uint32_t dbgStage = tilingData.baseParams.reserved[0];
    if (dbgStage == 1) {
        return;
    }

    const uint32_t coreNum = tilingData.baseParams.coreNum;
    const uint32_t baseM = tilingData.baseParams.baseM;
    const uint32_t baseN = tilingData.baseParams.baseN;
    const uint64_t kDim = tilingData.baseParams.k;
    const uint64_t nDim = tilingData.baseParams.n;

    uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    int64_t coreRation = GetTaskRation();
    if (coreRation > 1) {
        coreIdx /= coreRation;  // same shape as stock GMMProcess::Init (= 1 under AIC_ONLY)
    }

    GlobalTensor<int32_t> tableGm;
    tableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_table));
    GlobalTensor<int32_t> tbGm;
    tbGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(total_blocks));
    const uint32_t totalBlocks = static_cast<uint32_t>(tbGm.GetValue(0));

    if (dbgStage == 2) {
        // Read table only, write no result: accumulate all of this core's
        // entries into workspace[0] to defeat optimization
        GlobalTensor<int32_t> wsGm;
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace));
        int32_t acc = static_cast<int32_t>(totalBlocks);
        for (uint32_t curBlock = coreIdx; curBlock < totalBlocks; curBlock += coreNum) {
            const uint32_t entBase = curBlock * sglang::npu_kernel::VGMM_BLOCK_ENTRY_INT32;
            for (uint32_t f = 0; f < 7; ++f) {
                acc += tableGm.GetValue(entBase + f);
            }
        }
        wsGm.SetValue(0, acc);
        return;
    }

    using matmulType = sglang::npu_kernel::vgmm::MMImplType<sglang::npu_kernel::vgmm::XType<false>,
                                                            sglang::npu_kernel::vgmm::WType<true>,
                                                            sglang::npu_kernel::vgmm::YType,
                                                            sglang::npu_kernel::vgmm::BiasType,
                                                            sglang::npu_kernel::vgmm::VGMM_MM_CFG>;
    TPipe tPipe;
    // Alignment safety: copy the packed member into a standalone stack object
    // before taking its address (same as tp_ascendc_fusion
    // fused_norm_qkv_proj_scatter_kernel.cpp:159, validated on-device)
    AscendC::tiling::TCubeTiling cubeTiling = tilingData.cubeTiling;
    matmulType::MT mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&cubeTiling, &tPipe);
    if (dbgStage == 3) {
        return;
    }

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> yGm;
    GlobalTensor<bfloat16_t> wGmLocal;

    AscendC::WaitPreTaskEnd();  // same position as stock GMMProcess::Process (grouped_matmul.h:259)
    for (uint32_t curBlock = coreIdx; curBlock < totalBlocks; curBlock += coreNum) {
        // Table entry 8x int32: {groupId, row0group, mIdx, nIdx, blockRows, curSingleN, groupRows, reserved}
        const uint32_t entBase = curBlock * sglang::npu_kernel::VGMM_BLOCK_ENTRY_INT32;
        const uint32_t groupIdx = static_cast<uint32_t>(tableGm.GetValue(entBase + 0));
        const uint32_t row0Group = static_cast<uint32_t>(tableGm.GetValue(entBase + 1));
        const uint32_t mIdx = static_cast<uint32_t>(tableGm.GetValue(entBase + 2));
        const uint32_t nIdx = static_cast<uint32_t>(tableGm.GetValue(entBase + 3));
        const uint32_t blockRows = static_cast<uint32_t>(tableGm.GetValue(entBase + 4));
        const uint32_t curSingleN = static_cast<uint32_t>(tableGm.GetValue(entBase + 5));
        const uint32_t groupRows = static_cast<uint32_t>(tableGm.GetValue(entBase + 6));

        // ---- stock GMMCompute::MMCompute replica (singleX=singleWeight=singleY=1,
        //      groupType=0 splitM, bf16 ND transB, no bias) ----
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x) + static_cast<uint64_t>(row0Group) * kDim);
        // SetWOffset: ND transB -> wOffset = tailN * k (tailN = nIdx * singleN)
        wGmLocal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weight) +
                                 static_cast<uint64_t>(groupIdx) * kDim * nDim +
                                 static_cast<uint64_t>(nIdx) * baseN * kDim);
        if (groupRows <= baseM) {  // blockDimM==1 (same L2 bypass as grouped_matmul.h:508-512)
            wGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y) + static_cast<uint64_t>(row0Group) * nDim);

        mm.SetOrgShape(groupRows, nDim, kDim);
        mm.SetSingleShape(blockRows, curSingleN, kDim);
        mm.SetTensorA(xGm[static_cast<uint64_t>(mIdx) * baseM * kDim], false);
        mm.SetTensorB(wGmLocal, true);
        if (dbgStage == 4) {
            continue;  // validate addressing/shape setup only; no DMA, no compute
        }
        // outOffset = mIdx * singleM * n + tailN (grouped_matmul.h:530)
        mm.template IterateAll<false>(
            yGm[static_cast<uint64_t>(mIdx) * baseM * nDim + static_cast<uint64_t>(nIdx) * baseN], 0);
    }
    AscendC::SetNextTaskStart();  // same position as stock (grouped_matmul.h:283)
}
