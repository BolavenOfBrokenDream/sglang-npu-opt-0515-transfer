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
 * \file vendored_gmm_tiling_data.h
 * \brief Tiling data shared between host and kernel for the two vendored GMM
 *        ops (plan B: offsets/block-schedule-table front kernel + vendored
 *        GMM scan-skipping). Layout follows the
 *        lora/op_host/tiling/sgemmc_tiling_data.h and tp_ascendc_fusion
 *        fused_*_tiling_data.h convention: custom scalar fields first,
 *        TCubeTiling embedded at the tail (the front sched op has no matmul
 *        and does not embed it). sizeof() must stay divisible by 4 bytes
 *        (kernel_utils::CopyTiling copies by uint32).
 *
 * Block schedule table entry (block_table, 8x int32 = 32B per block,
 * naturally aligned for DataCopy):
 *   [0] groupId     expert id g
 *   [1] row0group   start row of this group in packed x/y (= exclusive cumsum
 *                   of counts, i.e. offsets)
 *   [2] mIdx        M-direction block index within the group
 *   [3] nIdx        N-direction block index within the group
 *   [4] blockRows   actual rows of this block (= stock MMCompute's curSingleM)
 *   [5] curSingleN  actual columns of this block (= stock MMCompute's curSingleN)
 *   [6] groupRows   total rows m_g of this group (= stock SetOrgShape's orgM
 *                   semantics)
 *   [7] reserved
 */

#ifndef VENDORED_GMM_TILING_DATA_H
#define VENDORED_GMM_TILING_DATA_H

#include <cstdint>

namespace AscendC {
namespace tiling {

struct TCubeTiling;

}  // namespace tiling
}  // namespace AscendC

namespace sglang {
namespace npu_kernel {

constexpr uint32_t VGMM_BLOCK_ENTRY_INT32 = 8;  // 8 int32 per table entry (32B)

#pragma pack(push, 1)
struct VgmmSchedTilingData {
    uint32_t groupNum;    // E (production 256)
    uint32_t baseM;       // matches the main op's mmTiling.baseM (same host function)
    uint32_t baseN;       // same as above
    uint32_t n;           // N
    uint32_t maxBlocks;   // block_table capacity (driver upper bound total_M + E)
    uint32_t reserved[3]; // pad to 32B
};

// C1 (sched absorbs the cumsum, promoted after the tp_ascendc_fusion_v3.2
// second-round decision): input changed from counts int64[E] to v22
// rank_hist's partials int32[partNum, partStride]; the kernel column-reduces
// counts in-kernel, then builds the table exactly like vgmm1_sched_pre.
struct VgmmSchedPartialTilingData {
    uint32_t groupNum;    // E (production 256; <= partStride)
    uint32_t partNum;     // partials row count NP (= M//bm, <= 63 when M<=512)
    uint32_t partStride;  // partials row stride PE (= pow2_ceil(E), production 256)
    uint32_t baseM;       // matches the main op's mmTiling.baseM (same host function)
    uint32_t baseN;       // same as above
    uint32_t n;           // N
    uint32_t maxBlocks;   // block_table capacity (driver upper bound total_M + E)
    uint32_t reserved[1]; // pad to 32B
};

struct VgmmBaseParams {
    uint32_t coreNum;                 // block dispatch stride (= AIC count, stock gmmBaseParams.coreNum)
    uint32_t baseM;                   // = mmTiling.baseM (redundant copy, kernel touches tiling once less)
    uint32_t baseN;                   // = mmTiling.baseN
    uint32_t k;                       // K
    uint32_t n;                       // N
    uint32_t isOutputDisableL2Cache;  // stock same-named field (production ND path = 0)
    uint32_t reserved[2];             // [0]=VGMM_STAGE debug level (0=full, delivered by host via env); [1] pad to 32B
};

struct VgmmTilingData {
    VgmmBaseParams baseParams;
    AscendC::tiling::TCubeTiling cubeTiling;  // host: filled by matmul_tiling; kernel: consumed by MatmulImpl::Init
};
#pragma pack(pop)

}  // namespace npu_kernel
}  // namespace sglang

#endif  // VENDORED_GMM_TILING_DATA_H
