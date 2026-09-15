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
 * \file vendored_gmm_host_common.h
 * \brief vendored GMM host-side common logic: replica of the stock
 *        GroupedMatmul non-quantized bf16 path's baseM/baseN/baseK and
 *        stepKa/stepKb computation.
 *        Source: ops-transformer-9.0.0 gmm/grouped_matmul/op_host/op_tiling/
 *        grouped_matmul_tiling.cpp CalMMTiling (:1451-1522) and CalcStepKaKb
 *        (:1325-1365), keeping only the non-quantized, ND, A16W16 branches.
 *        The sched op and the main op must obtain values through the same
 *        function so both kernels block identically.
 */

#ifndef VENDORED_GMM_HOST_COMMON_H
#define VENDORED_GMM_HOST_COMMON_H

#include <algorithm>
#include <cstdint>

#include <torch/all.h>

#include "tiling/platform/platform_ascendc.h"

namespace sglang {
namespace npu_kernel {

struct VgmmHwInfo {
    uint32_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
};

inline VgmmHwInfo VgmmQueryHw()
{
    auto pf = platform_ascendc::PlatformAscendCManager::GetInstance();
    VgmmHwInfo hw;
    hw.aicNum = pf->GetCoreNumAic();
    pf->GetCoreMemSize(platform_ascendc::CoreMemType::L1, hw.l1Size);
    pf->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, hw.l0aSize);
    pf->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, hw.l0bSize);
    pf->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, hw.l0cSize);
    pf->GetCoreMemSize(platform_ascendc::CoreMemType::UB, hw.ubSize);
    return hw;
}

inline uint32_t VgmmAlign16Up(int64_t v)  // = stock SixteenAlign(x, true) (round up)
{
    return static_cast<uint32_t>((v + 15) / 16 * 16);
}

struct VgmmBaseTile {
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t stepKa = 0;
    uint32_t stepKb = 0;
};

// Replica of stock CalMMTiling + CalcStepKaKb for non-quantized bf16;
// maxM = total row count of x
// (SplitMSingleXSingleWeightSingleY: maxM_ = totalM_ = GMMGetBS(xShape)).
inline VgmmBaseTile VgmmComputeBaseTile(int64_t maxM, const VgmmHwInfo &hw)
{
    constexpr uint32_t BF16_SIZE = 2;
    constexpr uint32_t FP32_SIZE = 4;
    constexpr uint64_t DOUBLE_BUFFER = 2;
    constexpr uint32_t BEST_BASEN = 256;   // grouped_matmul_host_util.h:38
    constexpr uint32_t MAX_BASEM = 256;    // grouped_matmul_host_util.h:46
    constexpr uint64_t L1_PARTA_SIZE = 256ULL * 1024;  // corroborated by the "less than 256k" error line

    VgmmBaseTile t;
    t.baseN = BEST_BASEN;
    // baseK: L0B double buffer / (baseN * bf16)
    int64_t baseK = static_cast<int64_t>((hw.l0bSize / DOUBLE_BUFFER) / (t.baseN * BF16_SIZE));
    t.baseK = VgmmAlign16Up(baseK);
    // baseM: min(L0A double buffer / (baseK * bf16), L0C / (baseN * fp32))
    uint32_t maxBaseM = static_cast<uint32_t>(hw.l0cSize / (t.baseN * FP32_SIZE));
    int64_t baseM = std::min<int64_t>((hw.l0aSize / DOUBLE_BUFFER) / (t.baseK * BF16_SIZE), maxBaseM);
    // baseM_ > maxM_ ? SixteenAlign(maxM_, true) : SixteenAlign(baseM_)
    baseM = baseM > maxM ? VgmmAlign16Up(maxM) : VgmmAlign16Up(baseM);
    if (baseM > MAX_BASEM) {
        baseM = MAX_BASEM;
    }
    t.baseM = static_cast<uint32_t>(baseM);

    TORCH_CHECK(t.baseK > 0, "vendored_gmm: baseK computed as 0 (l0bSize=", hw.l0bSize, ")");
    TORCH_CHECK(t.baseM > 0, "vendored_gmm: baseM computed as 0 (l0aSize=", hw.l0aSize, ")");

    // stepKa/stepKb: L1 is split into A/B halves depending on baseM>baseN,
    // each half double-buffered
    uint64_t l1ASize = t.baseM > t.baseN ? L1_PARTA_SIZE : hw.l1Size - L1_PARTA_SIZE;
    uint64_t l1BSize = hw.l1Size - l1ASize;
    t.stepKa = static_cast<uint32_t>((l1ASize / DOUBLE_BUFFER) /
                                     (static_cast<uint64_t>(t.baseM) * t.baseK * BF16_SIZE));
    t.stepKb = static_cast<uint32_t>((l1BSize / DOUBLE_BUFFER) /
                                     (static_cast<uint64_t>(t.baseN) * t.baseK * BF16_SIZE));
    TORCH_CHECK(t.stepKa > 0 && t.stepKb > 0,
                "vendored_gmm: stepKa/stepKb computed as 0 (l1Size=", hw.l1Size, ")");
    // Stock CalcStepKaKb tail normalization (grouped_matmul_tiling.cpp:1361-1365,
    // must not be dropped): the larger is truncated to an integer multiple of
    // the smaller, guaranteeing the outerKaIter/outerKbIter multiple relation
    // in the kernel MDL K-loop.
    // Dropping it lets combos like stepKa=7/stepKb=4 make Ceil(K,baseK*stepKa)=5
    // and Ceil(K,baseK*stepKb)=8 non-multiples of each other — kernel-side
    // k_loop_mdl_base.h has an ASCENDC_ASSERT guarding this case, but it is
    // compiled away in release, and the A/B L1 K-chunk misalignment continues
    // silently: numerics are guaranteed wrong (first on-device run measured
    // stepKa=7 stepKb=4, normalized to 4/4, back to the README expected
    // values).
    if (t.stepKa > t.stepKb) {
        t.stepKa = t.stepKa / t.stepKb * t.stepKb;
    } else if (t.stepKa < t.stepKb) {
        t.stepKb = t.stepKb / t.stepKa * t.stepKa;
    }
    return t;
}

}  // namespace npu_kernel
}  // namespace sglang

#endif  // VENDORED_GMM_HOST_COMMON_H
