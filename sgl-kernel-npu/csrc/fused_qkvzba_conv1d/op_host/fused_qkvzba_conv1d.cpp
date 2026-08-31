// fused_qkvzba_conv1d (GDN decode: fused split + causal_conv1d) host implementation
/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_qkvzba_conv1d.cpp
 * \brief fused_qkvzba_conv1d host-side implementation
 *
 * Fused semantics (matching fla/utils.py fused_qkvzba_split_reshape_cat_contiguous + torch.ops.npu.causal_conv1d):
 *   y[B, qkvWidth]   = causal_conv1d(qkvz[:, :qkvWidth], ...)   -- x rows are read with stride xRowStride = qkvz.stride(0)
 *   z[B, nv, head_v] = qkvz[:, qkvWidth : qkvWidth+nv*head_v]   -- copied by the same kernel
 *   b[B, nv]         = mixed_ba[:, :nv]
 *   a[B, nv]         = mixed_ba[:, nv : 2nv]
 * The conv part is bit-exact with causal_conv1d run_mode=1 (UPDATE): same tiling struct, same device class,
 * only dim = qkvWidth and xRowStride = physical row stride of qkvz. Decode empty-call convention:
 * has_initial_state / num_accepted_tokens are always treated as empty.
 *
 * qkvz / mixed_ba accept row-stride views (stride(1)==1, row stride >= logical width), e.g. non-contiguous
 * slices of packed GEMM output: xRowStride = stride(0) (== size(1) when contiguous); zWidth is carried
 * explicitly in tiling (it cannot be derived as xRowStride - dim when the row stride includes pack padding);
 * the ba row stride is carried as baRowStride. Alignment contract: xRowStride must still be a multiple of 16
 * (32B alignment of z row starts); the ba copy uses DataCopyPad and requires no alignment.
 */

#include <cstring>
#include <limits>
#include <tuple>
#include <unordered_map>
#include "acl/acl.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "tiling/platform/platform_ascendc.h"
#include "stub/aclrtlaunch_fused_qkvzba_conv1d.h"
#include "defines.h"
#include "torch_helper.h"
#include "common.h"
#include "fused_qkvzba_conv1d.h"
#include "../../causal_conv1d/op_kernel/causal_conv1d_tiling_data.h"  // shared tiling struct (with xRowStride/numVHeads)

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr int64_t MAX_DIM_TILE = 4096;
constexpr int32_t MAX_WIDTH = 4;
constexpr int32_t MIN_WIDTH = 2;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;
constexpr int64_t Z_COPY_MAX_BYTES = 65535;  // single-block DataCopy blockLen limit (bytes); z row copy is never split

constexpr uint32_t CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE = 1;

// Op-local tiling cache (same field-hash + global device buffer scheme as the causal_conv1d host,
// with xRowStride / numVHeads added so entries never collide with the standalone op or other shapes)
static uint32_t g_fusedQkvzbaConv1dCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> g_fusedQkvzbaConv1dCaptureMap;

struct FusedQkvzbaConv1dTilingKey {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t batch;
    int64_t inputMode;
    int64_t width;
    int64_t stateLen;
    int64_t numCacheLines;
    int64_t activationMode;
    int64_t padSlotId;
    int64_t runMode;
    int64_t hasBias;
    int64_t hasCacheIndices;
    int64_t hasInitialState;
    int64_t hasNumAccept;
    int64_t xRowStride;
    int64_t numVHeads;
    int64_t zWidth;       // hashed explicitly (not derived as xRowStride - dim)
    int64_t baRowStride;  // ba physical row stride (== 2*numVHeads when contiguous)
};

struct FusedQkvzbaConv1dTilingKeyHash {
    static inline std::size_t HashCombine(std::size_t seed, std::size_t value)
    {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }

    std::size_t operator()(const FusedQkvzbaConv1dTilingKey &k) const
    {
        std::size_t h = 0;
        h = HashCombine(h, static_cast<std::size_t>(k.dim));
        h = HashCombine(h, static_cast<std::size_t>(k.cuSeqlen));
        h = HashCombine(h, static_cast<std::size_t>(k.seqLen));
        h = HashCombine(h, static_cast<std::size_t>(k.batch));
        h = HashCombine(h, static_cast<std::size_t>(k.inputMode));
        h = HashCombine(h, static_cast<std::size_t>(k.width));
        h = HashCombine(h, static_cast<std::size_t>(k.stateLen));
        h = HashCombine(h, static_cast<std::size_t>(k.numCacheLines));
        h = HashCombine(h, static_cast<std::size_t>(k.activationMode));
        h = HashCombine(h, static_cast<std::size_t>(k.padSlotId));
        h = HashCombine(h, static_cast<std::size_t>(k.runMode));
        h = HashCombine(h, static_cast<std::size_t>(k.hasBias));
        h = HashCombine(h, static_cast<std::size_t>(k.hasCacheIndices));
        h = HashCombine(h, static_cast<std::size_t>(k.hasInitialState));
        h = HashCombine(h, static_cast<std::size_t>(k.hasNumAccept));
        h = HashCombine(h, static_cast<std::size_t>(k.xRowStride));
        h = HashCombine(h, static_cast<std::size_t>(k.numVHeads));
        h = HashCombine(h, static_cast<std::size_t>(k.zWidth));
        h = HashCombine(h, static_cast<std::size_t>(k.baRowStride));
        return h;
    }
};

namespace {

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

struct UpdateDimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

// Mirrors the causal_conv1d host's ChooseUpdateBaseDimChoice (GE update-mode tiling policy);
// here dim = qkvWidth participates in the choice, identical to the old mixed_qkv path.
UpdateDimTileChoice ChooseUpdateBaseDimChoice(int64_t batch, int64_t dim, int32_t numCores)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};
    const int64_t coreNum = (numCores > 0) ? static_cast<int64_t>(numCores) : 1;

    auto chooseOnce = [&](bool requireExactDiv) -> UpdateDimTileChoice {
        UpdateDimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        UpdateDimTileChoice bestUnder;

        for (int64_t candBaseDim : candidates) {
            if (candBaseDim <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % candBaseDim != 0)) {
                continue;
            }
            const int64_t baseDimCnt = requireExactDiv ? (dim / candBaseDim) : CeilDiv(dim, candBaseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }
            if (gridSize >= coreNum) {
                const int64_t gap = gridSize - coreNum;
                if (gap < bestOverGap) {
                    bestOver = {candBaseDim, baseDimCnt, gridSize};
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && candBaseDim < bestUnder.baseDim)) {
                bestUnder = {candBaseDim, baseDimCnt, gridSize};
            }
        }
        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    UpdateDimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    return result;
}

// Mirrors the causal_conv1d host's ComputeTilingData, keeping only the decode (run_mode=1 UPDATE) path,
// with hasInitialState=false / hasNumAccept=false per the empty-call convention.
// zWidth / baRowStride go into tiling explicitly (cannot be derived from xRowStride for row-stride views).
void ComputeTilingData(int64_t dim, int64_t cuSeqlen, int64_t seqLen, int64_t batch, int64_t inputMode, int64_t width,
                       int64_t stateLen, int64_t numCacheLines, int64_t activationMode, int64_t padSlotId, bool hasBias,
                       bool hasCacheIndices, bool isBf16, int32_t numCores, int64_t xRowStride, int64_t numVHeads,
                       int64_t zWidth, int64_t baRowStride, CausalConv1dTilingData &td)
{
    (void)padSlotId;
    std::memset(&td, 0, sizeof(td));

    td.dim = dim;
    td.xRowStride = xRowStride;  // x rows are read with this stride (= qkvz.stride(0))
    td.numVHeads = numVHeads;    // head count for the b/a copy
    td.zWidth = zWidth;          // z column width carried explicitly (= numVHeads * head_v_dim)
    td.baRowStride = baRowStride;  // mixed_ba physical row stride (== 2*numVHeads when contiguous)
    td.cuSeqlen = cuSeqlen;
    td.seqLen = seqLen;
    td.inputMode = inputMode;
    td.width = width;
    td.stateLen = stateLen;
    td.numCacheLines = numCacheLines;
    td.batch = batch;
    td.activationMode = activationMode;
    td.padSlotId = padSlotId;
    td.hasBias = hasBias ? 1 : 0;
    td.hasCacheIndices = hasCacheIndices ? 1 : 0;
    td.hasInitialStateMode = 0;   // decode empty call: no has_initial_state
    td.hasInitStateWorkspace = 0;
    td.hasNumAcceptedTokens = 0;  // decode empty call: no num_accepted_tokens (non-speculative path)

    td.dtypeKey = isBf16 ? 0 : 1;
    td.runModeKey = CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE;
    td.widthKey = (width == 2) ? 1 : (width == 3) ? 2 : 3;  // same WIDTH_2/3/4 encoding as the old host

    UpdateDimTileChoice choice = ChooseUpdateBaseDimChoice(batch, dim, numCores);
    if (choice.baseDim <= 0 || choice.baseDimCnt <= 0) {
        choice.baseDim = (dim > 0 && dim <= MAX_DIM_TILE) ? dim : MAX_DIM_TILE;
        choice.baseDimCnt = (choice.baseDim > 0) ? CeilDiv(dim, choice.baseDim) : 1;
        if (choice.baseDimCnt <= 0) {
            choice.baseDimCnt = 1;
        }
    }
    td.baseDim = choice.baseDim;
    td.baseDimCnt = choice.baseDimCnt;
    td.fnPlanKey = 0;  // CAUSAL_CONV1D_TPL_FN_PLAN_INVALID
    td.tokenBlockSize = 0;
    td.tokenBlockCnt = 0;

    td.hasExplicitTokenSeqRanges = 0;
    td.explicitTokenSeqRangeCount = 0;
}

}  // namespace

HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_qkvzba_conv1d_impl(
    const at::Tensor &qkvz, const at::Tensor &weight, const at::Tensor &conv_states, const at::Tensor &mixed_ba,
    int64_t num_k_heads, int64_t num_v_heads, int64_t head_k_dim, int64_t head_v_dim, const at::Tensor &bias,
    const at::Tensor &query_start_loc, const at::Tensor &cache_indices, int64_t activation_mode, int64_t pad_slot_id)
{
    TORCH_CHECK(qkvz.defined(), "qkvz tensor must be defined");
    TORCH_CHECK(weight.defined(), "weight tensor must be defined");
    TORCH_CHECK(conv_states.defined(), "conv_states tensor must be defined");
    TORCH_CHECK(mixed_ba.defined(), "mixed_ba tensor must be defined");

    TORCH_CHECK(qkvz.dim() == 2, "qkvz must be 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D tensor");
    TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D tensor");
    TORCH_CHECK(mixed_ba.dim() == 2, "mixed_ba must be 2D tensor");

    const at::ScalarType dtype = qkvz.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported");
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match qkvz dtype");
    TORCH_CHECK(conv_states.scalar_type() == dtype, "conv_states dtype must match qkvz dtype");
    TORCH_CHECK(mixed_ba.scalar_type() == dtype, "mixed_ba dtype must match qkvz dtype");

    // Row-stride views (non-contiguous slices of packed GEMM output) are accepted: contiguous within a row
    // (stride(1)==1) and row stride >= logical width; contiguous tensors satisfy this trivially.
    TORCH_CHECK(qkvz.stride(1) == 1 && qkvz.stride(0) >= qkvz.size(1),
                "qkvz must be row-contiguous (stride(1)==1 and stride(0)>=size(1)); contiguous or pack row-stride view");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_states.is_contiguous(), "conv_states must be contiguous");
    TORCH_CHECK(mixed_ba.stride(1) == 1 && mixed_ba.stride(0) >= mixed_ba.size(1),
                "mixed_ba must be row-contiguous (stride(1)==1 and stride(0)>=size(1))");

    TORCH_CHECK(num_k_heads > 0 && num_v_heads > 0 && head_k_dim > 0 && head_v_dim > 0,
                "num_k_heads/num_v_heads/head_k_dim/head_v_dim must be positive");

    const int64_t qkvWidth = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    const int64_t zWidth = num_v_heads * head_v_dim;
    TORCH_CHECK(qkvz.size(1) == qkvWidth + zWidth, "qkvz.size(1) must equal qkvWidth + zWidth = ",
                qkvWidth + zWidth);
    TORCH_CHECK(mixed_ba.size(1) == 2 * num_v_heads, "mixed_ba.size(1) must equal 2 * num_v_heads");
    TORCH_CHECK(mixed_ba.size(0) == qkvz.size(0), "mixed_ba and qkvz must have the same row count");

    const int64_t dim = qkvWidth;  // conv consumes only the first qkvWidth columns of each row (= the old path's split-out mixed_qkv)
    const int64_t xRowStride = qkvz.stride(0);       // physical row stride (== size(1) when contiguous)
    const int64_t baRowStride = mixed_ba.stride(0);  // physical row stride (== 2*numVHeads when contiguous)

    const int64_t width = weight.size(0);
    TORCH_CHECK(width >= MIN_WIDTH && width <= MAX_WIDTH, "Only support width in [2,4]");
    TORCH_CHECK(weight.size(1) == dim, "weight must be [width, qkvWidth]");

    // Alignment contract for the z/b/a copy: z row start byte offset = element_size * (t * xRowStride + qkvWidth)
    // and z row bytes = element_size * zWidth must both be 32B aligned (the kernel uses plain DataCopy);
    // for bf16/fp16 this means qkvWidth / xRowStride must be multiples of 16 (the pack side pads the packed N
    // to a multiple of 16; when contiguous, xRowStride == size(1) == qkvWidth + zWidth).
    TORCH_CHECK(qkvWidth % 16 == 0 && xRowStride % 16 == 0,
                "qkvWidth and qkvz.stride(0) must be multiples of 16 (32B alignment for z DataCopy)");
    TORCH_CHECK(zWidth * static_cast<int64_t>(qkvz.element_size()) <= Z_COPY_MAX_BYTES,
                "z row bytes exceed single-block DataCopy limit");

    // decode is fixed to 2D varlen semantics (the old host's inputMode=0 path): batch is derived from query_start_loc
    const int64_t inputMode = 0;
    const int64_t seqLen = 0;
    const int64_t cuSeqlen = qkvz.size(0);
    TORCH_CHECK(query_start_loc.defined() && query_start_loc.numel() >= 2,
                "query_start_loc must have at least 2 elements");
    const int64_t batch = query_start_loc.size(0) - 1;
    TORCH_CHECK(batch == cuSeqlen, "decode requires qkvz.size(0) == query_start_loc.size(0) - 1");

    const int64_t numCacheLines = conv_states.size(0);
    const int64_t stateLen = conv_states.size(1);
    TORCH_CHECK(conv_states.size(2) == dim, "conv_states must be [slots, stateLen, qkvWidth]");

    bool hasBias = bias.defined() && bias.numel() > 0;
    bool hasCacheIndices = cache_indices.defined() && cache_indices.numel() > 0;
    bool isBf16 = (dtype == at::kBFloat16);

    at::Tensor y = at::empty({batch, qkvWidth}, qkvz.options());
    at::Tensor z = at::empty({batch, num_v_heads, head_v_dim}, qkvz.options());
    at::Tensor b = at::empty({batch, num_v_heads}, qkvz.options());
    at::Tensor a = at::empty({batch, num_v_heads}, qkvz.options());

    // None -> empty conversion for optional args, matching the old host (causal_conv1d.cpp)
    at::Tensor bias_tensor = hasBias ? bias : at::empty({0}, qkvz.options());
    at::Tensor query_start_loc_tensor = query_start_loc.to(at::kLong);
    at::Tensor cache_indices_tensor =
        hasCacheIndices ? cache_indices.to(at::kLong) : at::empty({0}, qkvz.options().dtype(at::kLong));
    at::Tensor has_initial_state_tensor = at::empty({0}, qkvz.options().dtype(at::kLong));  // decode empty call
    at::Tensor num_accepted_tokens_tensor = at::empty({0}, qkvz.options().dtype(at::kInt));  // decode empty call

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t maxAivCore = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());

    CausalConv1dTilingData tilingData;
    ComputeTilingData(dim, cuSeqlen, seqLen, batch, inputMode, width, stateLen, numCacheLines, activation_mode,
                      pad_slot_id, hasBias, hasCacheIndices, isBf16, maxAivCore, xRowStride, num_v_heads,
                      zWidth, baRowStride, tilingData);

    // run_mode=1 (UPDATE): totalBlocks = batch * baseDimCnt (matching the old host)
    int64_t totalBlocks = tilingData.batch * tilingData.baseDimCnt;
    int32_t blockDim = std::min(maxAivCore, static_cast<int32_t>(totalBlocks));
    if (blockDim <= 0) {
        blockDim = 1;
    }

    int32_t libApiWorkspaceSize = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    // decode empty call has no hasInitialState workspace: totalWorkspace = libApiWorkspaceSize (matching the old host)
    int64_t totalWorkspace = std::max(static_cast<int64_t>(libApiWorkspaceSize), static_cast<int64_t>(0));
    if (totalWorkspace <= 0) {
        totalWorkspace = libApiWorkspaceSize;
    }

    int32_t tilingSize =
        (static_cast<int32_t>(sizeof(CausalConv1dTilingData)) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;

    FusedQkvzbaConv1dTilingKey key{dim,
                                   cuSeqlen,
                                   seqLen,
                                   batch,
                                   inputMode,
                                   width,
                                   stateLen,
                                   numCacheLines,
                                   activation_mode,
                                   pad_slot_id,
                                   static_cast<int64_t>(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE),
                                   hasBias ? 1 : 0,
                                   hasCacheIndices ? 1 : 0,
                                   0,  // hasInitialState: decode empty call
                                   0,  // hasNumAccept: decode empty call
                                   xRowStride,
                                   num_v_heads,
                                   zWidth,
                                   baRowStride};
    uint64_t hashValue = FusedQkvzbaConv1dTilingKeyHash{}(key);

    static auto globalTilingBuffer = at::empty({tilingSize * static_cast<int64_t>(MAX_CAPTURE_NUM)},
                                               at::TensorOptions().dtype(at::kByte).device(qkvz.options().device()));

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(CausalConv1dTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    at::Tensor tilingTensor;
    if (g_fusedQkvzbaConv1dCaptureMap.find(hashValue) != g_fusedQkvzbaConv1dCaptureMap.end()) {
        tilingTensor = at::from_blob(
            globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * g_fusedQkvzbaConv1dCaptureMap[hashValue]),
            tilingSize, at::kByte);
    } else if (g_fusedQkvzbaConv1dCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        g_fusedQkvzbaConv1dCaptureMap[hashValue] = g_fusedQkvzbaConv1dCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer
            .slice(0, g_fusedQkvzbaConv1dCaptureNum * tilingSize, g_fusedQkvzbaConv1dCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        g_fusedQkvzbaConv1dCaptureNum++;
        tilingTensor = at::from_blob(
            globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * g_fusedQkvzbaConv1dCaptureMap[hashValue]),
            tilingSize, at::kByte);
    }

    auto workspaceTensor =
        at::empty({totalWorkspace}, at::TensorOptions().dtype(at::kByte).device(qkvz.options().device()));

    EXEC_KERNEL_CMD(fused_qkvzba_conv1d, blockDim, qkvz, weight, conv_states, mixed_ba, bias_tensor,
                    query_start_loc_tensor, cache_indices_tensor, has_initial_state_tensor,
                    num_accepted_tokens_tensor, y, z, b, a, workspaceTensor, tilingTensor);

    return std::make_tuple(y, z, b, a);
}

}  // namespace npu_kernel
}  // namespace sglang
