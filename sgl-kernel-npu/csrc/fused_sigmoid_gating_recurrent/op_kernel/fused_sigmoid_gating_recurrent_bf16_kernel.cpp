// fused_sigmoid_gating_recurrent bf16 entry (bf16 pool, production arm).
// One kernel per file: KERNEL_TASK_TYPE_DEFAULT does not coexist with multiple
// kernels in one file (only the first registers; the rest fail with
// RegisterAscendBinary ret 107000 — see the structural constraint in the lib header).

#include "fused_sigmoid_gating_recurrent_kernel_lib.h"

extern "C" __global__ __aicore__ void fused_sigmoid_gating_recurrent_bf16(FGR_KERNEL_ARGS)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    FusedSigmoidGatingRecurrent<bfloat16_t> op(n, h, hv, qRowStride, kRowStride, vRowStride, scale, softplusBeta,
                                               invSoftplusBeta, softplusThreshold, useQkL2norm);
    op.Init(A_log, a, dt_bias, q, k, v, b, o, pool, cache_indices, cu_seqlens, &pipe);
    op.Process();
}
