// fused_sigmoid_gating_recurrent fp32 入口（pool fp32，精度回归臂）。
// 一 kernel 一文件——KERNEL_TASK_TYPE_DEFAULT 与单文件多 kernel 不共存（仅首个注册
// 成功，其余 RegisterAscendBinary ret 107000；见 lib 头注释的结构约束说明）。

#include "fused_sigmoid_gating_recurrent_kernel_lib.h"

extern "C" __global__ __aicore__ void fused_sigmoid_gating_recurrent_fp32(FGR_KERNEL_ARGS)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    FusedSigmoidGatingRecurrent<float> op(n, h, hv, qRowStride, kRowStride, vRowStride, scale, softplusBeta,
                                          invSoftplusBeta, softplusThreshold, useQkL2norm);
    op.Init(A_log, a, dt_bias, q, k, v, b, o, pool, cache_indices, cu_seqlens, &pipe);
    op.Process();
}
