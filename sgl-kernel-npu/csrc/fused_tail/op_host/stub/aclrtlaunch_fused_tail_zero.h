// fused_tail: aclrtlaunch declaration for fused_tail_zero (handwritten stub,
// same convention as fused_qkvzba_conv1d; fallback when auto_gen is absent —
// signature matches the kernel entry parameter by parameter, workspace slot is
// a dummy position).
#ifndef HEADER_ACLRTLAUNCH_CUSTOM_FUSED_TAIL_ZERO_H
#define HEADER_ACLRTLAUNCH_CUSTOM_FUSED_TAIL_ZERO_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_fused_tail_zero(uint32_t numBlocks, aclrtStream stream, void *x, void *workspace,
                                                void *tiling);
#endif
