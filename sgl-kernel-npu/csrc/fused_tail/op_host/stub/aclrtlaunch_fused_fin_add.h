// fused_tail: aclrtlaunch declaration for fused_fin_add (handwritten stub,
// same convention as fused_qkvzba_conv1d; fallback when auto_gen is absent —
// signature matches the kernel entry parameter by parameter, workspace slot is
// a dummy position).
#ifndef HEADER_ACLRTLAUNCH_CUSTOM_FUSED_FIN_ADD_H
#define HEADER_ACLRTLAUNCH_CUSTOM_FUSED_FIN_ADD_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_fused_fin_add(uint32_t numBlocks, aclrtStream stream, void *x, void *scales,
                                              void *skip1, void *out, void *workspace, void *tiling);
#endif
