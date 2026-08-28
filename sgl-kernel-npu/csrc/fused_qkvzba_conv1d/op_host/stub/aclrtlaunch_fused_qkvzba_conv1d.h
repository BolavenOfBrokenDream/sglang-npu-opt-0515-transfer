// fused_qkvzba_conv1d 的 aclrtlaunch 声明（照 causal_conv1d 的 stub 约定手写）
#ifndef HEADER_ACLRTLAUNCH_CUSTOM_FUSED_QKVZBA_CONV1D_H
#define HEADER_ACLRTLAUNCH_CUSTOM_FUSED_QKVZBA_CONV1D_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_fused_qkvzba_conv1d(uint32_t numBlocks, aclrtStream stream, void *x, void *weight,
                                                    void *convStates, void *ba, void *bias, void *queryStartLoc,
                                                    void *cacheIndices, void *hasInitialState,
                                                    void *numAcceptedTokens, void *y, void *z, void *b, void *a,
                                                    void *workspace, void *tiling);
#endif
