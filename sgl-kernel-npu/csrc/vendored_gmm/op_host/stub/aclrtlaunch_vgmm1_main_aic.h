#ifndef HEADER_ACLRTLAUNCH_VGMM1_MAIN_AIC_H
#define HEADER_ACLRTLAUNCH_VGMM1_MAIN_AIC_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_vgmm1_main_aic(uint32_t numBlocks, aclrtStream stream, void *x, void *weight,
                                               void *block_table, void *total_blocks, void *y, void *workspace,
                                               void *tiling);
#endif
