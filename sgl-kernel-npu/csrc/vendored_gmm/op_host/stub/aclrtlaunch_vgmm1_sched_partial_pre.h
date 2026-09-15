#ifndef HEADER_ACLRTLAUNCH_VGMM1_SCHED_PARTIAL_PRE_H
#define HEADER_ACLRTLAUNCH_VGMM1_SCHED_PARTIAL_PRE_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_vgmm1_sched_partial_pre(uint32_t numBlocks, aclrtStream stream, void *partials,
                                                        void *block_table, void *row_offsets, void *total_blocks,
                                                        void *counts, void *workspace, void *tiling);
#endif
