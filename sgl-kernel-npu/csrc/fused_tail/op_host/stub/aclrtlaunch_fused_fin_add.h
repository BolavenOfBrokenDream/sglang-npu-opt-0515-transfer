// fused_tail (tp_ascendc_fusion_v4.1): aclrtlaunch declaration for fused_fin_add
// (follows fused_qkvzba_conv1d's handwritten stub convention; fallback when
// auto_gen does not emit it — signature matches the kernel entry parameter by
// parameter, workspace slot = dummy parameter position).
#ifndef HEADER_ACLRTLAUNCH_CUSTOM_FUSED_FIN_ADD_H
#define HEADER_ACLRTLAUNCH_CUSTOM_FUSED_FIN_ADD_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_fused_fin_add(uint32_t numBlocks, aclrtStream stream, void *x, void *scales,
                                              void *skip1, void *out, void *eri, void *workspace, void *tiling);
#endif
