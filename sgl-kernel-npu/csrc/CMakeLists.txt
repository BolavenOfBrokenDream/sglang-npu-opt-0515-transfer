# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/pytorch_extensions.cpp
    ${PROJECT_OP_SRC_BASE}/helloworld/op_host/helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_host/cache_loc_assign.cpp
    ${PROJECT_OP_SRC_BASE}/alloc_extend/op_host/alloc_extend_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_host/assign_cache.cpp
    ${PROJECT_OP_SRC_BASE}/build_tree/op_host/build_tree.cpp
    ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_host/mla_preprocess.cpp
    ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_host/batch_matmul_transpose.cpp
    ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_host/tiling/tiling_data.cpp
    ${PROJECT_OP_SRC_BASE}/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/bgmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/bgmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmc_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmc_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/tiling/sgemmc_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/lightning_indexer.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/tiling/lightning_indexer_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/tri_inv/op_host/tri_inv.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/causal_conv1d_update.cpp
    ${PROJECT_OP_SRC_BASE}/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_host/causal_conv1d.cpp
    ${PROJECT_OP_SRC_BASE}/apply_token_bitmask/op_host/apply_token_bitmask.cpp
    # TP_FUSION: fused_qkvzba_conv1d + fused_sigmoid_gating_recurrent
    ${PROJECT_OP_SRC_BASE}/fused_qkvzba_conv1d/op_host/fused_qkvzba_conv1d.cpp
    ${PROJECT_OP_SRC_BASE}/fused_sigmoid_gating_recurrent/op_host/fused_sigmoid_gating_recurrent.cpp
    )
if(BUILD_CATLASS_MODULE)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_matmul_basic.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_op_utils.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_gmm_fp8.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_matmul_fp8.cpp
    )
endif()
list(APPEND OP_SRCS
    ${PROJECT_OP_SRC_BASE}/mega_chunk_gdn/op_host/mega_chunk_gdn.cpp
)

# set the so name
set(OP_PLUGIN_NAME sgl_kernel_npu)

# kernel side files without workspace
set(NO_WORKSPACE_KERNEL_SRCS
    ${PROJECT_OP_SRC_BASE}/helloworld/op_kernel/kernel_helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_kernel/assign_cache_op.cpp
    ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/bgmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/bgmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/tri_inv/op_kernel/tri_inv_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp
)

if(BUILD_CATLASS_MODULE)
    list(APPEND NO_WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_gmm_fp8_kernel.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_matmul_fp8_kernel.cpp
    )
endif()
ascendc_library(no_workspace_kernel STATIC ${NO_WORKSPACE_KERNEL_SRCS})
ascendc_include_directories(no_workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils/kernel
)
if(BUILD_CATLASS_MODULE)
    ascendc_include_directories(no_workspace_kernel PRIVATE
        ${CATLASS_DIR}/include
    )
endif()

# kernel side files with workspace
set(WORKSPACE_KERNEL_SRCS
    ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/alloc_extend/op_kernel/alloc_extend_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/build_tree/op_kernel/build_tree_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_kernel/lightning_indexer_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel/causal_conv1d_update.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_kernel/causal_conv1d.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmc_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmc_shrink_kernel.cpp
)
if(BUILD_CATLASS_MODULE)
    list(APPEND WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_matmul_basic_kernel.cpp
    )
endif()
ascendc_library(workspace_kernel STATIC ${WORKSPACE_KERNEL_SRCS})
ascendc_include_directories(workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_kernel
)
if(BUILD_CATLASS_MODULE)
    ascendc_include_directories(workspace_kernel PRIVATE
        ${CATLASS_DIR}/include
    )
endif()

ascendc_include_directories(workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils/kernel
)

ascendc_compile_definitions(workspace_kernel PRIVATE
       -DHAVE_WORKSPACE
       -DHAVE_TILING
)

# The two TP_FUSION kernels live in dedicated libraries with --cce-auto-sync
# pinned explicitly so the global default cannot drift. Both kernels use fully
# manual synchronization (explicit PipeBarrier/SetFlag) and are correct either
# way.
#
# The two kernels must be in separate libraries: under
# HAVE_WORKSPACE/HAVE_TILING the toolchain auto_gen launch wrapper expands the
# signature per the "last two params = workspace/tiling" convention.
# fused_qkvzba_conv1d ends in GM_ADDR workspace/tiling and matches;
# fused_sigmoid_gating_recurrent has no workspace/tiling and ends in 8
# by-value scalar params, so in the same library auto_gen would mistake the
# second-to-last float softplusThreshold for a workspace pointer. Hence
# recurrent goes alone into tp_fusion_recurrent_kernel, without
# HAVE_WORKSPACE/HAVE_TILING.
ascendc_library(tp_fusion_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/fused_qkvzba_conv1d/op_kernel/fused_qkvzba_conv1d_kernel.cpp
)
ascendc_include_directories(tp_fusion_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils/kernel
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_kernel
)
ascendc_compile_definitions(tp_fusion_kernel PRIVATE
       -DHAVE_WORKSPACE
       -DHAVE_TILING
)
ascendc_compile_options(tp_fusion_kernel PRIVATE
       --cce-auto-sync=on
)

# The bf16/fp32 recurrent entries must be one kernel per file:
# KERNEL_TASK_TYPE_DEFAULT does not coexist with multiple kernels in one file
# — only the first kernel registers; the rest fail with RegisterAscendBinary
# aiv ret 107000 and launch hangs with the kernel not found.
# Shared classes live in op_kernel/fused_sigmoid_gating_recurrent_kernel_lib.h.
ascendc_library(tp_fusion_recurrent_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/fused_sigmoid_gating_recurrent/op_kernel/fused_sigmoid_gating_recurrent_bf16_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/fused_sigmoid_gating_recurrent/op_kernel/fused_sigmoid_gating_recurrent_fp32_kernel.cpp
)
# Pin =off: this kernel is a double-buffered pipeline with fully manual event
# discipline (AllocEventID events + DrainEvents draining at the end) and does
# not use the TQue programming model — auto-sync only applies to the canonical
# model (EnQue/DeQue/AllocTensor/FreeTensor). =on gains nothing here and only
# inserts extra compiler-owned events at scalar GetValue/SetValue (same
# hardware pool as the manual events).
ascendc_compile_options(tp_fusion_recurrent_kernel PRIVATE
       --cce-auto-sync=off
)

set(MEGA_CHUNK_GDN_KERNEL_TARGETS)
set(MEGA_CHUNK_GDN_KERNEL_TARGETS mega_chunk_gdn_kernel)
ascendc_library(mega_chunk_gdn_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/mega_chunk_gdn/op_kernel/mega_kernel.cpp
)
ascendc_include_directories(mega_chunk_gdn_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/mega_chunk_gdn/op_kernel
    ${PROJECT_SOURCE_DIR}/third_party/pto-isa/include
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/experiment/runtime
    ${ASCEND_INCLUDE_DIR}/experiment/msprof
)
ascendc_compile_definitions(mega_chunk_gdn_kernel PRIVATE
    -DGDN_KERNEL_NAME=launch_mega_kernel
    -DGDN_D=128
    -DGDN_C=128
)

# create shared library libsgl_kernel_npu.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        ${MEGA_CHUNK_GDN_KERNEL_TARGETS}
        workspace_kernel
        no_workspace_kernel
        tp_fusion_kernel
        tp_fusion_recurrent_kernel
        torch_npu
        ascendcl
        tiling_api
        register
        platform
        ascendalog
        dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/tiling
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/stub
        ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_host
        ${PROJECT_OP_SRC_BASE}/fused_qkvzba_conv1d/op_host
        ${PROJECT_OP_SRC_BASE}/fused_sigmoid_gating_recurrent/op_host
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${ASCEND_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
