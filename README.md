# 0515 基线部署指南

代码：`git clone git@github.com:BolavenOfBrokenDream/sglang-npu-opt-0515-transfer.git`，放到容器可挂载的共享盘（下文 `<ROOT>` 指仓目录）。

## 1. 挂载（docker run 三个挂载点）

```bash
-v <ROOT>/sglang/python:/sgl-workspace/sglang/python \
-v <ROOT>/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu:/usr/local/python3.11.15/lib/python3.11/site-packages/sgl_kernel_npu \
-v <ROOT>/sgl-kernel-npu:/sgl-workspace/sgl-kernel-npu-src \
```

自检：`ls /sgl-workspace/sglang/python` 应见 `sglang/ setup.py`；`ls .../site-packages/sgl_kernel_npu` 应见 `__init__.py fla/`。

## 2. build 前准备（在 `<ROOT>/sgl-kernel-npu` 下）

若代码是从 Windows 拷贝的（直接 git clone 则跳过），先清 CRLF 和未来时间戳：

```bash
find . -type f -name '*.sh' -exec sed -i 's/\r$//' {} +
find . -type f \( -name '*.py' -o -name '*.cmake' -o -name 'CMakeLists.txt' -o -name '*.cpp' -o -name '*.h' \) -exec sed -i 's/\r$//' {} +
find . -exec touch {} +
```

下载第三方依赖（仓内不含子模块内容，缺了必编不过；无网则从可编译环境整目录拷入）：

```bash
git clone https://gitcode.com/cann/catlass.git third_party/catlass
git -C third_party/catlass checkout 70eda02b1271ff0e8c47db94469fae576feba97b
git clone https://gitcode.com/cann/pto-isa.git third_party/pto-isa
git -C third_party/pto-isa checkout af66b6ada43c40ad9583731883fb9bef65ac840a
```

## 3. 编译（容器内，仅 kernels）

```bash
cd /sgl-workspace/sgl-kernel-npu-src
./build.sh -a kernels Ascend910_9382
```

- 门禁：kernels 模块仅限 Ascend910_93xx；`-a kernels` 下 attentions 随编为脚本固有行为。
- 产物 `libsgl_kernel_npu.so` 直接写入仓内 python 包 `lib/`，经挂载自动生效，无需 pip install。
- 日志中 `fatal: not a git repository` 为良性警告（拷贝的树无 .git）。
- 若报 `failed to get run python script to get ENVS`：先 `python3 -c "import torch, torch_npu, pybind11"` 验证环境，必要时 `source /usr/local/Ascend/cann-9.0.0/set_env.sh` 后重跑。

## 4. 自检

```bash
python3 -c "
import sgl_kernel_npu, torch
print(hasattr(torch.ops.npu, 'fused_qkvzba_conv1d'), hasattr(torch.ops.npu, 'fused_sigmoid_gating_recurrent'))
"
```

输出 `True True` 即可起 relax rollout 验证。

## 附：开关速查

- 默认开：`SGLANG_NPU_GDN_QKVZBA_PACK`（门控 `_MAX_M=256`）、`SGLANG_NPU_FULL_ATTN_FUSION`（q∈{1,2,4}）
- 默认关：`SGLANG_NPU_TP_ASCENDC_FUSION`、`SGLANG_NPU_GDN_RECURRENT_ASCENDC`、`SGLANG_NPU_MOE_PREFETCH`、`SGLANG_NPU_EXP_RACE_TRITON`
- 本分支（qwen35_k9_moe_prefetch）新增，均默认关：
  - `SGLANG_NPU_MAINSTREAM_SHARED_EXPERT`（k9：shared expert 并入 routed GMM 第 257 槽；仅 `SGLANG_NPU_USE_MULTI_STREAM`+`SGLANG_NPU_VGMM1`+`SGLANG_GMM2_TRITON`+`SGLANG_NPU_MOE_TAIL_FUSION`(spin) 全开时生效，否则 warning 回退）
  - MoE prefetch v2：`SGLANG_NPU_MOE_PREFETCH_LAUNCH_POINT`（gdn/moe 可同时开，默认 gdn）、每发射点 `..._GDN_OPS`/`..._MOE_OPS`（单 OPS，默认 gmm1/gmm2）、`..._GDN_BUDGET_MIB`/`..._MOE_BUDGET_MIB`（单条 CMO 预取量，默认 32MiB）、`..._L2_CAP_RATIO`（预取量占 L2 上限，默认 0.7）；moe 发射点另需 `SGLANG_MOE_FRONT_FUSION`+k9 开启。旧 `..._OPS`/`..._MODE`/`..._CHUNK_MIB`/`..._BUDGET_MIB` 已删除
- 全量见 `sglang/python/sglang/srt/environ.py`

已知遗留：现场基线 `ascend_gdn_backend.py` forward_extend 尾部有疑似 patch 残留（未动），extend 段 AttributeError 先查该处。
