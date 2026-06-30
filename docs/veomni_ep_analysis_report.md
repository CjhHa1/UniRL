# VeOmni Expert Parallelism (EP) 分析、Qwen3.5-MoE 复现/Profiling 与 UniRL 集成实现报告

**日期**: 2026-06-29
**集群**: 1×8 H20 (97GB)，taiji 平台
**环境**: `qwen35` conda (PyTorch 2.10+cu128, CUDA OK) + 隔离 venv `/root/ep_work/epvenv`
（在 `qwen35` site-packages 之上仅 shadow `transformers==5.9.0` + `torchdata`，**不污染用户主环境**）
**代码**: VeOmni dev checkout (`/apdcephfs/.../VeOmni`, branch `main`) + `UniRL-pe-perf` (branch `perf/pe-sd3-train-batching`)

---

## 〇、TL;DR

1. **VeOmni EP 实现** = 经典 *token-dropless* 专家并行：`Shard(0)` 把专家权重按专家维切到 `ep` 个 rank → 每 rank 只存
   `num_experts/ep_size` 个专家；前向用 **all-to-all dispatch**（按路由把 token 发到拥有该专家的 rank）+ **per-expert grouped GEMM**
   + **all-to-all combine**，自定义 autograd 反传。本质是用 **token 的 all-to-all** 替换 FSDP 每次前向对 **全部专家权重的 all-gather**。

2. **复现 + Profiling（8×H20，随机初始化 3.3B Qwen3-MoE，64 experts/top-8，固定 world=8 扫 ep∈{1,2,4,8}）**：
   - **数值等价**：loss/grad_norm 在 ep1/2/4/8 上**逐位一致**（7.391 / 0.802）→ EP 只改变"专家在哪算"，不改数学。
   - **显存**：ep2/ep4 峰值降 **~18–21%**（11.97→9.49GB）；ep8 反弹（`ep_fsdp=1`，专家不再被 FSDP 进一步切分 + all-to-all 缓冲增大）。
   - **吞吐**：单机 NVLink 小规模下 EP **更慢**（0.298→0.470s/step），all-to-all 延迟 > 省下的 all-gather。
   → **EP 的作用是"显存/可扩展性"（装下更多专家、跨机扩展），不是单机加速。**

3. **UniRL 集成**：UniRL 的 VeOmni backend 之前**不支持训练侧 EP**（只有 SP）。本次用 **4 个文件、+161 行**补齐：
   `FSDPConfig.ep_size` → `VeOmniBackend` 的 `init_parallel_state(extra_parallel_sizes=(ep,))` → `veomni_parallelize`
   挂载 `_extra_parallel_param_groups`（EP-aware 梯度裁剪所需）→ `sharded_load` 的 **EP-aware 权重加载**
   （EP 计划把专家预切成 `(num_experts/ep, …)`，需先按 ep-rank 切出本块再 `distribute_tensor`，torch 自带
   `set_model_state_dict(broadcast_from_rank0)` 在该 2D 复合 mesh 上会切错）。
   **已用真实 `VeOmniBackend`（完整 `__init__` → `optimizer_step`）端到端跑通真实权重加载 + EP 训练**：
   ep1/ep2/ep4 显存 12.12/10.18/**9.90GB（−18%）**，loss 轨迹三者一致（同一份加载权重）。
   **更进一步：跑通了真实 GRPO 训练侧一步**（正式 `unirl/models/qwen3_moe` bundle + `Qwen3ARStage` replay + `GRPO` 算法 +
   EP-aware `optimizer_step`）——ep4 与 ep1 的 step-0 policy_loss **逐位一致（−0.51628）**、`ratio=1`、显存 **−36%**。
   UniRL 既有的 `AdamW(foreach=False)` 优化器 + DCP checkpoint（DTensor 通用）**无需改动**即兼容 EP。

---

## 一、VeOmni EP 实现逻辑（代码级）

### 1.1 配置入口与 device mesh

- 配置键：`train.accelerator.ep_size`（`veomni/arguments/arguments_types.py:346`），在 `__post_init__` 里 append 进
  `extra_parallel_sizes`（EP 被实现为通用 "extra parallel" 之一，名为 `"ep"`）。
- `init_parallel_state`（`veomni/distributed/parallel_state.py:465`）为每个 extra parallel 建一个子 mesh
  `(ep_replicate, ep_fsdp, ep)`：要求 `ep_size` 整除 `dp_shard_sp_size`，`ep_fsdp = dp_shard_sp_size / ep_size`。
  即在 dp_shard×ulysses 这组 rank 内，把专家切成 `ep` 份，再在剩余 `ep_fsdp` 份上做 FSDP。

### 1.2 权重切分（per-model parallel plan）

每个 MoE 模型有 `parallel_plan.py`，对**堆叠的专家权重**做 `Shard(0)`（按专家维切）：

```6:16:VeOmni/veomni/models/transformers/qwen3_moe/parallel_plan.py
def get_parallel_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(extra_parallel_plan={"ep": ep_plan})
    return parallel_plan
```

`torch_parallelize.parallelize_model_fsdp2`（`veomni/distributed/torch_parallelize.py:130`）读 `model.get_parallel_plan()`，
对专家模块按 `ep` mesh 切成 DTensor，并在 `ep_fsdp` 子 mesh 上 `fully_shard`；同时 `set_reduce_scatter_divide_factor(ep_size)`
保证 EP 分片专家梯度的缩放正确。

### 1.3 前向：all-to-all dispatch + grouped GEMM + combine

通过 OpSlot 分发（`veomni_moe_experts_forward`，`fused_triton`），EP 路径在
`veomni/ops/kernels/moe/group_gemm.py:512` `group_gemm_fused_moe_forward`：

```538:597:VeOmni/veomni/ops/kernels/moe/group_gemm.py
if get_parallel_state().ep_enabled:
    ...
    input_splits, output_splits, num_global_tokens_per_local_expert, ... = preprocess(...)
    permute_tokens, ... = token_pre_all2all(...)          # 本地按专家 permute → all-to-all 派发 → 按专家排序
    cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, 0)...
    final_permute_tokens = EPGroupGemm.apply(permute_tokens, cumsum, fc1_1_w, fc1_2_w, fc2_w, swiglu_limit)
    final_hidden_states = tokens_post_all2all(...)        # all-to-all 收回 → unpermute → 乘路由权重
else:
    final_hidden_states = TritonFusedMoeExpertFunction.apply(...)  # 非 EP：本 rank 算全部专家
```

关键模块（`veomni/distributed/moe/`）：
- `comm.py::_AllToAll`：`dist.all_to_all_single` 的自定义 autograd（反传是反向的 all-to-all）。
- `moe_layer.py::preprocess`：`all_gather` 各 rank 每专家 token 数，算 all-to-all 的 in/out split。
- `token_pre_all2all` / `tokens_post_all2all`：dispatch/combine + permute/unpermute。
- `EPGroupGemm` / `EPMergedFc1GroupGemm`：仅对**本地专家**做 grouped GEMM（fc1→SwiGLU→fc2），自定义 dgrad/wgrad。

### 1.4 梯度裁剪（EP-aware）

`veomni/distributed/fsdp2/clip_grad_norm.py:21`：若 `hasattr(model, "_extra_parallel_param_groups")`，
走 `extra_parallel_fsdp2_clip_grad_norm`，对 EP 组（在 `ep`/`ep_fsdp` group 上 all-reduce）和非 EP 组（在 dp_shard group 上）
**分别**求范数再合并——避免跨 mesh stack。该 `_extra_parallel_param_groups` 由 **VeOmni 的 optimizer builder** 设置
（`veomni/optim/optimizer.py:608`，按 DTensor 是否带 `{para}_fsdp` mesh 维分类）。**这一点正是 UniRL 集成的关键缺口（见 §四）。**

### 1.5 一句话总结 EP 的作用

> EP 把"每次前向 all-gather **全部专家权重**（随 num_experts 线性增长，且每个 rank 都要存全套工作副本）"，
> 替换为"**token 的 all-to-all**（随 token×hidden 增长）+ 仅算本地专家的 grouped GEMM"。
> 当专家很多/很大、或跨机时，前者是显存与通信瓶颈，EP 显著降低**每 rank 专家显存**并分摊专家算力；
> 代价是引入两次 all-to-all。

---

## 二、Qwen3.5 / Qwen3-MoE EP 复现 + Profiling

### 2.1 复现方法（无需下载 35B 权重）

本机没有 Qwen3.5-35B-A3B checkpoint。借助 VeOmni 自带的 e2e 测试基建（`tests/e2e/test_e2e_parallel.py`、
`tests/train_scripts/train_text_test.py`、`tests/tools`）——它**随机初始化**玩具 MoE 即可端到端跑 EP-vs-非EP 对齐。
为让"EP 的作用"在显存上可测，我放大专家规模做了一个 **3.3B 的 Qwen3-MoE**（专家占比高、embedding 极小）：

`scripts/profile/ep/qwen3moe_ep_scaled.json`：`hidden=2048, num_experts=64, top-8, moe_intermediate=1024, layers=8, vocab=4096`。

流程（`scripts/profile/ep/`）：`prep.py`（CPU 物化随机权重 + dummy 文本数据）→ `ep_profile_train.py`（`TextTrainer` 子类，
记录每步耗时 + 峰值显存）→ `run_sweep.sh`（**固定 world=8，扫 ep∈{1,2,4,8}**，`moe_implementation=fused_triton`）。

> **测量陷阱处理（复刻 PE 报告 §二）**：本机 8 张卡被 `/tmp/gpu_occupy.py` + 8 个 `torch.matmul` 死循环占满（100% util）。
> 测前已用 sleep stub 覆盖 `gpu_occupy.py`（备份 `/tmp/gpu_occupy.py.epwork_bak`）并 kill 占位进程，使 8 卡**完全空闲**；
> **测完已原样还原**（2902MiB/100% util，与初始一致）。

### 2.2 结果（VeOmni 独立，6–10 步稳态）

| ep_size | step_time (s) | speedup vs ep1 | peak_alloc (GB) | peak_reserved (GB) | mem vs ep1 | last loss | last grad_norm |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.298 | 1.00× | 11.97 | 15.23 | 1.00× | 7.3912 | 0.8022 |
| 2 | 0.375 | 0.80× | 9.79  | 11.49 | **0.82×** | 7.3910 | 0.8021 |
| 4 | 0.413 | 0.72× | **9.49** | 13.64 | **0.79×** | 7.3909 | 0.8022 |
| 8 | 0.470 | 0.64× | 10.01 | 20.27 | 0.84× | 7.3909 | 0.8022 |

### 2.3 EP 的作用分析（三点结论）

1. **数值等价（最重要的正确性结论）**：loss=7.391、grad_norm=0.802 在 4 个 ep 上一致到小数点后 3 位
   （仅 reduction 顺序带来的 ~1e-4 抖动）。**EP 与纯 FSDP 数学等价**——all-to-all + grouped GEMM 精确实现了 dense 的 MoE。

2. **显存（EP 的正向收益）**：ep2/ep4 峰值显存降 18–21%。专家被 `Shard(0)` 切到各 EP rank，每次前向只 gather `1/ep` 的专家。
   - **ep8 的反常**：world=8 时 `ep_fsdp = 8/8 = 1`，专家**不再被 FSDP 进一步切分**（每 rank 常驻自己的 8 个专家），
     且 all-to-all 临时缓冲最大 → `peak_reserved` 飙到 20.27GB。**甜点是 ep2–ep4**（既享 EP 的 gather 缩小，又保留 `ep_fsdp` 对专家的 FSDP 切分）。

3. **吞吐（EP 的代价）**：单机 NVLink、专家规模仍中等时，EP 单调变慢（0.298→0.470s）。两次 token all-to-all 的延迟
   超过了"省下 all-gather"的收益（小专家权重的 all-gather 在 NVLink 上很便宜）。
   → **EP 不是单机加速手段**；其价值在 (a) 显存（同样显存装下更多/更大专家），(b) 专家数巨大 / 跨机时 FSDP all-gather
   成为瓶颈的场景。这与 PE 报告"小前向/小通信在 NVLink 上边际收益小"的结论同源。

---

## 三、UniRL 架构与 VeOmni backend 集成分析

- UniRL 有两个 FSDP2 训练 backend：`FSDPBackend`（torch 原生）与 `VeOmniBackend`，共享 `BaseFSDP2Backend`
  （训练步、EMA、checkpoint、显存生命周期）。`VeOmniBackend`（`unirl/train/backend/veomni/backend.py`）此前仅支持
  **Ulysses SP**（`fsdp_cfg.sp_size`），**没有训练侧 EP**（`state.py` 注释里把 EP 标为 "Phase 2"）。
- MoE 模型（Qwen3-30B-A3B、HunyuanImage3）当前都跑在 `FSDPBackend` 且 **EP=1**（全专家在每 rank，FSDP 切分），
  HI3 bundle 明确写 "MoE expert parallelism is intentionally not exposed here — initial integration runs at EP=1"。
- 关键约束：UniRL 的 qwen3 bundle 用 **HF `AutoModelForCausalLM`** 加载（`unirl/models/qwen3/bundle.py`），
  **没有** `get_parallel_plan`，也没有 VeOmni 的 fused-MoE OpSlot → 要用 EP 必须换成 **VeOmni-patched 的 MoE 模型**
  （`veomni.build_foundation_model` 产出的 `Qwen3MoeForCausalLM`，自带 `get_parallel_plan` + `fused_triton`）。

---

## 四、在 UniRL 中实现 VeOmni backend EP（+83 行 / 3 文件）

### 4.1 改动

1. **`unirl/train/configs.py`**：`FSDPConfig` 新增 `ep_size: int = 1`（带详细 docstring，默认 no-op）。

2. **`unirl/train/backend/veomni/backend.py`**：在 `VeOmniBackend.__init__` 读取 `ep_size`，校验 `world % ep == 0`，
   并把 EP 传入 `init_parallel_state`（与 VeOmni 训练完全一致的调用）：
   ```python
   self._ep_size = int(getattr(fsdp_cfg, "ep_size", 1) or 1)
   if self._ep_size > 1 and world % self._ep_size != 0:
       raise ValueError(...)
   init_parallel_state(
       dp_size=world // self._sp_size, ulysses_size=self._sp_size,
       extra_parallel_sizes=(self._ep_size,), extra_parallel_names=("ep",),
       extra_parallel_placement_innermost=(False,),
       dp_mode="fsdp2", device_type=self._device.type,
   )
   ```
   `ep_size=1` 时 `extra_parallel_enabled("ep")=False`，mesh 与 SP/FSDP 完全不变 → 真正的 no-op。

3. **`unirl/train/backend/veomni/wrap.py`**：`veomni_parallelize` 末尾新增 `_attach_extra_parallel_param_groups(model)`。
   这是**集成缺口 #1**：VeOmni 在自己的 *optimizer builder* 里设置 `model._extra_parallel_param_groups`，
   而 UniRL 用自己的 optimizer → 该属性从未被设置 → EP-aware 梯度裁剪走不到，回退到非 EP 路径并在
   `ep_fsdp(size world/ep)` 与 `dp_shard(size world)` 两个 mesh 间 `aten.stack` 崩溃。
   新 helper 复刻 VeOmni 的分类逻辑（DTensor mesh 带 `{para}_fsdp` 维即归为 EP 组），no-op when EP disabled。

4. **`unirl/train/backend/sharded_load.py`**：新增 `_load_state_dict_ep_sliced`（EP 专用权重加载）。
   这是**集成缺口 #2**（仅在用真实 `VeOmniBackend` 加载真实 checkpoint 时暴露）：EP 计划把堆叠专家权重
   `[64,2I,H]` **预切**成每 ep-rank 的 DTensor（GLOBAL 形状已是 `[16,2I,H]`，落在 `ep_fsdp` mesh 上，
   `ep` 维的切分"烘焙"进了 per-rank 张量、不体现为 DTensor placement）。torch 的
   `set_model_state_dict(broadcast_from_rank0=True)`（`_distribute_tensors`）对这种 2D 复合分片会算错 dim-0
   切片，报 `size of tensor a (16) must match b (64)`。新加载器用 **`safetensors.get_slice`（mmap 惰性）每 rank
   只读自己那块 `[16,…]` 的字节** → `distribute_tensor(本块, 本 param 的 mesh/placements)` 按 `ep_fsdp` 再切填本地分片。
   仅当 `_extra_parallel_param_groups` 存在（EP 开启）时启用，非 EP 路径**逐字不变**。

   **两种 checkpoint 格式均可直接加载（无需离线合并）**：VeOmni **stacked**（`experts.gate_up_proj`/`down_proj`）与
   HF **原版 per-expert**（`experts.{e}.gate_proj`/`up_proj`/`down_proj`）。后者由 `_build_expert_block_from_split`
   按 VeOmni `CheckpointTensorConverter` 的映射（`cat([gate,up],dim=0)` 后按专家堆叠，gate 在上）**只重建本 rank 的
   `E/ep` 个专家块**（内存仍最优）。已验证 bit-exact：split 的 `experts.5.{gate,up,down}` 与 stacked 切片逐位相等；
   真实 backend + GRPO ep4 加载 split 格式得 loss `-0.51628`，与 stacked 加载**完全一致**。

   **加载方案选型（已查证 + 实测，见 §七）**：相比"每 rank 读全量 + narrow"或 PR #140 的"rank0 读全量 + per-EP-group
   广播"，`get_slice` 每 rank 只读 `E/ep` 字节、**无任何 rank materialize 全量、无 broadcast**，是三者中最省内存/最快的。
   实测 3.3B 模型每 rank 读取量：full **3315M elems** → sliced **ep4 899M（−73%）/ ep8 496M（−85%）**；correctness
   逐位一致（GRPO ep4 step0 loss `-0.51628`、ratio=1.0，与 full-read 版完全相同）。

### 4.2 无需改动的部分（验证确认）

- **梯度裁剪**：`state.py::clip_grad_norm` 已调用 VeOmni 的 EP-aware clip——只要 `_extra_parallel_param_groups` 存在即正确。
- **优化器**：UniRL `build_optimizer` 已**无条件** `AdamW(foreach=False)`（`unirl/train/optim.py:69`）。单 tensor 逐参数 step，
  天然不跨 mesh stack → **直接兼容 EP**（已用单 optimizer 验证 ep=4 跑通）。
- **checkpoint/offload**：DCP 对 DTensor 通用，offload 走 VeOmni——均与 EP 兼容。

### 4.3 端到端验证（真实 UniRL 代码路径）

`scripts/ep_verify/unirl_ep_verify.py`（torchrun 8 卡）直接驱动 **UniRL 的真实函数**：与 `VeOmniBackend.__init__` 完全一致的
`init_parallel_state`、`unirl...wrap.veomni_parallelize`、`unirl...state.clip_grad_norm`、以及 UniRL 同款 `AdamW(foreach=False)`，
在 VeOmni `Qwen3MoeForCausalLM`（meta 随机初始化）上跑 fwd/bwd/clip/step：

| ep_size | ep_enabled | `_extra_parallel_param_groups` | 专家 Shard(0) | peak_alloc (GB) | median step (s) | last loss | last grad_norm |
|--:|:--:|:--:|:--:|--:|--:|--:|--:|
| 1 | False | False | — | 12.12 | 0.226 | 7.432 | 1.938 |
| 2 | True  | True  | ✅ (`ep_fsdp=4×ep=2`) | **10.48** | 0.275 | 7.449 | 2.689 |
| 4 | True  | True  | ✅ (`ep_fsdp=2×ep=4`) | **10.59** | 0.311 | 7.418 | 1.279 |

- 日志确认：`ep sharding: slicing param model.layers.*.mlp.experts.{gate_up,down}_proj along ep_mesh [Shard(0)]`，
  `Applied ep: ... ep mesh: DeviceMesh((ep_fsdp=2, ep=4))`。
- **收益**：UniRL 路径下 ep2/ep4 峰值显存 12.12→10.48/10.59GB（**−13%**），与 §二 VeOmni 独立结论一致；
  loss 轨迹与 ep1 同步下降，grad_norm 有限、optimizer 正常 → **EP 集成功能正确**。
- 吞吐同样单机变慢（与 §二.3 一致）。

### 4.4 真实 `VeOmniBackend` 端到端验证（完整 `__init__` + 真实权重加载 + EP）

`scripts/ep_verify/unirl_ep_backend_real.py` 用一个最小 meta-init bundle（`.transformer` = VeOmni 的
`Qwen3MoeForCausalLM`，`._transformer_weights_path` = stacked safetensors）实例化**真实的 `VeOmniBackend`**，
完整跑 `VeOmniBackend.__init__`（`init_parallel_state(ep)` → `veomni_parallelize` → `_attach_extra_parallel_param_groups`
→ `load_trainable_weights` 加载真实权重 → 建 optimizer），再用 `backend.zero_grad / loss.backward /
backend.optimizer_step(max_grad_norm)`（含 EP-aware clip）做真实训练步：

| ep_size | ep_enabled | ep_param_groups | 真实权重加载 | peak_alloc (GB) | median step (s) | loss step0→7 |
|--:|:--:|:--:|:--:|--:|--:|--:|
| 1 | False | False | ✅ (broadcast 路径) | 12.12 | 0.248 | 8.68 → 7.25 |
| 2 | True  | True  | ✅ (EP distribute_tensor) | **10.18** | 0.305 | 8.76 → 7.28 |
| 4 | True  | True  | ✅ (EP distribute_tensor) | **9.90** | 0.316 | 8.72 → 7.27 |

- **完整产品代码路径**：这次走的是真实 `VeOmniBackend` 类（非 §4.3 的手搓 harness），含**真实 checkpoint 加载**。
- **显存收益 −18%**（ep4），比 §4.3（随机初始化、未加载）更明显——加载后专家确实只在每 ep-rank 持有 16/64 个专家、
  并经 `ep_fsdp` FSDP 进一步切分。
- **数值一致**：loss 起点 ≈ ln(vocab=4096)=8.32（随机权重的正确交叉熵），三档 ep 的 loss 轨迹一致（同一份加载权重）
  → EP 与 FSDP 在真实权重上数值等价。
- loss `start≈8.7 → 7.25` 单调下降 = 真实训练（拟合随机标签），grad_norm 有限、clip/step 正常。

### 4.5 真实 GRPO 训练步 + EP（贯穿 UniRL 完整训练侧栈）

新增**正式 bundle** `unirl/models/qwen3_moe/`（`Qwen3MoeBundle`）：经 `veomni.build_foundation_model` 构建 VeOmni
`Qwen3MoeForCausalLM`（meta-init，自带 `get_parallel_plan` + fused MoE），stash stacked safetensors 路径，
并 stamp 一个 deferred op 在加载后重算非持久 RoPE `inv_freq`（`apply_deferred_ops` 自动 drain）。复用 dense Qwen3 的
`Qwen3ARStage`/`Qwen3ARConditions`（replay forward 与架构无关，只需 `.model`+`.lm_head`）。

`scripts/ep_verify/unirl_grpo_ep_real.py` 跑**真实 GRPO 训练侧一步**贯穿完整 UniRL 栈：
`Qwen3MoeBundle → 真实 VeOmniBackend(ep) → Qwen3ARStage(replay) → GRPO.compute_loss_and_backward
（= stage.replay 策略前向 + PPO clip 损失 + backward）→ backend.optimizer_step（EP-aware clip）`。
rollout/reward 用合成数据（随机 prompt/response/advantage —— EP 只影响训练 backend，不影响 reward 信号）；
`old_logp` 由一次 no-grad replay 播种，使 step-0 的 ratio 精确 = 1（干净的 GRPO ratio）。

| ep_size | step0 policy_loss | step0 ratio_mean | step0 grad_norm | peak_alloc (GB) | median step (s) |
|--:|--:|--:|--:|--:|--:|
| 1 | **−0.51628** | 1.0 | 6.875 | 7.80 | 0.131 |
| 4 | **−0.51628** | 1.0 | 6.867 | **4.99** | 0.128 |

- **数值精确等价**：ep4 与 ep1 的 step-0 `policy_loss` **逐位一致（−0.51628）**、`ratio_mean=1.0`、grad_norm 几乎一致
  （6.867 vs 6.875）→ EP 在**真实 GRPO 训练路径**上与 dense/FSDP 数学等价（`ratio=1` 也证实 EP 下 replay 的 log-prob 正确）。
- **显存 −36%**（4.99 vs 7.80GB；这里序列短 P+R=192，专家权重在工作集中占比更高，故相对收益更大）。
- 这是**最完整的"真实"验证**：真实 bundle + 真实 `VeOmniBackend` EP + 真实 `Qwen3ARStage` replay + 真实 `GRPO` 算法 +
  EP-aware `optimizer_step`，一步不少地跑通了 GRPO 的训练侧。

### 4.6 正式 recipe（接入 pipeline）

`examples/ar/qwen3_moe_grpo_30b_a3b_veomni_ep_sglang.yaml`：把 `Qwen3MoeBundle` 接入完整 GRPO recipe
（bundle / pipeline / backend / rollout / reward / algorithm / sync / stack / data_source / sampling），单开关
`backend.fsdp_cfg.ep_size: 8`。**`Qwen3Pipeline` 直接复用**（其 chat-template + AR stage 只依赖 `.transformer`/`.tokenizer`，
架构无关），无需 MoE 专属 pipeline。已用 hydra `compose` 校验：配置合法、所有 `_target_` 可导入解析、
`ep_size=8`、`block_class_names=["Qwen3MoeDecoderLayer"]`。`backend.py` 同时采纳了 **ep>1 才传 `extra_parallel_*`** 的写法
（见 §七 与 PR #140 的对比），使 ep_size=1 的其它 recipe 完全不受影响、也不依赖 pinned veomni 接受这些 kwarg。

> recipe 端到端长跑仍需两项前置（YAML 注释已写明）：① 真实 Qwen3-30B-A3B 的 **stacked-format** 权重（HF per-expert
> → 用 VeOmni `CheckpointTensorConverter` 合并）；② EP-aware 的权重同步（向 SGLang 推送 EP 分片专家需 gather + stacked→per-expert remap）。

---

## 五、改动文件与复现

**UniRL（branch `perf/pe-sd3-train-batching`，`git diff --stat`，+216 行 / 4 文件）**
```
unirl/train/backend/sharded_load.py   | 121 +  # EP get_slice 权重加载 (_load_state_dict_ep_sliced)
unirl/train/backend/veomni/backend.py |  26 +  # init_parallel_state 传 ep (ep>1 才传)
unirl/train/backend/veomni/wrap.py    |  57 +  # _attach_extra_parallel_param_groups
unirl/train/configs.py                |  12 +  # FSDPConfig.ep_size
```
新增 UniRL 模块（未跟踪）：`unirl/models/qwen3_moe/`（`Qwen3MoeBundle`，§4.5 的正式 EP-capable MoE bundle）、
`examples/ar/qwen3_moe_grpo_30b_a3b_veomni_ep_sglang.yaml`（§4.6 接入 pipeline 的 GRPO+EP recipe，已 hydra-compose 校验）。
新增脚本（未跟踪）：`scripts/ep_verify/unirl_ep_verify.py`（手搓 harness，§4.3）、
`unirl_ep_backend_real.py`（真实 `VeOmniBackend` 端到端，§4.4）、
`unirl_grpo_ep_real.py`（真实 GRPO 训练步 + EP，§4.5）、
`measure_ep_load.py`（加载内存 full-vs-sliced 实测，§七.1）。

**VeOmni（dev checkout，profiling 脚本，未改动 VeOmni 源码）**：`scripts/profile/ep/`
（`qwen3moe_ep_scaled.json`、`prep.py`、`ep_profile_train.py`、`run_sweep.sh`、`summarize.py`）。

**复现**：
```bash
# 0. 隔离 venv（不动 qwen35 主环境）
/data/miniconda3/envs/qwen35/bin/python -m venv --system-site-packages /root/ep_work/epvenv
/root/ep_work/epvenv/bin/pip install "transformers==5.9.0" torchdata
# 1. 物化随机 MoE 权重 + dummy 数据（CPU）
cd VeOmni && PYTHONPATH=$(pwd) /root/ep_work/epvenv/bin/python scripts/profile/ep/prep.py \
   scripts/profile/ep/qwen3moe_ep_scaled.json /root/ep_work/qwen3moe_scaled_weights /root/ep_work/dummy_text 4096
# 2.（务必先 neutralize /tmp/gpu_occupy.py 与 matmul 占位循环，测完还原）
# 3. VeOmni EP 扫描
EPS="1 2 4 8" MAXSTEPS=10 bash scripts/profile/ep/run_sweep.sh
# 4. UniRL backend EP 端到端验证
cd UniRL-pe-perf && PYTHONPATH=../VeOmni:. /root/ep_work/epvenv/bin/python -m torch.distributed.run \
  --nproc_per_node=8 scripts/ep_verify/unirl_ep_verify.py ../VeOmni/scripts/profile/ep/qwen3moe_ep_scaled.json 4 /root/ep_work/unirl_runs/ep4.json
```

---

## 六、结论与建议

1. **EP 的本质与作用**：用 token all-to-all 替换专家权重 all-gather；与 dense/FSDP **数学等价**；
   **正向收益是显存与可扩展性**（甜点 ep2–ep4），单机 NVLink/中等规模下吞吐反而下降。
   要把 EP 用对，应在**专家数大、单卡显存吃紧、或跨机**的真实 Qwen3.5-MoE（256 experts/A3B）上启用，
   并与 rollout 侧 EP（如 vLLM-Omni `enable_expert_parallel`）的并行度对齐以保 on-policy。

2. **UniRL 集成已具备，且真实 GRPO 训练侧已端到端跑通**：4 文件 +161 行让 `VeOmniBackend` 支持 `ep_size>1`，
   外加正式 bundle `unirl/models/qwen3_moe/`。已通过**完整 `VeOmniBackend.__init__` → 真实权重加载 → `optimizer_step`**（§4.4）
   以及**真实 GRPO 训练步**（§4.5：真实 bundle + backend EP + `Qwen3ARStage` replay + `GRPO` 算法 + EP-aware clip）端到端验证：
   ep4 与 ep1 的 step-0 policy_loss **逐位一致（−0.51628）**、ratio=1、显存 −18%~−36%。
   既有 `foreach=False` 优化器与 DCP checkpoint 天然兼容；过程中发现并修复了两个集成缺口：
   EP-aware 梯度裁剪的 `_extra_parallel_param_groups`、以及 EP 2D 复合 mesh 的权重加载。

3. **落地到生产 RL recipe 仍需（未在本次实现，明确声明）**：
   - 把 `Qwen3MoeBundle` 接入 UniRL 的 pipeline 注册 + recipe YAML（`backend.fsdp_cfg.ep_size`、
     `block_class_names=["Qwen3MoeDecoderLayer"]`、`moe_implementation=fused_triton`），并配真实 rollout（SGLang 服务 MoE）+ reward。
   - **HF 原版 split-format MoE checkpoint**（`experts.N.gate_proj`）：✅ **已接线**——EP 加载器
     (`_build_expert_block_from_split`) 按 VeOmni `CheckpointTensorConverter` 映射在加载时逐 rank 重建融合专家块，
     无需离线合并（bit-exact 已验证）。真实 Qwen3-30B-A3B HF checkpoint 可直接喂入（仅需本地下载，sharded_load 不拉 HF repo id）。
   - 权重同步（`TensorWeightSync`）对 EP 分片专家需 EP-aware gather；rollout 侧 EP 与 train 侧 EP 度对齐以保 on-policy。

**未验证项（如实声明）**：在随机初始化的中等 MoE（3.3B，stacked-format 真实 safetensors）上做了 step 级
（loss/grad/ratio/显存/吞吐/真实加载/真实 GRPO 损失反传）验证并走通了**真实 `VeOmniBackend` + 真实 GRPO 训练侧**；
未做真实 Qwen3.5-35B-A3B 权重下载、HF split-format converter、完整 GRPO RL（真实 rollout+reward）长跑与 reward 收敛对齐。

---

---

## 七、与上游 PR #140（HunyuanImage-3 EP）的实现对比

上游 [Tencent-Hunyuan/UniRL#140](https://github.com/Tencent-Hunyuan/UniRL/pull/140)
（`feat(hi3): veomni expert parallelism for HunyuanImage-3`）也给 UniRL 的 VeOmni backend 加 EP，但目标模型与训练范式
不同（80B/64-expert 的 HI3、**LoRA、冻结专家**）。本节逐项对比（本工作 = "本实现"）。该 PR **不在**当前 checkout
（`perf/pe-sd3-train-batching`）里，两者独立实现。

| 维度 | PR #140 (HI3) | 本实现 (Qwen3-MoE) |
|---|---|---|
| **训练范式** | **LoRA**，专家**冻结**；EP 仅是冻结专家的显存/算力优化，梯度只到非 EP 的 attention LoRA | **全参数**，专家**可训练**；梯度流经 EP 专家 |
| **目标模型/专家融合** | HI3 是 HF per-expert `ModuleList` → meta 期**手动 swap** 成 fused `FusedHunyuanMoE`（`[E,2I,H]`,Shard(0)），靠 `prepare_for_expert_parallel()` bundle hook | 直接用 VeOmni 原生 `qwen3_moe` patched model —— **本就 stacked + 自带 `get_parallel_plan`**，无需融合 swap |
| **单开关 / no-op** | `ep_size>1` 才启 EP；`init_parallel_state` 仅 ep>1 传 `extra_parallel_*` | 一致；**已采纳** PR 的 ep>1-才传 写法（§4.6） |
| **权重加载** | 绕过 DCP，**per-EP-group 广播**（rank0 仍 materialize 全量 ~80B 再广播；PR 自列 OOM 风险）| **`safetensors.get_slice` 每 rank 只读 `E/ep` 字节**（mmap 惰性，无任何 rank 持全量、无 broadcast、并行 I/O）→ `distribute_tensor`。**严格优于 PR 的广播**（见下）。两者都绕开 torch `set_model_state_dict(broadcast)` 在 EP 2D-mesh 的切片 bug |
| **root 参数 (wte/ln_f/lm_head)** | HI3 在 FSDP forward **外**调用 → 显式 all-gather hooks (`register_unsharded_param_hooks`) | `Qwen3ARStage` replay 把 lm_head 跑在 **root forward 内**（`_replay_aware_forward`）→ **无需** hook |
| **EP-aware 梯度裁剪** | 专家冻结 → 无专家梯度 → 不涉及（只裁 LoRA） | 专家可训练 → **必须** `_extra_parallel_param_groups` + VeOmni EP-aware clip（否则跨 mesh stack 崩溃）—— 本实现独有 |
| **权重同步到 rollout** | 只 sync attention LoRA（小），并修了 fused-qkv 的 **GQA de-interleave**；专家冻结不 sync | 全参 → 需 sync EP 分片专家到 SGLang（HF per-expert 命名）→ recipe 标注的 **OPEN ITEM**；PR 因冻结专家天然回避 |
| **附带修复** | FSDP2 下 activation-checkpointing 静默失效（class-name match）、fused-qkv LoRA 误切 | 未涉及（本实现 AC 同样按 class-name 匹配，未单测该 bug） |
| **结果** | e2e 单 rollout −29%，**train step >2× 快**（80B 冻结 MoE 在 FSDP 下全量 all-gather 是大头，EP 后大幅省） | 单机中等规模 EP **吞吐略慢**（all-to-all > gather 节省），**显存 −18%~−36%**；数值与 FSDP 逐位等价 |

**结论**：两套实现在"EP=单开关、专家 `Shard(0)`+all-to-all+grouped-GEMM、绕开 torch 广播加载 bug"上同源；核心差异来自
**范式**（PR=LoRA/冻结专家 → EP 是纯显存/吞吐优化、回避了专家梯度裁剪与专家权重同步；本实现=全参/可训练专家 → 多做了
EP-aware clip 与 EP 专家权重加载，并把专家权重同步列为待办）与**模型**（PR 自定义 HF 需手动融合专家；本实现借力 VeOmni 原生
`qwen3_moe`，零融合）。吞吐结论相反则源于规模：PR 的 80B 冻结 MoE 是 EP 吞吐受益的典型场景，本实现的中等可训练 MoE 在
单机 NVLink 上 all-to-all 延迟占主导（与 §二.3 一致）。

### 七.1 权重加载方案对比（已查证 + 实测）

加载 EP 分片专家权重的几种方案，按"任意 rank 峰值内存 / 通信 / 适配 VeOmni 预切布局"排序：

| 方案 | 任意 rank 峰值 host 内存 | 跨 rank 通信 | 适配 VeOmni `ep` 预切 |
|---|---|---|---|
| **① `get_slice` 每 rank 只读本块**（本实现，最优）| `E/ep` | **无** | ✅ 手动切 dim0 后 `distribute_tensor` |
| ② PR #140 per-EP-group 广播 | **rank0=全量**（80B OOM 风险）| broadcast | ✅ |
| ③ 每 rank 读全量 + distribute（本实现旧版）| **每 rank=全量**（8× host RAM）| 无 | ✅ |
| ④ DCP `HuggingFaceStorageReader` / `set_model_state_dict` | 低（offset 读）| 少 | ❌ global `[E]≠[E/ep]` 直接挂 |

- **实测**（3.3B，fp32 stacked safetensors，单 rank 读取量）：full **3315M elems** → `get_slice` **ep4 899M（−73%）/ ep8 496M（−85%）**。
  旧版还让**全部 8 rank 各读一份全量**（聚合 ~8×13GB host RAM），新版每 rank 只读自己的 `E/ep` 切片。
- **结论**：`get_slice`（①）严格优于 PR 的广播（②）—— 没有任何 rank materialize 全量、零 broadcast、各 rank 并行只读自己的专家字节，
  且 mmap 只 fault-in 所需页。**已采纳为本实现的 EP 加载器**（`_load_state_dict_ep_sliced`，correctness 与 full-read 逐位一致）。
- **查证依据**：safetensors `get_slice` "only the requested byte ranges are read from storage"
  （[safetensors docs](https://deepwiki.com/huggingface/safetensors/2.3-tensor-slicing)）；
  torch DCP 已原生支持 HF safetensors 的 offset 读
  （[PyTorch 博客](https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/)），
  但其按 global shape 匹配，无法直接吃 VeOmni 的 `[E/ep]` 预切布局（故 ④ 不适用，除非把专家改成真正的 3D DTensor）。

**仍可借鉴 PR 的一点**：root 参数（wte/ln_f/lm_head）若在 FSDP forward **之外**被调用，需补 all-gather hook
（`register_unsharded_param_hooks`）。本实现因 `Qwen3ARStage` replay 把 lm_head 跑在 root forward 内而天然回避；
未来若加 forward 外访问 root 参数的路径则需补。

---

*报告生成: 2026-06-29（§七 PR#140 对比 + §4.5/4.6 GRPO/recipe 于 2026-06-30 追加）｜ UniRL `perf/pe-sd3-train-batching` ｜ VeOmni dev checkout*
