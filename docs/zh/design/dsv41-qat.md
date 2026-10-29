# DSV4.1 QAT：按原始量化格式做 rollout 的训推对齐

## 1. 目标

DeepSeek-V4.1-Flash 的发布 checkpoint 本身就是量化格式：dense 层是 FP8 e4m3，每 32x32 一个 ue8m0 scale；routed experts 是 MXFP4，每 32 个输入元素一个 ue8m0 scale。我们希望 SGLang rollout 直接用这个原始精度来跑，换取推理吞吐和显存。

问题在于 Megatron 用 bf16 训练，rollout 却用量化权重和量化激活，两侧数值不一致，k3 会明显变大。QAT 的做法是让训练前向看到的权重和激活，与 rollout 实际执行的数值一致：

- 权重：训练用的权重就是 rollout 收到的那份量化权重，逐位相同。
- 激活：凡是 rollout 会做激活量化的 linear，训练侧都用同一个 kernel 做 fake-quant。

整套方案有四个约束：

- 训练和 rollout 都直接读原始 checkpoint，不需要再转一份 bf16。
- 量化格式从 checkpoint 的 safetensors header 里推断，代码里不写死层名。模型相关的差异放在一个小 adapter 里。
- 一个开关 `trainer.qat`，只控制训练侧。发给 rollout 的权重是否需要编码，由 checkpoint 是否量化自动决定。
- `save_hf` 导出的 HF 模型与原始 checkpoint 格式完全一致，可以直接替换原 checkpoint 使用。

当前不训练 MTP，也不支持 vision。

## 2. 数据流

```
原始 ckpt (FP8/MXFP4 + .scale, wrapped text/vision config)
  │
  ├─ Megatron-Bridge import ── 配置展平为纯文本模型、去掉 MTP，权重反量化为 bf16
  │     └─ Megatron 训练 (bf16 params, fp32 master)
  │          ├─ [qat] 每次 optimizer.step 后：量化参数 snap 到网格上   (权重 QAT)
  │          ├─ [qat] W8A8 linear 的输入加 forward pre-hook fake-quant (激活 QAT)
  │          ├─ update_weights：bridge 导出 bf16 → QuantPlan.encode → (q, scale) → TransferMesh
  │          └─ save_hf：bridge 导出 → encode → save_like_source(按源 ckpt 逐 shard 重写)
  │
  └─ SGLang 从同一目录启动 (vision_n_layers=0)，使用 ckpt 自带的 quantization_config
        └─ 热更新：dense FP8 走原生 weight_loader；MXFP4 experts 用 patch 做原地 interleave
```

## 3. 原理

### 3.1 量化格式与识别

格式定义在 [coda/utils/quant_formats.py](../../../coda/utils/quant_formats.py)。`QuantScheme(kind, block)` 中，`block` 表示一个 scale 覆盖的元素范围（行数, 列数）。

| kind | 权重存储 | scale | V4.1 中对应的层 |
|---|---|---|---|
| `fp8_block` | e4m3 `[O, I]` | e8m0 `[O/br, I/bc]` | attn wq_a/wq_b/wkv/wo_a/wo_b、indexer.wq_b、shared_experts w1/w2/w3、engram.wkv，均为 32x32；engram embed 表为 1x32 |
| `mxfp4` | int8 `[O, I/2]`，每字节两个 e2m1，低 nibble 存偶数列 | e8m0 `[O, I/32]` | routed experts w1/w2/w3 |

量化公式：`s = 2^ceil(log2(amax / qmax))`，FP8 的 qmax 取 448，FP4 取 6；之后用 RNE（round to nearest even）落到网格上。用这个量化器去量化 bf16 checkpoint，反量化值与原始 checkpoint 逐位相同。字节层面有少量差异：约 5% 的 fp8 块 scale 取了 s/2，fp4 会出现 -0 编码，但反量化后的数值一样。

`read_ckpt_quant(path)` 只读 safetensors header，从每一对 `X.weight` / `X.scale` 的 dtype 和 shape 推断格式，完整模型约 0.6 s：

- weight 是 `F8_E4M3` → `fp8_block`，tile = `(O/sO, I/sI)`；
- weight 是 `I8`/`U8` 且 `sO == O` → `mxfp4`，group = `2I/sI`；
- scale 不是 `F8_E8M0` 时直接报错。只有 2 的幂 scale 才能保证 snap 后的值在 bf16 中精确表示。

关键性质：使用 ue8m0 scale 时，`dequant(quant(w))` 在 bf16 中可以精确表示，而且是量化器的不动点。把它再量化一次，得到的数值完全相同。这一点是整个 QAT 成立的前提，后面几节都依赖它。

### 3.2 Megatron 参数与 checkpoint tensor 的对应

`QuantPlan`（[coda/backends/megatron/qat.py](../../../coda/backends/megatron/qat.py)）不按层名匹配，而是遍历 `bridge.get_conversion_tasks(model)`。每个 task 给出 Megatron 参数对应的 HF 名字（mapping.hf_param），再拿这个名字去 `read_ckpt_quant` 的结果里查表：

- 查得到：这个 tensor 在 checkpoint 中是量化存储的，标记 `quantized=True`，导出时需要编码。
- 如果同时开了 `qat`，还会把这个参数加入 `snap_params`，并在需要时把对应模块加入 `act_modules`（见 3.4）。

TP/EP 分片的正确性由 `_local_layout_ok` 保证：只有当本地分片是完整 HF tensor 的连续切片，并且切在 tile 边界上（按 dim-0 或 dim-1 切，GatedMLP 是 n 段 dim-0 拼接），对本地分片做 snap 才等价于对完整 tensor 做 snap。不满足时直接报错，比如 head 交错排布的 QKV，或者 Fused expert mapping。这类情况需要为该模型写 adapter。V4.1 在 TP4/8、EP 切分下全部满足条件。

### 3.3 权重 QAT：原地 snap

开启 `trainer.qat` 后，以下时机会对 `snap_params` 原地执行 `w.copy_(dequant(quant(w)))`：

1. 训练 worker 初始化，checkpoint 加载完成后，在任何 TensorBackuper 备份之前；
2. ref 模型加载完成后；
3. `train_minibatch` 中每次 `optimizer.step()` 成功之后（[model.py](../../../coda/backends/megatron/model.py) 的 `qat_plan.snap_()`）。

原理：

- fp32 master 权重不做 snap，持续累积未量化的更新；bf16 模型参数每步都被覆盖成 Q(W)。所以前向看到的是 Q(W)，梯度作用在 master 上，效果等价于直通估计（STE），前向没有额外开销。
- 不在前向里替换权重，是因为 TE linear 开启 `gradient_accumulation_fusion` 后，`main_grad` 绑定在具体的 param 对象上，替换 param 会破坏梯度累积。原地 `copy_` 不改变对象。
- 由于不动点性质，导出时 `encode` 再量化一次，得到的就是训练前向用的那份权重。rollout 收到的权重与训练权重逐位一致。

### 3.4 激活 QAT：使用 SGLang 自己的 kernel 做 fake-quant

checkpoint 里只有 `activation_scheme: dynamic`，不包含激活 scale，所以激活无法和 checkpoint 对齐，对齐的对象是 SGLang 运行时的行为。SGLang 在 dense FP8 GEMM 之前会对输入做量化：每个 token、沿 K 维每 `weight_block_size[1]` 个元素一组、scale 为 ue8m0。训练侧在对应 linear 上注册 forward pre-hook（`_FakeQuantActFP8`）：

- 前向：`q, s = sglang_per_token_group_quant_fp8(x, group, scale_ue8m0=True)`，再反量化回 bf16。调用的就是 SGLang 的同一个函数。group 取权重 tile 的列宽 `scheme.block[1]`，V4.1 为 32。
- 反向：梯度直接透传（STE）。
- 反量化回 bf16 不损失精度：e4m3 只有 3 位尾数，bf16 有 7 位，scale 又是 2 的幂。所以训练 GEMM 的输入在数值上就等于 SGLang 量化后的激活。

哪些层挂 hook：

- 默认规则：`fp8_block` 格式的权重对应的 linear 挂 hook，MXFP4 experts 不挂（按 W4A16 处理）。
- `DeepseekV41Adapter` 例外：`attn.wo_a` 不挂。SGLang 在加载时把 wo_a 反量化成 bf16，它的输入也是 bf16。
- V4.1 最终挂上 hook 的层：wq_a、wq_b、wkv、wo_b、indexer.wq_b、engram.wkv、shared_experts fc1/fc2。
- 不量化的层（compressor、gate、indexer wk/weights_proj、embed、head）两侧都是 bf16。KV cache 和 indexer 的 fp8 fake-quant 由已有的 kvnorm 逻辑处理，不属于本方案。

分片下的正确性：

- column-parallel + SP 时，hook 量化的是 all-gather 之前的序列分片。量化按 token 独立进行，先量化后 gather 与先 gather 后量化结果相同。
- row-parallel 时，输入是 K 维的本地分片。`_local_layout_ok` 已保证分片宽度是 32 的倍数，量化分组不会跨 TP 边界。
- norm 与 linear 融合的模块（带 `layer_norm_weight`）会直接报错，因为它的输入是 norm 之前的值。

对齐到什么程度：两侧使用的量化函数相同，但 GEMM 结果不会逐位相同。SGLang 的 FP8 kernel 以 fp32 累加 `q_x·q_w·s_x·s_w`，Megatron 用反量化后的 bf16 做 GEMM，两者累加顺序和 kernel 不同；上游层的数值也存在差异。

**临时开关 `CODA_QAT_DISABLE_ACT`**。实测在 DeepSeek-V4.1 上，激活 QAT 会让 `train/is_approx_k3_kl` 变大：H20 完整模型约 +25%，B200 toy 约 +27%。原因是训推两侧的激活差异逐层累积，超过约 1% 后，fake-quant 反而多出一份量化误差。在查清差异来源之前，可以在 `trainer.env_vars` 中设置 `CODA_QAT_DISABLE_ACT: "1"`，只保留权重 QAT。值必须写成字符串，否则 Ray 的 runtime_env 会报类型错误。启动日志会打印 `[QAT] activation QAT disabled by CODA_QAT_DISABLE_ACT=1, weight QAT only`。这个开关是临时的，找到原因后会删除。

### 3.5 各硬件上 rollout 的实际数值路径

| | H20 (SM90) | B200 (SM100) |
|---|---|---|
| dense FP8 32x32 | `Fp8LinearMethod`，Triton block-FP8 GEMM，W8A8 | SGLang 自动选择 `flashinfer_cutedsl`，以 MXFP8 方式执行（32x32 scale 无损映射为逐行 e8m0）。MXFP8 激活量化与 QAT hook 的输出逐字节相同 |
| wo_a | 加载时反量化为 bf16 | 同 H20 |
| routed experts MXFP4 | `Mxfp4FlashinferCutlassMoE`，W4A16 | `Mxfp4FlashinferTrtllmMoE`。默认是 W4A8，必须设置 `flashinfer_mxfp4_moe_precision=bf16` 才是 W4A16 |
| shared experts | dense FP8 linear（`disable_shared_experts_fusion`），TP1（2304/16 无法按 32 对齐） | 同 H20 |

B200 上 dense 层没有使用 Triton。原因：单个 GEMM 慢 5–15 倍；另外整模型以 triton 后端启动时会崩溃。

### 3.6 权重同步编码

`update_weights` 中，只要 `QuantPlan.quantized` 为真（即 checkpoint 是量化格式），就执行 `hf_weights = self._quant.encode(hf_weights)`：bridge 导出的 bf16 tensor 如果在 checkpoint 中是量化存储的，就转成 `(name, q)` 和 `(scale_name, s)` 两项依次输出；其他 tensor 原样输出。这一步和 `trainer.qat` 无关，所以 `qat=false` 时得到的就是“FP8/FP4 rollout、不做 QAT”的 baseline。如果 checkpoint 是 bf16，plan 为空，走普通 bf16 同步。

TransferMesh 以原始字节传输 fp8/e8m0/int8，不需要额外处理。SGLang 端的接收方式：

- dense FP8：使用原生 `weight_loader`。SM90 上 32x32 格式的 `process_weights_after_loading` 不做任何操作，所以热更新不需要修改。SM100 的 MXFP8 需要 patch 刷新派生的 swizzled scale，见第 5 节。
- MXFP4 experts：原生实现在加载后会做 interleave 并替换 Parameter，`weight_loader` 随之丢失，因此用 patch 改成按 expert 原地更新，见第 5 节。

### 3.7 save_hf：与原始量化格式保持一致

当 checkpoint 为量化格式时，`save_model(hf=...)` 调用 `_save_hf_like_source`，最终执行 [coda/utils/hf_ckpt.py](../../../coda/utils/hf_ckpt.py) 中的 `save_like_source`：

- 计算 expected 集合：对所有 rank 上 conversion task 的 HF 名字做 `all_gather_object` 后取并集（EP 下各 rank 持有不同的 experts），再加上对应的 `.scale`。
- rank 0 消费 `encode(bridge.export_hf_weights(model))`，其他 rank 只负责参与导出所需的集合通信。
- 按源 checkpoint 逐个 shard 重写：header 字节原样复制；训练过的 tensor 写入新数据，dtype 和 shape 必须与源一致，float 会 cast 为源 dtype，量化 tensor 的 dtype 不一致时报错；其余 tensor（vision、aligner、MTP、`gate.bias_vl`、engram 表等）按 256 MB 分块从源文件复制。每个 shard 先写 `.tmp`，再 `os.replace`。
- 非权重文件（config、tokenizer、index、encoding 代码等）原样复制。
- 出现未知 tensor、重复 tensor 或缺失 tensor 时直接报错。

结果目录可以直接替换原 checkpoint：shard 文件名、tensor 名、dtype、shape、config 都相同。

## 4. 模型适配：QATAdapter

无法从 checkpoint 中读出的 rollout 数值行为，放在 adapter 里，按 `config.architectures` 选择：

```python
class QATAdapter:
    def act_quant(self, hf_name, scheme) -> bool: ...   # 默认：fp8_block 的 linear 做激活量化
    def validate(self, config) -> None: ...             # 拒绝 QAT 未建模的 rollout 配置

_ADAPTERS = {"DeepseekV41ForCausalLM": DeepseekV41Adapter()}
```

`DeepseekV41Adapter`：

- `act_quant`：排除 `attn.wo_a`。
- `validate`：开启 `qat` 时检查以下配置，不满足则报错：rollout 后端是 sglang；`flashinfer_mxfp4_moe_precision=bf16`；`SGLANG_OPT_FP8_WO_A_GEMM=0`；`disable_shared_experts_fusion=true`。各项的原因见 6.3。

接入新模型时：如果分片布局满足 `_local_layout_ok`，并且 rollout 行为符合默认规则（FP8 做 W8A8，FP4 做 W4A16），不需要写代码。否则在 `_ADAPTERS` 中注册一个 adapter。

## 5. 依赖的补丁

Dockerfile 会自动应用这两个补丁：

**`docker/patch/dsv41/megatron-bridge.patch`**

- `DeepseekV41Config`：把 wrapped schema（`text_config`/`vision_config`）展平成纯文本配置，丢弃 vision，强制 `num_nextn_predict_layers=0`，并截断 `compress_ratios`。展平后与 bf16 目录的 config 完全一致（`quantization_config` 除外）。
- `maybe_modify_loaded_hf_weight`：FP8 的块大小改为从 `quantization_config.weight_block_size` 读取。原代码把块大小写死为 128，导入 V4.1 的 32x32 时结果是错的，而且不报错，因为切片后 shape 恰好能对上。engram 表的 scale 是逐行的，走原来的通用路径。导入结果与 bf16 checkpoint 逐位一致。
- `maybe_modify_converted_hf_weight`：导出纯 bf16，量化编码统一交给 coda 的 `encode`。

**`docker/patch/sglang-0.5.21.patch`**（与 QAT 相关的部分）

- MXFP4 experts 热更新：在 `mxfp4_flashinfer_cutlass_moe.py`（SM90）和 `mxfp4_flashinfer_trtllm_moe.py`（SM100）中，按 expert 原地执行 interleave 并 `copy_` 到原有权重中，指针不变，兼容 CUDA graph。`deepseek_v4.py` 的 `load_weights` 增加检查，更新不完整时报错。
- SM100 MXFP8 dense：新增 `Fp8LinearMethod.refresh_block_fp8_mxfp8_scale`，在 DSV4 `post_load_weights` 中原地刷新 `weight_scale_inv_swizzled`。修复前热更新后这个派生 scale 还是旧值。
- 两个上游 bug，也影响普通 bf16 流程：
  - `_attn_sink_local` 在第一次前向时缓存，之后从不刷新。修复：在 `post_load_weights` 中调用 `refresh_attn_sink_local()`。
  - colocate 模式下，`_swiglu_limit_tensor` 等普通属性分配在 TMS weights 区域内，release/resume 之后变成垃圾值，k3 从第 2 步起由 0.03 涨到 2.5。修复：注册为 non-persistent buffer。

## 6. 用法

### 6.1 开关

`conf/default.yaml`：

```yaml
trainer:
  qat: false   # true：每步 snap 量化权重，并对 W8A8 linear 的输入做 fake-quant
```

| `hf_model_path` | `trainer.qat` | 行为 |
|---|---|---|
| bf16 checkpoint | false | 普通 bf16 训练和 rollout |
| bf16 checkpoint | true | 报错：checkpoint 不是量化格式 |
| 量化 checkpoint | false | rollout 跑 FP8/FP4，同步时自动编码，训练不做 QAT（baseline） |
| 量化 checkpoint | true | 完整 QAT |

### 6.2 配方（`conf/dsv41_flash/`）

```bash
bash examples/start.sh dsv41_flash/dapo_h20_6node \
  checkpoint_path=/root/ckpt/dsv41_dapo_qat \
  hf_model_path=/root/DeepSeek-V4.1-Flash-1017 \
  data_source.dataset.prompt_data_path=/root/dapo-math
```

`hf_model_path` 直接填写原始发布 checkpoint 的目录，不需要转成 bf16。

### 6.3 配方中必须保留的 rollout 设置

```yaml
rollout:
  env_vars:
    SGLANG_DSV4_FP4_EXPERTS: "1"
    SGLANG_SHARED_EXPERT_TP1: "1"
    SGLANG_OPT_FP8_WO_A_GEMM: "0"
    SGLANG_OPT_FUSE_WQA_WKV: "0"
  sglang_args:
    json_model_override_args: '{"vision_n_layers": 0}'
    moe_runner_backend: flashinfer_mxfp4
    flashinfer_mxfp4_moe_precision: bf16
    disable_shared_experts_fusion: true
trainer:
  qat: true
```

各项的原因（基于 SGLang 0.5.21 源码）：

| 设置 | QAT 是否必需 | 原因 |
|---|---|---|
| `SGLANG_DSV4_FP4_EXPERTS=1` | 是 | routed experts 按 checkpoint 的 MXFP4 格式运行 |
| `SGLANG_SHARED_EXPERT_TP1=1` | 是 | shared experts 的中间维度是 2304，TP16 时每卡 144，不是 32 的倍数，无法按 32x32 块切分；不切分则保持完整的块 |
| `json_model_override_args: {"vision_n_layers": 0}` | 是 | 以纯文本模型启动，否则 R3 拿到的 expert id 全为 0，见 7.2 |
| `moe_runner_backend: flashinfer_mxfp4` | 是 | patch 中的 MXFP4 原地热更新只实现了这个后端（SM90 cutlass、SM100 trtllm） |
| `flashinfer_mxfp4_moe_precision: bf16` | 是（B200） | 见下文 |
| `disable_shared_experts_fusion: true` | 是 | 不开时 shared experts 会并入 routed MoE 一起运行，数值路径变了；开启后作为 dense FP8 linear 运行，和训练侧的激活 hook 对应 |
| `SGLANG_OPT_FP8_WO_A_GEMM=0` | 当前版本不必需，防御性保留 | 见下文 |
| `SGLANG_OPT_FUSE_WQA_WKV=0` | 不必需 | 见下文 |

**`flashinfer_mxfp4_moe_precision: bf16`**。这个参数决定 MXFP4 experts 的激活精度，SGLang 默认值是 `default`，在两种硬件上含义不同：

- SM100（B200），`Mxfp4FlashinferTrtllmMoEMethod`：`default` 会先用 `_prepare_flashinfer_mxfp8_activations` 把 expert 输入量化成 MXFP8，也就是 W4A8；`bf16` 时直接传入 bf16 输入，是 W4A16（`moe_runner/flashinfer_trtllm.py` 中 `_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen`）。
- SM90（H20），`Mxfp4FlashinferCutlassMoEMethod`：只有 `fp8` 才启用 Humming W4A8；`default` 和 `bf16` 都是 W4A16。

训练侧没有给 experts 的输入挂激活 fake-quant，因为 expert 输入经过 token dispatch 和 grouped GEMM，hook 的位置和其他层不同，目前没有实现。所以 rollout 的 experts 必须是 W4A16，否则 rollout 多出一层激活量化误差，QAT 没有建模。adapter 统一要求 `bf16`，这样同一份配置在 H20 和 B200 上含义一致。代价是 B200 上 experts 不能用更快的 W4A8，并且会关闭只在 `default` 下生效的两个优化：routed experts 的输入预量化 stream（`deepseek_v2.py`）和 `fuse_finalize_all_reduce`。如果以后要支持 W4A8，需要在训练侧对 expert 输入加 MXFP8 fake-quant（per-token、每 32 一组、ue8m0）。

**`SGLANG_OPT_FP8_WO_A_GEMM=0`**。这个 env 默认值为 True，开启时 wo_a 的 absorb GEMM 走 DeepGEMM `fp8_einsum`，会对 wo_a 的输入做 FP8 量化。但在 0.5.21 中，`wo_a_fp8_gemm_enabled()` 还要求 `quant_config.weight_block_size == [128, 128]`，V4.1 的块大小是 `[32, 32]`，所以无论 env 取什么值，V4.1 的 wo_a 都会在加载时（包括每次热更新）由 `_dequant_fp8_wo_a_streaming` 反量化成 bf16，输入也是 bf16。当前设置成 0 是冗余的。保留并由 adapter 校验，是为了防止以后 SGLang 支持 32x32 的 FP8 wo_a GEMM 后，wo_a 的输入悄悄变成 FP8，而训练侧的 adapter 恰好把 wo_a 排除在激活 hook 之外。真到那时，需要改 adapter，给 wo_a 也挂 hook。

**`SGLANG_OPT_FUSE_WQA_WKV=0`**。这个 env 沿用自 bf16 配方（commit 989cc02 首次加入，没有记录原因），默认值为 True。开启时 SGLang 把 wq_a `[1280, 5120]` 和 wkv `[512, 5120]` 沿 dim 0 拼成一个 `wqkv_a` `[1792, 5120]`，weight 和 `weight_scale_inv` 都直接拼接。对 QAT 没有影响：

- 1280 和 512 都是 32 的倍数，拼接后 32x32 块的划分不变，每个块的 scale 与拆开时相同；
- 两个 linear 的输入是同一个 x，激活量化按 token 沿 K 维进行，与输出维度无关，结果不变；
- GEMM 的每一行输出只依赖对应的权重行。拼接后 kernel 可能选不同的 tile 配置，带来累加顺序上的差异，但这不属于 QAT 建模的量化误差；
- 热更新时，`load_weights` 会缓存 wq_a/wkv 及其 scale，凑齐一对才写入，结束时检查缓存为空。coda 的整个权重同步是一次 `load_weights(channel.get_iterator())` 调用，每层的 wq_a 和 wkv 都在同一次调用里，一定能凑齐。

B200 toy 上用原始 checkpoint 启动，开和关的输出逐位相同（token ids 和 logprobs）。开启后的热更新路径没有实际跑过。所以这个设置对 QAT 不是必需的，保留 0 也没有坏处；如果要去掉以换取一次 GEMM 的开销，建议先用 toy 跑一次同步对比。

不需要设置 `load_format: dummy`，也不需要注入 quantization_config：SGLang 直接加载原始 checkpoint，后续由权重同步覆盖。


## 7. 已知问题

- **vision 必须关闭**：用原始 config 启动时 `vision_n_layers>0`，SGLang MoE 改用 `vision_topk` 路由（带 VL 校正 bias `gate.bias_vl`）。这条路径不经过 routed-experts capturer，R3 拿到的 expert id 全为 0，Megatron all-to-all 报 "Split sizes doesn't match"。配方中用 `vision_n_layers: 0` 规避，规避后路由结果与 bf16 目录启动逐位相同。
- **不训练 MTP 和 vision**：`save_hf` 写出的 MTP 和 vision 权重是原始 checkpoint 中未改动的版本。
- **每次 save_hf 都写出完整 checkpoint**：包括未训练的部分，toy 约 100 GB，完整模型约 476 GB，请预留足够的磁盘空间。
- **未建模的 rollout 配置**：MXFP4 W4A8（B200 默认）、FP8 wo_a GEMM、shared experts fusion、dense trtllm 后端（会 reshuffle 权重，refresh 时直接报错）、EPLB。
- **激活 QAT 目前是负收益**：已逐层 dump 对比过 SGLang 实际的量化输入和 Megatron 的输入，51 个 hook 与 SGLang 一一对应，没有挂错。问题出在训推激活差异逐层累积。可用 `CODA_QAT_DISABLE_ACT=1` 临时关闭激活 QAT（见 3.4）。

## 8. 代码索引

| 文件 | 内容 |
|---|---|
| [coda/utils/quant_formats.py](../../../coda/utils/quant_formats.py) | 量化器、`QuantScheme`、`read_ckpt_quant` |
| [coda/backends/megatron/qat.py](../../../coda/backends/megatron/qat.py) | `QuantPlan`（snap、激活 hook、encode）、adapter |
| [coda/utils/hf_ckpt.py](../../../coda/utils/hf_ckpt.py) | `save_like_source` |
| [coda/backends/megatron/megatron_train_worker.py](../../../coda/backends/megatron/megatron_train_worker.py) | 构建 plan、ref 模型 snap、`update_weights` 编码、`_save_hf_like_source` |
| [coda/backends/megatron/model.py](../../../coda/backends/megatron/model.py) | `train_minibatch` 在 step 后 snap |
| [conf/dsv41_flash/](../../../conf/dsv41_flash/) | 配方 |
| `docker/patch/dsv41/megatron-bridge.patch`、`docker/patch/sglang-0.5.21.patch` | 依赖的补丁 |