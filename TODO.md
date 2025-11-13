# LightMem LoRA 集成计划与 TODO

## 目标

- 在不破坏现有检索增强（RAG）流程的前提下，引入 LoRA 适配器对开源指令模型进行增量训练，使模型在无检索时也能回答与用户记忆相关的问题（新知灌入）。
- 让模型学习 LightMem 的记忆更新策略（`update/delete/ignore` 的结构化决策），提高离线更新的一致性与稳定性。
- 以“本地 HuggingFace 模型 + LoRA 适配器”替换/新增 `MemoryManager` 推理后端，支持 `meta_text_extract` 与 `_call_update_llm` 两类调用。

## 可行性论证要点

- 数据可得性：
  - LightMem 已产出 `MemoryEntry`（事实）与离线更新结果（最终决策），可直接转为监督训练样本。
  - 评估脚本（如 `run_lightmem_gpt.py`）提供问答场景，可生成“事实问答”与“冲突更新”样本对。

- 训练成本与资源：
  - 7B 量级基座模型使用 QLoRA（4-bit）在消费级显卡上即可训练，`r=16~32`、`alpha=32~64`、`lr=2e-5~5e-5` 即可见效。
  - 采用梯度检查点与较短序列，能进一步降低显存压力。

- 效果预期：
  - 短期内显著提升“无检索”的事实问答准确率（新知内化）。
  - 更新决策输出更稳定，JSON 解析成功率提高，减少回退与手工修复。

- 风险与缓解：
  - 灾难性遗忘：混入少量通用指令数据；小学习率与早停；必要时适配器叠加（stacking）。
  - 结构化输出稳定性：在提示中强化 JSON schema 示例；推理侧做正则清洗与重试。
  - 许可与隐私：脱敏个人数据，遵循模型与数据的许可条款。

## 技术路线总览

1. 数据构建（JSONL）：
   - 事实灌入（SFT）：从 `MemoryEntry.memory` 自动生成若干问答样本，覆盖同义变体与负样本（过时事实）。
   - 更新决策：重放 `UPDATE_PROMPT` 的输入，把离线更新的最终结果作为监督 `assistant` 输出（结构化 JSON）。

2. 训练（QLoRA）：
   - 模型：`Qwen2.5-7B-Instruct` / `Llama-3.1-8B-Instruct` / `Mistral-7B-Instruct`。
   - LoRA 设定：`r=16/32`、`alpha=32/64`、`dropout=0.05`，模块覆盖 `q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj`。
   - 优化：`adamw`、`cosine`、`warmup_ratio=0.05`、`gradient_checkpointing=True`、`max_seq_len=2048~4096`。

3. 推理集成（新增后端）：
   - 在 `MemoryManagerFactory` 增加 `local_hf_lora`，加载基座模型 + LoRA 适配器，实现：
     - `meta_text_extract(messages)`：批量元数据抽取。
     - `_call_update_llm(system_prompt, user_prompt)`：返回结构化 JSON 更新决策。
   - 生成参数：`temperature=0.2`、`top_p=0.9`，`do_sample=False` 以提升格式稳定性。

## 环境准备（Windows）

建议使用虚拟环境管理依赖：

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install torch transformers peft trl bitsandbytes datasets accelerate jsonlines
```

如遇 `bitsandbytes` 与显卡/驱动兼容问题，可退回 CPU 或使用 `bnb` 的替代配置。

## 数据与样本格式

- 事实灌入（SFT）样本结构：

```json
{
  "text": "<s>[SYSTEM] 你是一个事实问答助手。\n[USER] 根据已知事实回答问题：\n- 2024-07-01 周一 张三的工作是工程师。\n- 2024-09-12 周四 张三搬到上海。\n问：张三现在住在哪里？\n[/USER]\n[ASSISTANT] 张三住在上海。</s>"
}
```

- 更新决策样本结构（简化）：

```json
{
  "text": "<s>[SYSTEM] 请根据候选记忆对目标记忆进行更新/删除/忽略，并返回 JSON。\n[USER] 目标: 2024-07-01 张三住在北京。\n候选: 2024-09-12 张三搬到上海。\n要求: 仅输出 JSON。\n[/USER]\n[ASSISTANT] {\n  \"action\": \"update\",\n  \"updated_text\": \"2024-09-12 张三住在上海。\"\n}</s>"
}
```

> 训练目标可形式化为最小化条件交叉熵：$L = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)$。

## 训练与导出

- 最小训练脚本建议：使用 `trl.SFTTrainer` 对上述 JSONL 数据进行 SFT 训练，保存 LoRA 适配器至 `./adapters/lightmem_mem/`。
- 关键超参范围：`lr=2e-5~5e-5`、`epochs=1~3`、`batch_size=1~4`（视显存而定）。

## 推理集成与配置示例

- 在 `BaseMemoryConfigs.memory_manager` 配置：

```json
{
  "memory_manager": {
    "model_name": "local_hf_lora",
    "config": {
      "base_model": "Qwen/Qwen2.5-7B-Instruct",
      "adapter_path": "./adapters/lightmem_mem",
      "device_map": "auto",
      "torch_dtype": "bfloat16",
      "max_seq_len": 4096,
      "gen_kwargs": {"temperature": 0.2, "top_p": 0.9}
    }
  }
}
```

- 评估可分别测试“关闭检索（检验新知内化）”与“打开检索（检索增强）”两种设置。

## 风险与缓解

- 灾难性遗忘：混入少量通用数据；降低学习率；早停；必要时 Adapter stacking。
- 结构化输出：提示中加入 JSON schema 示例；推理端做 JSON 修复与重试；设置较低温度与 `do_sample=False`。
- 显存与性能：降低 `r`、缩短序列、选小模型；使用梯度检查点。
- 合规与隐私：数据脱敏；遵循模型与数据许可。

## 分阶段 TODO

### 阶段 1：数据管道

- 从 `MemoryEntry` 与离线更新日志构建两类 JSONL 数据集（事实灌入、更新决策）。
- 做基础清洗、去重与一致性修正（时间戳优先最新事实）。
- 划分训练/验证集并统计规模与覆盖率。

### 阶段 2：QLoRA 训练

- 选择 7B 基座模型并配置 QLoRA 超参（`r`、`alpha`、`dropout`、目标模块）。
- 使用 `trl.SFTTrainer` 训练 1–3 epoch，记录验证集准确率与损失。
- 导出 LoRA 适配器到 `./adapters/lightmem_mem/`。

### 阶段 3：推理集成

- 在 `MemoryManagerFactory` 新增 `local_hf_lora` 后端，实现 `meta_text_extract` 与 `_call_update_llm`。
- 替换配置并在 `run_lightmem_*` 脚本上做小规模验收（含 JSON 解析稳定性测试）。

### 阶段 4：评估与调参

- 在 `longmemeval_s.json` 场景对比 LoRA 前后与是否检索增强的差异。
- 调整温度、`top_p`、序列长度与 LoRA 超参，平衡准确率与时延。

### 阶段 5：文档与整理

- 更新 README 与实验记录，说明环境、数据、训练、评估与结果。
- 输出基准对比与建议默认配置，便于开箱即用。

---

如需，我可以进一步提供：

- 最小 QLoRA 训练脚本（Python，遵循 snake_case 与 NumPy 注释风格）。
- `local_hf_lora` 后端代码骨架，与 `openai.py` 方法签名保持一致。
- 数据构建脚本，把 `MemoryEntry` 转成标准 SFT 数据集以直接训练。