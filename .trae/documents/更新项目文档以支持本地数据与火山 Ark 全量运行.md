## 文档范围

* 更新 `README.md`，补充端到端运行说明（最小与全量）、数据格式、转换脚本与依赖环境、火山 Ark API 配置、CPU/无CUDA运行、Qdrant 存储与结果说明、常见问题与性能建议。

* 原则上不新增多余文档，除非有可优化和分离的需要；所有说明集中在 `README.md`，并在必要处加入到具体代码位置的跳转提示（文件:行）。

## 目录结构与总览

* 介绍关键入口与模块：

  * 实验脚本：`experiments/run_lightmem_qwen.py`（运行入口、参数、配置位置：10-20、98-146、196-230）

  * 管线核心：`src/lightmem/memory/lightmem.py`（构建与索引：186-349、368-394、409-436）

  * 向量库适配：`src/lightmem/factory/retriever/embeddingretriever/qdrant.py`（集合与落盘：52-83；检索返回：146-153）

  * 文本嵌入器：`src/lightmem/factory/text_embedder/huggingface.py:13-16,31-35`

  * 火山 Ark/OpenAI 管理器：`src/lightmem/factory/memory_manager/openai.py:25-41,93-123`

  * 令牌编码映射：`src/lightmem/memory/utils.py:104-116`

## 环境与依赖

* Python 版本与虚拟环境：`>=3.10,<3.12`（`pyproject.toml:10`）与建议使用 `Python 3.9` 兼容 Ark SDK 与依赖。

* 依赖清单与安装命令（含国内镜像示例）：

  * 最小依赖：`openai==2.3.0`、`httpx==0.28.1`、`tiktoken==0.12.0`、`numpy>=2.0.2`

  * 完整管线：`torch==2.8.0`、`transformers==4.57.0`、`sentence-transformers==2.6.1`、`qdrant-client==1.15.1`、`llmlingua==0.2.2`

  * 示例命令与镜像：`py -3.9 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <packages>`

* 暂时无 CUDA 运行：将设备统一设置为 `cpu`（`experiments/run_lightmem_qwen.py:104-107,132-135`）。

## 火山 Ark API 配置

* 必需环境变量或脚本内配置：`API_KEY`、`API_BASE_URL`（`experiments/run_lightmem_qwen.py:10-11`）。

* 模型名示例：`deepseek-v3-1-250821`（LLM）与 `deepseek-r1-250528`（Judge）；支持通过 `OpenAI` 兼容接口直连（`openai.py:33-41`）。

* 常见错误与排查：鉴权失败、模型未开通、Endpoint 404（提供错误范例与解决建议）。

## 数据格式（LongMemEval）

* 字段定义与校验要求：

  * `question_id`、`question`、`answer`、`question_type`、`question_date`

  * `haystack_sessions`：每个会话由成对的 `{role:"user"|"assistant", content, sequence_number}` 组成（`experiments/run_lightmem_qwen.py:171-189`）

  * `haystack_dates`：与 `haystack_sessions` 等长（`experiments/run_lightmem_qwen.py:166-168`）；日期推荐 ISO 字符串

* JSON 最小示例与多会话示例（含正确的序号与日期对应）。

## 数据转换脚本与用法

* 脚本位置：`scripts/convert_conversations_to_longmemeval.py`

* 输入/输出：

  * 输入：`user.json`、`conversations.json`（默认读取项目根路径）；输出：`data/longmemeval_converted.json`

* 转换规则：

  * 从 `mapping` 链提取顺序消息；用户消息取 `fragments.type=="REQUEST"`；助手消息排除 `REQUEST/THINK`，缺失则为空字符串补齐成对结构

  * `question` 取 `title` 或首条用户消息；`answer` 取最后一条助手消息；`question_date` 取 `inserted_at/updated_at`

* 运行命令与示例输出（首行片段查看方式）。

## 快速运行（最小与全量）

* 最小数据集：`data/longmemeval_s_min.json`（用于验证）

* 全量数据：将 `DATA_PATH` 指向 `data/longmemeval_converted.json` 或你的 LongMemEval 文件（`experiments/run_lightmem_qwen.py:20`）

* 命令：

  * `Set-Location <repo>`；`$env:PYTHONPATH = (Join-Path $PWD 'src')`；`py -3.9 experiments\run_lightmem_qwen.py`

## 检索策略与存储

* 向量检索：`index_strategy='embedding'`、`retrieve_strategy='embedding'`

* 文本嵌入模型：`sentence-transformers/all-MiniLM-L6-v2`（维度 `384`）与设备 `cpu`

* Qdrant：集合名为 `question_id`；路径 `QDRANT_DATA_DIR/<question_id>`；建议 `on_disk=True`（`experiments/run_lightmem_qwen.py:139-145`，`qdrant.py:82`）

## 输出结果说明

* 每样本结果文件：`../results/result_<question_id>.json`（`experiments/run_lightmem_qwen.py:227-230`）

* 字段：`construction_time`、`generated_answer`、`ground_truth`、`correct`、`results`（含中间提示）

* 如何查看与汇总：提供 PowerShell/脚本示例汇总正确率与耗时均值

## 资源与性能建议

* 资源范围（代码支撑）：

  * 轻量默认：GPU 2–5 GB、CPU 2–6 GB（参考 `llmlingua_2.py:17-21`、`topic_segmenter.llmlingua_2.py:49-55`、`huggingface.py:13-16`）

  * 重量场景：GPU 8–24 GB、CPU 4–12 GB（大模型与高并行）

* 并行与稳定性：控制 `num_workers`；避免 `get_all(with_vectors=True)` 全量载入；网络下载失败的重试与镜像建议

## 常见问题（FAQ）

* HF 权重下载失败：网络/镜像/离线缓存处理（错误栈与解决）

* tiktoken 编码映射缺失：加入模型名映射（`utils.py:104-116`）

* Windows 编码问题：读取 JSON 使用 `encoding='utf-8'`（`experiments/run_lightmem_qwen.py:146`）

* Ark 模型不可用：检查模型开通、Endpoint、Key 权限

## 修改摘要（便于旁观者快速理解）

* 实验脚本：已支持 Ark 模型、CPU 设备与向量检索；`DATA_PATH` 可指向转换后的全量数据

* 转换脚本：从对话映射构造 LongMemEval 样本，保证成对消息与日期字段

* 管线兼容：增加 `deepseek-*` 的令牌编码映射，避免运行时错误

确认后我将：

1. 按上述章节结构更新 `README.md`（新增和强化对应说明与示例命令）
2. 在转换脚本顶部添加简短使用说明注释（不改变逻辑）
3. 验证渲染与目录跳转正确性，并提交改动供你审阅

