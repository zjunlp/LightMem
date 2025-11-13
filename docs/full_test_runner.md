# 🧪 Full Test Runner（统一入口）

统一“全量测试”入口脚本位于 `scripts/run_full_tests.py`，支持交互选择或通过参数指定测设，结束后自动生成并打印汇总（`reports/summary.json`）。

## 使用

```powershell
# 交互式选择测设
python scripts/run_full_tests.py

# 指定测设（qwen / gpt / both / summarize_only）
python scripts/run_full_tests.py --target qwen
python scripts/run_full_tests.py --target gpt
python scripts/run_full_tests.py --target both
python scripts/run_full_tests.py --target summarize_only
```

```bash
# Bash 示例
python scripts/run_full_tests.py --target both
```

## 行为说明

- 脚本将调用对应的实验脚本（`experiments/run_lightmem_qwen.py`、`experiments/run_lightmem_gpt.py`）。
- 两个实验脚本均把结果写入 `../results/result_<question_id>.json`，并在结束时自动触发汇总；统一入口脚本结束时也会再次汇总以确保最终统计同步。
- 输出位置：`reports/summary.json`。
 - 统一入口在运行前会加载仓库根目录的 `.env` 并注入环境变量。

## 常见问题

- 汇总为空：当 `results` 目录没有任何 `result_*.json` 时，`reports/summary.json` 的统计为 0。请先运行 `--target qwen` 或 `--target gpt` 再执行 `--target summarize_only`。
- 向量统计为 0：未安装 `qdrant-client` 或未启用向量检索时，`vector_counts` 为 0 属于正常现象；启用后将统计各集合向量数。
