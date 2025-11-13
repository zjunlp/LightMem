# 📊 Results Summary（reports/summary.json）

统一入口或单脚本运行结束后，会生成 `reports/summary.json`，字段说明如下。

## 字段说明

- `total_samples`：样本总数
- `correct_count`：预测正确样本数
- `accuracy`：准确率（`correct_count / total_samples`）
- `avg_construction_time`：平均构建耗时（秒）
- `total_vectors`：向量总数（需要安装并启用 Qdrant）
- `avg_vectors_per_collection`：每集合平均向量数
- `vector_counts`：各 `question_id` 的向量数量
- `top_collections_by_vectors`：按向量数排序的 Top10 集合
- `source`：汇总输入与输出路径

## 示例

```json
{
  "total_samples": 384,
  "correct_count": 279,
  "accuracy": 0.7266,
  "avg_construction_time": 5.406,
  "total_vectors": 0,
  "avg_vectors_per_collection": 0.0,
  "vector_counts": {"<question_id>": 0},
  "top_collections_by_vectors": [{"question_id": "<qid>", "vector_count": 0}],
  "source": {"results_dir": "../results", "qdrant_dir": "./qdrant_data"}
}
```
