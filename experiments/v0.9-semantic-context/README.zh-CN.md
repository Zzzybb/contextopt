# v0.9 自动语义上下文候选评测产物

本目录是 `versioned-v1+semantic` 可选上下文模式的完全离线 conformance 产物，不调用
任何模型 provider。

从仓库根目录重新生成：

```text
python -m contextopt semantic-context-eval \
  --repetitions 3 --budgets 128,256,512 \
  --output experiments/v0.9-semantic-context/report.json \
  --markdown experiments/v0.9-semantic-context/report.md \
  --html experiments/v0.9-semantic-context/report.html
```

fixture 包含两个带 scope 的 ACM/数学任务，以及两个可继承的 global 记忆。每个 cell
先对相同消息编译两次，再删除 live store、仅使用 receipt 里的 durable-memory snapshot
重编译。矩阵共 72 个 cell（4 个 policy × 3 个预算 × 2 个任务 × 3 次重复）。

当前提交报告的关键结果：

- `retrieved_recall_rate = 1.000`：每个标注目标都进入 lexical 候选集合；
- `selected_recall_rate = 0.542`：候选在 policy 和预算竞争后仍被保留的比例；
- `budget_compliant_rate = 1.000`、`deterministic_rate = 1.000`、`replayable_rate = 1.000`；
- 平均 retrieved/selected/evicted 候选数为 `3.000 / 1.167 / 1.833`；
- `model_calls = 0`，`failed_count = 0`。

这些是固定 fixture label 和 receipt 的工程指标，不是 embedding 质量、语义理解、模型
是否实际使用候选，也不是代码任务成功率。`candidate_tokens` 使用 ContextOpt 稳定的
provider-neutral 估算，不等于 provider tokenizer。真实 provider 多轮评测仍需单独获得
授权后进行。

文件说明：

- [`report.json`](report.json)：完整机器可读账本；
- [`report.md`](report.md)：便于审阅的表格摘要；
- [`report.html`](report.html)：包含 JSON 账本的自包含 dashboard；
- [`README.md`](README.md)：英文说明。
