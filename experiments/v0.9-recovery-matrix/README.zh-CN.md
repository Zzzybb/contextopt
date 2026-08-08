# v0.9 长程运行恢复矩阵

本目录是事件溯源代码 Agent runtime 的确定性故障注入实验。每个场景在一个 durable
事件之后停止第一个 runner，然后用全新的 runner、模型 adapter 和工具注册表恢复同一份
日志。

| 场景 | 注入边界 | 恢复契约 |
|---|---|---|
| `pending-model-request` | `model.requested` | 复用 pending request |
| `durable-final-response` | `model.responded` | 不再重复调用模型，直接完成 |
| `reconcile-write-before-effect` | `tool.started` / `replace_text` | 校验 precondition 后安全重试 |
| `durable-write-result` | `tool.completed` / `replace_text` | 避免重复写入 |
| `nonreplayable-test-pause` | `tool.started` / `run_tests` | 先暂停，再显式 `mark_failed` |

五个场景全部通过。JSON 报告保留 pending 状态重建、事件序列、模型调用数、工具事件数和
最终 fixture 摘要；HTML 报告是自包含的，并嵌入同一份 JSON ledger；manifest 记录协议和
claim boundary，不包含 secret。

从仓库根目录重新生成：

```text
python -m contextopt recovery-eval \
  --output experiments/v0.9-recovery-matrix/report.json \
  --markdown experiments/v0.9-recovery-matrix/report.md \
  --html experiments/v0.9-recovery-matrix/report.html \
  --manifest experiments/v0.9-recovery-matrix/manifest.json
```

这是 runtime 恢复一致性产物，不是模型代码能力、任意机器宕机恢复、provider exactly-once
副作用、sandbox 安全性或生产安全性的证据。
