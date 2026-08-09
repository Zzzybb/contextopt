# v0.9 首个有效候选胜出与 speculative cancellation artifact

这是一个不依赖 provider 的确定性演示，用来展示可选的“首个有效候选竞速”。lane 0 立即
返回协议合法的排序候选，lane 1 一直等待，直到 runner 发出取消请求。候选仍然必须通过
可执行 visible oracle 和 reviewer gate。

提交的文件包括 durable checkpoint、JSON ledger、Markdown 摘要、自包含 HTML dashboard，
以及记录 SHA-256 的 manifest。可以这样重建：

```text
python scripts/generate_speculative_cancellation_artifact.py
```

报告仍按配置 width 计费：`model_calls=4`、`solver_calls=2`，即使有一个 lane 被取消。
`speculative_winners=1` 和 `cancelled_solver_lanes=1` 描述本地调度效果。这不代表模型质量、
远端 provider 的 abort 延迟或 exactly-once 取消；这里的 `valid` 只表示候选协议解析成功。
