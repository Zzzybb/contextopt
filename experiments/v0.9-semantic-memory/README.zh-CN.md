# v0.9 语义记忆检索一致性评测

这个产物只评测持久化跨运行记忆边界，不调用模型。它在 append-only store 中创建五条带
类型的记忆，显式失效一条旧记忆，然后对五个带 scope/tag 的查询各重复三次。

## 复现

~~~text
python -m contextopt memory-eval \
  --repetitions 3 \
  --limit 3 \
  --output experiments/v0.9-semantic-memory/report.json \
  --markdown experiments/v0.9-semantic-memory/report.md \
  --html experiments/v0.9-semantic-memory/report.html
~~~

报告保留原始 fixture、结果 memory id、命中词、分数、重复查询 digest 和自包含 HTML 面板。
当前提交中的确定性结果为：

| 指标 | 结果 |
|---|---:|
| 正向 hit@1 | 1.000 |
| 正向 hit@k（`k=3`） | 1.000 |
| 平均倒数排名 MRR | 1.000 |
| 负查询通过率 | 1.000 |
| scope 隔离率 | 1.000 |
| 已失效记忆排除率 | 1.000 |
| 确定性重放率 | 1.000 |

## 指标和边界

- `hit@1` / `hit@k` 检查期望的 active memory 是否排在第一或配置的结果范围内；`MRR` 是
  第一个期望条目的倒数排名；
- 负查询通过表示没有期望的 active memory 时返回空结果；scope 隔离只允许 `global` 和
  当前 project scope 的结果；
- 已失效排除检查显式 invalidated 的 memory id 不会出现在检索结果；确定性重放要求重复
  查询的结果 id 和序列化 match digest 一致。

这些是无 provider 的检索和持久化检查，不是 embedding 质量、语义理解、模型是否会主动
查询记忆或生成代码成功率。实现明确报告 lexical baseline，不把固定 fixture 包装成学习型
记忆结论。
