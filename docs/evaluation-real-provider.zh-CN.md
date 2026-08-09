# 真实 provider 评测工作流

仓库新增手动 GitHub Actions 工作流 `.github/workflows/real-agent-eval.yml`。它不会定时运行，
也不会在 pull request 上自动运行，因此只有操作者明确点击后才会产生 provider 调用和费用。
如果该 workflow 是通过 pull request 引入的，GitHub 会在改动合并到仓库默认分支后才显示
**Actions → Run workflow** 表单；合并前请使用 README 中等价的本地 CLI 命令。

## 一次性配置

1. 在仓库或 environment 中添加名为 `CONTEXTOPT_API_KEY` 的 secret。
2. 打开 **Actions → real-agent-eval → Run workflow**。
3. 填写 OpenAI-compatible 的 `model` 与 `base_url`。第一次建议保留 `all` fixture、
   `single_pass,best_of_n,orchestrated` 策略和 `3` 次 repetition。
   工作流会拒绝少于 3 次的 repetition，以满足 Level 3 探索性对比规则。
4. 工作流完成后下载上传的 artifact bundle。

当前 revision 中，`all` 会展开为 `two-sum`、`extended-gcd`、`merge-intervals`、
`modular-inverse`。确定性的三次重复控制基线位于
[experiments/v1.0-acm-math-4-fixtures](../experiments/v1.0-acm-math-4-fixtures/README.zh-CN.md)；
与 provider 结果比较时保持 fixture ID、策略、预算和 prompt 配置不变。

bundle 包含 JSON ledger、Markdown 摘要、自包含 HTML dashboard、manifest 和原子 checkpoint。
如果 provider 调用或 runner 中断，可以在本地使用 `--resume --checkpoint` 继续同一矩阵，
或者用同样的 workflow 参数重新运行；不要把不同模型、prompt、fixture 或预算配置混在一份
对比里。

## 结果代表什么

矩阵会在相同 ACM/数学 fixture 和 repetition 上配对比较 `single_pass`、`best_of_n` 以及
planner/solver/reviewer 编排。报告包含 visible-test 成功率、独立 hidden grader 成功率、
模型/角色调用数、候选与测试成本、token 用量、Wilson 区间，以及带观测方差的配对差值。

这仍然是固定小样本的探索性 benchmark，不是通用能力结论、统计充分的研究、延迟 SLA、
provider 稳定性测试或安全评测。hidden grader 源码不会进入模型可见快照，manifest 记录
provider 身份但不会写入 API key。

CI 会通过 `tests/test_agent_search_eval.py` 中的本地确定性 HTTP fixture 走同一个
OpenAI-compatible adapter 边界，校验请求格式、authorization、idempotency header 和
oracle gate；这个 smoke 不访问真实 provider，也不代表模型质量。
