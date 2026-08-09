# 真实 provider 评测工作流

仓库新增手动 GitHub Actions 工作流 `.github/workflows/real-agent-eval.yml`。它不会定时运行，
也不会在 pull request 上自动运行，因此只有操作者明确点击后才会产生 provider 调用和费用。
如果该 workflow 是通过 pull request 引入的，GitHub 会在改动合并到仓库默认分支后才显示
**Actions → Run workflow** 表单；合并前请使用 README 中等价的本地 CLI 命令。

## 一次性配置

1. 在仓库或 environment 中添加名为 `CONTEXTOPT_API_KEY` 的 secret。
2. 打开 **Actions → real-agent-eval → Run workflow**。
3. 填写 OpenAI-compatible 的 `model` 与 `base_url`；如果 provider 暴露了文档化的取消协议，
   可以额外填写 provider-specific 的 `cancellation_url`。如果要控制候选执行环境，选择
   `sandbox=docker` 并把 `container_image` 固定到 digest。第一次建议保留 `all` fixture、
   `single_pass,best_of_n,orchestrated` 策略和 `3` 次 repetition。
   工作流会拒绝少于 3 次的 repetition，以满足 Level 3 探索性对比规则。
4. 工作流完成后下载上传的 artifact bundle。

如果想把本地真实模型矩阵保存成可审计的轨迹，可以使用 cassette：

```text
python -m contextopt agent-eval --fixtures all --repetitions 3 \
  --model <model-name> --base-url <endpoint> \
  --cancellation-url <provider-cancel-endpoint> \
  --sandbox docker --container-image <image@sha256:digest> \
  --record-transcript-dir .contextopt/provider-matrix \
  --output agent-eval-real.json --manifest agent-eval-real.manifest.json
python -m contextopt agent-eval --fixtures all --repetitions 3 \
  --replay-transcript-dir .contextopt/provider-matrix \
  --output agent-eval-replay.json --manifest agent-eval-replay.manifest.json
```

目录会为每个 fixture、strategy、repetition 和 role 保存一条严格 JSONL cassette。
重放会校验完整的规范化请求 hash；缺失或变化的请求会 fail closed。它是离线复现已
记录运行的方式，不是新的 provider 测量；每次矩阵录制都应使用新的目录。

`--cancellation-url` 是可选项。传入时，它必须是 provider-specific 的 `POST` endpoint，接受
`{"request_idempotency_key": "...", "model": "..."}`，并在 provider 自己的取消协议生效后返回
2xx。adapter 会把 404/405 记录为 `unsupported`，其他传输或 HTTP 错误记录为 `failed:*`。
Chat Completions 没有统一的该 endpoint，因此这只是集成 hook，不能证明远端生成已经停止，
也不能单独证明节省了计费。

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
OpenAI-compatible adapter 边界。一个 fixture 覆盖 single-pass 请求格式；多智能体 fixture
会对四个 ACM/数学任务完整驱动 planner → solver → reviewer，并校验角色模型、authorization、
idempotency header 和 oracle gate。两个 smoke 都不访问真实 provider，也不代表模型质量。
