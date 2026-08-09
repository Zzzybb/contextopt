# v1.0 ACM/数学四任务脚本化控制产物

本目录是 v1.0 代码 Agent 矩阵的确定性基线产物。

- fixture：`two-sum`、`merge-intervals`（ACM 算法），以及 `extended-gcd`、
  `modular-inverse`（数学）
- 策略：`single_pass`、`best_of_n`、`orchestrated`
- 重复：每个 fixture/策略组合重复 `3` 次（ledger 共 `36` 行）
- hidden grader：开启，且不进入模型可见 snapshot
- 文件：`report.json`、`report.md`、`report.html`、`checkpoint.json`、`manifest.json`

脚本化控制结果是刻意固定的：`single_pass` 在 `12` 个 cell 中接受 `0` 个，两个更强
策略均在 `12` 个 visible cell 中接受 `12` 个，并通过全部 `12/12` 个独立 hidden
grader。这证明的是 evaluator 的 oracle gate 和记账逻辑，不是模型能力或多智能体价值。

从仓库根目录重新生成：

```text
python -m contextopt agent-eval --fixtures all --repetitions 3 \
  --output experiments/v1.0-acm-math-4-fixtures/report.json \
  --markdown experiments/v1.0-acm-math-4-fixtures/report.md \
  --html experiments/v1.0-acm-math-4-fixtures/report.html \
  --checkpoint experiments/v1.0-acm-math-4-fixtures/checkpoint.json \
  --manifest experiments/v1.0-acm-math-4-fixtures/manifest.json
```

第一次真实 provider 运行必须保持相同 fixture ID、策略集合、预算、prompt，并至少重复
三次，之后才能与这份控制产物做对比。

manifest 固定了本基线使用的源码 revision：
`c27f5a291a3998eb0b0b40e16ca57a7ab66f2154`。本次在 orchestration knowledge-snapshot 完成后
重新生成了提交中的 report；其中仍只有 provider-free scripted control 数据。
