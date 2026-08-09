# v1.0 Level 4 运行时鲁棒性矩阵

本目录是确定性的、无需 provider 的 fault-injection ledger，覆盖长程运行时中不能被
happy path 演示掩盖的边界。

- 场景：超大 tool output 压缩、source-aware 陈旧记忆失效、重复 tool result 拒绝、
  陈旧 compare-and-swap 写冲突
- 结果：`4/4` 个场景通过
- 文件：`report.json`、`report.md`、`report.html`、`manifest.json`

矩阵证明的是本地 contract 行为并记录观察证据，不能声称机器掉电恢复、sandbox 安全、
provider 稳定性、exactly-once 外部副作用或代码能力。

manifest 固定了本基线使用的源码 revision：`330e047a3170a799fc4c434a94b2178f33d6791a`。

从仓库根目录重新生成：

```text
python -m contextopt robustness-eval --scenarios all \
  --output experiments/v1.0-robustness-matrix/report.json \
  --markdown experiments/v1.0-robustness-matrix/report.md \
  --html experiments/v1.0-robustness-matrix/report.html \
  --manifest experiments/v1.0-robustness-matrix/manifest.json
```
