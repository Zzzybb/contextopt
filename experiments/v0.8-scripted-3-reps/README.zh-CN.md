# v0.8 脚本化三次重复控制实验

这是 ContextOpt 评测 harness 的确定性控制运行，产物与英文说明位于同一目录。

- fixture：`two-sum`（ACM 算法）和 `extended-gcd`（数学）
- 策略：`single_pass`、`best_of_n`、`orchestrated`
- 重复：每个 fixture/策略组合重复 3 次
- hidden grader：开启，且不进入模型可见 snapshot
- 文件：`report.json`、`report.md`、`report.html`、`checkpoint.json`、`manifest.json`

该运行用于检查配对统计、visible/hidden oracle gate、token 和测试预算以及 checkpoint
结构。它不是实际模型 benchmark：响应由确定性的 `ScriptedModel` 生成，不能说明真实模型
代码能力、provider 稳定性或生产安全性。

从仓库根目录重新生成：

```text
python -m contextopt agent-eval --fixtures all --repetitions 3 \
  --output experiments/v0.8-scripted-3-reps/report.json \
  --markdown experiments/v0.8-scripted-3-reps/report.md \
  --html experiments/v0.8-scripted-3-reps/report.html \
  --checkpoint experiments/v0.8-scripted-3-reps/checkpoint.json \
  --manifest experiments/v0.8-scripted-3-reps/manifest.json
```
