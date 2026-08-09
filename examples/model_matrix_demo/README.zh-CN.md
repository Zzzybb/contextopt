# 无 provider key 的多模型矩阵演示

这个 demo 不需要 API key，就能跑通完整的“采集 → 配对 → 分析”链路：

```text
python scripts/run_model_matrix_demo.py
```

它用确定性的 `ScriptedModel` 对四个 ACM/数学 fixture 各跑三次，生成两份 bundle，再用
`agent-eval-compare` 以 `scripted-a` 和 `scripted-b` 为 label 做比较。输出位于 Git 忽略的
`build/model-matrix-demo/`，包括：

- `scripted-a/`、`scripted-b/`：report、Markdown/HTML dashboard、manifest 和 checkpoint；
- `model-matrix.json`：机器可读的 protocol fingerprint 和方向一致性摘要；
- `model-matrix.md`、`model-matrix.html`：可审查报告。

由于本地 shell 没有 `GITHUB_SHA`，demo 会把 provenance 字段固定为
`provider-free-model-matrix-demo`；它只是演示 revision 标记，不是源码 commit 声明。

两个 label 是控制副本，不是两个真实模型。它只能证明 evaluator、manifest 匹配、配对分析和
渲染器能工作，不能证明模型能力。要做真实比较，请替换成获授权 provider 运行产生的 bundle，
或使用手动 `.github/workflows/real-agent-model-matrix.yml` workflow。
