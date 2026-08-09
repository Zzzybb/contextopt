# Workspace 知识库

`knowledge-index` 是 ContextOpt durable memory 面向仓库的一半。它把有界的 UTF-8 workspace
快照转换为带 source 地址的分块，写入现有 append-only `SemanticMemoryStore`；不会调用模型，
也不会额外生成一个没有账本的索引。

## 索引快照

~~~text
contextopt knowledge-index \
  --workspace ./workspace \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --output .contextopt/knowledge-index.json \
  --markdown .contextopt/knowledge-index.md
~~~

默认策略覆盖常见源码、配置、文档和测试后缀；会跳过生成/缓存目录、点文件、符号链接、
非 UTF-8 文件、超过 256 KiB 的文件以及超过 500 个的候选文件。`--chunk-lines`、
`--max-chunk-bytes`、`--max-file-bytes`、`--max-files`、`--extensions`、`--ignore-dirs` 和
`--include-dotfiles` 可以显式固定边界。

每个分块正文包含相对路径和行号，带有 `knowledge-chunk-v1` tag，并在 `source_refs` 中记录
源路径。记忆 identity 包含分块正文、scope、kind 和 tags，因此不变快照重复索引会复用同一
条目。文件被修改或删除时，旧的 active 分块会追加 `memory.invalidated` 事件；Agent 在两次
索引之间直接编辑被引用文件时，现有 source-aware tool reconciliation 也会使对应分块失效。

## 在 Agent 运行中使用

附加同一个 store，并打开有界 semantic projection：

~~~text
contextopt run "Fix the parser" \
  --workspace ./workspace \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --script examples/runtime_demo/script.json
~~~

正常 `ContextCompiler` 会把索引分块当作 advisory block；最多 3 条候选进入普通 token budget
和策略选择，receipt 会记录 candidate id 与 selected id。模型仍必须读取当前文件并运行测试。
检索是 lexical 且不依赖 provider，所以这里证明的是 provenance、新鲜度、有界上下文和恢复，
不是 embedding 质量或代码成功率。

新建单 Agent 运行时，也可以在第一次模型调用前自动刷新索引：

~~~text
contextopt run "Fix the parser" \
  --workspace ./workspace \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --auto-index-knowledge \
  --knowledge-index-report .contextopt/knowledge-index.json \
  --script examples/runtime_demo/script.json
~~~

`--auto-index-knowledge` 必须同时提供 `--memory-store` 和 `--memory-scope`。它会在
`AgentRunner` 打开工具前执行，因此第一次编译请求就能看到最新源码投影；后续写操作仍由
正常的 source-aware 失效逻辑处理。`resume` 不会自动重新索引，因为修改 pending request
所依赖的 durable store 必须是显式配置决定。

## 在多智能体编排中使用

`orchestrate` 接收的是 `--root-files` JSON 完整快照，而不是文件系统路径。可以在 planner
第一次请求前刷新同一份投影：

~~~text
contextopt orchestrate \
  --task "Fix the parser" \
  --root-files root-files.json \
  --checkpoint .contextopt/orchestration.json \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --auto-index-knowledge \
  --knowledge-index-report .contextopt/orchestration-knowledge.json \
  --planner-script planner.json --solver-script solver.json --reviewer-script reviewer.json \
  --test-command "python -m unittest discover -s ." --allow-command
~~~

root-file 分块带有独立的 `knowledge-snapshot-v1` tag。因此同一个 scope 内，快照刷新和文件系统
索引是 source-isolated 的：修改 root 快照只会使 snapshot 分块失效，workspace 索引只会使
自己的分块失效。三个角色的 receipt 仍会记录 semantic context 策略实际选中的 lexical
候选。该 flag 只支持新建编排；`resume` 会保留 pending request 的 store 快照。
知识分块是 evidence 而不是 learned experience，因此编排结束时的 `--memory-feedback` 不会
更新它们的 retrieval signal。

## 可复现边界

JSON 报告记录规范化配置、文件数量、创建/复用/失效分块数量、跳过原因、active chunk id 和
快照 fingerprint。entry 内容及失效历史以 memory ledger 为准。索引可以在中断后安全重复：
写入按内容幂等，过期条目不会被静默删除。
