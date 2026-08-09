# ForgeAgent / ContextOpt（中文说明）

这是一个面向长程代码 Agent 的实验型开源项目。目标不是做一个只能聊天的
演示，而是把 Agent 在真实编码任务中需要的控制面拆出来并做成可验证的运行时：

当前包版本：`0.9.0a1`。

- 上下文编译：在有限预算下选择任务、代码、测试、工具观察和记忆证据；
- 长程运行：持久化事件、checkpoint、恢复、预算和 pending 请求；
- 代码搜索：生成完整候选工作区，执行可见测试，去重、beam search 和基于观测质量的 MCTS；
- 多角色编排：planner 形成假设，solver 生成候选，reviewer 审计证据；
- 安全边界：测试和 apply/rollback 都是显式操作，不能因为模型说成功就写盘。
- 跨运行记忆：可选的 append-only `SemanticMemoryStore`，由显式工具读写并保留 provenance、
  幂等 identity 和失效/替代状态。

当前状态是 v0.9。已经实现单 Agent 运行时、上下文选择、分支搜索、可恢复的
proposal/test session、顺序的 planner / solver / reviewer 编排，以及带独立隐藏测试的
固定 ACM/数学题代码 Agent 策略评测。评测也可以接入 OpenAI-compatible 模型做探索性
运行。现在三个角色还会把各自历史中的 assistant 摘要和当前请求交给同一个
ContextCompiler，持久化 ContextReceipt、消息哈希、版本化观察记忆指纹和 workspace
generation，因此 checkpoint 里能审计“本轮到底给了角色什么上下文”。候选 oracle 还支持
`max_parallel_tests > 1`：每个候选使用独立临时工作区，逐个写入 requested/completed 事件，
每个结果都更新 checkpoint；恢复时只重跑尚未落账的观察。现在固定候选树还支持
`--search-policy mcts`：它把 visible-test 质量沿父链回传，用 UCT 选择下一条已生成分支，
并把选择事件写入 hash-chain。现在还支持可选的 disjoint 三方合并：只合并已经返回的
独立候选快照，冲突路径写入证据，不会做部分写入。`speculative_solver_width > 1` 还会在
共享模型/候选预算内并发调用多个 solver lane：每个 lane 有独立多样性指令，响应先分别
校验、记录哈希/用量/失败，再 namespace 后进入 oracle；checkpoint 记录观察到的 provider
并发度和已经落账的 lane response。恢复时复用已落账 lane，只重发没有 response 的 lane；
planner 和 reviewer 仍然顺序调用，仍不声称 exactly-once。还可以打开
`speculative_solver_stop_on_valid`（CLI 参数 `--speculative-solver-stop-on-valid`）：
第一个通过候选协议解析的 lane 会被记录为 winner，其余未完成 lane 会收到取消请求，
并通过 `solver.speculative.winner` / `solver.speculative.cancelled` 事件留下可审计证据。
这里的 valid 只表示协议可解析，不代表测试通过；取消是 best-effort，不能假设 provider
一定已经停止远端 HTTP 请求。provider-specific adapter 可以实现可选的
`request_cancellation(request)` hook，事件会记录 `acknowledged`、`not_observed`、
`unsupported` 或 `failed:*`。默认的串行 OpenAI-compatible adapter 可以关闭本地活动 HTTP response，
因此在本地传输确实被打断时记录 `acknowledged`；但通用 Chat Completions 没有标准 abort
endpoint，不能据此证明 provider 已停止服务端生成。
当前 v0.9 已经提供显式、可审计的跨运行语义记忆 notebook，以及一个可选的、受预算约束的
自动候选上下文模式；它仍然不是 embedding 检索、自动总结或学习型置信度校准。OS sandbox
和统计严谨的真实模型评测仍在后续计划中。
另外提供不调用模型的 `semantic-context-eval` 矩阵，固定 ACM/数学任务，比较
`recent`、`topk`、`density`、`submodular` 在候选上下文上的检索、预算淘汰、receipt
确定性和无 store 重放。这里的 recall 只是固定 fixture id 的保留率，不是语义理解、模型
使用证据或代码成功率。

另外新增了 `recovery-eval` 长程恢复矩阵：在 `model.requested`、`model.responded`、
`tool.started`、`tool.completed` 等 durable 边界注入 process-like stop，再用全新的
runner/model/tools 恢复同一份日志。它覆盖 pending request 复用、已落账最终响应不重复
调用、写操作 reconciliation、重复写入避免，以及 `run_tests` 的暂停和显式
`mark_failed`。五个场景的 JSON、Markdown、HTML、manifest 产物在
[`experiments/v0.9-recovery-matrix`](experiments/v0.9-recovery-matrix/README.zh-CN.md)。
每个 durable `model.requested` 事件还会记录稳定的 request idempotency key；
OpenAI-compatible adapter 默认通过 `Idempotency-Key` 发送它，但是否去重取决于 provider
是否真正支持该 header，项目不把本地 header 包装成 exactly-once 保证。
另有一个不依赖 provider 的首个有效候选取消演示，产物在
[`experiments/v0.9-speculative-cancellation`](experiments/v0.9-speculative-cancellation/README.zh-CN.md)，
展示 winner/cancelled lane ledger 和按配置 width 计费的预算口径，但不冒充远端 abort 评测。

现在还可以为多个运行挂载同一个 `SemanticMemoryStore`。它是 append-only、带 hash-chain
和 lease 的 JSONL notebook，保存 `fact`、`decision`、`procedure`、`failure` 四类短记忆，
并记录 scope、tags、confidence、source run/reference。Agent 必须显式调用
`memory_search`；只有打开 `--allow-write` 时才有 `memory_save`、`memory_invalidate` 和
`memory_feedback`。检索是可复现的 lexical 匹配，写入按内容 identity 幂等，`memory_save`
可以显式 supersede 旧记忆，`memory_invalidate` 可以写入失效原因。`memory_feedback` 用
tool-call id 做幂等键，把 helpful / not_helpful 信号以有界的排序调整写回，但不会修改记忆
正文。默认模式下 Agent 仍需显式调用 `memory_search`；如果显式设置
`--context-memory versioned-v1+semantic`，每轮最多会把 3 条命中渲染成带标签的 advisory
assistant block，并和普通上下文一起竞争 token 预算。候选可能被淘汰，receipt 会保存完整候选
快照和 store fingerprint，因此挂起请求可以在 store 变化后重放原候选。它不是当前 workspace
文件状态的证明，也不会覆盖观察记忆账本。
`orchestrate` 也可以用 `--memory-store`、`--memory-scope` 和
`--context-memory versioned-v1+semantic` 给 planner、solver、reviewer 挂载同一个 store；
每个角色仍有独立历史和 ContextReceipt，进程在 role request 边界停止后会重放原候选快照，
不会因为 live store 失效而偷偷换一组候选。
加上 `orchestrate --memory-feedback` 后，语义上下文可以显式闭环：终态 visible-test/oracle
结果会为每个角色实际选中的记忆候选写入一条幂等的 `helpful` 或 `not_helpful` 反馈；事件账本
保存 feedback id、信号和记忆已失效时的跳过原因，恢复不会重复计票。这是运行结果相关性
信号，不是“某条记忆导致补丁成功”的因果证明。
如果记忆带有 `source_refs`，成功的 `create_file` / `replace_text` 会在对应文件变更后
自动追加失效事件；进程在写入后、记忆失效前停止时，reconcile 也会补做这一步。非文件
原因导致的过期仍可通过 `memory_invalidate` 显式记录。

## 为什么适合面试 Agent 开发岗

项目中的技术点可以直接讲清楚：

1. 事件溯源状态机：模型请求、工具观察、预算、恢复和终态都有严格 schema；
2. 上下文工程：协议原子性、token 估算、freshness、版本化观察记忆和选择收据；
3. 代码生成 Agent：模型只能输出完整快照，必须经过路径校验和可执行测试；
4. 多 Agent 协作：三个角色使用独立模型指纹，但共享 token、候选和测试预算；
5. 角色上下文与记忆：每个角色的历史摘要独立编译，receipt 记录选择块、消息哈希、
   memory fingerprint 和 workspace generation；
6. 并行候选调度：限制 in-flight 数量，隔离临时工作区，并在每个测试结果后持久化；
7. speculative solver：在共享预算内并发发起独立 solver 请求，记录 lane 级上下文收据、
   响应哈希、token 用量、失败和 `max_provider_in_flight`，支持首个可解析候选胜出并请求
   取消其余 lane，再合并进入同一个可见 oracle；
8. MCTS 调度：使用真实 oracle 质量而不是模型自报置信度选择后续候选，记录 UCT、访问次数和
   reward；
9. 评测边界：reviewer 不能绕过可见测试，脚本 conformance 与模型能力明确分开。
10. 跨运行记忆：显式的 `memory_search` / `memory_save` / `memory_invalidate`、scope 继承、内容幂等、
    provenance、supersession/invalidation 和 lease，让“记住上一轮经验”变成可恢复、可审计
    的运行时协议，而不是 prompt 里凭空塞一段摘要。

编排也可以使用 `merge_policy=disjoint`：对相同根快照下的独立 solver 候选做有界三方合并，
合并候选仍必须经过可见测试；同一路径的不同修改只记录 conflict，不会猜测如何拼接。

这些设计让演示可以回答“状态是什么、失败如何恢复、指标如何计算、谁有权
接受结果”，而不是只展示一段角色扮演对话。

`trace` 还可以生成本地自包含 HTML 时间线，展开查看每个事件 JSON，并汇总模型请求、
工具结果和 ContextReceipt，适合排查长程运行或放入作品集：

~~~text
python -m contextopt trace events.jsonl --html trace.html
~~~

如果希望让后续运行读取同一份记忆，可以把 store 路径加入 `run`（resume 时也要传同一条
路径）：

~~~text
contextopt run "修复 parser" \
  --workspace <temporary-workspace-copy> \
  --script examples/runtime_demo/script.json \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --allow-write --allow-command \
  --test-command "python -m unittest discover -s tests -v" \
  --event-log <temporary-events.jsonl>
~~~

不加 `--allow-write` 时仍可搜索但不能保存或失效；记忆结果会进入普通 tool observation 和事件
账本，因此能在 trace 中检查查询内容、命中项、revision 和 memory id。semantic context 模式
会把最多 3 条候选作为普通上下文 block，`durable_memory_selected_ids` 显示实际保留的条目。
当前实现不依赖 embedding service，也不宣称 memory 本身已经提升真实模型成功率。resume 时
需要传入同一条 store 路径和 `--memory-scope`。

完整的“两次全新运行”离线演示在
[`examples/semantic_memory_demo`](examples/semantic_memory_demo/README.md)：第一次运行保存
procedure，第二次运行打开同一个 store 查询，reader trace 会显示命中词、memory id 和 revision。

还可以单独评测记忆检索边界：

~~~text
python -m contextopt memory-eval \
  --output experiments/v0.9-semantic-memory/report.json \
  --markdown experiments/v0.9-semantic-memory/report.md \
  --html experiments/v0.9-semantic-memory/report.html
~~~

固定报告把 hit@1/hit@k、MRR、负查询通过率、scope 隔离、失效记忆排除和重复检索确定性与
代码成功率分开；提交中的产物说明在
[`experiments/v0.9-semantic-memory`](experiments/v0.9-semantic-memory/README.zh-CN.md)。

多智能体编排也可以打开同一份 durable memory：

~~~text
python -m contextopt orchestrate \
  --task "Fix the parser" \
  --root-files root-files.json \
  --checkpoint orchestration.json \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --planner-script planner.json --solver-script solver.json --reviewer-script reviewer.json \
  --test-command "python -m unittest discover -s ." --allow-command
~~~

三个角色各自保存候选上下文 receipt；挂起的 role request 恢复时使用原快照，而不是使用
变化后的 live store 重新检索。

还可以评测自动 durable-memory 候选和上下文预算的边界：

~~~text
python -m contextopt semantic-context-eval \
  --repetitions 3 --budgets 128,256,512 \
  --output experiments/v0.9-semantic-context/report.json \
  --markdown experiments/v0.9-semantic-context/report.md \
  --html experiments/v0.9-semantic-context/report.html
~~~

提交中的 JSON、Markdown、HTML 产物说明在
[`experiments/v0.9-semantic-context`](experiments/v0.9-semantic-context/README.zh-CN.md)。
该评测始终 `model_calls=0`，只验证候选投影、token 预算、收据确定性和重放，不能推出
embedding 质量、模型是否使用记忆或生成 patch 是否更好。

## 离线验证

项目需要 Python 3.11 或更高版本，没有运行时依赖：

~~~text
python -m pip install -e .
python -m unittest discover -s tests -v
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/contextopt
~~~

planner / solver / reviewer 的协议测试：

~~~text
python -m unittest tests.test_orchestrator -v
~~~

候选搜索 session 也支持可恢复的并行 oracle：

~~~text
python -m contextopt search-session \
  --task "Implement solve so it returns ascending values" \
  --root-files examples/branch_demo/root-files.json \
  --checkpoint session.json \
  --script examples/branch_demo/proposal.json \
  --test-command "python -m unittest discover -s ." \
  --allow-command --max-parallel-tests 4 \
  --output session-report.json --markdown session-report.md
~~~

每个候选都在独立临时工作区运行；`max_parallel_tests` 限制同时运行的 oracle 进程，
每个结果落盘后才算进入账本。恢复时只会重新执行尚未持久化的候选，设为 `1` 即为
串行基线。增加 `--scheduler-policy adaptive` 后，后续批次会根据已观测父分支质量动态
排序，并在第一批通过后停止；`fixed` 则保持确定性顺序。分支树本身可以另用
`--search-policy mcts` 做 UCT 选择；它不负责让模型生成新候选。

完整的 orchestrate 命令需要三个 ScriptedModel JSON 文件、根目录快照和可信的
可见测试命令。它会输出角色调用数、实际测试进程数、缓存复用数、分支状态、
reviewer 决策、oracle gate 和 checkpoint。

接入真实 OpenAI-compatible solver 时，可以用 `--speculative-solver-width 3` 并发发起
三个独立 solver lane；总调用仍受 `--max-solver-calls` 和 `--max-model-calls` 约束，
每个返回快照都会经过同一套路径/大小/可见测试门禁：

~~~text
python -m contextopt orchestrate \
  --task "修复算法实现" --root-files root.json --checkpoint run.json \
  --solver-model <model-name> --base-url <endpoint> \
  --speculative-solver-width 3 --speculative-solver-stop-on-valid \
  --max-solver-calls 3 \
  --test-command "python -m unittest discover -s ." --allow-command
~~~

离线 ScriptedModel 也支持该开关，只需为 solver script 准备足够多的 response；
报告里的 `solver_variants`、`solver.speculative.*` 事件、winner/cancelled lane 数和
`max_provider_in_flight` 可以直接检查并发与提前停止是否真的发生。

代码 Agent 策略评测可以直接离线运行：

~~~text
python -m contextopt agent-eval \
  --fixtures all --repetitions 1 \
  --output agent-eval.json --markdown agent-eval.md
~~~

默认包含 `two-sum`（ACM 算法）和 `extended-gcd`（数论/数学）两个可执行 fixture，
比较 `single_pass`、`best_of_n` 与 `orchestrated`。每个 fixture 都有完整根快照、
故意失败的候选、正确候选、可见测试和不进入模型快照的独立隐藏 grader；报告会分开
记录 visible success、hidden success、模型/角色调用、候选数、实际测试进程、缓存复用
和 token 用量；同时给出描述性 Wilson 95% 区间，以及按相同 fixture/repetition 配对的
胜/负/平、可见结果差值以及测试/token 成本均值和观测方差，不隐藏原始 ledger。每个
run 还会记录本地墙钟 `duration_ms` 和配对耗时差值；它只是诊断指标，不是 provider
延迟 SLA。

如果要接入 OpenAI-compatible 模型，可使用：

~~~text
python -m contextopt agent-eval \
  --model <model-name> --base-url <endpoint> \
  --api-key-env CONTEXTOPT_API_KEY --repetitions 3 \
  --output agent-eval-real.json --markdown agent-eval-real.md \
  --html agent-eval-real.html \
  --checkpoint agent-eval-real.checkpoint.json \
  --manifest agent-eval-real.manifest.json
~~~

真实模型路径会记录 provider token，但仍是固定小样本的探索性评测，不能直接当成统计
严谨的模型能力结论。HTML 会生成自包含 dashboard，展示 visible/hidden 成功率、成本
指标和嵌入式 JSON ledger，适合放在 PR 或作品集里。
`--manifest` 会记录 adapter、模型名、去掉 query/fragment 的 endpoint、运行时设置、
评测配置和显式提供的 revision（`CONTEXTOPT_GIT_REVISION` 或 `GITHUB_SHA`），不会写入 API key。
`--checkpoint` 会在每个 fixture/strategy/repetition cell 完成后原子写入；provider 或进程
中断后，使用相同参数加 `--resume --checkpoint ...`，已完成 cell 会复用，只重跑缺失 cell。
这是 at-least-once provider 执行语义，不声称 exactly-once。

## 评测指标

主要指标不是 reviewer 的自信度，而是：

- model_calls，以及 planner / solver / reviewer 的角色调用分解；
- candidate_proposals、test_calls、test_reuses；
- 每轮分支的 visible-test 通过情况、去重、剪枝和 best branch；
- total_tokens、checkpoint 事件链和恢复后的状态一致性；
- oracle gate：只有 reviewer accept 且候选可见测试通过才会 accepted。
- 策略评测的 visible/hidden `success_rate`、`mean_model_calls`、`mean_test_calls`、
  `mean_candidate_proposals`、`mean_hidden_test_calls`、`mean_total_tokens` 和本地
  `mean_duration_ms`。
- 编排角色的 ContextReceipt：`policy`、选择/淘汰 block、`messages_sha256`、
  `memory_fingerprint` 和 `workspace_generation`，用于解释长程上下文是否真的被使用。

脚本模型只能证明协议、预算、持久化、恢复和策略控制流正确，不能证明真实模型的
编码能力。`agent-eval` 会在可见测试通过后运行独立隐藏 grader，但固定脚本通过仍不
代表泛化能力、延迟或生产安全。真实模型比较必须固定模型版本、提示词、仓库快照、
工具和预算，并保留多次运行的完整 ledger。

## 项目文档

- 英文架构：[docs/architecture.md](docs/architecture.md)
- 英文评测：[docs/evaluation.md](docs/evaluation.md)
- 真实 provider 工作流：[docs/evaluation-real-provider.zh-CN.md](docs/evaluation-real-provider.zh-CN.md)
- 运行时说明：[docs/runtime.md](docs/runtime.md)
- PR 变更说明约定：[docs/pr/README.md](docs/pr/README.md)
- PR #1 中文回顾：[docs/pr/0001-contextopt-evolution.zh-CN.md](docs/pr/0001-contextopt-evolution.zh-CN.md)
- v0.8 中文变更说明：[docs/pr/0001-v0.8-evaluation-addendum.zh-CN.md](docs/pr/0001-v0.8-evaluation-addendum.zh-CN.md)
- v0.8 角色上下文补充：[docs/pr/0001-v0.8-context-memory-addendum.zh-CN.md](docs/pr/0001-v0.8-context-memory-addendum.zh-CN.md)
- v0.8 并行调度补充：[docs/pr/0001-v0.8-parallel-scheduler-addendum.zh-CN.md](docs/pr/0001-v0.8-parallel-scheduler-addendum.zh-CN.md)
- v0.8 评测面板补充：[docs/pr/0001-v0.8-evaluation-dashboard-addendum.zh-CN.md](docs/pr/0001-v0.8-evaluation-dashboard-addendum.zh-CN.md)
- v0.8 MCTS 调度补充：[docs/pr/0001-v0.8-mcts-addendum.zh-CN.md](docs/pr/0001-v0.8-mcts-addendum.zh-CN.md)
- v0.8 评测 checkpoint 补充：[docs/pr/0001-v0.8-evaluation-checkpoint-addendum.zh-CN.md](docs/pr/0001-v0.8-evaluation-checkpoint-addendum.zh-CN.md)
- v0.8 评测统计补充：[docs/pr/0001-v0.8-evaluation-statistics-addendum.zh-CN.md](docs/pr/0001-v0.8-evaluation-statistics-addendum.zh-CN.md)
- v0.8 trace 可视化补充：[docs/pr/0001-v0.8-trace-dashboard-addendum.zh-CN.md](docs/pr/0001-v0.8-trace-dashboard-addendum.zh-CN.md)
- v0.8 合并感知快照补充：[docs/pr/0001-v0.8-merge-aware-snapshots-addendum.zh-CN.md](docs/pr/0001-v0.8-merge-aware-snapshots-addendum.zh-CN.md)
- v0.8 评测 manifest 补充：[docs/pr/0001-v0.8-evaluation-manifest-addendum.zh-CN.md](docs/pr/0001-v0.8-evaluation-manifest-addendum.zh-CN.md)
- v0.8 speculative solver 并发补充：[docs/pr/0001-v0.8-speculative-solver-addendum.zh-CN.md](docs/pr/0001-v0.8-speculative-solver-addendum.zh-CN.md)

离线三次重复的 ACM/数学控制实验产物在
[`experiments/v0.8-scripted-3-reps`](experiments/v0.8-scripted-3-reps/README.md)，用于审查
协议、预算、隐藏测试和配对统计；它不是真实模型能力证据。
本次对应的 [PR 变更说明](docs/pr/0001-v0.8-scripted-control-artifact.zh-CN.md) 和
[英文版](docs/pr/0001-v0.8-scripted-control-artifact.md) 记录了复现命令与 claim boundary。
- v0.9 长程恢复矩阵补充：[docs/pr/0001-v0.9-recovery-matrix.zh-CN.md](docs/pr/0001-v0.9-recovery-matrix.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-recovery-matrix.md)
- v0.9 model request 幂等钩子：[docs/pr/0001-v0.9-idempotency-hook.zh-CN.md](docs/pr/0001-v0.9-idempotency-hook.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-idempotency-hook.md)
- v0.9 首个有效候选取消：[docs/pr/0001-v0.9-speculative-cancellation.zh-CN.md](docs/pr/0001-v0.9-speculative-cancellation.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-speculative-cancellation.md)
- v0.9 真实 provider 手动工作流：[docs/pr/0001-v0.9-real-provider-workflow.zh-CN.md](docs/pr/0001-v0.9-real-provider-workflow.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-real-provider-workflow.md)
- v0.9 评测本地耗时记账：[docs/pr/0001-v0.9-evaluation-duration.zh-CN.md](docs/pr/0001-v0.9-evaluation-duration.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-evaluation-duration.md)
- v0.9 包版本对齐：[docs/pr/0001-v0.9-version-alignment.zh-CN.md](docs/pr/0001-v0.9-version-alignment.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-version-alignment.md)
- v0.9 本地 HTTP 传输取消：[docs/pr/0001-v0.9-http-transport-cancellation.zh-CN.md](docs/pr/0001-v0.9-http-transport-cancellation.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-http-transport-cancellation.md)
- v0.9 真实 provider secret 作用域：[docs/pr/0001-v0.9-real-provider-secret-scope.zh-CN.md](docs/pr/0001-v0.9-real-provider-secret-scope.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-real-provider-secret-scope.md)
- v0.9 跨运行语义记忆：[docs/pr/0001-v0.9-semantic-memory.zh-CN.md](docs/pr/0001-v0.9-semantic-memory.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-semantic-memory.md)
- v0.9 语义记忆检索评测：[docs/pr/0001-v0.9-semantic-memory-evaluation.zh-CN.md](docs/pr/0001-v0.9-semantic-memory-evaluation.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-semantic-memory-evaluation.md)
- v0.9 基于 source_refs 的记忆自动失效：[docs/pr/0001-v0.9-source-aware-memory-invalidation.zh-CN.md](docs/pr/0001-v0.9-source-aware-memory-invalidation.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-source-aware-memory-invalidation.md)
- v0.9 幂等 memory feedback：[docs/pr/0001-v0.9-memory-feedback.zh-CN.md](docs/pr/0001-v0.9-memory-feedback.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-memory-feedback.md)
- v0.9 有界 durable 记忆上下文：[docs/pr/0001-v0.9-semantic-memory-context.zh-CN.md](docs/pr/0001-v0.9-semantic-memory-context.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-semantic-memory-context.md)
- v0.9 自动语义上下文候选评测：[docs/pr/0001-v0.9-semantic-context-evaluation.zh-CN.md](docs/pr/0001-v0.9-semantic-context-evaluation.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-semantic-context-evaluation.md)
- v0.9 编排接入语义记忆：[docs/pr/0001-v0.9-orchestration-semantic-memory.zh-CN.md](docs/pr/0001-v0.9-orchestration-semantic-memory.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-orchestration-semantic-memory.md)
- v0.9 provider adapter 本地 smoke：[docs/pr/0001-v0.9-provider-adapter-smoke.zh-CN.md](docs/pr/0001-v0.9-provider-adapter-smoke.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-provider-adapter-smoke.md)
- v0.9 终态语义记忆反馈闭环：[docs/pr/0001-v0.9-semantic-memory-feedback-loop.zh-CN.md](docs/pr/0001-v0.9-semantic-memory-feedback-loop.zh-CN.md)
  和 [英文版](docs/pr/0001-v0.9-semantic-memory-feedback-loop.md)
- v0.7 编排补充的中文回顾：[docs/pr/0001-v0.7-orchestration-addendum.zh-CN.md](docs/pr/0001-v0.7-orchestration-addendum.zh-CN.md)

本中文文件是当前英文 README 的工程化摘要。英文文档和代码中的 schema、命令、
指标名称是权威定义。
