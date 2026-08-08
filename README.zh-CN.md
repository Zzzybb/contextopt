# ForgeAgent / ContextOpt（中文说明）

这是一个面向长程代码 Agent 的实验型开源项目。目标不是做一个只能聊天的
演示，而是把 Agent 在真实编码任务中需要的控制面拆出来并做成可验证的运行时：

- 上下文编译：在有限预算下选择任务、代码、测试、工具观察和记忆证据；
- 长程运行：持久化事件、checkpoint、恢复、预算和 pending 请求；
- 代码搜索：生成完整候选工作区，执行可见测试，去重和 beam search；
- 多角色编排：planner 形成假设，solver 生成候选，reviewer 审计证据；
- 安全边界：测试和 apply/rollback 都是显式操作，不能因为模型说成功就写盘。

当前状态是 v0.8。已经实现单 Agent 运行时、上下文选择、分支搜索、可恢复的
proposal/test session、顺序的 planner / solver / reviewer 编排，以及带独立隐藏测试的
固定 ACM/数学题代码 Agent 策略评测。评测也可以接入 OpenAI-compatible 模型做探索性
运行。现在三个角色还会把各自历史中的 assistant 摘要和当前请求交给同一个
ContextCompiler，持久化 ContextReceipt、消息哈希、版本化观察记忆指纹和 workspace
generation，因此 checkpoint 里能审计“本轮到底给了角色什么上下文”。并行工作区、
PatchTree/MCTS 和统计严谨的真实模型评测仍在后续计划中。

## 为什么适合面试 Agent 开发岗

项目中的技术点可以直接讲清楚：

1. 事件溯源状态机：模型请求、工具观察、预算、恢复和终态都有严格 schema；
2. 上下文工程：协议原子性、token 估算、freshness、版本化观察记忆和选择收据；
3. 代码生成 Agent：模型只能输出完整快照，必须经过路径校验和可执行测试；
4. 多 Agent 协作：三个角色使用独立模型指纹，但共享 token、候选和测试预算；
5. 角色上下文与记忆：每个角色的历史摘要独立编译，receipt 记录选择块、消息哈希、
   memory fingerprint 和 workspace generation；
6. 评测边界：reviewer 不能绕过可见测试，脚本 conformance 与模型能力明确分开。

这些设计让演示可以回答“状态是什么、失败如何恢复、指标如何计算、谁有权
接受结果”，而不是只展示一段角色扮演对话。

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

完整的 orchestrate 命令需要三个 ScriptedModel JSON 文件、根目录快照和可信的
可见测试命令。它会输出角色调用数、实际测试进程数、缓存复用数、分支状态、
reviewer 决策、oracle gate 和 checkpoint。

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
和 token 用量。

如果要接入 OpenAI-compatible 模型，可使用：

~~~text
python -m contextopt agent-eval \
  --model <model-name> --base-url <endpoint> \
  --api-key-env CONTEXTOPT_API_KEY --repetitions 3 \
  --output agent-eval-real.json --markdown agent-eval-real.md
~~~

真实模型路径会记录 provider token，但仍是固定小样本的探索性评测，不能直接当成统计
严谨的模型能力结论。

## 评测指标

主要指标不是 reviewer 的自信度，而是：

- model_calls，以及 planner / solver / reviewer 的角色调用分解；
- candidate_proposals、test_calls、test_reuses；
- 每轮分支的 visible-test 通过情况、去重、剪枝和 best branch；
- total_tokens、checkpoint 事件链和恢复后的状态一致性；
- oracle gate：只有 reviewer accept 且候选可见测试通过才会 accepted。
- 策略评测的 visible/hidden `success_rate`、`mean_model_calls`、`mean_test_calls`、
  `mean_candidate_proposals`、`mean_hidden_test_calls` 和 `mean_total_tokens`。
- 编排角色的 ContextReceipt：`policy`、选择/淘汰 block、`messages_sha256`、
  `memory_fingerprint` 和 `workspace_generation`，用于解释长程上下文是否真的被使用。

脚本模型只能证明协议、预算、持久化、恢复和策略控制流正确，不能证明真实模型的
编码能力。`agent-eval` 会在可见测试通过后运行独立隐藏 grader，但固定脚本通过仍不
代表泛化能力、延迟或生产安全。真实模型比较必须固定模型版本、提示词、仓库快照、
工具和预算，并保留多次运行的完整 ledger。

## 项目文档

- 英文架构：[docs/architecture.md](docs/architecture.md)
- 英文评测：[docs/evaluation.md](docs/evaluation.md)
- 运行时说明：[docs/runtime.md](docs/runtime.md)
- PR 变更说明约定：[docs/pr/README.md](docs/pr/README.md)
- PR #1 中文回顾：[docs/pr/0001-contextopt-evolution.zh-CN.md](docs/pr/0001-contextopt-evolution.zh-CN.md)
- v0.8 中文变更说明：[docs/pr/0001-v0.8-evaluation-addendum.zh-CN.md](docs/pr/0001-v0.8-evaluation-addendum.zh-CN.md)
- v0.8 角色上下文补充：[docs/pr/0001-v0.8-context-memory-addendum.zh-CN.md](docs/pr/0001-v0.8-context-memory-addendum.zh-CN.md)

本中文文件是当前英文 README 的工程化摘要。英文文档和代码中的 schema、命令、
指标名称是权威定义。
