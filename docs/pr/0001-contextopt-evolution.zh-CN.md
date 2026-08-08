# PR #1：ContextOpt / ForgeAgent 演进回顾

## 改动内容

PR #1 把仓库从上下文选择 baseline 演进成一个可审计的代码 Agent 实验平台：

- v0.2：加入 AgentRunner、工具权限、事件日志和终态；
- v0.3：加入 live ContextCompiler、协议原子性和 versioned observed memory；
- v0.4：加入不可变候选快照、可见测试 oracle、beam search、去重和报告；
- v0.5：加入严格的 model-to-candidate JSON boundary；
- v0.6：加入跨轮 proposal/test session、失败反馈、测试缓存、checkpoint、
  apply/rollback 和篡改检测；
- v0.7：加入 planner / solver / reviewer 顺序编排、角色级模型指纹、共享预算、
  reviewer oracle gate 和中文文档。

## 为什么改

单次生成 patch 无法解释长程 Agent 最重要的工程问题：上下文如何裁剪、失败如何
反馈、进程中断如何恢复、同一个候选是否被重复测试，以及多角色意见能否绕过测试
事实。本 PR 把这些边界变成可序列化状态和可执行测试。

## 如何验证

~~~text
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/contextopt
python -m unittest discover -s tests -q
python -m unittest tests.test_orchestrator -v
python scripts/ci_checks.py
git diff --check
~~~

所有脚本模型和测试都是离线、可重复的；可见测试在候选临时工作区中执行，真实
工作区写入必须通过显式 apply-best。

## 指标和边界

model_calls、candidate_proposals、test_calls、test_reuses、rounds、total_tokens、
分支通过率和 oracle gate 是控制流、预算、安全或 conformance 指标。它们不能证明
模型推理质量、隐藏测试成功或多 Agent 比单 Agent 更强。Reviewer 的 confidence
只是结构化审计字段；accepted 必须同时满足 reviewer accept 和可见测试通过。

事件哈希链可以发现意外篡改，但没有外部信任锚，不能抵抗能够重写整个文件的攻击者。
测试命令是可信宿主进程，不是 OS sandbox；apply/rollback 也不是通用版本控制系统。
