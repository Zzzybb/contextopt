# 候选代码执行隔离策略

这份策略把生成代码的信任边界和正确性 oracle 分开。运行时始终校验候选快照并记录测试结果；
执行环境决定恶意或意外候选最多能影响什么。

## 执行模式

| 模式 | 适用信任级别 | 网络 | 文件系统 | 操作者要求 |
|---|---|---|---|---|
| `host` | 受信本地实验 | 继承宿主机 | 一次性临时工作区 | 不能当作安全 sandbox；清理凭据并审查命令 |
| `docker` | 有界 CI/benchmark smoke | `--network=none` | 只读镜像 root + 可写候选工作区 | 核验 Docker daemon、镜像 digest、rootless/daemon 策略和挂载 |
| VM（外部 runner） | 不受信生成代码或受控付费 benchmark | 默认关闭，只允许显式白名单 | 一次性 VM 磁盘，不挂载宿主机工作区 | 独立 kernel、快照/回滚、配额、审计日志和明确 egress 策略 |

内置 Docker executor 会启用 `--cap-drop=ALL`、`--security-opt=no-new-privileges`、`--read-only`、
`--pids-limit 256`、`--memory 1g`、`--cpus 2`，以及 `64m`、`noexec`/`nosuid` 的 `/tmp`。唯一可写
的 bind mount 是一次性 `/workspace`；镜像 root 和网络保持只读/离线。宿主机侧还会用 timeout 和
进程组清理限制候选进程。

## 操作者检查清单

1. `host` 只用于受信代码和本地调试。
2. Docker 模式把 `--container-image` 固定到不可变 digest，核验镜像来源，并确认 daemon 不能访问
   宿主机敏感 socket 或目录。
3. 不受信 benchmark 应放在一次性 VM 内运行，而不是把 container 当成 kernel 边界。除非实验明确需要
   并经过审查，否则关闭网络 egress。
4. provider 凭据放在 runner secret store。候选子进程会清理常见 API/token/password/secret 变量，
   但这只是纵深防御，不能证明任意宿主进程无法读取凭据。
5. 在比较 model bundle 前，把 sandbox 模式、镜像/VM 身份、资源限制和 policy revision 写进 manifest。

## Claim boundary

Docker 路径和 CI smoke 是工程/conformance 证据，不是通用安全保证。本仓库不会自动配置 VM、认证
Docker daemon、证明侧信道隔离，也不声称本地取消后 provider 一定停止生成。真实模型 benchmark 必须
绑定经过审查的 VM/container 策略，并在 ledger 中保留不可变环境身份。
