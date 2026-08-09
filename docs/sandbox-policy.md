# Candidate execution sandbox policy

This policy separates the trust boundary of generated code from the correctness oracle. The
runtime always validates a candidate snapshot and records the test result; the execution
environment determines what a malicious or accidental candidate can affect.

## Execution modes

| Mode | Intended trust | Network | Filesystem | Required operator stance |
|---|---|---|---|---|
| `host` | trusted local experiments | inherited | disposable temporary workspace | never treat as a security sandbox; scrub credentials and review commands |
| `docker` | bounded CI/benchmark smoke | `--network=none` | read-only image root plus writable candidate workspace | verify Docker daemon, image digest, rootless/daemon policy, and host mounts |
| VM (external runner) | untrusted generated code or controlled paid benchmark | disabled by default, explicit allowlist only | ephemeral VM disk; no host workspace mounts | separate kernel, snapshot/rollback, quotas, audit logs, and explicit egress policy |

The built-in Docker executor applies `--cap-drop=ALL`, `--security-opt=no-new-privileges`,
`--read-only`, `--pids-limit 256`, `--memory 1g`, `--cpus 2`, and a `64m` `noexec`/`nosuid`
`/tmp`. The only writable bind mount is the disposable `/workspace`; the image root and network
remain read-only/offline. Candidate processes are still bounded by a timeout and process-group
cleanup on the host side.

## Operator checklist

1. Use `host` only for trusted code and local debugging.
2. For Docker, pin `--container-image` to an immutable digest, verify the image provenance, and
   ensure the daemon cannot reach host-sensitive sockets or directories.
3. For an untrusted benchmark, run the candidate executor inside an ephemeral VM instead of
   relying on a container as a kernel boundary. Disable network egress unless the experiment
   explicitly requires a reviewed endpoint.
4. Keep provider credentials in the runner secret store. The candidate child environment removes
   common API/token/password/secret variables, but this is defense-in-depth rather than proof that
   an arbitrary host process cannot access secrets.
5. Preserve the sandbox mode, image/VM identity, resource limits, and policy revision in the
   evaluation manifest before comparing model bundles.

## Claim boundary

The Docker path and its CI smoke are engineering/conformance evidence, not a universal security
guarantee. This repository does not provision a VM, certify a Docker daemon, prove side-channel
isolation, or claim that an external provider stopped generation after local cancellation. A real
model benchmark must attach a reviewed VM/container policy and retain its immutable environment
identity in the ledger.
