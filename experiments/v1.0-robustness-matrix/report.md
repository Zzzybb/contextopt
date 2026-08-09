# ContextOpt Level 4 robustness evaluation

This is a deterministic Level 4 runtime robustness evaluation. It measures local fault-injection outcomes for context compaction, source-aware memory invalidation, duplicate tool-result rejection, and compare-and-swap write conflicts without a model provider; it does not prove machine-loss recovery, sandbox security, provider reliability, exactly-once external effects, or coding quality.

| Scenario | Fault contract | Result | Observed evidence | Error |
|---|---|---|---|---|
| `tool-output-compaction` | Compact oversized tool output while preserving sentinels | PASS | `{"compacted_block_ids": ["block-000001"], "compacted_tokens": 80, "deterministic": true, "has_head_sentinel": true, "has_tail_sentinel": true, "max_tool_output_tokens": 80, "selected_tokens": 132}` |  |
| `stale-memory-invalidation` | Invalidate source-aware memory after a workspace change | PASS | `{"invalidated_ids": ["mem-3141790e193f0985e529479539bd848a"], "memory_id": "mem-3141790e193f0985e529479539bd848a", "persisted_status": "invalidated", "retrieved_after": [], "retrieved_before": ["mem-3141790e193f0985e529479539bd848a"], "revision": 2}` |  |
| `duplicate-tool-result` | Reject a repeated tool result inside one exchange | PASS | `{"error": "duplicate tool result at message index 0", "rejected": true}` |  |
| `cas-write-conflict` | Refuse a stale compare-and-swap workspace write | PASS | `{"error_code": "content_conflict", "read_ok": true, "stale_sha_present": true, "workspace_unchanged": true, "write_ok": false}` |  |

## Summary

- Scenarios: `4`; passed: `4`; failed: `0`.
- The matrix is provider-free and intentionally does not claim machine-loss recovery, exactly-once effects, security isolation, or coding quality.
