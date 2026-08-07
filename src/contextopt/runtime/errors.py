"""Stable error types used at runtime boundaries."""

from __future__ import annotations


class RuntimeContractError(RuntimeError):
    """The model or runtime violated a deterministic protocol invariant."""


class ModelError(RuntimeError):
    """A normalized model-provider failure."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
