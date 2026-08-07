"""Built-in context selection policies."""

from contextopt.policies.base import ContextPolicy, UnsupportedProblemError
from contextopt.policies.knapsack import KnapsackPolicy
from contextopt.policies.oracle import ExactOraclePolicy
from contextopt.policies.rank import DensityPolicy, TopKPolicy
from contextopt.policies.submodular import SubmodularGreedyPolicy

POLICIES: dict[str, type[ContextPolicy]] = {
    "topk": TopKPolicy,
    "density": DensityPolicy,
    "knapsack": KnapsackPolicy,
    "submodular": SubmodularGreedyPolicy,
    "oracle": ExactOraclePolicy,
}


def create_policy(name: str) -> ContextPolicy:
    try:
        policy_type = POLICIES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown policy {name!r}; choose from {', '.join(sorted(POLICIES))}"
        ) from exc
    return policy_type()


__all__ = [
    "POLICIES",
    "ContextPolicy",
    "DensityPolicy",
    "ExactOraclePolicy",
    "KnapsackPolicy",
    "SubmodularGreedyPolicy",
    "TopKPolicy",
    "UnsupportedProblemError",
    "create_policy",
]
