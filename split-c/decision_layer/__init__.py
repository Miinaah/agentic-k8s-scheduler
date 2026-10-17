"""
decision_layer package.

Decision layer for an agentic Kubernetes scheduler.
"""

from decision_layer.lexicographic import select_node
from decision_layer.schemas import (
    ClusterState,
    NodeScores,
    NodeState,
    Objective,
    PendingPod,
    PriorityOrder,
    SchedulingDecision,
    SingleNodeScore,
    VALID_OBJECTIVES,
)
from decision_layer.heuristic_scorer import score_nodes
from decision_layer.llm_agent import (
    AnthropicLLMClient,
    FakeLLMClient,
    LLMClientProtocol,
    SYSTEM_PROMPT,
    get_priority_order,
)
from decision_layer.stress import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    WEIGHT_FAILURES,
    WEIGHT_QUEUE,
    WEIGHT_UTILISATION,
    WEIGHT_VARIANCE,
    compute_stress,
    rule_based_priority,
    stress_level,
)

__all__ = [
    "ClusterState",
    "NodeScores",
    "NodeState",
    "Objective",
    "PendingPod",
    "PriorityOrder",
    "SchedulingDecision",
    "SingleNodeScore",
    "VALID_OBJECTIVES",
    "score_nodes",
    "select_node",
    "compute_stress",
    "stress_level",
    "rule_based_priority",
    "get_priority_order",
    "AnthropicLLMClient",
    "FakeLLMClient",
    "LLMClientProtocol",
    "SYSTEM_PROMPT",
    "decide",
    "WEIGHT_UTILISATION",
    "WEIGHT_FAILURES",
    "WEIGHT_QUEUE",
    "WEIGHT_VARIANCE",
    "DEFAULT_LOW_THRESHOLD",
    "DEFAULT_HIGH_THRESHOLD",
]


def __getattr__(name: str):
    if name == "decide":
        from decision_layer.pipeline import decide
        return decide
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
