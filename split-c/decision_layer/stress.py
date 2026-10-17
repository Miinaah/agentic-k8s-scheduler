"""
decision_layer/stress.py

Cluster stress metric calculation and rule-based priority mapping.

Components combined in stress calculation:
1. Average utilisation across all nodes (CPU and Memory)
2. Utilisation variance across nodes (imbalance metric)
3. Pending pod queue length (backlog metric)
4. Total recent failures across all nodes (reliability/instability metric)

Each component is normalized to [0, 1]. Documented weights sum to 1.0.
"""

from __future__ import annotations

import math
from typing import Any, Literal
from decision_layer.schemas import (
    ClusterState,
    PriorityOrder,
)

# ==============================================================================
# Documented Component Weights (Sum = 1.0)
# ==============================================================================
# Primary emphasis on current resource utilization
WEIGHT_UTILISATION: float = 0.40

# Significant weight on recent failure events (node flapping, pod evictions, OOMs)
WEIGHT_FAILURES: float = 0.35

# Pending scheduling queue backlog
WEIGHT_QUEUE: float = 0.15

# Imbalance / fragmentation across nodes
WEIGHT_VARIANCE: float = 0.10

# ==============================================================================
# Normalization Parameters
# ==============================================================================
# Failures normalization cap: 5 recent failures across cluster represents critical instability
MAX_RECENT_FAILURES: float = 5.0

# Queue backlog normalization cap: 20 pending pods represents heavy scheduler backlog
MAX_PENDING_QUEUE: float = 20.0

# Maximum possible variance for numbers bounded in [0, 1] is 0.25 (Popoviciu's inequality)
MAX_VARIANCE: float = 0.25

# ==============================================================================
# Stress Level Thresholds
# ==============================================================================
DEFAULT_LOW_THRESHOLD: float = 0.25
DEFAULT_HIGH_THRESHOLD: float = 0.50

StressLevel = Literal["low", "medium", "high"]


def _extract_node_metrics(node: Any) -> tuple[float, int]:
    """
    Extract (node_utilisation, recent_failures) for a given node.
    Utilisation is the balanced mean of CPU and memory utilisation in [0, 1].
    """
    if isinstance(node, dict):
        cpu_cap = float(node.get("cpu_capacity", 1.0))
        cpu_used = float(node.get("cpu_used", 0.0))
        mem_cap = float(node.get("mem_capacity_gb", 1.0))
        mem_used = float(node.get("mem_used_gb", 0.0))
        failures = int(node.get("recent_failures", 0))
    else:
        cpu_cap = float(node.cpu_capacity)
        cpu_used = float(node.cpu_used)
        mem_cap = float(node.mem_capacity_gb)
        mem_used = float(node.mem_used_gb)
        failures = int(node.recent_failures)

    cpu_ratio = min(1.0, max(0.0, cpu_used / cpu_cap if cpu_cap > 0 else 0.0))
    mem_ratio = min(1.0, max(0.0, mem_used / mem_cap if mem_cap > 0 else 0.0))
    node_util = 0.5 * cpu_ratio + 0.5 * mem_ratio

    return node_util, max(0, failures)


def compute_stress(
    cluster_state: ClusterState | dict[str, Any],
    pending_queue_len: int = 0,
) -> float:
    """
    Compute aggregate cluster stress in [0, 1].

    Combines:
    - Average node resource utilisation (CPU + Memory)
    - Utilisation variance across nodes (resource imbalance)
    - Pending queue backlog length
    - Total recent failures across all nodes

    Parameters:
    - cluster_state: ClusterState instance or snapshot dict.
    - pending_queue_len: Count of pods waiting in scheduling queue (>= 0).

    Returns:
    - stress: Float in [0, 1].
    """
    if isinstance(cluster_state, ClusterState):
        nodes = cluster_state.nodes
    elif isinstance(cluster_state, dict):
        nodes = cluster_state.get("nodes", [])
    else:
        raise TypeError(f"Expected ClusterState or dict, got {type(cluster_state).__name__}")

    if not nodes:
        return 0.0

    # 1. Collect per-node utilisation and failure metrics
    node_utils: list[float] = []
    total_failures = 0
    for node in nodes:
        util, failures = _extract_node_metrics(node)
        node_utils.append(util)
        total_failures += failures

    # 2. Average utilisation component in [0, 1]
    avg_util = sum(node_utils) / len(node_utils)

    # 3. Variance component in [0, 1]
    if len(node_utils) > 1:
        variance = sum((u - avg_util) ** 2 for u in node_utils) / len(node_utils)
    else:
        variance = 0.0
    norm_variance = min(1.0, max(0.0, variance / MAX_VARIANCE))

    # 4. Pending queue backlog component in [0, 1]
    queue_len = max(0, int(pending_queue_len))
    norm_queue = min(1.0, max(0.0, queue_len / MAX_PENDING_QUEUE))

    # 5. Total recent failures component in [0, 1]
    norm_failures = min(1.0, max(0.0, total_failures / MAX_RECENT_FAILURES))

    # Weighted combination
    stress = (
        WEIGHT_UTILISATION * avg_util
        + WEIGHT_VARIANCE * norm_variance
        + WEIGHT_QUEUE * norm_queue
        + WEIGHT_FAILURES * norm_failures
    )

    return max(0.0, min(1.0, float(stress)))


def stress_level(
    stress: float,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> StressLevel:
    """
    Map continuous stress metric to categorical stress level: "low" | "medium" | "high".

    Parameters:
    - stress: Float in [0, 1].
    - low_threshold: Upper bound for "low" level (exclusive).
    - high_threshold: Lower bound for "high" level (inclusive).

    Returns:
    - "low" if stress < low_threshold
    - "high" if stress >= high_threshold
    - "medium" otherwise
    """
    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be strictly less than high_threshold ({high_threshold})"
        )

    # Handle numeric bounds clamping
    val = max(0.0, min(1.0, float(stress)))

    if val < low_threshold:
        return "low"
    elif val >= high_threshold:
        return "high"
    else:
        return "medium"


def rule_based_priority(stress: str | float) -> PriorityOrder:
    """
    Map stress level to a deterministic PriorityOrder with source="rule".

    Priority mappings:
    - High stress -> fault_tolerance first: ["fault_tolerance", "utilisation", "cost"]
    - Medium stress -> utilisation first: ["utilisation", "fault_tolerance", "cost"]
    - Low stress -> cost first: ["cost", "utilisation", "fault_tolerance"]

    Parameters:
    - stress: Categorical level ("low", "medium", "high") or numeric stress float in [0, 1].

    Returns:
    - PriorityOrder instance with validated objectives and justification.
    """
    if isinstance(stress, (int, float)):
        level: str = stress_level(float(stress))
    elif isinstance(stress, str):
        level = stress.strip().lower()
    else:
        raise TypeError(f"Expected str or float for stress, got {type(stress).__name__}")

    if level == "high":
        return PriorityOrder(
            priority_order=["fault_tolerance", "utilisation", "cost"],
            justification=(
                "High cluster stress detected. Prioritizing fault tolerance to preserve "
                "workload stability and prevent cascading node failures."
            ),
            confidence=0.95,
            source="rule",
        )
    elif level == "medium":
        return PriorityOrder(
            priority_order=["utilisation", "fault_tolerance", "cost"],
            justification=(
                "Moderate cluster stress detected. Prioritizing resource utilisation to optimize "
                "bin-packing efficiency while maintaining resilience."
            ),
            confidence=0.90,
            source="rule",
        )
    elif level == "low":
        return PriorityOrder(
            priority_order=["cost", "utilisation", "fault_tolerance"],
            justification=(
                "Low cluster stress detected. Prioritizing cost efficiency to minimize cloud "
                "infrastructure expenditure under calm operational conditions."
            ),
            confidence=0.95,
            source="rule",
        )
    else:
        raise ValueError(
            f"Invalid stress_level '{stress}'. Expected 'low', 'medium', or 'high'."
        )
