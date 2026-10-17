"""
decision_layer/lexicographic.py

Lexicographic filtering with tolerance (epsilon) for Kubernetes scheduling.

Logic:
1. Validate candidate nodes, priority order, and epsilon.
2. For each objective in priority_order:
   - Identify the maximum score among surviving nodes.
   - Retain all nodes whose score is within epsilon of that maximum (score >= max_score - epsilon).
   - If only one node survives, continue or break.
3. If more than one node remains after all objectives, break ties deterministically
   by selecting the lowest node_id (lexicographical string order).
"""

from __future__ import annotations

import math
from typing import Any, Sequence
from decision_layer.schemas import (
    NodeScores,
    Objective,
    PriorityOrder,
    SingleNodeScore,
    VALID_OBJECTIVES,
)


def _get_node_id(node: SingleNodeScore | dict[str, Any] | Any) -> str:
    """Extract node_id from SingleNodeScore, dict, or object with node_id attribute."""
    if isinstance(node, dict):
        if "node_id" not in node:
            raise KeyError("Candidate dict missing required 'node_id' key")
        return str(node["node_id"])
    if hasattr(node, "node_id"):
        return str(node.node_id)
    raise TypeError(f"Cannot extract 'node_id' from candidate of type {type(node).__name__}")


def _get_score(node: SingleNodeScore | dict[str, Any] | Any, objective: str) -> float:
    """Extract numeric score for a given objective."""
    if isinstance(node, dict):
        if objective not in node:
            raise KeyError(f"Candidate dict missing required objective '{objective}'")
        val = node[objective]
    elif hasattr(node, objective):
        val = getattr(node, objective)
    else:
        raise AttributeError(f"Candidate {node} has no attribute or key '{objective}'")

    if not isinstance(val, (int, float)):
        raise TypeError(f"Score for objective '{objective}' must be numeric, got {type(val).__name__}")
    return float(val)


def _normalize_priority_order(
    priority_order: PriorityOrder | Sequence[str] | Any,
) -> list[str]:
    """Validate and return priority_order as a list of strings."""
    if isinstance(priority_order, PriorityOrder):
        order = list(priority_order.priority_order)
    elif isinstance(priority_order, (list, tuple)):
        order = list(priority_order)
    else:
        raise ValueError(
            f"Invalid priority_order type: {type(priority_order).__name__}. "
            "Must be a PriorityOrder instance or list/tuple of objective strings."
        )

    if len(order) != 3:
        raise ValueError(
            f"priority_order must contain exactly 3 objectives, got {len(order)}: {order}"
        )

    order_set = set(order)
    expected_set = set(VALID_OBJECTIVES)
    if order_set != expected_set:
        raise ValueError(
            f"priority_order must contain exactly the 3 objectives {VALID_OBJECTIVES}. Got: {order}"
        )

    # Check for duplicates (in case len is 3 but has duplicates like ['cost', 'cost', 'utilisation'])
    if len(order_set) != 3:
        raise ValueError(f"priority_order contains duplicate objectives: {order}")

    return order


def _normalize_node_scores(
    node_scores: NodeScores | Sequence[SingleNodeScore | dict[str, Any]] | Any,
) -> list[Any]:
    """Validate and return candidate nodes list."""
    if isinstance(node_scores, NodeScores):
        candidates = list(node_scores.scores)
    elif isinstance(node_scores, (list, tuple)):
        candidates = list(node_scores)
    else:
        raise ValueError(
            f"Invalid node_scores type: {type(node_scores).__name__}. "
            "Must be a NodeScores instance or a list/tuple of node score objects."
        )

    if len(candidates) == 0:
        raise ValueError("Cannot select node from an empty list of candidate scores.")

    return candidates


def select_node(
    node_scores: NodeScores | Sequence[SingleNodeScore | dict[str, Any]],
    priority_order: PriorityOrder | Sequence[str],
    epsilon: float = 0.05,
) -> str:
    """
    Select the winning node using lexicographic filtering with tolerance (epsilon).

    Parameters:
    - node_scores: NodeScores instance or sequence of SingleNodeScore / dicts.
    - priority_order: PriorityOrder instance or sequence of 3 valid objective names.
    - epsilon: Tolerance margin within which scores are considered practically tied (default: 0.05).

    Returns:
    - node_id: Selected node's string identifier.

    Edge Cases:
    - Empty candidate list: raises ValueError.
    - Single candidate node: returns its node_id (after validating priority_order).
    - Invalid priority_order (wrong count, unknown objectives, duplicates): raises ValueError.
    - Negative epsilon: raises ValueError.
    - Ties after all objectives: picks lowest node_id lexicographically.
    """
    if epsilon < 0.0 or math.isnan(epsilon):
        raise ValueError(f"epsilon must be a non-negative real number, got {epsilon}")

    # Validate priority order first
    order = _normalize_priority_order(priority_order)

    # Validate candidate nodes
    survivors = _normalize_node_scores(node_scores)

    # Single node edge case
    if len(survivors) == 1:
        return _get_node_id(survivors[0])

    # Float comparison tolerance to prevent precision edge-case dropouts
    precision_tolerance = 1e-9

    for obj in order:
        if len(survivors) <= 1:
            break

        # Calculate maximum score for current objective
        max_score = max(_get_score(node, obj) for node in survivors)
        cutoff = max_score - epsilon

        # Retain nodes within epsilon of the max
        survivors = [
            node for node in survivors
            if _get_score(node, obj) >= (cutoff - precision_tolerance)
        ]

    # If only one survivor remains, return it
    if len(survivors) == 1:
        return _get_node_id(survivors[0])

    # Multiple survivors remaining: tie-break deterministically by lowest node_id
    best_node = min(survivors, key=_get_node_id)
    return _get_node_id(best_node)
