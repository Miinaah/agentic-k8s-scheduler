"""
decision_layer/tests/test_lexicographic.py

Unit tests for lexicographic filtering with tolerance (select_node).
"""

import pytest
from pydantic import ValidationError

from decision_layer.lexicographic import select_node
from decision_layer.schemas import (
    NodeScores,
    PriorityOrder,
    SingleNodeScore,
)


def test_clear_winner_on_top_objective():
    """
    Scenario 1: Clear winner on the top objective.
    node-alpha beats node-beta by more than epsilon (0.95 vs 0.85, diff 0.10 > 0.05).
    Even though node-beta has vastly superior utilisation and cost, node-alpha wins immediately.
    """
    scores = [
        SingleNodeScore(node_id="node-alpha", fault_tolerance=0.95, utilisation=0.20, cost=0.10),
        SingleNodeScore(node_id="node-beta", fault_tolerance=0.85, utilisation=0.99, cost=0.99),
        SingleNodeScore(node_id="node-gamma", fault_tolerance=0.70, utilisation=0.95, cost=0.95),
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]
    winner = select_node(scores, priority, epsilon=0.05)
    assert winner == "node-alpha"


def test_near_tie_within_epsilon_decided_by_second_objective():
    """
    Scenario 2: Near-tie within epsilon on objective 1, decided by objective 2.
    Top objective is fault_tolerance:
      - node-alpha: 0.90
      - node-beta:  0.88 (within 0.02 of 0.90 <= epsilon 0.05) -> survives!
      - node-gamma: 0.75 (0.15 behind > 0.05) -> eliminated
    Second objective is utilisation:
      - node-beta:  0.95
      - node-alpha: 0.50 (0.45 behind > 0.05) -> eliminated
    Winner: node-beta.
    """
    scores = [
        SingleNodeScore(node_id="node-alpha", fault_tolerance=0.90, utilisation=0.50, cost=0.30),
        SingleNodeScore(node_id="node-beta", fault_tolerance=0.88, utilisation=0.95, cost=0.30),
        SingleNodeScore(node_id="node-gamma", fault_tolerance=0.75, utilisation=0.98, cost=0.90),
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]
    winner = select_node(scores, priority, epsilon=0.05)
    assert winner == "node-beta"


def test_tie_broken_by_third_objective():
    """
    Scenario 3: Ties within epsilon across the first two objectives, broken by the third.
    Priority order: ["fault_tolerance", "utilisation", "cost"], epsilon = 0.05.
    Obj 1 (fault_tolerance):
      - max = 0.92 (alpha). Cutoff = 0.87.
      - alpha=0.92, beta=0.90, gamma=0.91 -> all survive.
    Obj 2 (utilisation):
      - max = 0.85 (alpha). Cutoff = 0.80.
      - alpha=0.85, beta=0.84, gamma=0.82 -> all survive.
    Obj 3 (cost):
      - max = 0.95 (gamma). Cutoff = 0.90.
      - gamma=0.95 survives.
      - alpha=0.40, beta=0.50 are eliminated.
    Winner: node-gamma.
    """
    scores = [
        SingleNodeScore(node_id="node-alpha", fault_tolerance=0.92, utilisation=0.85, cost=0.40),
        SingleNodeScore(node_id="node-beta", fault_tolerance=0.90, utilisation=0.84, cost=0.50),
        SingleNodeScore(node_id="node-gamma", fault_tolerance=0.91, utilisation=0.82, cost=0.95),
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]
    winner = select_node(scores, priority, epsilon=0.05)
    assert winner == "node-gamma"


def test_same_scores_different_winners_under_different_priority_orders():
    """
    Scenario 4: The same score matrix produces different winners under different priority orders.
    node-resilient excels at fault_tolerance.
    node-packed excels at utilisation.
    node-thrifty excels at cost.
    """
    scores = [
        SingleNodeScore(node_id="node-resilient", fault_tolerance=0.95, utilisation=0.40, cost=0.30),
        SingleNodeScore(node_id="node-packed", fault_tolerance=0.40, utilisation=0.95, cost=0.30),
        SingleNodeScore(node_id="node-thrifty", fault_tolerance=0.30, utilisation=0.40, cost=0.95),
    ]

    # Order 1: fault_tolerance first -> node-resilient wins
    order_ft = ["fault_tolerance", "utilisation", "cost"]
    assert select_node(scores, order_ft, epsilon=0.05) == "node-resilient"

    # Order 2: utilisation first -> node-packed wins
    order_ut = ["utilisation", "cost", "fault_tolerance"]
    assert select_node(scores, order_ut, epsilon=0.05) == "node-packed"

    # Order 3: cost first -> node-thrifty wins
    order_cost = ["cost", "fault_tolerance", "utilisation"]
    assert select_node(scores, order_cost, epsilon=0.05) == "node-thrifty"


def test_all_equal_scores_deterministic_tie_breaker():
    """
    Scenario 5: All-equal scores across all objectives.
    All nodes survive all three filters.
    Tie must be broken deterministically by lowest node_id (lexicographical).
    """
    scores = [
        SingleNodeScore(node_id="node-zeta", fault_tolerance=0.80, utilisation=0.80, cost=0.80),
        SingleNodeScore(node_id="node-alpha", fault_tolerance=0.80, utilisation=0.80, cost=0.80),
        SingleNodeScore(node_id="node-mu", fault_tolerance=0.80, utilisation=0.80, cost=0.80),
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]

    # Regardless of input list order, lowest node_id "node-alpha" must win
    assert select_node(scores, priority, epsilon=0.05) == "node-alpha"

    # Shuffled input order
    shuffled = [scores[1], scores[0], scores[2]]
    assert select_node(shuffled, priority, epsilon=0.05) == "node-alpha"


def test_single_node_edge_case():
    """
    Scenario 6: Single node edge case.
    Should immediately return that node's id.
    """
    scores = [
        SingleNodeScore(node_id="node-solo", fault_tolerance=0.50, utilisation=0.50, cost=0.50)
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]
    assert select_node(scores, priority, epsilon=0.05) == "node-solo"


def test_empty_list_raises_value_error():
    """
    Scenario 7: Empty list edge cases raise ValueError.
    """
    priority = ["fault_tolerance", "utilisation", "cost"]

    with pytest.raises(ValueError, match="empty"):
        select_node([], priority, epsilon=0.05)

    with pytest.raises(ValueError, match="empty"):
        select_node(NodeScores(pod_id="pod-test", scores=[]), priority, epsilon=0.05)


def test_invalid_priority_order():
    """
    Scenario 8: Invalid priority order edge cases raise ValueError.
    """
    scores = [
        SingleNodeScore(node_id="node-1", fault_tolerance=0.8, utilisation=0.8, cost=0.8)
    ]

    # Fewer than 3
    with pytest.raises(ValueError, match="exactly 3"):
        select_node(scores, ["fault_tolerance", "utilisation"], epsilon=0.05)

    # More than 3
    with pytest.raises(ValueError, match="exactly 3"):
        select_node(scores, ["fault_tolerance", "utilisation", "cost", "speed"], epsilon=0.05)

    # Unknown objective
    with pytest.raises(ValueError, match="objectives"):
        select_node(scores, ["fault_tolerance", "utilisation", "latency"], epsilon=0.05)

    # Duplicate objective
    with pytest.raises(ValueError):
        select_node(scores, ["cost", "cost", "utilisation"], epsilon=0.05)

    # Invalid type
    with pytest.raises(ValueError):
        select_node(scores, None, epsilon=0.05)  # type: ignore


def test_negative_epsilon_raises_value_error():
    """
    Negative epsilon should raise ValueError.
    """
    scores = [
        SingleNodeScore(node_id="node-1", fault_tolerance=0.8, utilisation=0.8, cost=0.8)
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]

    with pytest.raises(ValueError, match="non-negative"):
        select_node(scores, priority, epsilon=-0.01)


def test_pydantic_model_integration():
    """
    Integration test with Pydantic NodeScores and PriorityOrder models.
    """
    node_scores = NodeScores(
        pod_id="pod-production-101",
        scores=[
            SingleNodeScore(node_id="worker-01", fault_tolerance=0.92, utilisation=0.60, cost=0.40),
            SingleNodeScore(node_id="worker-02", fault_tolerance=0.90, utilisation=0.95, cost=0.50),
        ],
    )
    priority_order = PriorityOrder(
        priority_order=["fault_tolerance", "utilisation", "cost"],
        justification="System is healthy, balance utilisation next",
        confidence=0.90,
        source="rule",
    )

    # worker-01 (0.92) and worker-02 (0.90) tie on fault_tolerance within epsilon=0.05
    # worker-02 wins on utilisation (0.95 vs 0.60)
    winner = select_node(node_scores, priority_order, epsilon=0.05)
    assert winner == "worker-02"


def test_dict_inputs_compatibility():
    """
    select_node should accept plain dictionaries matching the node score schema.
    """
    scores_dict = [
        {"node_id": "node-A", "fault_tolerance": 0.95, "utilisation": 0.30, "cost": 0.20},
        {"node_id": "node-B", "fault_tolerance": 0.70, "utilisation": 0.90, "cost": 0.90},
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]
    assert select_node(scores_dict, priority, epsilon=0.05) == "node-A"


def test_exact_epsilon_boundary():
    """
    Verify exact epsilon boundary condition:
    max_score = 0.90, epsilon = 0.05 -> score 0.85 must survive.
    """
    scores = [
        SingleNodeScore(node_id="node-top", fault_tolerance=0.90, utilisation=0.50, cost=0.50),
        SingleNodeScore(node_id="node-exact-cutoff", fault_tolerance=0.85, utilisation=0.95, cost=0.50),
        SingleNodeScore(node_id="node-below-cutoff", fault_tolerance=0.849, utilisation=0.99, cost=0.50),
    ]
    priority = ["fault_tolerance", "utilisation", "cost"]
    # node-exact-cutoff survives round 1 (0.85 >= 0.90 - 0.05) and then beats node-top on utilisation
    winner = select_node(scores, priority, epsilon=0.05)
    assert winner == "node-exact-cutoff"
