"""
decision_layer/tests/test_stress.py

Unit tests for cluster stress calculation and rule-based priority mapping.
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from decision_layer.schemas import (
    ClusterState,
    NodeState,
    PendingPod,
    PriorityOrder,
    VALID_OBJECTIVES,
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


def _make_node(
    node_id: str,
    cpu_cap: float = 16.0,
    cpu_used: float = 1.6,
    mem_cap: float = 64.0,
    mem_used: float = 6.4,
    failures: int = 0,
) -> NodeState:
    return NodeState(
        node_id=node_id,
        cpu_capacity=cpu_cap,
        cpu_used=cpu_used,
        mem_capacity_gb=mem_cap,
        mem_used_gb=mem_used,
        disk_capacity_gb=500.0,
        disk_used_gb=50.0,
        health_score=1.0,
        cost_per_hour=0.24,
        zone="us-central1-a",
        pod_count=5,
        recent_failures=failures,
    )


def _make_cluster(nodes: list[NodeState]) -> ClusterState:
    return ClusterState(
        timestamp="2026-10-11T16:00:00Z",
        nodes=nodes,
        pending_pod=PendingPod(
            pod_id="test-pod",
            cpu_request=1.0,
            mem_request_gb=2.0,
            priority_class="default",
            qos_class="Burstable",
            estimated_duration_sec=300.0,
        ),
    )


def test_weights_sum_to_one():
    """Verify that module component weights sum to 1.0."""
    total_weights = WEIGHT_UTILISATION + WEIGHT_FAILURES + WEIGHT_QUEUE + WEIGHT_VARIANCE
    assert pytest.approx(total_weights, abs=1e-6) == 1.0


def test_idle_cluster_reads_low():
    """
    Test 1: Idle cluster reads low.
    A cluster with low CPU and RAM usage (~10%), 0 failures, and 0 queue
    should produce a low stress metric and map to 'low'.
    """
    nodes = [
        _make_node("n1", cpu_used=1.0, mem_used=4.0, failures=0),  # ~6%
        _make_node("n2", cpu_used=1.5, mem_used=6.0, failures=0),  # ~9%
        _make_node("n3", cpu_used=1.2, mem_used=5.0, failures=0),  # ~7%
    ]
    cluster = _make_cluster(nodes)
    stress = compute_stress(cluster, pending_queue_len=0)

    assert 0.0 <= stress < DEFAULT_LOW_THRESHOLD
    assert stress_level(stress) == "low"


def test_calm_mock_cluster_reads_low():
    """Verify the realistic calm_cluster.json mock reads as 'low' stress."""
    calm_path = Path(__file__).resolve().parent.parent / "mocks" / "calm_cluster.json"
    data = json.loads(calm_path.read_text())
    cluster = ClusterState.model_validate(data)

    stress = compute_stress(cluster, pending_queue_len=0)
    assert stress < DEFAULT_LOW_THRESHOLD
    assert stress_level(stress) == "low"


def test_high_utilisation_with_three_failures_reads_high():
    """
    Test 2: A cluster with 3 recent failures and high utilisation reads high.
    Nodes operating near ~85% resource utilisation combined with 3 failures across
    the cluster should evaluate to 'high' stress.
    """
    nodes = [
        _make_node("n1", cpu_cap=16.0, cpu_used=13.6, mem_cap=64.0, mem_used=54.4, failures=2),
        _make_node("n2", cpu_cap=16.0, cpu_used=13.6, mem_cap=64.0, mem_used=54.4, failures=1),
        _make_node("n3", cpu_cap=16.0, cpu_used=13.6, mem_cap=64.0, mem_used=54.4, failures=0),
    ]
    cluster = _make_cluster(nodes)
    stress = compute_stress(cluster, pending_queue_len=0)

    assert stress >= DEFAULT_HIGH_THRESHOLD
    assert stress_level(stress) == "high"


def test_stressed_mock_cluster_reads_high():
    """Verify the realistic stressed_cluster.json mock reads as 'high' stress."""
    stressed_path = Path(__file__).resolve().parent.parent / "mocks" / "stressed_cluster.json"
    data = json.loads(stressed_path.read_text())
    cluster = ClusterState.model_validate(data)

    stress = compute_stress(cluster, pending_queue_len=0)
    assert stress >= DEFAULT_HIGH_THRESHOLD
    assert stress_level(stress) == "high"


def test_stress_rises_monotonically_with_failures():
    """
    Test 3: Stress rises monotonically when adding failures.
    Holding utilisation and queue constant, each additional failure must strictly
    increase (or non-decrease when capped) the stress metric.
    """
    stresses: list[float] = []
    for f in range(7):
        nodes = [
            _make_node("n1", cpu_used=8.0, mem_used=32.0, failures=f),
            _make_node("n2", cpu_used=8.0, mem_used=32.0, failures=0),
        ]
        cluster = _make_cluster(nodes)
        stresses.append(compute_stress(cluster))

    # Stress strictly increases for failures below normalisation cap
    for i in range(len(stresses) - 1):
        assert stresses[i] <= stresses[i + 1], f"Failed monotonicity at {i}: {stresses[i]} > {stresses[i+1]}"

    # Prior to capping (0 to 5 failures), stress strictly rises
    for i in range(5):
        assert stresses[i] < stresses[i + 1], f"Expected strict increase: {stresses[i]} vs {stresses[i+1]}"


def test_stress_rises_with_pending_queue():
    """Adding pending pods to the queue should increase the computed stress."""
    nodes = [_make_node("n1", cpu_used=8.0, mem_used=32.0, failures=0)]
    cluster = _make_cluster(nodes)

    stress_q0 = compute_stress(cluster, pending_queue_len=0)
    stress_q5 = compute_stress(cluster, pending_queue_len=5)
    stress_q10 = compute_stress(cluster, pending_queue_len=10)

    assert stress_q0 < stress_q5 < stress_q10


def test_rule_based_priority_mappings():
    """
    Test 4: The mapping returns a valid PriorityOrder for each level.
    - High -> fault_tolerance first
    - Low -> cost first
    - Medium -> utilisation first
    """
    # High stress level
    po_high = rule_based_priority("high")
    assert isinstance(po_high, PriorityOrder)
    assert po_high.source == "rule"
    assert po_high.priority_order[0] == "fault_tolerance"
    assert set(po_high.priority_order) == set(VALID_OBJECTIVES)
    assert len(po_high.justification) > 10
    assert 0.0 <= po_high.confidence <= 1.0

    # Medium stress level
    po_med = rule_based_priority("medium")
    assert isinstance(po_med, PriorityOrder)
    assert po_med.source == "rule"
    assert po_med.priority_order[0] == "utilisation"
    assert set(po_med.priority_order) == set(VALID_OBJECTIVES)
    assert len(po_med.justification) > 10
    assert 0.0 <= po_med.confidence <= 1.0

    # Low stress level
    po_low = rule_based_priority("low")
    assert isinstance(po_low, PriorityOrder)
    assert po_low.source == "rule"
    assert po_low.priority_order[0] == "cost"
    assert set(po_low.priority_order) == set(VALID_OBJECTIVES)
    assert len(po_low.justification) > 10
    assert 0.0 <= po_low.confidence <= 1.0


def test_rule_based_priority_with_float_inputs():
    """rule_based_priority should also directly accept float stress values."""
    po_low = rule_based_priority(0.10)
    assert po_low.priority_order[0] == "cost"

    po_med = rule_based_priority(0.35)
    assert po_med.priority_order[0] == "utilisation"

    po_high = rule_based_priority(0.75)
    assert po_high.priority_order[0] == "fault_tolerance"


def test_configurable_thresholds():
    """Verify stress_level supports custom low and high thresholds."""
    # With custom thresholds: [0, 0.40) low, [0.40, 0.70) med, [0.70, 1.0] high
    assert stress_level(0.35, low_threshold=0.40, high_threshold=0.70) == "low"
    assert stress_level(0.50, low_threshold=0.40, high_threshold=0.70) == "medium"
    assert stress_level(0.72, low_threshold=0.40, high_threshold=0.70) == "high"

    # Inverted thresholds should raise ValueError
    with pytest.raises(ValueError, match="strictly less than"):
        stress_level(0.50, low_threshold=0.60, high_threshold=0.40)


def test_invalid_stress_inputs():
    """Invalid stress inputs should raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid stress_level"):
        rule_based_priority("critical")

    with pytest.raises(TypeError):
        rule_based_priority(None)  # type: ignore

    with pytest.raises(TypeError):
        compute_stress("invalid_cluster_state")  # type: ignore


def test_empty_cluster_nodes():
    """ClusterState with empty nodes dict should return 0.0 stress."""
    assert compute_stress({"nodes": []}) == 0.0
