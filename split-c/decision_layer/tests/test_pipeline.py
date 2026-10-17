"""
decision_layer/tests/test_pipeline.py

End-to-end scenario tests for the scheduling decision pipeline (decide).
Validates input/expected-output pairs across the 5 required scenarios:
(1) Two nodes just failed and maintenance in 10 min -> FT first.
(2) Calm cluster and overspend flagged -> Cost first.
(3) Bursty traffic forecast with uneven utilisation -> Utilisation first.
(4) LLM returns garbage -> Rule fallback produces valid decision.
(5) Scenario where the static rule and the LLM choose different nodes.
"""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from decision_layer.heuristic_scorer import score_nodes
from decision_layer.llm_agent import FakeLLMClient
from decision_layer.pipeline import decide
from decision_layer.schemas import (
    ClusterState,
    NodeScores,
    NodeState,
    PendingPod,
    SchedulingDecision,
    SingleNodeScore,
)


def _make_node(
    node_id: str,
    cpu_cap: float = 16.0,
    cpu_used: float = 3.2,
    mem_cap: float = 64.0,
    mem_used: float = 12.8,
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
        health_score=1.0 if failures == 0 else 0.6,
        cost_per_hour=0.24,
        zone="us-central1-a",
        pod_count=10,
        recent_failures=failures,
    )


def _make_cluster(nodes: list[NodeState], pod_id: str = "pod-test-101") -> ClusterState:
    return ClusterState(
        timestamp="2026-10-11T17:00:00Z",
        nodes=nodes,
        pending_pod=PendingPod(
            pod_id=pod_id,
            cpu_request=2.0,
            mem_request_gb=4.0,
            priority_class="standard",
            qos_class="Burstable",
            estimated_duration_sec=600.0,
        ),
    )


def test_scenario_1_two_nodes_failed_and_maintenance_in_10_min():
    """
    Scenario 1: Two nodes just failed and maintenance in 10 min -> FT first.
    Input:
      - Telemetry reporting 2 recent node failures and imminent maintenance window.
      - Fake LLM prioritizing fault_tolerance.
      - 3 candidate nodes where node-stable dominates fault_tolerance.
    Expected Output:
      - priority_order_used begins with 'fault_tolerance'
      - selected_node_id == 'node-stable'
      - source == 'llm'
    """
    cluster = _make_cluster([
        _make_node("w1", failures=1),
        _make_node("w2", failures=1),
        _make_node("w3", failures=0),
    ])

    scores = NodeScores(
        pod_id="pod-resilient-app",
        scores=[
            SingleNodeScore(node_id="node-stable", fault_tolerance=0.96, utilisation=0.35, cost=0.20),
            SingleNodeScore(node_id="node-packed", fault_tolerance=0.60, utilisation=0.95, cost=0.50),
            SingleNodeScore(node_id="node-thrifty", fault_tolerance=0.50, utilisation=0.30, cost=0.95),
        ],
    )

    telemetry = (
        "Two worker nodes (w1, w2) suffered kernel panics in the last 15 minutes. "
        "Scheduled maintenance window on rack-B starts in 10 minutes. "
        "Critical workloads require maximum resilience."
    )

    fake_llm = FakeLLMClient(
        response=json.dumps({
            "priority_order": ["fault_tolerance", "utilisation", "cost"],
            "justification": "Active hardware failures and imminent maintenance window require fault tolerance first.",
            "confidence": 0.96,
        })
    )

    decision = decide(
        cluster_state=cluster,
        node_scores=scores,
        telemetry_summary=telemetry,
        use_llm=True,
        llm_client=fake_llm,
    )

    assert isinstance(decision, SchedulingDecision)
    assert decision.priority_order_used.source == "llm"
    assert decision.priority_order_used.priority_order[0] == "fault_tolerance"
    assert decision.selected_node_id == "node-stable"
    assert decision.decision_latency_ms >= 0.0


def test_scenario_2_calm_cluster_and_overspend_flagged():
    """
    Scenario 2: Calm cluster and overspend flagged -> cost first.
    Input:
      - Calm cluster with zero failures and low utilisation.
      - Telemetry reporting strict monthly cloud budget overspend alert.
      - Fake LLM prioritizing cost.
      - 3 candidate nodes where node-cheap dominates cost.
    Expected Output:
      - priority_order_used begins with 'cost'
      - selected_node_id == 'node-cheap'
      - source == 'llm'
    """
    cluster = _make_cluster([
        _make_node("w1", cpu_used=1.6, mem_used=6.4, failures=0),  # 10%
        _make_node("w2", cpu_used=1.6, mem_used=6.4, failures=0),  # 10%
    ])

    scores = NodeScores(
        pod_id="pod-batch-crawler",
        scores=[
            SingleNodeScore(node_id="node-cheap", fault_tolerance=0.60, utilisation=0.50, cost=0.98),
            SingleNodeScore(node_id="node-packed", fault_tolerance=0.60, utilisation=0.95, cost=0.30),
            SingleNodeScore(node_id="node-safe", fault_tolerance=0.95, utilisation=0.50, cost=0.30),
        ],
    )

    telemetry = (
        "Calm cluster conditions. FinOps overspend alert flagged: monthly AWS budget exceeded by 18%. "
        "Prioritize cost efficiency for non-urgent tasks."
    )

    fake_llm = FakeLLMClient(
        response=json.dumps({
            "priority_order": ["cost", "utilisation", "fault_tolerance"],
            "justification": "Cluster is calm with zero failures; active budget overspend requires cost minimization.",
            "confidence": 0.94,
        })
    )

    decision = decide(
        cluster_state=cluster,
        node_scores=scores,
        telemetry_summary=telemetry,
        use_llm=True,
        llm_client=fake_llm,
    )

    assert isinstance(decision, SchedulingDecision)
    assert decision.priority_order_used.source == "llm"
    assert decision.priority_order_used.priority_order[0] == "cost"
    assert decision.selected_node_id == "node-cheap"


def test_scenario_3_bursty_traffic_forecast_with_uneven_utilisation():
    """
    Scenario 3: Bursty traffic forecast with uneven utilisation -> utilisation first.
    Input:
      - Cluster with fragmented utilisation across nodes.
      - Telemetry reporting incoming traffic surge needing bin-packing.
      - Fake LLM prioritizing utilisation.
      - 3 candidate nodes where node-balanced dominates utilisation.
    Expected Output:
      - priority_order_used begins with 'utilisation'
      - selected_node_id == 'node-balanced'
      - source == 'llm'
    """
    cluster = _make_cluster([
        _make_node("w1", cpu_used=14.4, mem_used=57.6),  # 90%
        _make_node("w2", cpu_used=3.2, mem_used=12.8),   # 20%
    ])

    scores = NodeScores(
        pod_id="pod-microservice-api",
        scores=[
            SingleNodeScore(node_id="node-balanced", fault_tolerance=0.50, utilisation=0.96, cost=0.40),
            SingleNodeScore(node_id="node-isolated", fault_tolerance=0.90, utilisation=0.45, cost=0.40),
            SingleNodeScore(node_id="node-frugal", fault_tolerance=0.50, utilisation=0.45, cost=0.95),
        ],
    )

    telemetry = (
        "Traffic forecast model predicts a 2.5x bursty surge in 15 minutes. "
        "Severe node utilisation imbalance detected across cluster. "
        "Bin-pack to clear contiguous capacity for surge pods."
    )

    fake_llm = FakeLLMClient(
        response=json.dumps({
            "priority_order": ["utilisation", "fault_tolerance", "cost"],
            "justification": "Imminent traffic burst and node imbalance demand bin-packing utilisation first.",
            "confidence": 0.91,
        })
    )

    decision = decide(
        cluster_state=cluster,
        node_scores=scores,
        telemetry_summary=telemetry,
        use_llm=True,
        llm_client=fake_llm,
    )

    assert isinstance(decision, SchedulingDecision)
    assert decision.priority_order_used.source == "llm"
    assert decision.priority_order_used.priority_order[0] == "utilisation"
    assert decision.selected_node_id == "node-balanced"


def test_scenario_4_llm_returns_garbage_rule_fallback_succeeds():
    """
    Scenario 4: LLM returns garbage -> rule fallback still produces a valid decision.
    Input:
      - Stressed cluster state (85%+ utilisation, 3 failures).
      - Fake LLM returns malformed HTML / non-JSON garbage.
    Expected Output:
      - Decision is created cleanly without throwing an exception.
      - priority_order_used has source == 'rule'.
      - Under high stress, fallback prioritizes fault_tolerance.
      - Best node on fault_tolerance is selected.
    """
    stressed_cluster = _make_cluster([
        _make_node("w1", cpu_used=14.0, mem_used=56.0, failures=2),
        _make_node("w2", cpu_used=13.6, mem_used=54.0, failures=1),
    ])

    scores = NodeScores(
        pod_id="pod-payment-gateway",
        scores=[
            SingleNodeScore(node_id="node-ft-champ", fault_tolerance=0.95, utilisation=0.60, cost=0.40),
            SingleNodeScore(node_id="node-other", fault_tolerance=0.70, utilisation=0.90, cost=0.90),
        ],
    )

    # LLM outputs completely broken content
    fake_llm = FakeLLMClient(response="<html><body><h1>500 Internal Server Error</h1><p>Crash</p></body></html>")

    decision = decide(
        cluster_state=stressed_cluster,
        node_scores=scores,
        telemetry_summary="Cluster experiencing stress...",
        use_llm=True,
        llm_client=fake_llm,
    )

    assert isinstance(decision, SchedulingDecision)
    assert decision.priority_order_used.source == "rule"
    assert decision.priority_order_used.priority_order[0] == "fault_tolerance"
    assert decision.selected_node_id == "node-ft-champ"
    assert decision.decision_latency_ms >= 0.0
    assert "Fallback rule applied" in decision.priority_order_used.justification


def test_scenario_5_static_rule_and_llm_choose_different_nodes():
    """
    Scenario 5: Static rule and LLM choose different nodes.
    Context:
      - Cluster has high resource utilisation (~85%), but ZERO failures.
      - Static rule: High utilisation triggers high stress (>= 0.50),
        so the rule blindly picks fault_tolerance first -> selects node-resilient.
      - LLM agent: Understands telemetry showing overnight batch processing with
        an expiring electricity discount rate. Decides cost is paramount -> selects node-frugal.
    Expected Output:
      - rule_decision.selected_node_id == 'node-resilient'
      - llm_decision.selected_node_id == 'node-frugal'
      - rule_decision.selected_node_id != llm_decision.selected_node_id
    """
    # 85% utilization across nodes with 3 failures -> static rule evaluates as HIGH stress (0.55 >= 0.50)
    batch_cluster = _make_cluster([
        _make_node("w1", cpu_cap=16.0, cpu_used=13.6, mem_cap=64.0, mem_used=54.4, failures=2),
        _make_node("w2", cpu_cap=16.0, cpu_used=13.6, mem_cap=64.0, mem_used=54.4, failures=1),
    ])

    scores = NodeScores(
        pod_id="pod-reproducible-batch-job",
        scores=[
            SingleNodeScore(node_id="node-resilient", fault_tolerance=0.98, utilisation=0.40, cost=0.20),
            SingleNodeScore(node_id="node-frugal", fault_tolerance=0.40, utilisation=0.40, cost=0.98),
        ],
    )

    telemetry = (
        "Overnight batch processing window for non-critical, fully reproducible jobs. "
        "Failures occurred on isolated test rack that does not host these pods. "
        "Electricity spot market discount expires in 45 minutes! "
        "Prioritize cost efficiency to complete batch runs under the discount rate."
    )

    fake_llm = FakeLLMClient(
        response=json.dumps({
            "priority_order": ["cost", "utilisation", "fault_tolerance"],
            "justification": "Expiring electricity spot discount and reproducible batch workload warrant cost prioritization over fault tolerance.",
            "confidence": 0.95,
        })
    )

    # 1. Decision with static rule (use_llm=False)
    rule_decision = decide(
        cluster_state=batch_cluster,
        node_scores=scores,
        telemetry_summary=telemetry,
        use_llm=False,
    )

    # 2. Decision with LLM reasoning (use_llm=True)
    llm_decision = decide(
        cluster_state=batch_cluster,
        node_scores=scores,
        telemetry_summary=telemetry,
        use_llm=True,
        llm_client=fake_llm,
    )

    # Verify static rule picked fault_tolerance -> node-resilient
    assert rule_decision.priority_order_used.source == "rule"
    assert rule_decision.priority_order_used.priority_order[0] == "fault_tolerance"
    assert rule_decision.selected_node_id == "node-resilient"

    # Verify LLM agent picked cost -> node-frugal
    assert llm_decision.priority_order_used.source == "llm"
    assert llm_decision.priority_order_used.priority_order[0] == "cost"
    assert llm_decision.selected_node_id == "node-frugal"

    # Crucial assertion: Static rule and LLM picked DIFFERENT nodes
    assert rule_decision.selected_node_id != llm_decision.selected_node_id


def test_cli_execution_no_feasible_node():
    """Test CLI invocation via python -m decision_layer.pipeline when cluster has no feasible nodes."""
    cmd = [
        sys.executable,
        "-m",
        "decision_layer.pipeline",
        "--cluster",
        "mocks/stressed.json",
        "--scores",
        "mocks/scores.json",
        "--no-llm",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    parsed = json.loads(res.stdout)
    assert parsed["selected_node_id"] is None
    assert parsed["priority_order_used"]["source"] == "rule"


def test_cli_execution_with_heuristic_scorer():
    """Test CLI invocation using --scorer heuristic."""
    cmd = [
        sys.executable,
        "-m",
        "decision_layer.pipeline",
        "--cluster",
        "decision_layer/mocks/calm_cluster.json",
        "--scorer",
        "heuristic",
        "--no-llm",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    parsed = json.loads(res.stdout)
    assert parsed["selected_node_id"] is not None
    assert parsed["selected_node_id"].startswith("node-worker-")
    assert parsed["priority_order_used"]["source"] == "rule"


def test_cli_execution_with_gnn_scorer():
    """Test CLI invocation using --scorer gnn."""
    cmd = [
        sys.executable,
        "-m",
        "decision_layer.pipeline",
        "--cluster",
        "decision_layer/mocks/calm_cluster.json",
        "--scorer",
        "gnn",
        "--no-llm",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    parsed = json.loads(res.stdout)
    assert parsed["selected_node_id"] is not None
    assert parsed["selected_node_id"].startswith("node-worker-")


def test_full_node_is_never_selected():
    """
    Test that a node with insufficient CPU or RAM capacity is never selected,
    even if it has higher multi-objective scores than all other nodes.
    """
    cluster = ClusterState(
        timestamp="2026-10-12T10:00:00Z",
        nodes=[
            NodeState(
                node_id="node-full",
                cpu_capacity=16.0,
                cpu_used=15.5,  # Only 0.5 CPU remaining < 2.0 required!
                mem_capacity_gb=64.0,
                mem_used_gb=20.0,
                disk_capacity_gb=500.0,
                disk_used_gb=100.0,
                health_score=1.0,
                cost_per_hour=0.10,
                zone="us-central1-a",
                pod_count=20,
                recent_failures=0,
            ),
            NodeState(
                node_id="node-avail",
                cpu_capacity=16.0,
                cpu_used=4.0,  # 12.0 CPU remaining >= 2.0 required
                mem_capacity_gb=64.0,
                mem_used_gb=16.0,
                disk_capacity_gb=500.0,
                disk_used_gb=50.0,
                health_score=0.9,
                cost_per_hour=0.50,
                zone="us-central1-b",
                pod_count=5,
                recent_failures=0,
            ),
        ],
        pending_pod=PendingPod(
            pod_id="pod-cpu-hungry",
            cpu_request=2.0,
            mem_request_gb=4.0,
            priority_class="standard",
            qos_class="Burstable",
            estimated_duration_sec=300.0,
        ),
    )

    # node-full dominates all 3 objectives with top scores
    scores = NodeScores(
        pod_id="pod-cpu-hungry",
        scores=[
            SingleNodeScore(node_id="node-full", fault_tolerance=0.99, utilisation=0.99, cost=0.99),
            SingleNodeScore(node_id="node-avail", fault_tolerance=0.50, utilisation=0.50, cost=0.50),
        ],
    )

    decision = decide(
        cluster_state=cluster,
        node_scores=scores,
        use_llm=False,
    )

    assert decision is not None
    assert decision.selected_node_id != "node-full"
    assert decision.selected_node_id == "node-avail"


def test_no_feasible_node():
    """
    Test that when all candidate nodes lack sufficient CPU or RAM capacity,
    decide() returns a SchedulingDecision with selected_node_id=None and logs reason.
    """
    cluster = ClusterState(
        timestamp="2026-10-12T10:00:00Z",
        nodes=[
            NodeState(
                node_id="node-1",
                cpu_capacity=8.0,
                cpu_used=7.5,  # 0.5 CPU remaining < 4.0 CPU requested
                mem_capacity_gb=32.0,
                mem_used_gb=10.0,
                disk_capacity_gb=200.0,
                disk_used_gb=50.0,
                health_score=1.0,
                cost_per_hour=0.20,
                zone="us-central1-a",
                pod_count=15,
                recent_failures=0,
            ),
            NodeState(
                node_id="node-2",
                cpu_capacity=8.0,
                cpu_used=2.0,
                mem_capacity_gb=32.0,
                mem_used_gb=30.0,  # 2.0 GB RAM remaining < 16.0 GB requested
                disk_capacity_gb=200.0,
                disk_used_gb=50.0,
                health_score=0.95,
                cost_per_hour=0.20,
                zone="us-central1-b",
                pod_count=10,
                recent_failures=0,
            ),
        ],
        pending_pod=PendingPod(
            pod_id="pod-heavy",
            cpu_request=4.0,
            mem_request_gb=16.0,
            priority_class="high",
            qos_class="Guaranteed",
            estimated_duration_sec=1200.0,
        ),
    )

    decision = decide(
        cluster_state=cluster,
        use_llm=False,
        scorer="heuristic",
    )

    assert isinstance(decision, SchedulingDecision)
    assert decision.selected_node_id is None
    assert decision.pod_id == "pod-heavy"


def test_idle_cluster():
    """
    Test that when cluster_state.pending_pod is None, decide() returns early
    with no decision (None) and logs an idle event.
    """
    cluster = ClusterState(
        timestamp="2026-10-12T10:00:00Z",
        nodes=[
            NodeState(
                node_id="node-idle-1",
                cpu_capacity=16.0,
                cpu_used=2.0,
                mem_capacity_gb=64.0,
                mem_used_gb=8.0,
                disk_capacity_gb=500.0,
                disk_used_gb=50.0,
                health_score=1.0,
                cost_per_hour=0.25,
                zone="us-central1-a",
                pod_count=5,
                recent_failures=0,
            ),
        ],
        pending_pod=None,
    )

    assert cluster.pending_pod is None

    decision = decide(
        cluster_state=cluster,
        use_llm=False,
    )

    assert decision is None


def test_heuristic_scores_differ_across_objectives():
    """
    Test that score_nodes produces NodeScores where fault_tolerance,
    utilisation, and cost genuinely differ across the three objectives.
    """
    cluster = ClusterState(
        timestamp="2026-10-12T10:00:00Z",
        nodes=[
            NodeState(
                node_id="node-resilient",
                cpu_capacity=32.0,
                cpu_used=6.0,
                mem_capacity_gb=128.0,
                mem_used_gb=24.0,
                disk_capacity_gb=1000.0,
                disk_used_gb=100.0,
                health_score=1.0,
                cost_per_hour=0.80,  # Most expensive ($0.80/hr)
                zone="us-central1-c",  # isolated zone with 2 pods
                pod_count=2,
                recent_failures=0,
            ),
            NodeState(
                node_id="node-budget",
                cpu_capacity=16.0,
                cpu_used=12.0,
                mem_capacity_gb=64.0,
                mem_used_gb=40.0,
                disk_capacity_gb=500.0,
                disk_used_gb=200.0,
                health_score=0.70,
                cost_per_hour=0.10,  # Cheapest ($0.10/hr)
                zone="us-central1-a",  # crowded zone with 50 pods
                pod_count=50,
                recent_failures=3,
            ),
            NodeState(
                node_id="node-balanced",
                cpu_capacity=16.0,
                cpu_used=4.0,
                mem_capacity_gb=64.0,
                mem_used_gb=16.0,
                disk_capacity_gb=500.0,
                disk_used_gb=100.0,
                health_score=0.95,
                cost_per_hour=0.30,
                zone="us-central1-b",
                pod_count=15,
                recent_failures=0,
            ),
        ],
        pending_pod=PendingPod(
            pod_id="pod-analytics",
            cpu_request=4.0,
            mem_request_gb=16.0,
            priority_class="standard",
            qos_class="Burstable",
            estimated_duration_sec=600.0,
        ),
    )

    scores = score_nodes(cluster)
    assert isinstance(scores, NodeScores)
    assert len(scores.scores) == 3

    scores_by_node = {s.node_id: s for s in scores.scores}

    # All scores must be in [0, 1]
    for s in scores.scores:
        assert 0.0 <= s.fault_tolerance <= 1.0
        assert 0.0 <= s.utilisation <= 1.0
        assert 0.0 <= s.cost <= 1.0

    resilient = scores_by_node["node-resilient"]
    budget = scores_by_node["node-budget"]

    # Verify objectives genuinely differ:
    # 1. Budget node is cheapest (cost=1.0) but has degraded fault tolerance due to failures/health
    assert budget.cost > resilient.cost
    assert resilient.fault_tolerance > budget.fault_tolerance

    # 2. On resilient node, cost score is 0.0 while fault_tolerance is very high (> 0.8)
    assert abs(resilient.fault_tolerance - resilient.cost) > 0.50

    # 3. Across all objectives for each node, the objectives do not collapse to identical values
    for s in scores.scores:
        assert not (s.fault_tolerance == s.utilisation == s.cost)

