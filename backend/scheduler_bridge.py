"""
backend/scheduler_bridge.py

Bridges the FastAPI application directly to Split C's multi-objective decision
pipeline, stress analysis, scoring engines (heuristic + GNN), and lexicographic
filtering with comprehensive step-by-step telemetry.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure paths are configured
ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PACKAGES = ROOT_DIR / ".venv" / "Lib" / "site-packages"
SPLIT_C_DIR = ROOT_DIR / "split-c"
SPLIT_AB_DIR = ROOT_DIR / "split-ab"

if VENV_PACKAGES.is_dir() and str(VENV_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_PACKAGES))
if str(SPLIT_C_DIR) not in sys.path:
    sys.path.insert(0, str(SPLIT_C_DIR))
if str(SPLIT_AB_DIR) not in sys.path:
    sys.path.insert(0, str(SPLIT_AB_DIR))

from decision_layer.heuristic_scorer import score_nodes as heuristic_score_nodes
from decision_layer.llm_agent import get_priority_order
from decision_layer.schemas import (
    ClusterState,
    NodeScores,
    NodeState,
    PendingPod,
    PriorityOrder,
    SingleNodeScore,
)
from decision_layer.stress import (
    compute_stress,
    rule_based_priority,
    stress_level,
    WEIGHT_UTILISATION,
    WEIGHT_FAILURES,
    WEIGHT_QUEUE,
    WEIGHT_VARIANCE,
    MAX_RECENT_FAILURES,
    MAX_PENDING_QUEUE,
    MAX_VARIANCE,
)

logger = logging.getLogger(__name__)


def compute_detailed_stress(cluster_state: ClusterState, pending_queue_len: int = 0) -> Dict[str, Any]:
    """Calculate cluster stress with individual factor breakdowns."""
    nodes = cluster_state.nodes
    if not nodes:
        return {
            "value": 0.0,
            "level": "low",
            "breakdown": {
                "avg_utilisation": 0.0,
                "utilisation_variance": 0.0,
                "recent_failures_total": 0,
                "queue_length": 0,
            },
        }

    node_utils: List[float] = []
    total_failures = 0

    for n in nodes:
        cpu_cap = max(n.cpu_capacity, 1e-6)
        mem_cap = max(n.mem_capacity_gb, 1e-6)
        u_cpu = min(1.0, max(0.0, n.cpu_used / cpu_cap))
        u_mem = min(1.0, max(0.0, n.mem_used_gb / mem_cap))
        node_utils.append((u_cpu + u_mem) / 2.0)
        total_failures += n.recent_failures

    avg_util = sum(node_utils) / len(node_utils)
    if len(node_utils) > 1:
        variance = sum((u - avg_util) ** 2 for u in node_utils) / len(node_utils)
    else:
        variance = 0.0

    norm_variance = min(1.0, max(0.0, variance / MAX_VARIANCE))
    norm_queue = min(1.0, max(0.0, float(pending_queue_len) / MAX_PENDING_QUEUE))
    norm_failures = min(1.0, max(0.0, float(total_failures) / MAX_RECENT_FAILURES))

    stress_val = (
        WEIGHT_UTILISATION * avg_util
        + WEIGHT_VARIANCE * norm_variance
        + WEIGHT_QUEUE * norm_queue
        + WEIGHT_FAILURES * norm_failures
    )
    stress_val = max(0.0, min(1.0, float(stress_val)))
    lvl = stress_level(stress_val)

    return {
        "value": round(stress_val, 4),
        "level": lvl,
        "breakdown": {
            "avg_utilisation": round(avg_util, 4),
            "utilisation_variance": round(variance, 4),
            "recent_failures_total": total_failures,
            "queue_length": pending_queue_len,
            "weights": {
                "utilisation": WEIGHT_UTILISATION,
                "failures": WEIGHT_FAILURES,
                "queue": WEIGHT_QUEUE,
                "variance": WEIGHT_VARIANCE,
            },
        },
    }


def execute_scoring(
    cluster_state: ClusterState,
    scorer_type: str = "heuristic",
) -> Tuple[NodeScores, str, Optional[str]]:
    """Score nodes using GNN or heuristic, with safe fallback."""
    warning = None
    actual_scorer = scorer_type

    if scorer_type == "gnn":
        try:
            from decision_layer.pipeline import _score_nodes_gnn
            scores = _score_nodes_gnn(cluster_state)
            return scores, "gnn", None
        except Exception as exc:
            logger.warning("GNN inference unavailable (%s); falling back to heuristic scorer", exc)
            actual_scorer = "heuristic"
            warning = f"GNN scorer could not be executed: {exc}. Gracefully fell back to Heuristic Scorer."

    scores = heuristic_score_nodes(cluster_state)
    return scores, actual_scorer, warning


def run_decision_trace(
    cluster_state_dict: Dict[str, Any],
    use_llm: bool = False,
    scorer: str = "heuristic",
    epsilon: float = 0.05,
) -> Dict[str, Any]:
    """
    Executes the full scheduling decision pipeline while recording detailed
    intermediate telemetry for every stage in the decision lifecycle.
    """
    start_time = time.perf_counter()
    cluster_state = ClusterState.model_validate(cluster_state_dict)
    pending_pod = cluster_state.pending_pod

    if pending_pod is None:
        return {
            "error": "No pending pod specified in cluster state",
            "latency_ms": 0.0,
        }

    # 1. Stress assessment
    stress_data = compute_detailed_stress(cluster_state)

    # 2. Candidate scoring
    node_scores, scorer_used, scorer_warning = execute_scoring(cluster_state, scorer_type=scorer)
    scores_list = [
        {
            "node_id": s.node_id,
            "fault_tolerance": round(s.fault_tolerance, 4),
            "utilisation": round(s.utilisation, 4),
            "cost": round(s.cost, 4),
        }
        for s in node_scores.scores
    ]

    # 3. Priority determination
    if use_llm:
        try:
            priority_order_obj = get_priority_order(cluster_state)
        except Exception as exc:
            logger.warning("LLM priority failed (%s); using rule-based", exc)
            priority_order_obj = rule_based_priority(stress_data["value"])
    else:
        priority_order_obj = rule_based_priority(stress_data["value"])

    priority_data = {
        "order": priority_order_obj.priority_order,
        "justification": priority_order_obj.justification,
        "confidence": priority_order_obj.confidence,
        "source": priority_order_obj.source,
    }

    # 4. Feasibility filtering
    req_cpu = pending_pod.cpu_request
    req_mem = pending_pod.mem_request_gb
    feasibility_results: List[Dict[str, Any]] = []
    feasible_scores: List[SingleNodeScore] = []

    scores_by_id = {s.node_id: s for s in node_scores.scores}

    for node in cluster_state.nodes:
        free_cpu = round(node.cpu_capacity - node.cpu_used, 2)
        free_mem = round(node.mem_capacity_gb - node.mem_used_gb, 2)
        reasons = []

        if free_cpu < req_cpu:
            reasons.append(f"Insufficient CPU: {free_cpu:.2f} cores free < {req_cpu:.2f} required")
        if free_mem < req_mem:
            reasons.append(f"Insufficient RAM: {free_mem:.2f} GB free < {req_mem:.2f} GB required")

        is_feasible = len(reasons) == 0
        feasibility_results.append({
            "node_id": node.node_id,
            "feasible": is_feasible,
            "free_cpu": free_cpu,
            "free_mem_gb": free_mem,
            "req_cpu": req_cpu,
            "req_mem_gb": req_mem,
            "zone": node.zone,
            "reason": "; ".join(reasons) if not is_feasible else "Sufficient CPU & Memory available",
        })

        if is_feasible and node.node_id in scores_by_id:
            feasible_scores.append(scores_by_id[node.node_id])

    # 5. Lexicographic elimination trace
    elimination_trace: List[Dict[str, Any]] = []
    selected_node_id: Optional[str] = None

    if not feasible_scores:
        selected_node_id = None
        decision_reason = "No candidate nodes had sufficient free CPU and Memory capacity."
    else:
        survivors = list(feasible_scores)

        for step_idx, obj in enumerate(priority_order_obj.priority_order, start=1):
            if len(survivors) <= 1:
                break

            max_score = max(getattr(s, obj) for s in survivors)
            cutoff = max_score - epsilon

            next_survivors: List[SingleNodeScore] = []
            eliminated: List[Dict[str, Any]] = []

            for s in survivors:
                score_val = getattr(s, obj)
                if score_val >= (cutoff - 1e-9):
                    next_survivors.append(s)
                else:
                    eliminated.append({
                        "node_id": s.node_id,
                        "score": round(score_val, 4),
                        "deficit": round(cutoff - score_val, 4),
                    })

            elimination_trace.append({
                "step": step_idx,
                "objective": obj,
                "max_score": round(max_score, 4),
                "epsilon": epsilon,
                "cutoff_threshold": round(cutoff, 4),
                "surviving_count": len(next_survivors),
                "survivor_node_ids": [s.node_id for s in next_survivors],
                "eliminated": eliminated,
            })

            survivors = next_survivors

        # Break ties deterministically
        if len(survivors) > 1:
            survivors.sort(key=lambda s: s.node_id)
            tie_broken = True
        else:
            tie_broken = False

        selected_node_id = survivors[0].node_id
        decision_reason = f"Node {selected_node_id} selected as optimal match according to {priority_order_obj.priority_order[0]}-first lexicographic ranking."

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return {
        "pod_id": pending_pod.pod_id,
        "selected_node_id": selected_node_id,
        "decision_reason": decision_reason,
        "latency_ms": latency_ms,
        "scorer_used": scorer_used,
        "scorer_warning": scorer_warning,
        "stress": stress_data,
        "priority": priority_data,
        "scores": scores_list,
        "feasibility": feasibility_results,
        "elimination_trace": elimination_trace,
    }


def check_kubernetes_status() -> Dict[str, Any]:
    """Inspect local/remote Kubernetes environment connectivity."""
    try:
        from kubernetes import client, config
        try:
            config.load_incluster_config()
            in_cluster = True
        except Exception:
            config.load_kube_config()
            in_cluster = False

        v1 = client.CoreV1Api()
        nodes = v1.list_node(timeout_seconds=3).items
        node_names = [n.metadata.name for n in nodes]
        return {
            "available": True,
            "in_cluster": in_cluster,
            "node_count": len(node_names),
            "nodes": node_names,
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "in_cluster": False,
            "node_count": 0,
            "nodes": [],
            "error": str(exc),
        }
