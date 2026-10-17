"""
decision_layer/heuristic_scorer.py

Heuristic multi-objective scoring for candidate Kubernetes nodes.

Computes NodeScores across three genuine scheduling objectives:
1. fault_tolerance:
   Derived from node health_score, recent_failures penalty, and zone spread
   (favoring availability zones with lower pod concentration to maximize resilience).
2. utilisation:
   Evaluates resource balance after placing the pending pod:
   - Node-level balance between post-placement CPU and memory utilisation (mitigating stranded resources).
   - Cluster-level balance (minimizing utilisation variance across nodes post-placement).
3. cost:
   Inverse of cost_per_hour, min-max normalized across candidate nodes in the cluster
   (cheapest node scores 1.0, most expensive scores 0.0).

All scores are bounded in [0, 1], with 1.0 being optimal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from decision_layer.schemas import (
    ClusterState,
    NodeScores,
    NodeState,
    SingleNodeScore,
)


def _load_cluster_state(state_input: ClusterState | str | Path | dict[str, Any]) -> ClusterState:
    """Parse or validate input into a ClusterState instance."""
    if isinstance(state_input, ClusterState):
        return state_input
    if isinstance(state_input, (str, Path)):
        path = Path(state_input)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            return ClusterState.model_validate_json(content)
        return ClusterState.model_validate_json(str(state_input))
    if isinstance(state_input, dict):
        return ClusterState.model_validate(state_input)
    raise TypeError(f"Cannot load ClusterState from type {type(state_input).__name__}")


def score_nodes(cluster_state: ClusterState | str | Path | dict[str, Any]) -> NodeScores:
    """
    Compute multi-objective scores for candidate nodes in cluster_state.

    Parameters:
    - cluster_state: ClusterState instance, dict, file path, or JSON string.

    Returns:
    - NodeScores: Candidate scores across fault_tolerance, utilisation, and cost, all in [0, 1].
    """
    state = _load_cluster_state(cluster_state)
    nodes = state.nodes
    pending_pod = state.pending_pod

    pod_id = pending_pod.pod_id if pending_pod else "unknown"

    if not nodes:
        return NodeScores(pod_id=pod_id, scores=[])

    # -------------------------------------------------------------------------
    # 1. Fault Tolerance: health_score, recent_failures, zone spread
    # -------------------------------------------------------------------------
    # Tally total running pods per availability zone
    zone_pod_counts: dict[str, int] = {}
    for node in nodes:
        zone_pod_counts[node.zone] = zone_pod_counts.get(node.zone, 0) + node.pod_count

    min_zone_pods = min(zone_pod_counts.values()) if zone_pod_counts else 0
    max_zone_pods = max(zone_pod_counts.values()) if zone_pod_counts else 0

    ft_scores: list[float] = []
    for node in nodes:
        health = max(0.0, min(1.0, float(node.health_score)))
        # Failure penalty: 0 failures -> 1.0, 5+ failures -> 0.0
        failure_penalty = max(0.0, 1.0 - (float(node.recent_failures) / 5.0))

        # Zone spread score: fewer existing pods in the zone gives higher spread score
        if max_zone_pods > min_zone_pods:
            zone_pods = zone_pod_counts.get(node.zone, min_zone_pods)
            zone_spread = 1.0 - ((zone_pods - min_zone_pods) / (max_zone_pods - min_zone_pods))
        else:
            zone_spread = 1.0

        ft = 0.40 * health + 0.35 * failure_penalty + 0.25 * zone_spread
        ft_scores.append(max(0.0, min(1.0, ft)))

    # -------------------------------------------------------------------------
    # 2. Utilisation: Balance after placing the pod
    # -------------------------------------------------------------------------
    pod_cpu = pending_pod.cpu_request if pending_pod else 0.0
    pod_mem = pending_pod.mem_request_gb if pending_pod else 0.0

    # Precalculate baseline utilisation per node
    base_utils: list[float] = []
    for node in nodes:
        cpu_u = min(1.0, max(0.0, node.cpu_used / max(node.cpu_capacity, 1e-6)))
        mem_u = min(1.0, max(0.0, node.mem_used_gb / max(node.mem_capacity_gb, 1e-6)))
        base_utils.append(0.5 * cpu_u + 0.5 * mem_u)

    util_scores: list[float] = []
    num_nodes = len(nodes)

    for i, node in enumerate(nodes):
        # Post-placement CPU & RAM utilisation on candidate node i
        post_cpu = min(1.0, max(0.0, (node.cpu_used + pod_cpu) / max(node.cpu_capacity, 1e-6)))
        post_mem = min(1.0, max(0.0, (node.mem_used_gb + pod_mem) / max(node.mem_capacity_gb, 1e-6)))

        # Node-level balance: penalize divergence between post-placement CPU and RAM
        node_balance = 1.0 - abs(post_cpu - post_mem)

        # Cluster-level balance: variance across cluster nodes after placing pod on node i
        post_candidate_util = 0.5 * post_cpu + 0.5 * post_mem
        cluster_post_utils = [
            post_candidate_util if j == i else base_utils[j]
            for j in range(num_nodes)
        ]

        if num_nodes > 1:
            mean_u = sum(cluster_post_utils) / num_nodes
            variance = sum((u - mean_u) ** 2 for u in cluster_post_utils) / num_nodes
            # Maximum variance for bounded [0, 1] variables is 0.25
            cluster_balance = max(0.0, min(1.0, 1.0 - (variance / 0.25)))
        else:
            cluster_balance = 1.0

        util = 0.50 * node_balance + 0.50 * cluster_balance
        util_scores.append(max(0.0, min(1.0, util)))

    # -------------------------------------------------------------------------
    # 3. Cost: Inverse of cost_per_hour, min-max across nodes
    # -------------------------------------------------------------------------
    inv_costs = [1.0 / max(float(node.cost_per_hour), 1e-6) for node in nodes]
    min_inv = min(inv_costs)
    max_inv = max(inv_costs)

    cost_scores: list[float] = []
    for inv in inv_costs:
        if max_inv > min_inv:
            norm_cost = (inv - min_inv) / (max_inv - min_inv)
        else:
            norm_cost = 1.0
        cost_scores.append(max(0.0, min(1.0, norm_cost)))

    # -------------------------------------------------------------------------
    # Assemble SingleNodeScore objects
    # -------------------------------------------------------------------------
    node_scores_list: list[SingleNodeScore] = []
    for i, node in enumerate(nodes):
        node_scores_list.append(
            SingleNodeScore(
                node_id=node.node_id,
                fault_tolerance=round(ft_scores[i], 4),
                utilisation=round(util_scores[i], 4),
                cost=round(cost_scores[i], 4),
            )
        )

    return NodeScores(pod_id=pod_id, scores=node_scores_list)
