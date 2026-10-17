"""
decision_layer/pipeline.py

Orchestrates the agentic Kubernetes scheduling decision pipeline.

Workflow:
1. Validates cluster state and checks for idle cluster (pending_pod is None).
2. Resolves candidate node scores (via heuristic scorer, GNN model, or pre-computed scores).
3. Computes aggregate cluster stress metric.
4. Determines priority order:
   - When use_llm=True: queries LLM reasoning agent (falling back to rule-based on failure/timeout).
   - When use_llm=False: evaluates deterministic rule-based priority mapping.
5. Filters candidate nodes: drops nodes where cpu_capacity - cpu_used < cpu_request or
   mem_capacity_gb - mem_used_gb < mem_request_gb.
6. If no nodes are feasible, returns a SchedulingDecision with selected_node_id=None and logs reason.
7. Executes lexicographic node filtering (select_node) with tolerance epsilon on feasible nodes.
8. Emits SchedulingDecision with scores snapshot and execution latency.
9. Emits structured JSON log of the full reasoning.

Provides a CLI:
python -m decision_layer.pipeline --cluster mocks/calm_cluster.json --scorer heuristic [--no-llm]
python -m decision_layer.pipeline --cluster mocks/stressed.json --scores mocks/scores.json [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from decision_layer.heuristic_scorer import score_nodes as heuristic_score_nodes
from decision_layer.lexicographic import select_node
from decision_layer.llm_agent import LLMClientProtocol, get_priority_order
from decision_layer.schemas import (
    ClusterState,
    NodeScores,
    PriorityOrder,
    SchedulingDecision,
    SingleNodeScore,
)
from decision_layer.stress import compute_stress, rule_based_priority, stress_level

logger = logging.getLogger(__name__)


def _load_cluster_state(state_input: ClusterState | str | Path | dict[str, Any]) -> ClusterState:
    """Parse or validate input into a ClusterState instance."""
    if isinstance(state_input, ClusterState):
        return state_input
    if isinstance(state_input, (str, Path)):
        path = Path(state_input)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            return ClusterState.model_validate_json(content)
        # Maybe raw JSON string
        return ClusterState.model_validate_json(str(state_input))
    if isinstance(state_input, dict):
        return ClusterState.model_validate(state_input)
    raise TypeError(f"Cannot load ClusterState from type {type(state_input).__name__}")


def _load_node_scores(scores_input: NodeScores | str | Path | dict[str, Any]) -> NodeScores:
    """Parse or validate input into a NodeScores instance."""
    if isinstance(scores_input, NodeScores):
        return scores_input
    if isinstance(scores_input, (str, Path)):
        path = Path(scores_input)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            return NodeScores.model_validate_json(content)
        # Maybe raw JSON string
        return NodeScores.model_validate_json(str(scores_input))
    if isinstance(scores_input, dict):
        return NodeScores.model_validate(scores_input)
    raise TypeError(f"Cannot load NodeScores from type {type(scores_input).__name__}")


def _score_nodes_gnn(
    cluster_state: ClusterState,
    model_dir: str | Path | None = None,
) -> NodeScores:
    """Run inference with the trained GNN + actor model to produce NodeScores."""
    import numpy as np
    import torch

    # Resolve checkpoints directory
    if model_dir is None:
        candidates = [
            Path(__file__).resolve().parents[2] / "split-ab" / "checkpoints",
            Path("checkpoints"),
        ]
        for c in candidates:
            if c.is_dir():
                model_dir = str(c)
                break
        if model_dir is None:
            raise FileNotFoundError("Could not locate 'a/checkpoints' directory for GNN scorer.")

    checkpoint_path = os.path.join(model_dir, "cluster_sage_gnn_final.pt")
    if not os.path.exists(checkpoint_path):
        pt_files = sorted(
            [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")],
            key=os.path.getmtime,
            reverse=True,
        )
        if pt_files:
            checkpoint_path = pt_files[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in directory '{model_dir}'")

    a_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "a"))
    project_root = Path(__file__).resolve().parents[2]
    split_ab_dir = project_root / "split-ab"

    if str(split_ab_dir) not in sys.path:
        sys.path.insert(0, str(split_ab_dir))

    from models.actor import NodeActor
    from models.gnn_encoder import ClusterGNNEncoder

    device = torch.device("cpu")
    hidden_dim = 64
    gnn_encoder = ClusterGNNEncoder(
        node_feat_dim=8,
        hidden_dim=hidden_dim,
        gnn_layers=2,
        gnn_type="sage",
    ).to(device)

    actor = NodeActor(
        embed_dim=hidden_dim * 2,
        pod_dim=6,
        action_dim=3,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    gnn_encoder.load_state_dict(state_dict["gnn_encoder"])
    actor.load_state_dict(state_dict["actor"])
    gnn_encoder.eval()
    actor.eval()

    nodes = cluster_state.nodes
    num_nodes = len(nodes)
    pending_pod = cluster_state.pending_pod

    node_features = np.zeros((num_nodes, 8), dtype=np.float32)
    zones: list[str] = []
    for i, n in enumerate(nodes):
        cpu_cap = max(n.cpu_capacity, 1e-6)
        mem_cap = max(n.mem_capacity_gb, 1e-6)
        disk_cap = max(n.disk_capacity_gb, 1e-6)
        cpu_util = min(1.0, n.cpu_used / cpu_cap)
        mem_util = min(1.0, n.mem_used_gb / mem_cap)
        disk_util = min(1.0, n.disk_used_gb / disk_cap)
        running_pods = min(1.0, n.pod_count / 50.0)
        health_score = n.health_score
        cost_per_hour = n.cost_per_hour
        zone = n.zone
        if zone not in zones:
            zones.append(zone)
        zone_id = zones.index(zone)
        zone_id_norm = zone_id / max(1, 3)
        recent_failures = float(n.recent_failures)
        node_features[i] = [
            cpu_util, mem_util, disk_util, running_pods,
            health_score, cost_per_hour, zone_id_norm, recent_failures
        ]

    pod_id = pending_pod.pod_id if pending_pod else "unknown"
    if pending_pod:
        cpu_req = pending_pod.cpu_request
        mem_req = pending_pod.mem_request_gb
        disk_req = 0.05
        priority = 1.0 if pending_pod.priority_class == "high" else 0.5
        ft_req = 1.0 if pending_pod.qos_class == "Guaranteed" else 0.0
        duration = pending_pod.estimated_duration_sec
    else:
        cpu_req, mem_req, disk_req, priority, ft_req, duration = 0.1, 0.1, 0.05, 0.5, 0.0, 1.0

    pod_feat = np.array([cpu_req, mem_req, disk_req, priority, ft_req, duration], dtype=np.float32)

    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    if not src:
        edge_index = torch.zeros((2, 1), dtype=torch.int64)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.int64)

    with torch.no_grad():
        x = torch.tensor(node_features, dtype=torch.float32)
        pod = torch.tensor(pod_feat, dtype=torch.float32)
        emb = gnn_encoder(x, edge_index=edge_index)
        h0 = gnn_encoder.input_proj(x)
        cat_embeddings = torch.cat([emb, h0], dim=-1)
        actions = actor(cat_embeddings, pod).numpy()

    scores_list = [
        SingleNodeScore(
            node_id=n.node_id,
            fault_tolerance=round(float(np.clip(actions[i, 0], 0.0, 1.0)), 4),
            utilisation=round(float(np.clip(actions[i, 1], 0.0, 1.0)), 4),
            cost=round(float(np.clip(actions[i, 2], 0.0, 1.0)), 4),
        )
        for i, n in enumerate(nodes)
    ]

    return NodeScores(pod_id=pod_id, scores=scores_list)


def _build_default_telemetry_summary(cluster_state: ClusterState, stress: float) -> str:
    """Construct an informative telemetry summary from cluster state."""
    total_failures = sum(n.recent_failures for n in cluster_state.nodes)
    avg_cpu_pct = sum((n.cpu_used / n.cpu_capacity) for n in cluster_state.nodes) / len(cluster_state.nodes) * 100.0
    avg_mem_pct = sum((n.mem_used_gb / n.mem_capacity_gb) for n in cluster_state.nodes) / len(cluster_state.nodes) * 100.0

    if cluster_state.pending_pod:
        pod_info = (
            f"Pending pod: {cluster_state.pending_pod.pod_id} "
            f"(CPU: {cluster_state.pending_pod.cpu_request}, RAM: {cluster_state.pending_pod.mem_request_gb}GB, "
            f"Priority: {cluster_state.pending_pod.priority_class}, QoS: {cluster_state.pending_pod.qos_class})."
        )
    else:
        pod_info = "Pending pod: None."

    return (
        f"Cluster telemetry snapshot at {cluster_state.timestamp}: "
        f"{len(cluster_state.nodes)} nodes total. "
        f"Average CPU utilisation: {avg_cpu_pct:.1f}%, RAM utilisation: {avg_mem_pct:.1f}%. "
        f"Total recent failure events: {total_failures}. "
        f"Calculated aggregate stress: {stress:.2f} ({stress_level(stress)}). "
        f"{pod_info}"
    )


def decide(
    cluster_state: ClusterState | str | Path | dict[str, Any],
    node_scores: NodeScores | str | Path | dict[str, Any] | None = None,
    telemetry_summary: Any = None,
    use_llm: bool = True,
    llm_client: LLMClientProtocol | None = None,
    epsilon: float = 0.05,
    scorer: str | None = None,
) -> SchedulingDecision | None:
    """
    Execute end-to-end scheduling decision pipeline.

    Parameters:
    - cluster_state: Cluster snapshot (ClusterState instance, dict, or file path).
    - node_scores: Candidate scores across objectives (NodeScores instance, dict, or file path).
    - telemetry_summary: Optional telemetry summary string or dict (auto-generated if None).
    - use_llm: Whether to invoke the LLM agent (True) or static rule-based policy (False).
    - llm_client: Optional mock/custom LLM client.
    - epsilon: Lexicographic tolerance for near-tied scores (default: 0.05).
    - scorer: Optional scorer name ('heuristic' or 'gnn') to dynamically evaluate scores.

    Returns:
    - SchedulingDecision | None: Emitted decision or None if cluster is idle.
    """
    start_time = time.perf_counter()

    # 1. Parse and validate cluster state
    state = _load_cluster_state(cluster_state)

    # 2. Check for idle cluster (pending_pod is None)
    if state.pending_pod is None:
        idle_log = {
            "event": "idle",
            "timestamp": state.timestamp,
            "message": "Cluster has no pending pod to schedule.",
        }
        logger.info("Cluster idle event:\n%s", json.dumps(idle_log, indent=2))
        return None

    # 3. Resolve candidate node scores
    if scorer == "heuristic":
        scores = heuristic_score_nodes(state)
    elif scorer == "gnn":
        scores = _score_nodes_gnn(state)
    elif node_scores is not None:
        scores = _load_node_scores(node_scores)
    else:
        scores = heuristic_score_nodes(state)

    pod_id = scores.pod_id or state.pending_pod.pod_id

    # 4. Compute aggregate cluster stress
    stress = compute_stress(state)
    current_stress_level = stress_level(stress)

    # 5. Determine telemetry summary string
    if telemetry_summary is None or (isinstance(telemetry_summary, str) and not telemetry_summary.strip()):
        telemetry = _build_default_telemetry_summary(state, stress)
    else:
        telemetry = telemetry_summary

    # 6. Determine PriorityOrder
    if use_llm:
        priority_order = get_priority_order(telemetry, client=llm_client)
    else:
        priority_order = rule_based_priority(stress)

    # 7. Drop infeasible nodes before select_node
    # Drop nodes where cpu_capacity - cpu_used < cpu_request or mem_capacity_gb - mem_used_gb < mem_request_gb
    pod_cpu = state.pending_pod.cpu_request
    pod_mem = state.pending_pod.mem_request_gb

    infeasible_node_ids = {
        node.node_id for node in state.nodes
        if (node.cpu_capacity - node.cpu_used < pod_cpu)
        or (node.mem_capacity_gb - node.mem_used_gb < pod_mem)
    }

    cluster_node_ids = {node.node_id for node in state.nodes}
    feasible_cluster_nodes = cluster_node_ids - infeasible_node_ids

    # Match candidate scores
    has_matching_ids = any(s.node_id in cluster_node_ids for s in scores.scores)
    if has_matching_ids:
        feasible_scores = [s for s in scores.scores if s.node_id in feasible_cluster_nodes]
    else:
        if len(infeasible_node_ids) == len(state.nodes) and len(state.nodes) > 0:
            feasible_scores = []
        else:
            feasible_scores = [s for s in scores.scores if s.node_id not in infeasible_node_ids]

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 8. If none are feasible, return a SchedulingDecision with selected_node_id=None and reason logged
    if not feasible_scores:
        decision = SchedulingDecision(
            pod_id=pod_id,
            selected_node_id=None,
            priority_order_used=priority_order,
            scores_snapshot=scores,
            decision_latency_ms=round(latency_ms, 2),
        )
        structured_log = {
            "event": "scheduling_decision",
            "pod_id": decision.pod_id,
            "selected_node_id": None,
            "reason": "no_feasible_nodes",
            "message": "No candidate node has sufficient CPU and memory capacity for the requested pod.",
            "stress_metric": round(stress, 4),
            "stress_level": current_stress_level,
            "use_llm": use_llm,
            "priority_source": decision.priority_order_used.source,
            "priority_order": decision.priority_order_used.priority_order,
            "confidence": decision.priority_order_used.confidence,
            "justification": decision.priority_order_used.justification,
            "decision_latency_ms": decision.decision_latency_ms,
        }
        logger.info("Scheduling decision (no feasible node):\n%s", json.dumps(structured_log, indent=2))
        return decision

    # 9. Run lexicographic node selection on feasible nodes
    filtered_scores_obj = NodeScores(pod_id=pod_id, scores=feasible_scores)
    selected_node = select_node(
        node_scores=filtered_scores_obj,
        priority_order=priority_order.priority_order,
        epsilon=epsilon,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 10. Assemble final SchedulingDecision
    decision = SchedulingDecision(
        pod_id=pod_id,
        selected_node_id=selected_node,
        priority_order_used=priority_order,
        scores_snapshot=scores,
        decision_latency_ms=round(latency_ms, 2),
    )

    # 11. Log full reasoning as structured JSON
    structured_log = {
        "event": "scheduling_decision",
        "pod_id": decision.pod_id,
        "selected_node_id": decision.selected_node_id,
        "stress_metric": round(stress, 4),
        "stress_level": current_stress_level,
        "use_llm": use_llm,
        "priority_source": decision.priority_order_used.source,
        "priority_order": decision.priority_order_used.priority_order,
        "confidence": decision.priority_order_used.confidence,
        "justification": decision.priority_order_used.justification,
        "decision_latency_ms": decision.decision_latency_ms,
    }
    logger.info("Scheduling decision completed:\n%s", json.dumps(structured_log, indent=2))

    return decision


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def _find_path(candidate_str: str) -> Path:
    """Resolve file path across working directory and decision_layer package directory."""
    raw_path = Path(candidate_str)
    if raw_path.is_file():
        return raw_path

    # Try relative to decision_layer
    dl_path = Path("decision_layer") / raw_path
    if dl_path.is_file():
        return dl_path

    # Search in decision_layer/mocks/ if filename matches
    name = raw_path.name
    for search_dir in [Path("decision_layer/mocks"), Path("mocks")]:
        direct = search_dir / name
        if direct.is_file():
            return direct
        # Try appending .json or adding _cluster
        with_json = search_dir / f"{name}.json"
        if with_json.is_file():
            return with_json
        with_cluster = search_dir / f"{name.replace('.json', '')}_cluster.json"
        if with_cluster.is_file():
            return with_cluster

    raise FileNotFoundError(f"Could not locate cluster or scores file at '{candidate_str}'")


def main(argv: list[str] | None = None) -> int:
    """CLI handler for running decision pipeline from command line."""
    parser = argparse.ArgumentParser(
        description="Kubernetes Decision Layer Scheduling Pipeline CLI",
        prog="python -m decision_layer.pipeline",
    )
    parser.add_argument(
        "--cluster",
        required=True,
        help="Path to ClusterState JSON file (e.g. mocks/stressed.json)",
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="Path to NodeScores JSON file (e.g. mocks/scores.json)",
    )
    parser.add_argument(
        "--scorer",
        choices=["heuristic", "gnn"],
        default=None,
        help="Node scorer to use: 'heuristic' or 'gnn' (optional if --scores is specified)",
    )
    parser.add_argument(
        "--telemetry",
        default=None,
        help="Optional telemetry summary text or path to telemetry file",
    )
    parser.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        default=True,
        help="Disable LLM call and force rule-based priority mapping",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Lexicographic filtering tolerance epsilon (default: 0.05)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.scores and not args.scorer:
        parser.error("Either --scores or --scorer must be specified.")

    cluster_path = _find_path(args.cluster)
    scores_path = _find_path(args.scores) if args.scores else None

    telemetry = args.telemetry
    if telemetry and Path(telemetry).is_file():
        telemetry = Path(telemetry).read_text(encoding="utf-8")

    decision = decide(
        cluster_state=cluster_path,
        node_scores=scores_path,
        telemetry_summary=telemetry,
        use_llm=args.use_llm,
        epsilon=args.epsilon,
        scorer=args.scorer,
    )

    if decision is None:
        print(json.dumps({"event": "idle", "status": "no_pending_pod"}, indent=2))
        return 0

    # Print clean formatted JSON decision to stdout
    print(decision.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
