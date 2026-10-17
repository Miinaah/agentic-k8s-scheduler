"""
cluster_env.py

Discrete-time, multi-node Kubernetes cluster simulator for multi-agent RL
experiments (PettingZoo ParallelEnv-style API).

Design assumptions (flagged for review against project plan section 6):
- One agent per node ("node_0", ..., "node_{N-1}"). At each step there is a
  bounded queue of pending pods; each node-agent chooses whether to bid on
  one of the pending pods (or bid on nothing). Bids are resolved by the
  environment (highest-priority QoS pod wins ties; a node can only accept
  a pod that fits its remaining capacity).
- Action space per agent: Discrete(MAX_QUEUE + 1), where 0 = "pass" and
  k in [1, MAX_QUEUE] = "bid on pending_queue[k-1]".
- Reward per agent: a shaped combination of the three cluster-level scores
  plus a small per-node local utilization term, computed after all bids for
  the step are resolved. Adjust `ClusterEnv._compute_rewards` to match your
  reward design once it's finalized.
- Pod lifetime: pods run for a sampled duration and release resources on
  completion; QoS class affects both request-size distribution and
  eviction priority under pressure (BestEffort evicted first).

None of this is load-bearing for the parts you asked for (workload
generator, stress injection, scoring functions, JSON schema) — those are
usable independently of the exact MARL action design above.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# --------------------------------------------------------------------------
# Core data model
# --------------------------------------------------------------------------

class QoSClass(str, Enum):
    GUARANTEED = "Guaranteed"
    BURSTABLE = "Burstable"
    BEST_EFFORT = "BestEffort"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class PodStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    EVICTED = "evicted"
    COMPLETED = "completed"


@dataclass
class Node:
    node_id: str
    cpu_capacity: float
    cpu_used: float
    memory_capacity: float
    memory_used: float
    disk_capacity: float
    disk_used: float
    health_score: float = 1.0
    cost_per_hour: float = 1.0
    zone: str = "zone-a"
    recent_failures: int = 0
    status: NodeStatus = NodeStatus.ACTIVE
    hosted_pods: List[str] = field(default_factory=list)

    def remaining(self, resource: str) -> float:
        cap = getattr(self, f"{resource}_capacity")
        used = getattr(self, f"{resource}_used")
        return max(0.0, cap - used)

    def utilization(self, resource: str) -> float:
        cap = getattr(self, f"{resource}_capacity")
        if cap <= 0:
            return 0.0
        used = getattr(self, f"{resource}_used")
        return min(1.0, used / cap)

    def can_fit(self, cpu_req: float, mem_req: float) -> bool:
        if self.status != NodeStatus.ACTIVE:
            return False
        return self.remaining("cpu") >= cpu_req and self.remaining("memory") >= mem_req


@dataclass
class Pod:
    pod_id: str
    cpu_request: float
    memory_request: float
    qos_class: QoSClass
    priority_class: str = "medium"
    arrival_time: int = 0
    duration: int = 1
    status: PodStatus = PodStatus.PENDING
    node_id: Optional[str] = None
    remaining_duration: Optional[int] = None

    def __post_init__(self):
        if self.remaining_duration is None:
            self.remaining_duration = self.duration


# --------------------------------------------------------------------------
# Workload generator
# --------------------------------------------------------------------------

class WorkloadGenerator:
    """
    Generates pod arrivals via a Poisson process and samples request sizes
    per QoS class from configurable distributions.

    Parameters
    ----------
    arrival_rate : float
        Expected number of pod arrivals per timestep (Poisson lambda).
    qos_weights : dict
        Relative sampling probability for each QoS class. Normalized
        internally, so absolute values don't matter.
    cpu_request_params, memory_request_params : dict
        Per-QoS-class (mean, sigma) for a lognormal request-size
        distribution, in "cores" and "GiB" units respectively.
    duration_params : tuple
        (mean, sigma) for a lognormal pod duration in timesteps.
    rng : random.Random or None
        Injectable RNG for reproducibility.
    """

    DEFAULT_CPU_PARAMS = {
        QoSClass.GUARANTEED: (0.6, 0.3),
        QoSClass.BURSTABLE: (0.3, 0.4),
        QoSClass.BEST_EFFORT: (0.1, 0.5),
    }
    DEFAULT_MEM_PARAMS = {
        QoSClass.GUARANTEED: (1.0, 0.3),
        QoSClass.BURSTABLE: (0.5, 0.4),
        QoSClass.BEST_EFFORT: (0.2, 0.5),
    }

    def __init__(
        self,
        arrival_rate: float = 2.0,
        qos_weights: Optional[Dict[QoSClass, float]] = None,
        cpu_request_params: Optional[Dict[QoSClass, Tuple[float, float]]] = None,
        memory_request_params: Optional[Dict[QoSClass, Tuple[float, float]]] = None,
        duration_params: Tuple[float, float] = (3.5, 0.8),
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ):
        if arrival_rate < 0:
            raise ValueError("arrival_rate must be non-negative")
        self.arrival_rate = arrival_rate
        self._np_rng = np.random.default_rng(seed)
        if rng is None and seed is not None:
            rng = random.Random(seed)
        self.qos_weights = qos_weights or {
            QoSClass.GUARANTEED: 0.2,
            QoSClass.BURSTABLE: 0.5,
            QoSClass.BEST_EFFORT: 0.3,
        }
        total = sum(self.qos_weights.values())
        self.qos_weights = {k: v / total for k, v in self.qos_weights.items()}
        self.cpu_params = cpu_request_params or self.DEFAULT_CPU_PARAMS
        self.mem_params = memory_request_params or self.DEFAULT_MEM_PARAMS
        self.duration_params = duration_params
        self.rng = rng or random.Random()
        self._pod_counter = 0

    def _sample_qos(self) -> QoSClass:
        classes = list(self.qos_weights.keys())
        weights = list(self.qos_weights.values())
        return self.rng.choices(classes, weights=weights, k=1)[0]

    def _lognormal(self, mean: float, sigma: float) -> float:
        # mean here is the target *median*-ish scale, not the lognormal's
        # underlying mu; we derive mu so exp(mu) == mean for interpretability.
        mu = math.log(max(mean, 1e-6))
        return float(self.rng.lognormvariate(mu, sigma))

    def next_arrivals(self, current_time: int) -> List[Pod]:
        """Sample the number and attributes of pods arriving at current_time."""
        n_arrivals = self._np_rng.poisson(self.arrival_rate) if self.arrival_rate > 0 else 0
        pods = []
        for _ in range(n_arrivals):
            qos = self._sample_qos()
            cpu_mean, cpu_sigma = self.cpu_params[qos]
            mem_mean, mem_sigma = self.mem_params[qos]
            cpu_req = round(self._lognormal(cpu_mean, cpu_sigma), 3)
            mem_req = round(self._lognormal(mem_mean, mem_sigma), 3)
            dur_mean, dur_sigma = self.duration_params
            duration = max(1, int(round(self._lognormal(dur_mean, dur_sigma))))
            self._pod_counter += 1
            priority_class = {
                QoSClass.GUARANTEED: "high",
                QoSClass.BURSTABLE: "medium",
                QoSClass.BEST_EFFORT: "low",
            }[qos]
            pods.append(
                Pod(
                    pod_id=f"pod_{self._pod_counter}",
                    cpu_request=cpu_req,
                    memory_request=mem_req,
                    qos_class=qos,
                    priority_class=priority_class,
                    arrival_time=current_time,
                    duration=duration,
                )
            )
        return pods


# --------------------------------------------------------------------------
# Scoring functions
# --------------------------------------------------------------------------

def fault_tolerance_score(nodes: List[Node]) -> float:
    """
    Score in [0, 1] combining node health and the cluster's spare-capacity
    redundancy (how much headroom exists to absorb a node failure).

    Heuristic: mean health of active nodes, penalized if losing the single
    most-loaded active node would exceed the remaining nodes' spare
    capacity (i.e. no redundancy to absorb that node's load).
    """
    active = [n for n in nodes if n.status == NodeStatus.ACTIVE]
    if not active:
        return 0.0

    mean_health = statistics.mean(n.health_score for n in active)

    if len(active) == 1:
        redundancy = 0.0
    else:
        loads = [(n.cpu_used, n) for n in active]
        heaviest_load, heaviest_node = max(loads, key=lambda t: t[0])
        others = [n for n in active if n.node_id != heaviest_node.node_id]
        spare_cpu = sum(n.remaining("cpu") for n in others)
        redundancy = 1.0 if heaviest_load <= spare_cpu else spare_cpu / max(heaviest_load, 1e-6)
        redundancy = min(1.0, redundancy)

    failed_ratio_penalty = 1.0 - (len([n for n in nodes if n.status == NodeStatus.FAILED]) / len(nodes))

    return round(0.5 * mean_health + 0.3 * redundancy + 0.2 * failed_ratio_penalty, 4)


def utilization_balance_score(nodes: List[Node]) -> float:
    """
    Score in [0, 1] rewarding even CPU+memory utilization across active
    nodes. 1.0 = perfectly balanced, 0.0 = maximally skewed.

    Computed as 1 - normalized standard deviation of per-node average
    (cpu_util, mem_util) across active nodes.
    """
    active = [n for n in nodes if n.status == NodeStatus.ACTIVE]
    if len(active) < 2:
        return 1.0 if active else 0.0

    avg_utils = [(n.utilization("cpu") + n.utilization("memory")) / 2 for n in active]
    mean_u = statistics.mean(avg_utils)
    stdev_u = statistics.pstdev(avg_utils)
    # Max possible pstdev for values in [0,1] is 0.5 (half at 0, half at 1)
    normalized_spread = min(1.0, stdev_u / 0.5)
    return round(1.0 - normalized_spread, 4)


def cost_efficiency_score(nodes: List[Node]) -> float:
    """
    Score in [0, 1]: useful work (aggregate CPU+memory utilization) per
    dollar of running cost, normalized against the cluster's own max
    possible efficiency (i.e. relative efficiency, not absolute $/unit).

    Nodes with zero cost are treated as maximally efficient for their
    utilization (avoids division by zero) and excluded from the
    normalization denominator's cost sum.
    """
    active = [n for n in nodes if n.status == NodeStatus.ACTIVE]
    if not active:
        return 0.0

    total_cost = sum(n.cost_per_hour for n in active)
    total_util_units = sum(
        n.utilization("cpu") * n.cpu_capacity + n.utilization("memory") * n.memory_capacity
        for n in active
    )
    total_capacity_units = sum(n.cpu_capacity + n.memory_capacity for n in active)

    if total_cost <= 0:
        # No cost incurred; efficiency is just how much of the free
        # capacity is being used.
        return round(total_util_units / max(total_capacity_units, 1e-6), 4)

    raw_efficiency = total_util_units / total_cost
    # Normalize against a theoretical max (100% utilization at current cost)
    max_efficiency = total_capacity_units / total_cost
    return round(min(1.0, raw_efficiency / max(max_efficiency, 1e-6)), 4)


# --------------------------------------------------------------------------
# Stress injection
# --------------------------------------------------------------------------

def inject_node_failure(env: "ClusterEnv", node_id: Optional[str] = None, duration: Optional[int] = None) -> str:
    """
    Marks a node as FAILED (health_score forced to 0, status FAILED). Its
    hosted pods are evicted and re-queued as pending. If node_id is None,
    a random active node is chosen. Returns the affected node_id.

    If duration is given, the env's `active_incidents` list is used to
    auto-recover the node after `duration` timesteps (recovery handled in
    ClusterEnv.step()).
    """
    active_nodes = [n for n in env.nodes.values() if n.status == NodeStatus.ACTIVE]
    if not active_nodes:
        raise RuntimeError("No active nodes available to fail")
    node = env.nodes[node_id] if node_id else env.rng.choice(active_nodes)

    node.status = NodeStatus.FAILED
    node.health_score = 0.0
    evicted_pod_ids = list(node.hosted_pods)
    for pid in evicted_pod_ids:
        pod = env.pods[pid]
        pod.status = PodStatus.EVICTED
        pod.node_id = None
        node.cpu_used -= pod.cpu_request
        node.memory_used -= pod.memory_request
        env.pending_queue.append(pod)
    node.hosted_pods.clear()
    node.cpu_used = 0.0
    node.memory_used = 0.0

    if duration is not None:
        env.active_incidents.append(
            {"type": "node_failure", "node_id": node.node_id, "ends_at": env.current_time + duration}
        )
    return node.node_id


def inject_traffic_spike(env: "ClusterEnv", multiplier: float = 3.0, duration: int = 5) -> None:
    """
    Temporarily multiplies the workload generator's arrival_rate by
    `multiplier` for `duration` timesteps, then reverts. Registers the
    incident on env.active_incidents for automatic reversion in step().
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    original_rate = env.workload_generator.arrival_rate
    env.workload_generator.arrival_rate = original_rate * multiplier
    env.active_incidents.append(
        {
            "type": "traffic_spike",
            "ends_at": env.current_time + duration,
            "restore_value": original_rate,
        }
    )


def inject_maintenance_window(env: "ClusterEnv", node_id: str, duration: int = 5) -> None:
    """
    Puts a node into MAINTENANCE status (unschedulable, but not treated as
    a failure for fault-tolerance scoring purposes). Existing pods on the
    node continue running; new pods will not be scheduled there until the
    window ends.
    """
    if node_id not in env.nodes:
        raise KeyError(f"Unknown node_id: {node_id}")
    node = env.nodes[node_id]
    node.status = NodeStatus.MAINTENANCE
    env.active_incidents.append(
        {"type": "maintenance", "node_id": node_id, "ends_at": env.current_time + duration}
    )


# --------------------------------------------------------------------------
# JSON serialization
# --------------------------------------------------------------------------

def _iso_timestamp(env: "ClusterEnv") -> str:
    """
    Renders env.current_time (an integer step count) as an ISO-8601-ish
    string, matching the schema's timestamp format. Simulated time starts
    at an arbitrary epoch and advances one second per timestep -- swap
    this out if your simulator should map steps to a different real-time
    scale (e.g. one timestep = one minute).
    """
    import datetime

    base = datetime.datetime(2026, 1, 1)
    return (base + datetime.timedelta(seconds=env.current_time)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _node_to_schema(n: Node) -> Dict[str, Any]:
    """Serializes a single Node to the frozen ClusterState node shape (section 6)."""
    return {
        "node_id": n.node_id,
        "cpu_capacity": n.cpu_capacity,
        "cpu_used": round(n.cpu_used, 4),
        "mem_capacity_gb": n.memory_capacity,
        "mem_used_gb": round(n.memory_used, 4),
        "disk_capacity_gb": n.disk_capacity,
        "disk_used_gb": round(n.disk_used, 4),
        "health_score": n.health_score,
        "cost_per_hour": n.cost_per_hour,
        "zone": n.zone,
        "pod_count": len(n.hosted_pods),
        "recent_failures": n.recent_failures,
    }


def _pending_pod_to_schema(p: Optional[Pod]) -> Optional[Dict[str, Any]]:
    """Serializes the head-of-queue Pod to the frozen PodRequest shape, or None if idle."""
    if p is None:
        return None
    return {
        "pod_id": p.pod_id,
        "cpu_request": p.cpu_request,
        "mem_request_gb": p.memory_request,
        "priority_class": p.priority_class,
        "qos_class": p.qos_class.value,
        "estimated_duration_sec": p.duration,
    }


def cluster_state_to_dict(env: "ClusterEnv") -> Dict[str, Any]:
    """
    Serializes the full cluster state to a dict matching the frozen
    ClusterState schema (project plan, section 6): {timestamp, nodes[],
    pending_pod}. The schema models one pending pod at a time (the pod
    Split B/C are currently scoring nodes for) -- here that's the head of
    `env.pending_queue`; the rest of the queue is exposed separately under
    `pending_queue_depth` for debugging/monitoring, since it isn't part of
    the frozen contract itself.

    `scores` and `active_incidents` are also included as extra top-level
    keys beyond the frozen schema -- useful for Split D's evaluation
    harness and for your own debugging, but not required by Split B/C, so
    drop them if strict schema conformance is being checked downstream.
    """
    head_pod = env.pending_queue[0] if env.pending_queue else None
    return {
        "timestamp": _iso_timestamp(env),
        "nodes": [_node_to_schema(n) for n in env.nodes.values()],
        "pending_pod": _pending_pod_to_schema(head_pod),
        "pending_queue_depth": len(env.pending_queue),
        "scores": {
            "fault_tolerance": fault_tolerance_score(list(env.nodes.values())),
            "utilization_balance": utilization_balance_score(list(env.nodes.values())),
            "cost_efficiency": cost_efficiency_score(list(env.nodes.values())),
        },
        "active_incidents": env.active_incidents,
    }


def cluster_state_to_json(env: "ClusterEnv", **json_kwargs) -> str:
    """Serializes the cluster state to a JSON string matching section 6's ClusterState."""
    return json.dumps(cluster_state_to_dict(env), **json_kwargs)


# --------------------------------------------------------------------------
# ClusterEnv
# --------------------------------------------------------------------------

class ClusterEnv:
    """
    Discrete-time multi-agent Kubernetes cluster scheduling environment.

    Follows a PettingZoo ParallelEnv-shaped API (reset/step return dicts
    keyed by agent id) without importing pettingzoo directly, so it can be
    wrapped in `pettingzoo.utils.ParallelEnv` or a thin Gymnasium adapter
    without modification. See module docstring for the assumed action
    design.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (and agents) in the cluster.
    node_capacity_range : dict
        {"cpu": (min, max), "memory": (min, max), "disk": (min, max)}
        Uniform sampling range for each node's capacity at reset.
    cost_range : tuple
        (min, max) uniform range for cost_per_hour at reset.
    max_queue : int
        Max number of pending pods visible/biddable per step.
    workload_generator : WorkloadGenerator or None
        Injectable generator; a default is created if None.
    max_steps : int
        Episode length in timesteps.
    seed : int or None
        Seed for the environment's RNG (used for node generation and
        bid tie-breaking; does not control the workload generator's RNG
        unless you also seed that separately).
    """

    def __init__(
        self,
        num_nodes: int = 5,
        node_capacity_range: Optional[Dict[str, Tuple[float, float]]] = None,
        cost_range: Tuple[float, float] = (0.5, 2.0),
        max_queue: int = 8,
        workload_generator: Optional[WorkloadGenerator] = None,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        self.num_nodes = num_nodes
        self.node_capacity_range = node_capacity_range or {
            "cpu": (4.0, 16.0),
            "memory": (8.0, 64.0),
            "disk": (50.0, 500.0),
        }
        self.cost_range = cost_range
        self.max_queue = max_queue
        self.workload_generator = workload_generator or WorkloadGenerator()
        self.max_steps = max_steps

        self.agents: List[str] = [f"node_{i}" for i in range(num_nodes)]
        self.possible_agents = list(self.agents)

        self.rng = random.Random(seed)
        self.nodes: Dict[str, Node] = {}
        self.pods: Dict[str, Pod] = {}
        self.pending_queue: List[Pod] = []
        self.active_incidents: List[Dict[str, Any]] = []
        self.current_time: int = 0

    # -- lifecycle -----------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to a fresh cluster with no pods and an
        empty pending queue. Returns (observations, infos) dicts keyed by
        agent id, matching the ParallelEnv reset() contract.
        """
        if seed is not None:
            self.rng = random.Random(seed)

        self.nodes = {}
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            cpu_cap = self.rng.uniform(*self.node_capacity_range["cpu"])
            mem_cap = self.rng.uniform(*self.node_capacity_range["memory"])
            disk_cap = self.rng.uniform(*self.node_capacity_range["disk"])
            cost = self.rng.uniform(*self.cost_range)
            self.nodes[node_id] = Node(
                node_id=node_id,
                cpu_capacity=round(cpu_cap, 2),
                cpu_used=0.0,
                memory_capacity=round(mem_cap, 2),
                memory_used=0.0,
                disk_capacity=round(disk_cap, 2),
                disk_used=0.0,
                health_score=1.0,
                cost_per_hour=round(cost, 3),
                status=NodeStatus.ACTIVE,
            )

        self.pods = {}
        self.pending_queue = []
        self.active_incidents = []
        self.current_time = 0
        self.agents = list(self.possible_agents)

        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        """
        Advances the simulation by one timestep.

        Parameters
        ----------
        actions : dict
            {agent_id: action_index}. action_index 0 = pass;
            k in [1, max_queue] = bid on pending_queue[k-1] (ignored if
            out of range or the pod no longer exists).

        Returns
        -------
        observations, rewards, terminations, truncations, infos
            Each a dict keyed by agent id, matching ParallelEnv.step().
        """
        self._resolve_incidents()
        self._advance_running_pods()
        self._resolve_bids(actions)

        new_pods = self.workload_generator.next_arrivals(self.current_time)
        for pod in new_pods:
            self.pods[pod.pod_id] = pod
        self.pending_queue.extend(new_pods)
        if len(self.pending_queue) > self.max_queue:
            self.pending_queue = self.pending_queue[: self.max_queue]

        self.current_time += 1
        terminated = self.current_time >= self.max_steps

        observations = {agent: self._observe(agent) for agent in self.agents}
        rewards = self._compute_rewards()
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # -- internals -------------------------------------------------------

    def _resolve_incidents(self) -> None:
        still_active = []
        for incident in self.active_incidents:
            if incident["ends_at"] > self.current_time:
                still_active.append(incident)
                continue
            if incident["type"] == "traffic_spike":
                self.workload_generator.arrival_rate = incident["restore_value"]
            elif incident["type"] == "maintenance":
                node = self.nodes.get(incident["node_id"])
                if node and node.status == NodeStatus.MAINTENANCE:
                    node.status = NodeStatus.ACTIVE
            elif incident["type"] == "node_failure":
                node = self.nodes.get(incident["node_id"])
                if node and node.status == NodeStatus.FAILED:
                    node.status = NodeStatus.ACTIVE
                    node.health_score = 1.0
        self.active_incidents = still_active

    def _advance_running_pods(self) -> None:
        for pod in list(self.pods.values()):
            if pod.status != PodStatus.RUNNING:
                continue
            pod.remaining_duration -= 1
            if pod.remaining_duration <= 0:
                pod.status = PodStatus.COMPLETED
                node = self.nodes.get(pod.node_id)
                if node:
                    node.cpu_used = max(0.0, node.cpu_used - pod.cpu_request)
                    node.memory_used = max(0.0, node.memory_used - pod.memory_request)
                    if pod.pod_id in node.hosted_pods:
                        node.hosted_pods.remove(pod.pod_id)

    def _resolve_bids(self, actions: Dict[str, int]) -> None:
        # Group bids by queue index; among nodes bidding on the same pod,
        # prefer the node with the most spare CPU (a simple, deterministic
        # tie-break rule -- swap out for your own scheduling policy).
        bids_by_index: Dict[int, List[str]] = {}
        for agent, action in actions.items():
            if action is None or action == 0:
                continue
            idx = action - 1
            if 0 <= idx < len(self.pending_queue):
                bids_by_index.setdefault(idx, []).append(agent)

        placed_indices = []
        # QoS priority: schedule Guaranteed pods' bids first, then Burstable,
        # then BestEffort, so higher-priority pods win resource contention.
        qos_priority = {QoSClass.GUARANTEED: 0, QoSClass.BURSTABLE: 1, QoSClass.BEST_EFFORT: 2}
        ordered_indices = sorted(
            bids_by_index.keys(),
            key=lambda i: qos_priority[self.pending_queue[i].qos_class],
        )

        for idx in ordered_indices:
            if idx >= len(self.pending_queue):
                continue
            pod = self.pending_queue[idx]
            candidates = [
                self.nodes[a] for a in bids_by_index[idx]
                if a in self.nodes and self.nodes[a].can_fit(pod.cpu_request, pod.memory_request)
            ]
            if not candidates:
                continue
            chosen = max(candidates, key=lambda n: n.remaining("cpu"))
            chosen.cpu_used += pod.cpu_request
            chosen.memory_used += pod.memory_request
            chosen.hosted_pods.append(pod.pod_id)
            pod.status = PodStatus.RUNNING
            pod.node_id = chosen.node_id
            placed_indices.append(idx)

        for idx in sorted(placed_indices, reverse=True):
            del self.pending_queue[idx]

    def _observe(self, agent: str) -> Dict[str, Any]:
        node = self.nodes[agent]
        return {
            "own_node": {
                "cpu_util": node.utilization("cpu"),
                "memory_util": node.utilization("memory"),
                "disk_util": node.utilization("disk"),
                "health_score": node.health_score,
                "cost_per_hour": node.cost_per_hour,
                "status": node.status.value,
            },
            "cluster_summary": {
                "num_active_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.ACTIVE),
                "mean_cpu_util": statistics.mean(n.utilization("cpu") for n in self.nodes.values()),
            },
            "pending_queue": [
                {
                    "cpu_request": p.cpu_request,
                    "memory_request": p.memory_request,
                    "qos_class": p.qos_class.value,
                }
                for p in self.pending_queue[: self.max_queue]
            ],
        }

    def _compute_rewards(self) -> Dict[str, float]:
        nodes_list = list(self.nodes.values())
        ft = fault_tolerance_score(nodes_list)
        ub = utilization_balance_score(nodes_list)
        ce = cost_efficiency_score(nodes_list)
        global_reward = 0.4 * ft + 0.3 * ub + 0.3 * ce
        rewards = {}
        for agent in self.agents:
            node = self.nodes[agent]
            local_term = (node.utilization("cpu") + node.utilization("memory")) / 2
            rewards[agent] = round(0.7 * global_reward + 0.3 * local_term, 4)
        return rewards

    # -- convenience -----------------------------------------------------

    def to_json(self, **kwargs) -> str:
        return cluster_state_to_json(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return cluster_state_to_dict(self)
