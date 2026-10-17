"""
decision_layer/schemas.py

Pydantic models and frozen contracts for the agentic Kubernetes scheduler
decision layer.

Frozen contracts:
- ClusterState
- NodeScores
- PriorityOrder
- SchedulingDecision
- Objectives: "fault_tolerance", "utilisation", "cost"
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Frozen objectives
Objective = Literal["fault_tolerance", "utilisation", "cost"]
VALID_OBJECTIVES: tuple[Objective, ...] = (
    "fault_tolerance",
    "utilisation",
    "cost",
)

PrioritySource = Literal["llm", "rule"]


class NodeState(BaseModel):
    """
    Representation of a single Kubernetes node's resource capacities,
    utilization metrics, health, cost, and failure history.
    """
    model_config = ConfigDict(extra="ignore")

    node_id: str
    cpu_capacity: float = Field(..., gt=0.0, description="Total CPU capacity in cores")
    cpu_used: float = Field(..., ge=0.0, description="CPU cores currently allocated or used")
    mem_capacity_gb: float = Field(..., gt=0.0, description="Total RAM capacity in GB")
    mem_used_gb: float = Field(..., ge=0.0, description="RAM currently used in GB")
    disk_capacity_gb: float = Field(..., gt=0.0, description="Total disk storage in GB")
    disk_used_gb: float = Field(..., ge=0.0, description="Disk storage currently used in GB")
    health_score: float = Field(..., ge=0.0, le=1.0, description="Node health score in [0, 1]")
    cost_per_hour: float = Field(..., ge=0.0, description="Monetary cost per node-hour")
    zone: str = Field(..., description="Availability zone, e.g. us-central1-a")
    pod_count: int = Field(..., ge=0, description="Number of pods currently running on the node")
    recent_failures: int = Field(..., ge=0, description="Count of recent node or pod failure events")


# Semantic aliases
Node = NodeState
ClusterNode = NodeState


class PendingPod(BaseModel):
    """
    Representation of a pending pod awaiting a scheduling decision.
    """
    model_config = ConfigDict(extra="ignore")

    pod_id: str
    cpu_request: float = Field(..., gt=0.0, description="Requested CPU cores")
    mem_request_gb: float = Field(..., gt=0.0, description="Requested RAM in GB")
    priority_class: str = Field(..., description="Kubernetes priority class (e.g. high, default)")
    qos_class: str = Field(..., description="QoS class (e.g. Guaranteed, Burstable, BestEffort)")
    estimated_duration_sec: float = Field(..., ge=0.0, description="Estimated execution duration in seconds")


# Semantic aliases
PodRequest = PendingPod
Pod = PendingPod


class ClusterState(BaseModel):
    """
    Snapshot of cluster state containing candidate nodes and the pending pod.
    """
    model_config = ConfigDict(extra="ignore")

    timestamp: str | float = Field(..., description="Snapshot timestamp (ISO 8601 string or epoch float)")
    nodes: list[NodeState] = Field(..., min_length=1, description="List of candidate nodes in the cluster")
    pending_pod: PendingPod | None = Field(default=None, description="The pod to be scheduled")


class SingleNodeScore(BaseModel):
    """
    Multi-objective scores for a single node with respect to a pending pod.
    All scores must be in [0, 1], where 1 represents the optimal score.
    """
    model_config = ConfigDict(extra="ignore")

    node_id: str
    fault_tolerance: float = Field(..., ge=0.0, le=1.0, description="Fault tolerance score in [0, 1]")
    utilisation: float = Field(..., ge=0.0, le=1.0, description="Utilisation / bin-packing score in [0, 1]")
    cost: float = Field(..., ge=0.0, le=1.0, description="Cost efficiency score in [0, 1]")


# Semantic alias
NodeScore = SingleNodeScore


class NodeScores(BaseModel):
    """
    Candidate node scores evaluated for a specific pending pod.
    """
    model_config = ConfigDict(extra="ignore")

    pod_id: str
    scores: list[SingleNodeScore] = Field(..., description="Evaluated scores per candidate node")


class PriorityOrder(BaseModel):
    """
    Strict priority ranking across the three scheduling objectives.
    """
    model_config = ConfigDict(extra="ignore")

    priority_order: list[Objective] = Field(
        ...,
        description="Permutation of ['fault_tolerance', 'utilisation', 'cost'] in descending priority order",
    )
    justification: str = Field(..., description="Reasoning behind the selected priority order")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level in [0, 1]")
    source: PrioritySource = Field(..., description="Origin of the priority decision: 'llm' or 'rule'")

    @field_validator("priority_order")
    @classmethod
    def validate_priority_order(cls, v: list[Objective]) -> list[Objective]:
        if len(v) != 3:
            raise ValueError(f"priority_order must contain exactly 3 objectives, got {len(v)}")
        if set(v) != set(VALID_OBJECTIVES):
            raise ValueError(
                f"priority_order must contain exactly the 3 objectives {VALID_OBJECTIVES}, got {v}"
            )
        return v


class SchedulingDecision(BaseModel):
    """
    Final decision emitted by the decision layer pipeline.
    """
    model_config = ConfigDict(extra="ignore")

    pod_id: str
    selected_node_id: str | None = None
    priority_order_used: PriorityOrder
    scores_snapshot: NodeScores
    decision_latency_ms: float = Field(..., ge=0.0, description="Decision pipeline latency in milliseconds")


__all__ = [
    "Objective",
    "VALID_OBJECTIVES",
    "PrioritySource",
    "NodeState",
    "Node",
    "ClusterNode",
    "PendingPod",
    "PodRequest",
    "Pod",
    "ClusterState",
    "SingleNodeScore",
    "NodeScore",
    "NodeScores",
    "PriorityOrder",
    "SchedulingDecision",
]
