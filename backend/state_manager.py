"""
backend/state_manager.py

Manages active cluster state in memory with predefined scenario presets,
fault injection / node metrics mutation, pod placement binding,
and scheduling history recording.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


def _load_mock_scenario(filename: str) -> Optional[Dict[str, Any]]:
    """Helper to load scenario directly from Split C's mock JSON files."""
    path = Path(__file__).resolve().parent.parent / "split-c" / "decision_layer" / "mocks" / filename
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "nodes": data.get("nodes", []),
                "default_pod": data.get("pending_pod"),
            }
        except Exception:
            return None
    return None

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "calm": {
        "name": "Calm Enterprise Cluster",
        "description": "Healthy balanced cluster across 3 availability zones with low resource pressure and zero failures.",
        "nodes": [
            {
                "node_id": "node-worker-01",
                "cpu_capacity": 32.0,
                "cpu_used": 7.2,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 34.5,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 185.0,
                "health_score": 1.0,
                "cost_per_hour": 0.48,
                "zone": "us-central1-a",
                "pod_count": 18,
                "recent_failures": 0,
            },
            {
                "node_id": "node-worker-02",
                "cpu_capacity": 32.0,
                "cpu_used": 8.5,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 41.0,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 220.0,
                "health_score": 0.99,
                "cost_per_hour": 0.48,
                "zone": "us-central1-b",
                "pod_count": 22,
                "recent_failures": 0,
            },
            {
                "node_id": "node-worker-03",
                "cpu_capacity": 16.0,
                "cpu_used": 3.8,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 18.2,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 110.0,
                "health_score": 1.0,
                "cost_per_hour": 0.24,
                "zone": "us-central1-c",
                "pod_count": 14,
                "recent_failures": 0,
            },
            {
                "node_id": "node-worker-04",
                "cpu_capacity": 16.0,
                "cpu_used": 4.6,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 22.8,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 135.0,
                "health_score": 0.98,
                "cost_per_hour": 0.24,
                "zone": "us-central1-a",
                "pod_count": 16,
                "recent_failures": 0,
            },
            {
                "node_id": "node-worker-05",
                "cpu_capacity": 8.0,
                "cpu_used": 2.1,
                "mem_capacity_gb": 32.0,
                "mem_used_gb": 9.4,
                "disk_capacity_gb": 250.0,
                "disk_used_gb": 62.0,
                "health_score": 1.0,
                "cost_per_hour": 0.12,
                "zone": "us-central1-b",
                "pod_count": 9,
                "recent_failures": 0,
            },
        ],
        "default_pod": {
            "pod_id": "pod-analytics-batch-01",
            "cpu_request": 2.0,
            "mem_request_gb": 4.0,
            "priority_class": "standard",
            "qos_class": "Burstable",
            "estimated_duration_sec": 1800.0,
        },
    },
    "stressed": {
        "name": "High Stress Contention",
        "description": "High CPU and RAM utilization across cluster with multiple recent node failure events.",
        "nodes": [
            {
                "node_id": "node-worker-01",
                "cpu_capacity": 32.0,
                "cpu_used": 29.8,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 118.5,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 850.0,
                "health_score": 0.82,
                "cost_per_hour": 0.48,
                "zone": "us-central1-a",
                "pod_count": 48,
                "recent_failures": 2,
            },
            {
                "node_id": "node-worker-02",
                "cpu_capacity": 32.0,
                "cpu_used": 30.5,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 122.0,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 910.0,
                "health_score": 0.75,
                "cost_per_hour": 0.48,
                "zone": "us-central1-b",
                "pod_count": 52,
                "recent_failures": 3,
            },
            {
                "node_id": "node-worker-03",
                "cpu_capacity": 16.0,
                "cpu_used": 14.8,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 58.0,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 420.0,
                "health_score": 0.88,
                "cost_per_hour": 0.24,
                "zone": "us-central1-c",
                "pod_count": 28,
                "recent_failures": 1,
            },
            {
                "node_id": "node-worker-04",
                "cpu_capacity": 16.0,
                "cpu_used": 13.9,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 54.0,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 390.0,
                "health_score": 0.90,
                "cost_per_hour": 0.24,
                "zone": "us-central1-a",
                "pod_count": 25,
                "recent_failures": 1,
            },
            {
                "node_id": "node-worker-05",
                "cpu_capacity": 8.0,
                "cpu_used": 5.2,
                "mem_capacity_gb": 32.0,
                "mem_used_gb": 21.0,
                "disk_capacity_gb": 250.0,
                "disk_used_gb": 180.0,
                "health_score": 0.94,
                "cost_per_hour": 0.12,
                "zone": "us-central1-b",
                "pod_count": 14,
                "recent_failures": 0,
            },
        ],
        "default_pod": {
            "pod_id": "pod-payment-service-critical",
            "cpu_request": 2.0,
            "mem_request_gb": 6.0,
            "priority_class": "high",
            "qos_class": "Guaranteed",
            "estimated_duration_sec": 3600.0,
        },
    },
    "high_stress_feasible": {
        "name": "High Stress Feasible",
        "description": "High cluster stress with failures, where exactly one node (node-worker-01) has sufficient remaining CPU/RAM capacity.",
        "nodes": _load_mock_scenario("high_stress_feasible.json")["nodes"] if _load_mock_scenario("high_stress_feasible.json") else [
            {
                "node_id": "node-worker-01",
                "cpu_capacity": 32.0,
                "cpu_used": 26.0,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 118.4,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 892.0,
                "health_score": 0.65,
                "cost_per_hour": 0.48,
                "zone": "us-central1-a",
                "pod_count": 98,
                "recent_failures": 4,
            },
            {
                "node_id": "node-worker-02",
                "cpu_capacity": 32.0,
                "cpu_used": 28.2,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 113.6,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 855.0,
                "health_score": 0.72,
                "cost_per_hour": 0.48,
                "zone": "us-central1-b",
                "pod_count": 94,
                "recent_failures": 2,
            },
            {
                "node_id": "node-worker-03",
                "cpu_capacity": 16.0,
                "cpu_used": 15.3,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 60.1,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 468.0,
                "health_score": 0.55,
                "cost_per_hour": 0.24,
                "zone": "us-central1-c",
                "pod_count": 89,
                "recent_failures": 6,
            },
            {
                "node_id": "node-worker-04",
                "cpu_capacity": 16.0,
                "cpu_used": 14.1,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 54.8,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 435.0,
                "health_score": 0.68,
                "cost_per_hour": 0.24,
                "zone": "us-central1-a",
                "pod_count": 83,
                "recent_failures": 3,
            },
            {
                "node_id": "node-worker-05",
                "cpu_capacity": 8.0,
                "cpu_used": 7.5,
                "mem_capacity_gb": 32.0,
                "mem_used_gb": 28.9,
                "disk_capacity_gb": 250.0,
                "disk_used_gb": 226.0,
                "health_score": 0.58,
                "cost_per_hour": 0.12,
                "zone": "us-central1-b",
                "pod_count": 65,
                "recent_failures": 5,
            },
        ],
        "default_pod": _load_mock_scenario("high_stress_feasible.json")["default_pod"] if _load_mock_scenario("high_stress_feasible.json") else {
            "pod_id": "pod-payment-service-auth-901",
            "cpu_request": 4.0,
            "mem_request_gb": 8.0,
            "priority_class": "production-critical",
            "qos_class": "Guaranteed",
            "estimated_duration_sec": 7200.0,
        },
    },
    "zone_outage": {
        "name": "Zone Degraded / Outage",
        "description": "Zone us-central1-c suffers network degradation and node failures. Testing fault-tolerant re-routing.",
        "nodes": [
            {
                "node_id": "node-worker-01",
                "cpu_capacity": 32.0,
                "cpu_used": 12.0,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 45.0,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 300.0,
                "health_score": 1.0,
                "cost_per_hour": 0.48,
                "zone": "us-central1-a",
                "pod_count": 20,
                "recent_failures": 0,
            },
            {
                "node_id": "node-worker-02",
                "cpu_capacity": 32.0,
                "cpu_used": 14.0,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 52.0,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 310.0,
                "health_score": 1.0,
                "cost_per_hour": 0.48,
                "zone": "us-central1-b",
                "pod_count": 22,
                "recent_failures": 0,
            },
            {
                "node_id": "node-worker-03",
                "cpu_capacity": 16.0,
                "cpu_used": 15.5,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 62.0,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 490.0,
                "health_score": 0.20,
                "cost_per_hour": 0.24,
                "zone": "us-central1-c",
                "pod_count": 5,
                "recent_failures": 4,
            },
            {
                "node_id": "node-worker-04",
                "cpu_capacity": 16.0,
                "cpu_used": 6.0,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 24.0,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 150.0,
                "health_score": 0.99,
                "cost_per_hour": 0.24,
                "zone": "us-central1-a",
                "pod_count": 15,
                "recent_failures": 0,
            },
            {
                "node_id": "node-worker-05",
                "cpu_capacity": 8.0,
                "cpu_used": 3.0,
                "mem_capacity_gb": 32.0,
                "mem_used_gb": 12.0,
                "disk_capacity_gb": 250.0,
                "disk_used_gb": 80.0,
                "health_score": 0.98,
                "cost_per_hour": 0.12,
                "zone": "us-central1-b",
                "pod_count": 10,
                "recent_failures": 0,
            },
        ],
        "default_pod": {
            "pod_id": "pod-database-replica-42",
            "cpu_request": 3.0,
            "mem_request_gb": 8.0,
            "priority_class": "high",
            "qos_class": "Guaranteed",
            "estimated_duration_sec": 7200.0,
        },
    },
    "cost_optimized": {
        "name": "Cost Arbitrage Scenario",
        "description": "Cluster with diverse node pricing tiers ($0.10 to $0.96/hr) to showcase cost-saving decision arbiter.",
        "nodes": [
            {
                "node_id": "node-premium-01",
                "cpu_capacity": 64.0,
                "cpu_used": 12.0,
                "mem_capacity_gb": 256.0,
                "mem_used_gb": 48.0,
                "disk_capacity_gb": 2000.0,
                "disk_used_gb": 300.0,
                "health_score": 1.0,
                "cost_per_hour": 0.96,
                "zone": "us-central1-a",
                "pod_count": 15,
                "recent_failures": 0,
            },
            {
                "node_id": "node-standard-02",
                "cpu_capacity": 32.0,
                "cpu_used": 8.0,
                "mem_capacity_gb": 128.0,
                "mem_used_gb": 32.0,
                "disk_capacity_gb": 1000.0,
                "disk_used_gb": 200.0,
                "health_score": 0.98,
                "cost_per_hour": 0.48,
                "zone": "us-central1-b",
                "pod_count": 18,
                "recent_failures": 0,
            },
            {
                "node_id": "node-economy-03",
                "cpu_capacity": 16.0,
                "cpu_used": 5.0,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 20.0,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 110.0,
                "health_score": 0.97,
                "cost_per_hour": 0.20,
                "zone": "us-central1-c",
                "pod_count": 12,
                "recent_failures": 0,
            },
            {
                "node_id": "node-spot-04",
                "cpu_capacity": 16.0,
                "cpu_used": 4.0,
                "mem_capacity_gb": 64.0,
                "mem_used_gb": 18.0,
                "disk_capacity_gb": 500.0,
                "disk_used_gb": 90.0,
                "health_score": 0.95,
                "cost_per_hour": 0.10,
                "zone": "us-central1-a",
                "pod_count": 10,
                "recent_failures": 0,
            },
        ],
        "default_pod": {
            "pod_id": "pod-batch-crawler-99",
            "cpu_request": 2.0,
            "mem_request_gb": 4.0,
            "priority_class": "low",
            "qos_class": "BestEffort",
            "estimated_duration_sec": 600.0,
        },
    },
}

POD_TEMPLATES = [
    {
        "id": "critical-db",
        "name": "Critical Database Replica",
        "cpu_request": 4.0,
        "mem_request_gb": 8.0,
        "priority_class": "high",
        "qos_class": "Guaranteed",
        "estimated_duration_sec": 7200.0,
        "description": "High QoS, mission-critical persistence pod requiring supreme fault tolerance.",
    },
    {
        "id": "api-gateway",
        "name": "Production API Gateway",
        "cpu_request": 2.0,
        "mem_request_gb": 4.0,
        "priority_class": "high",
        "qos_class": "Burstable",
        "estimated_duration_sec": 3600.0,
        "description": "Customer-facing ingress tier requiring high availability and low latency.",
    },
    {
        "id": "ml-training",
        "name": "Distributed ML Worker",
        "cpu_request": 8.0,
        "mem_request_gb": 16.0,
        "priority_class": "standard",
        "qos_class": "Burstable",
        "estimated_duration_sec": 5400.0,
        "description": "Compute-heavy training task seeking bin-packed high utilization.",
    },
    {
        "id": "batch-etl",
        "name": "Batch Analytics ETL",
        "cpu_request": 2.0,
        "mem_request_gb": 4.0,
        "priority_class": "standard",
        "qos_class": "Burstable",
        "estimated_duration_sec": 1800.0,
        "description": "Standard batch transformation job, flexible on cost vs packing.",
    },
    {
        "id": "spot-crawler",
        "name": "Cost-Sensitive Web Crawler",
        "cpu_request": 1.0,
        "mem_request_gb": 2.0,
        "priority_class": "low",
        "qos_class": "BestEffort",
        "estimated_duration_sec": 600.0,
        "description": "Fault-tolerant background spider prioritizing lowest dollar cost.",
    },
]


class StateManager:
    """Manages active cluster state in memory."""

    def __init__(self):
        self.active_scenario: str = "calm"
        self.nodes: List[Dict[str, Any]] = copy.deepcopy(SCENARIOS["calm"]["nodes"])
        self.pending_pod: Optional[Dict[str, Any]] = copy.deepcopy(SCENARIOS["calm"]["default_pod"])
        self.history: List[Dict[str, Any]] = []

    def get_cluster_state_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nodes": copy.deepcopy(self.nodes),
            "pending_pod": copy.deepcopy(self.pending_pod),
            "scenario": self.active_scenario,
        }

    def reset(self, scenario_name: str = "calm") -> Dict[str, Any]:
        if scenario_name not in SCENARIOS:
            scenario_name = "calm"
        self.active_scenario = scenario_name
        self.nodes = copy.deepcopy(SCENARIOS[scenario_name]["nodes"])
        self.pending_pod = copy.deepcopy(SCENARIOS[scenario_name]["default_pod"])
        return self.get_cluster_state_dict()

    def update_node(self, node_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for node in self.nodes:
            if node["node_id"] == node_id:
                for k, v in updates.items():
                    if k in node and v is not None:
                        # preserve type
                        if isinstance(node[k], float):
                            node[k] = float(v)
                        elif isinstance(node[k], int):
                            node[k] = int(v)
                        else:
                            node[k] = v
                return copy.deepcopy(node)
        return None

    def set_pending_pod(self, pod_data: Dict[str, Any]) -> Dict[str, Any]:
        self.pending_pod = copy.deepcopy(pod_data)
        return self.pending_pod

    def bind_pod(self, pod: Dict[str, Any], node_id: str, decision_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Bind a pod to a node, updating resource allocation in the cluster."""
        for node in self.nodes:
            if node["node_id"] == node_id:
                cpu_req = float(pod.get("cpu_request", 0.0))
                mem_req = float(pod.get("mem_request_gb", 0.0))
                node["cpu_used"] = round(min(node["cpu_capacity"], node["cpu_used"] + cpu_req), 2)
                node["mem_used_gb"] = round(min(node["mem_capacity_gb"], node["mem_used_gb"] + mem_req), 2)
                node["pod_count"] += 1

                # Record in history
                event = {
                    "event_id": str(uuid.uuid4())[:8],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pod_id": pod.get("pod_id", "unknown"),
                    "node_id": node_id,
                    "cpu_request": cpu_req,
                    "mem_request_gb": mem_req,
                    "priority_class": pod.get("priority_class", "default"),
                    "qos_class": pod.get("qos_class", "BestEffort"),
                    "decision": decision_metadata or {},
                }
                self.history.insert(0, event)
                # Keep last 50 events
                if len(self.history) > 50:
                    self.history = self.history[:50]
                return True
        return False

    def get_history(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.history)

    def clear_history(self) -> None:
        self.history = []


# Global singleton instance
state_manager = StateManager()
