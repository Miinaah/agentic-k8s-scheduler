"""
backend/app.py

FastAPI REST API server for the Agentic Kubernetes Scheduler Studio.
Provides cluster topology management, interactive what-if fault injection,
multi-objective decision tracing, pod placement commit, and history feeds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from state_manager import SCENARIOS, POD_TEMPLATES, state_manager
from scheduler_bridge import run_decision_trace, check_kubernetes_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backend.app")

app = FastAPI(
    title="Agentic K8s Scheduler Studio API",
    description="Interactive control plane and observability API for the GNN + Multi-Objective Agentic Scheduler",
    version="1.0.0",
)

# Enable CORS for frontend Vite dev server and production builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NodeUpdateRequest(BaseModel):
    node_id: str
    cpu_used: Optional[float] = Field(None, ge=0.0)
    mem_used_gb: Optional[float] = Field(None, ge=0.0)
    health_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recent_failures: Optional[int] = Field(None, ge=0)
    cost_per_hour: Optional[float] = Field(None, ge=0.0)


class PodRequest(BaseModel):
    pod_id: str
    cpu_request: float = Field(..., gt=0.0)
    mem_request_gb: float = Field(..., gt=0.0)
    priority_class: str = "standard"
    qos_class: str = "Burstable"
    estimated_duration_sec: float = 1800.0


class ScheduleRequest(BaseModel):
    use_llm: bool = False
    scorer: str = "heuristic"  # 'heuristic' or 'gnn'
    epsilon: float = 0.05
    pod: Optional[PodRequest] = None


class ResetRequest(BaseModel):
    scenario: str = "calm"


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    return {"status": "ok", "service": "agentic-k8s-scheduler-api"}


@app.get("/api/cluster/state")
def get_cluster_state() -> Dict[str, Any]:
    return state_manager.get_cluster_state_dict()


@app.post("/api/cluster/reset")
def reset_cluster(req: ResetRequest) -> Dict[str, Any]:
    if req.scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario '{req.scenario}'")
    updated_state = state_manager.reset(req.scenario)
    return {
        "message": f"Cluster reset to scenario '{req.scenario}'",
        "state": updated_state,
    }


@app.post("/api/cluster/node/update")
def update_node(req: NodeUpdateRequest) -> Dict[str, Any]:
    updates = req.model_dump(exclude_unset=True, exclude={"node_id"})
    updated_node = state_manager.update_node(req.node_id, updates)
    if not updated_node:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found")
    return {
        "message": f"Node '{req.node_id}' updated successfully",
        "node": updated_node,
    }


@app.post("/api/schedule/evaluate")
def evaluate_schedule(req: ScheduleRequest) -> Dict[str, Any]:
    """Runs a dry-run scheduling decision trace without mutating cluster state."""
    state_dict = state_manager.get_cluster_state_dict()
    if req.pod:
        state_dict["pending_pod"] = req.pod.model_dump()

    trace = run_decision_trace(
        cluster_state_dict=state_dict,
        use_llm=req.use_llm,
        scorer=req.scorer,
        epsilon=req.epsilon,
    )
    return trace


@app.post("/api/schedule/apply")
def apply_schedule(req: ScheduleRequest) -> Dict[str, Any]:
    """
    Executes the decision pipeline and binds the pod to the selected node,
    updating cluster resource consumption and incrementing history.
    """
    state_dict = state_manager.get_cluster_state_dict()
    if req.pod:
        state_dict["pending_pod"] = req.pod.model_dump()

    trace = run_decision_trace(
        cluster_state_dict=state_dict,
        use_llm=req.use_llm,
        scorer=req.scorer,
        epsilon=req.epsilon,
    )

    selected_node = trace.get("selected_node_id")
    pod_data = state_dict.get("pending_pod")

    bound = False
    if selected_node and pod_data:
        bound = state_manager.bind_pod(
            pod=pod_data,
            node_id=selected_node,
            decision_metadata={
                "scorer": trace.get("scorer_used"),
                "priority_order": trace.get("priority", {}).get("order", []),
                "latency_ms": trace.get("latency_ms", 0.0),
                "decision_reason": trace.get("decision_reason"),
            },
        )

    return {
        "decision": trace,
        "bound": bound,
        "new_cluster_state": state_manager.get_cluster_state_dict(),
    }


@app.get("/api/history")
def get_scheduling_history() -> List[Dict[str, Any]]:
    return state_manager.get_history()


@app.post("/api/history/clear")
def clear_scheduling_history() -> Dict[str, str]:
    state_manager.clear_history()
    return {"message": "History cleared successfully"}


@app.get("/api/scenarios")
def get_scenarios() -> Dict[str, Any]:
    scenarios_summary = [
        {
            "id": key,
            "name": val["name"],
            "description": val["description"],
            "node_count": len(val["nodes"]),
        }
        for key, val in SCENARIOS.items()
    ]
    return {
        "scenarios": scenarios_summary,
        "pod_templates": POD_TEMPLATES,
    }


@app.get("/api/k8s/status")
def get_k8s_status() -> Dict[str, Any]:
    return check_kubernetes_status()


# Mount frontend dist build if present
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
