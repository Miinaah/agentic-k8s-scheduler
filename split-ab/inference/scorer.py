import argparse
import json
import os
import sys
import torch
import numpy as np
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.actor import NodeActor
from models.gnn_encoder import ClusterGNNEncoder

def score_nodes(state_json: Dict[str, Any], model_dir: str) -> Dict[str, Any]:
    # Extract nodes
    nodes = state_json.get("nodes", [])
    num_nodes = len(nodes)
    
    # Check if there is a pending pod
    pending_pod = state_json.get("pending_pod")
    if not pending_pod:
        return {"scores": {n["node_id"]: 0.0 for n in nodes}}
    
    # 8 features: cpu_util, mem_util, disk_util, running_pods, health_score, cost_per_hour, zone_id_norm, recent_failures
    node_features = np.zeros((num_nodes, 8), dtype=np.float32)
    zones = []
    
    for i, n in enumerate(nodes):
        cpu_cap = max(n.get("cpu_capacity", 1.0), 1e-6)
        mem_cap = max(n.get("mem_capacity_gb", 1.0), 1e-6)
        disk_cap = max(n.get("disk_capacity_gb", 1.0), 1e-6)
        
        cpu_util = min(1.0, n.get("cpu_used", 0.0) / cpu_cap)
        mem_util = min(1.0, n.get("mem_used_gb", 0.0) / mem_cap)
        disk_util = min(1.0, n.get("disk_used_gb", 0.0) / disk_cap)
        
        running_pods = min(1.0, n.get("pod_count", 0) / 50.0)
        health_score = n.get("health_score", 1.0)
        cost_per_hour = n.get("cost_per_hour", 1.0)
        
        # simplified zone mapping
        zone = n.get("zone", "zone-a")
        if zone not in zones:
            zones.append(zone)
        zone_id = zones.index(zone)
        zone_id_norm = zone_id / max(1, 3) # default 3 zones
        
        recent_failures = n.get("recent_failures", 0.0)
        
        node_features[i] = [cpu_util, mem_util, disk_util, running_pods, health_score, cost_per_hour, zone_id_norm, recent_failures]
        
    # Pod features
    cpu_req = pending_pod.get("cpu_request", 0.1)
    mem_req = pending_pod.get("mem_request_gb", 0.1)
    disk_req = 0.05
    priority = 1.0 if pending_pod.get("priority_class") == "high" else 0.5
    qos_class = pending_pod.get("qos_class", "Guaranteed")
    ft_req = 1.0 if qos_class == "Guaranteed" else 0.0
    duration = pending_pod.get("estimated_duration_sec", 1.0)
    
    pod_feat = np.array([cpu_req, mem_req, disk_req, priority, ft_req, duration], dtype=np.float32)
    
    # Graph edges
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
        
    # Load model
    device = torch.device("cpu")
    # For inference, just load the default configs
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
    
    checkpoint_path = os.path.join(model_dir, "cluster_sage_gnn_final.pt")
    if not os.path.exists(checkpoint_path) and os.path.isdir(model_dir):
        pt_files = sorted(
            [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")],
            key=os.path.getmtime,
            reverse=True,
        )
        if pt_files:
            checkpoint_path = pt_files[0]

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        gnn_encoder.load_state_dict(state["gnn_encoder"])
        actor.load_state_dict(state["actor"])
    else:
        raise FileNotFoundError(
            f"No checkpoint found in directory '{model_dir}'. Run training first or provide a valid model path."
        )
    
    gnn_encoder.eval()
    actor.eval()
    
    with torch.no_grad():
        x = torch.tensor(node_features, dtype=torch.float32)
        pod = torch.tensor(pod_feat, dtype=torch.float32)
        
        emb = gnn_encoder(x, edge_index=edge_index)
        h0 = gnn_encoder.input_proj(x)
        cat_embeddings = torch.cat([emb, h0], dim=-1)
        actions = actor(cat_embeddings, pod).numpy()
        
    # compute suitability
    # ft, util, cost
    w_ft, w_util, w_cost = 0.35, 0.35, 0.30
    suitability = w_ft * actions[:, 0] + w_util * actions[:, 1] + w_cost * actions[:, 2]
    
    result = {"scores": {}}
    for i, n in enumerate(nodes):
        result["scores"][n["node_id"]] = float(suitability[i])
        
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True, help="Path to ClusterState JSON")
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    with open(args.state, "r") as f:
        state_json = json.load(f)
        
    res = score_nodes(state_json, args.model_dir)
    print(json.dumps(res, indent=2))
