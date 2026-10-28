# Agentic Kubernetes Scheduler Studio 🚀

An end-to-end multi-agent, reinforcement learning (GNN + MADDPG), and LLM-arbitrated Kubernetes custom scheduler with an interactive observability and simulation dashboard.

---

## 🌟 Quick Start UI

Launch the full-stack UI Studio with a single command:

```bash
# Windows Batch:
start-ui.bat

# PowerShell:
.\start-ui.ps1

# Or directly with Python:
python backend/run.py --port 8000
```

Once running, navigate to **[http://localhost:8000](http://localhost:8000)** in your browser.

> For frontend development with instant hot-reload:
> ```bash
> cd frontend
> npm run dev
> ```
> Visit `http://localhost:5173` (requests are automatically proxied to the backend at port 8000).

---

## 🏗️ Architecture Overview

The system is organized into modular layers:

```
agentic-k8s-scheduler/
├── split-ab/             # Learning Core (GNN + MADDPG Policy)
│   ├── models/           # SAGEConv/GATConv Encoder, NodeActor, Pooled Critic
│   ├── env/              # BaseClusterEnv & MockClusterEnv simulator
│   └── checkpoints/      # Trained PyTorch GNN checkpoints
│
├── split-c/              # Decision Layer Pipeline
│   └── decision_layer/   # Stress calculation, Heuristic & GNN scoring,
│                         # LLM priority reasoning, and Lexicographic filtering
│
├── split-d/              # Kubernetes Custom Scheduler Integration
│   ├── scheduler_integrated.py # Controller watching pending pods & binding via V1Binding
│   └── resource_utils.py # CPU cores and memory byte conversions
│
├── backend/              # Studio FastAPI REST Control Plane
│   ├── app.py            # Endpoints for topology, decision tracing, binding
│   ├── state_manager.py  # In-memory scenarios, dynamic node mutation, history
│   └── scheduler_bridge.py # Interconnects UI with Split C, Split AB & K8s
│
└── frontend/             # Interactive React + Vite + Tailwind Dashboard
    ├── src/components/   # ClusterTopology, PodStudio, DecisionPipeline,
    │                     # WhatIfScenarios, MetricsOverview, SchedulingHistory
    └── dist/             # Production build served directly by FastAPI
```

---

## 🎯 Key Dashboard Features

1. **Multi-Zone Cluster Topology Grid**:
   - Visual nodes organized by availability zone (`us-central1-a`, `us-central1-b`, `us-central1-c`).
   - Real-time CPU, RAM allocation bars, health score badges, hourly pricing, and active pod counters.
   - Quick fault-injection actions on every node card (*+35% CPU Load*, *Simulate Node Outage*, *Heal Node*).

2. **Pod Workload Studio**:
   - Preset enterprise templates: *Critical Database Replica*, *Production API Gateway*, *Distributed ML Worker*, *Batch Analytics ETL*, *Cost-Sensitive Web Crawler*.
   - Customizable specifications: requested CPU cores, RAM, priority class (`high`, `standard`, `low`), and QoS class (`Guaranteed`, `Burstable`, `BestEffort`).
   - Configurable engine knobs: toggle between **Heuristic Scorer** and **GNN SAGE/GAT Deep Model**, switch between **Rule-Based Mapping** and **LLM Agent Arbiter**, and adjust the lexicographic tolerance $\epsilon$.

3. **Multi-Objective Decision Pipeline Walkthrough**:
   - **Stage 1 (Stress)**: Real-time composite stress metric ($0.0 - 1.0$) with breakdown of resource load ($40\%$), failures ($35\%$), backlog ($15\%$), and variance ($10\%$).
   - **Stage 2 (Candidate Scoring)**: Multi-objective score breakdown for each node across Fault Tolerance, Utilisation, and Cost.
   - **Stage 3 (Agentic Priority Arbiter)**: Strict priority hierarchy with confidence and natural language justification quote.
   - **Stage 4 (Feasibility & Lexicographic Ranking)**: Infeasible node filtering and step-by-step tolerance elimination trace with surviving nodes and score thresholds.
   - **Stage 5 (Winner Selection & Binding)**: Highlighted winner node with decision latency in milliseconds.

4. **What-If Chaos Sandbox**:
   - One-click disaster scenarios: *Cluster-Wide CPU Surge*, *Zone Outage*, *High Contention*, and *Calm Recovery*.

5. **Placement Event Feed**:
   - Audit trail of scheduled pods, target nodes, resource allocations, and latencies.
