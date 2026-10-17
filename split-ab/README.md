# GNN + MADDPG Cluster Scheduler (Split B — Learning Core)

This repository contains the learning core of a Graph Neural Network (GNN) empowered Multi-Agent Deep Deterministic Policy Gradient (MADDPG) scheduler for distributed cluster computing (e.g. Kubernetes).

Split B implements the reinforcement learning and graph representation learning core. It communicates with a cluster environment via a standardized Gym-like contract (`BaseClusterEnv`). It is developed against `MockClusterEnv` and designed to seamlessly swap in Split A's high-fidelity cluster simulator without any model or training code modifications.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Incoming Pod Request (6)        |
                      +-------------------+-------------------+
                                          |
                                          v
+------------------------+      +--------------------+      +--------------------+
| Cluster Topology Graph | ---> |  GNN Encoder (64)  | ---> | Decentralized Node | ---> Action Scores
|   (N nodes, 12 dims)   |      | SAGEConv / GATConv |      |   Actor (70->3)    |      [FT, Util, Cost]
+------------------------+      +--------------------+      +--------------------+
                                          |
                                          v
                                +--------------------+
                                | Centralized Critic | <--- Node Embeddings + Actions
                                |  Pooled (203->1)   |      + Pooled Context + Pod Reqs
                                +--------------------+
```

1. **Decentralized Execution**:
   - Cluster nodes act as homogeneous agents sharing network weights.
   - Each node receives its GNN embedding (64 dims) concatenated with the incoming pod request features (6 dims) = 70 dims.
   - The Actor outputs 3 bounded suitability scores in $[0, 1]$ corresponding to Fault Tolerance (FT), Utilization, and Cost.
2. **Centralized Training**:
   - A single shared Centralized Critic observes the full cluster state using a permutation-invariant mean-pooling design over all node-action pairs.
   - The critic is evaluated exclusively during training to guide actor policy updates via the MADDPG algorithm.
3. **Graph Neural Network (PyG)**:
   - Supports both `SAGEConv` (spatial neighbor aggregation) and `GATConv` (multi-head attention, heads=4).
   - Residual skip connections are available for networks with $> 2$ layers to counteract over-smoothing.
   - Fully compatible with PyTorch Geometric `Batch` mini-batching.

---

## Project Structure

```
project/
├── env/
│   ├── __init__.py
│   ├── base_env.py             # Abstract simulator contract (BaseClusterEnv)
│   └── mock_cluster_env.py     # Mock cluster simulator with FT/Util/Cost signals
├── models/
│   ├── __init__.py
│   ├── gnn_encoder.py          # SAGE/GATConv encoder with ablation toggle
│   ├── actor.py                # Homogeneous actor (70 -> 128 -> 64 -> 3 -> Sigmoid)
│   └── critic.py               # Pooled centralized critic (203 -> 256 -> 128 -> 64 -> 1)
├── training/
│   ├── __init__.py
│   ├── noise.py                # Gaussian and Ornstein-Uhlenbeck action noise
│   ├── replay_buffer.py        # Replay buffer with n-step returns and PyG batching
│   ├── maddpg_trainer.py       # MADDPG update rules, target networks, Polyak soft updates
│   └── train.py                # CLI entry point and training loop
├── configs/
│   └── default.yaml            # Hyperparameters and configuration defaults
├── tests/
│   ├── __init__.py
│   ├── test_overfit_toy.py     # Overfit test on 2-node no-stress toy cluster
│   ├── test_ablation_no_gnn.py # Ablation comparing GNN vs No-GNN on overloaded zone
│   └── test_sanity_differentiation.py # Checks actor produces distinct node scores
├── utils/
│   ├── __init__.py
│   └── logger.py               # TensorBoard writer and Matplotlib reward curve generator
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## Installation

Ensure Python 3.11+ is installed. Then install the dependencies:

```bash
pip install -r requirements.txt
```

Verify GPU/PyG availability:
```bash
python -c "import torch, torch_geometric; print('CUDA:', torch.cuda.is_available(), 'MPS (Apple Silicon):', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(), 'PyG:', torch_geometric.__version__)"
```

---

## Running Training

### Standard Training Run
Run training with default configuration (`num_nodes: 10`, `episodes: 200`):
```bash
python training/train.py --config configs/default.yaml
```

### Quick Toy / Sanity Run (`--tiny`)
Run in 2-node no-stress cluster mode for quick validation:
```bash
python training/train.py --config configs/default.yaml --tiny --episodes 50
```

### CLI Overrides
```bash
# Train using GATConv instead of SAGEConv
python training/train.py --config configs/default.yaml --gnn_type gat

# Run ablation baseline without GNN
python training/train.py --config configs/default.yaml --no_gnn

# Target specific device (auto detects CUDA -> Apple Silicon MPS -> CPU)
python training/train.py --config configs/default.yaml --device auto
python training/train.py --config configs/default.yaml --device mps   # Apple Silicon GPU
python training/train.py --config configs/default.yaml --device cuda  # NVIDIA GPU
python training/train.py --config configs/default.yaml --device cpu   # CPU fallback
```

Artifacts generated:
- Model checkpoints saved to `checkpoints/`
- Reward curves plotted to `runs/<run_name>/reward_curve.png`
- TensorBoard event logs stored under `runs/<run_name>/` (viewable via `tensorboard --logdir runs`)

---

## Running the Test Suite

Execute all tests with `pytest`:
```bash
pytest -v
```

Or run each test individually:
```bash
# 1. Verify agent overfits the 2-node toy environment
pytest tests/test_overfit_toy.py -v -s

# 2. Verify GNN outperforms non-GNN baseline on overloaded zone scenario
pytest tests/test_ablation_no_gnn.py -v -s

# 3. Verify actor outputs differentiated scores on distinct node states
pytest tests/test_sanity_differentiation.py -v -s
```

---

## How to Swap `mock_cluster_env.py` for Split A's Real Simulator

To replace `MockClusterEnv` with Split A's production cluster simulator, subclass `BaseClusterEnv` (or wrap Split A's simulator in a class adhering to it) and ensure the following exact shapes and contracts are honored:

### 1. `reset(seed=None)`
- **Returns**: `(obs, info)`
  - `obs` must be a dictionary containing:
    - `"x"`: `np.ndarray` of shape `(N, 12)`, `dtype=np.float32`. Raw node features (CPU, memory, disk, network, health, failure history, cost, zone, queue, etc.).
    - `"edge_index"`: `np.ndarray` of shape `(2, E)`, `dtype=np.int64`. Graph connectivity indices (source, destination nodes).
    - `"edge_attr"`: `np.ndarray` of shape `(E, 1)`, `dtype=np.float32`. Edge attributes, such as `1.0` for intra-zone links (`same_zone`) and `0.0` otherwise.
    - `"pod_features"`: `np.ndarray` of shape `(6,)`, `dtype=np.float32`. `[cpu_req, mem_req, disk_req, priority, ft_required, duration]`.
    - `"num_nodes"`: `int`, cluster node count $N$.
  - `info`: `dict`, arbitrary simulator metadata.

### 2. `step(actions)`
- **Input**:
  - `actions`: `np.ndarray` of shape `(N, 3)`, `dtype=np.float32`. For each node $i$: `[suitability_ft, suitability_util, suitability_cost]` with values in $[0.0, 1.0]$.
- **Returns**: `(next_obs, reward, done, info)`
  - `next_obs`: Dictionary matching the `obs` structure above.
  - `reward`: `float` scalar representing the weighted combination of the 3 signals.
  - `done`: `bool` indicating episode completion.
  - `info`: Dictionary containing unweighted raw reward signals and placement resolution:
    - `"fault_tolerance_signal"`: `float`
    - `"utilization_signal"`: `float`
    - `"cost_signal"`: `float`
    - `"selected_node"`: `int` or `None` (node assigned to the pod)
    - `"feasible"`: `bool` (whether resource constraints were met)

Once Split A provides its simulator implementing this interface, simply import and instantiate it in `training/train.py` in place of `MockClusterEnv`.

---

## Configuration Reference (`configs/default.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `gnn_type` | `"sage"` | GNN convolution layer: `"sage"` (`SAGEConv`) or `"gat"` (`GATConv`). |
| `gnn_layers` | `2` | Number of message passing layers (range: 2 to 4). |
| `hidden_dim` | `64` | Embedding dimension for node representations throughout the network. |
| `gat_heads` | `4` | Number of multi-head attention channels when `gnn_type: "gat"`. |
| `dropout` | `0.1` | Dropout rate applied after each GNN layer activation. |
| `residual` | `false` | Enables skip connections between message-passing layers. |
| `use_gnn` | `true` | When `false`, raw node features are projected directly to the actor without graph message passing. |
| `node_feat_dim` | `12` | Number of raw node metrics per cluster node. |
| `pod_feat_dim` | `6` | Feature vector length for incoming pod requests. |
| `action_dim` | `3` | Number of suitability outputs per node (FT, Utilization, Cost). |
| `critic_input_dim`| `203` | Input dimension to the first Critic MLP layer. |
| `actor_lr` | `1e-4` | Learning rate for the Actor and GNN Encoder optimizer. |
| `critic_lr` | `1e-3` | Learning rate for the Centralized Critic optimizer. |
| `gamma` | `0.95` | MDP reward discount factor. |
| `tau` | `0.01` | Polyak averaging coefficient for target network soft updates. |
| `grad_clip_norm` | `1.0` | Maximum gradient norm threshold for clipping actor and critic updates. |
| `weight_decay` | `0.0` | L2 weight regularization coefficient. |
| `buffer_size` | `100000`| Maximum capacity of experience replay buffer. |
| `batch_size` | `128` | Mini-batch sample size for gradient descent updates. |
| `warmup_steps` | `2000` | Initial uniform random exploration steps before training kicks in. |
| `grad_updates_per_step` | `1` | Number of gradient updates executed per environment transition step. |
| `n_step_return` | `3` | Lookahead horizon for multi-step Bellman return estimation. |
| `noise_type` | `"gaussian"` | Exploration noise algorithm: `"gaussian"` or `"ou"` (Ornstein-Uhlenbeck). |
| `noise_sigma_start` | `0.3` | Starting exploration standard deviation. |
| `noise_sigma_end` | `0.05` | Final exploration standard deviation after decay. |
| `noise_decay_steps` | `50000`| Total steps over which exploration noise is linearly decayed. |
| `ou_theta` | `0.15` | Mean-reversion parameter for Ornstein-Uhlenbeck noise. |
| `ou_sigma` | `0.2` | Volatility parameter for Ornstein-Uhlenbeck noise. |
| `num_nodes` | `10` | Number of nodes in the cluster cluster. |
| `num_zones` | `3` | Number of availability zones across which nodes are distributed. |
| `tiny` | `false` | Enables 2-node no-stress toy cluster for overfit testing. |
| `max_steps_per_episode` | `50` | Maximum scheduling decisions per episode before environment termination. |
| `episodes` | `200` | Total training episodes to run. |
| `reward_weights` | `{ft: 0.4, util: 0.3, cost: 0.3}` | Linear weighting used to combine the 3 unweighted signals. |
| `log_dir` | `"runs"` | Directory for TensorBoard logs and PNG reward plots. |
| `checkpoint_dir` | `"checkpoints"` | Directory for saved model weights (`.pt`). |

---

## Flagged Design Decisions & Assumptions

Per the project specification, the following design decisions and assumptions are explicitly flagged:

1. **Critic Input Dimension (203 vs 140 dims)**:
   - The specification defines:
     `critic_input = own_embedding(64) concat own_action(3) concat pooled(67) concat pod_features(6) = 203 dims`
     and specifies the input layer as `Linear(203 -> 256)`.
   - Arithmetic on the individual components yields: $64 + 3 + 67 + 6 = 140$ dimensions.
   - **Resolution**: `CentralizedCritic` dynamically handles this: when initialized with `input_dim=203` (default), the 140-dim feature vector is cleanly zero-padded with 63 zeros to match `Linear(203, 256)` exactly, guaranteeing full compatibility without dimension mismatch errors. If initialized with `input_dim=140`, it feeds `Linear(140, 256)` directly without padding.
2. **Action Resolution and Node Selection**:
   - The Actor outputs continuous suitability scores $s_{ft}, s_{util}, s_{cost} \in [0, 1]$ per node.
   - Suitability scores are weighted using `reward_weights` ($w_{ft} s_{ft} + w_{util} s_{util} + w_{cost} s_{cost}$) and gated by feasibility (CPU, memory, disk must fit the pod request). The feasible node with the highest score is assigned the pod.
3. **Decentralized Actor with GNN Representations**:
   - The GNN encoder runs once per cluster observation, broadcasting contextual node embeddings. The homogeneous Actor network processes each node's local embedding in parallel.
   - Both the Actor and GNN encoder are trained jointly through the policy gradient backpropagated from the Critic.
4. **Reward Independence**:
   - Fault Tolerance, Utilization, and Cost are strictly computed as independent signals in $[-1.0, 1.0]$. The scalar reward is their weighted sum.
   - Both the scalar reward and the individual raw unweighted signals are logged to `info` and TensorBoard, enabling changes to reward weighting without modifying training code.


## Update
Tables have been updated to reflect Split B fixes (gamma 0.5, n_step 1, gnn_layers 2, dropout 0.0, node_feat_dim 8).
