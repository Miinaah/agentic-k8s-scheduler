"""
Mock Cluster Environment (Split B Stand-in).

Simulates a Kubernetes-like cluster with N nodes across availability zones,
streaming pod placement requests with resource requirements, priority, and
fault tolerance requirements.

Exposes separable reward signals:
- Fault Tolerance (FT)
- Utilization (Load balancing & capacity)
- Cost (Economic efficiency)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from env.base_env import BaseClusterEnv


class MockClusterEnv(BaseClusterEnv):
    """Mock cluster simulator implementing BaseClusterEnv."""

    def __init__(
        self,
        num_nodes: int = 10,
        num_zones: int = 3,
        reward_weights: Optional[Dict[str, float]] = None,
        max_steps_per_episode: int = 50,
        tiny: bool = False,
        scenario: str = "default",
        seed: Optional[int] = None,
    ):
        """Initialize mock cluster environment.

        Args:
            num_nodes: Number of nodes in the cluster (default 10).
            num_zones: Number of availability zones (default 3).
            reward_weights: Dict with keys 'fault_tolerance', 'utilization', 'cost'.
            max_steps_per_episode: Maximum pod scheduling steps per episode.
            tiny: If True, activates 2-node no-stress toy cluster for overfit testing.
            scenario: "default", "tiny", or "overloaded_zone" for GNN ablation testing.
            seed: Optional seed for reproducibility.
        """
        self.tiny = tiny
        self.scenario = "tiny" if tiny else scenario

        if self.scenario == "tiny":
            self.num_nodes = 2
            self.num_zones = 1
        elif self.scenario == "overloaded_zone":
            self.num_nodes = 4
            self.num_zones = 2
        else:
            self.num_nodes = num_nodes
            self.num_zones = max(1, num_zones)

        if reward_weights is None:
            self.reward_weights = {
                "fault_tolerance": 0.4,
                "utilization": 0.3,
                "cost": 0.3,
            }
        else:
            self.reward_weights = reward_weights

        self.max_steps_per_episode = max_steps_per_episode
        self.step_count = 0
        self.max_pods_per_node = 50.0
        self.rng = np.random.default_rng(seed)

        # Graph topology initialization
        self.node_zones = np.zeros(self.num_nodes, dtype=np.int64)
        self.edge_index = np.empty((2, 0), dtype=np.int64)
        self.edge_attr = np.empty((0, 1), dtype=np.float32)
        self._build_graph()

        # State storage: node features (N, 12)
        self.node_features = np.zeros((self.num_nodes, 8), dtype=np.float32)
        self.current_pod: np.ndarray = np.zeros(6, dtype=np.float32)
        # Tracking active allocations to simulate container completions
        self.active_allocations: List[List[Dict[str, float]]] = [[] for _ in range(self.num_nodes)]

    def _build_graph(self):
        """Construct fully connected cluster graph with same_zone edge attribute."""
        if self.scenario == "tiny":
            self.node_zones = np.array([0, 0], dtype=np.int64)
        elif self.scenario == "overloaded_zone":
            # Nodes 0, 1 in Zone 0; Nodes 2, 3 in Zone 1
            self.node_zones = np.array([0, 0, 1, 1], dtype=np.int64)
        else:
            # Distribute nodes evenly across zones
            self.node_zones = np.array(
                [i % self.num_zones for i in range(self.num_nodes)], dtype=np.int64
            )

        src, dst = [], []
        same_zone_attrs = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    if self.scenario == "overloaded_zone":
                        # Zone-restricted edges: only connect nodes within the same zone.
                        # This is the key design: GNN aggregates zone-local context,
                        # so Node 0 (Zone 0, failing neighbor) gets a different embedding
                        # than Node 2 (Zone 1, healthy neighbor) even with identical raw features.
                        if self.node_zones[i] == self.node_zones[j]:
                            src.append(i)
                            dst.append(j)
                            same_zone_attrs.append(1.0)
                    else:
                        # Default: fully connected with same_zone attribute
                        src.append(i)
                        dst.append(j)
                        same_zone_attrs.append(1.0 if self.node_zones[i] == self.node_zones[j] else 0.0)

        if len(src) == 0:
            # Fallback for 1 node edge case
            self.edge_index = np.zeros((2, 1), dtype=np.int64)
            self.edge_attr = np.ones((1, 1), dtype=np.float32)
        else:
            self.edge_index = np.array([src, dst], dtype=np.int64)
            self.edge_attr = np.array(same_zone_attrs, dtype=np.float32).reshape(-1, 1)

    def _generate_pod_request(self) -> np.ndarray:
        """Generate incoming 6-dim pod request vector:

        [cpu_req, mem_req, disk_req, priority, ft_required, expected_duration]
        """
        if self.scenario == "tiny":
            # Tiny mode: minuscule request so neither node ever runs out of capacity
            return np.array([0.02, 0.02, 0.01, 1.0, 1.0, 0.5], dtype=np.float32)

        if self.scenario == "overloaded_zone":
            # Demands FT and balanced allocation
            return np.array([0.05, 0.05, 0.05, 1.0, 1.0, 0.5], dtype=np.float32)

        cpu_req = float(self.rng.uniform(0.05, 0.25))
        mem_req = float(self.rng.uniform(0.05, 0.25))
        disk_req = float(self.rng.uniform(0.02, 0.15))
        priority = float(self.rng.choice([0.2, 0.5, 0.8, 1.0]))
        ft_required = float(self.rng.choice([0.0, 1.0], p=[0.4, 0.6]))
        duration = float(self.rng.uniform(0.1, 1.0))
        return np.array(
            [cpu_req, mem_req, disk_req, priority, ft_required, duration],
            dtype=np.float32,
        )

    def _init_node_features(self):
        """Initialize 8-dimensional node state vector."""
        self.node_features = np.zeros((self.num_nodes, 8), dtype=np.float32)
        self.active_allocations = [[] for _ in range(self.num_nodes)]

        if self.scenario == "tiny":
            # Node 0: High utilization (0.85), low health (0.5), high cost (0.8) -> Bad choice
            # Node 1: Low utilization (0.10), high health (0.99), low cost (0.1) -> Obvious best choice
            self.node_features[0] = np.array([0.85, 0.85, 0.80, 0.8, 0.50, 0.80, 0.0, 0.20], dtype=np.float32)
            self.node_features[1] = np.array([0.10, 0.10, 0.10, 0.1, 0.99, 0.10, 0.0, 0.95], dtype=np.float32)
            return

        if self.scenario == "overloaded_zone":
            # GNN ablation scenario: Node 0 and Node 2 have IDENTICAL raw features.
            # A No-GNN agent cannot distinguish them. Only GNN message passing
            # reveals that Node 0's zone-neighbor (Node 1) is failing, while
            # Node 2's zone-neighbor (Node 3) is healthy.
            #
            # Zone 0: Node 0 (target, util=0.20) + Node 1 (failing, util=0.99, health=0.05)
            # Zone 1: Node 2 (target, util=0.20) + Node 3 (healthy, util=0.20, health=0.95)
            #
            # With zone-restricted edges, GNN sees:
            #   Node 0 embedding = f(Node0_raw, aggregate(Node1_failing))  → "bad zone"
            #   Node 2 embedding = f(Node2_raw, aggregate(Node3_healthy))  → "good zone"
            #
            # Without GNN: Node 0 and Node 2 are indistinguishable → random 50/50

            # Node 0 (Zone 0 target) — identical to Node 2
            self.node_features[0] = np.array([0.20, 0.20, 0.20, 0.2, 0.90, 0.40, 0.0, 0.70], dtype=np.float32)
            self.node_features[1] = np.array([0.99, 0.99, 0.95, 0.9, 0.05, 0.40, 0.0, 0.01], dtype=np.float32)
            self.node_features[2] = np.array([0.20, 0.20, 0.20, 0.2, 0.90, 0.40, 0.0, 0.70], dtype=np.float32)
            self.node_features[3] = np.array([0.99, 0.99, 0.95, 0.9, 0.95, 0.40, 0.0, 0.90], dtype=np.float32)
            return

        # Default randomized initialization
        for i in range(self.num_nodes):
            cpu_util = float(self.rng.uniform(0.1, 0.6))
            mem_util = float(self.rng.uniform(0.1, 0.6))
            disk_util = float(self.rng.uniform(0.1, 0.5))
            running_pods = float(self.rng.integers(1, 20) / self.max_pods_per_node)
            health_score = float(self.rng.uniform(0.8, 1.0))
            cost_per_hour = float(self.rng.uniform(0.2, 0.9))
            zone_id_norm = float(self.node_zones[i] / max(1, self.num_zones))
            recent_failures = float(self.rng.uniform(0.0, 0.5))

            self.node_features[i] = np.array(
                [cpu_util, mem_util, disk_util, running_pods, health_score, cost_per_hour, zone_id_norm, recent_failures],
                dtype=np.float32,
            )

    def _get_obs(self) -> Dict[str, Any]:
        """Construct observation dictionary."""
        return {
            "x": self.node_features.copy(),
            "edge_index": self.edge_index.copy(),
            "edge_attr": self.edge_attr.copy(),
            "pod_features": self.current_pod.copy(),
            "num_nodes": self.num_nodes,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._init_node_features()
        self.current_pod = self._generate_pod_request()
        return self._get_obs(), {"step": 0}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take scheduling decision actions for all nodes.

        actions: (N, 3) matrix containing [s_ft, s_util, s_cost] suitability scores.
        """
        self.step_count += 1
        actions = np.asarray(actions, dtype=np.float32)

        # Decay finished pods slightly to simulate job progression
        if self.scenario not in ["tiny", "overloaded_zone"]:
            self._progress_allocations()

        w_ft = self.reward_weights["fault_tolerance"]
        w_util = self.reward_weights["utilization"]
        w_cost = self.reward_weights["cost"]

        cpu_req = self.current_pod[0]
        mem_req = self.current_pod[1]
        disk_req = self.current_pod[2]
        ft_required = self.current_pod[4]

        # Feasibility check per node
        feasible_mask = np.ones(self.num_nodes, dtype=bool)
        for i in range(self.num_nodes):
            if (
                self.node_features[i, 0] + cpu_req > 1.0
                or self.node_features[i, 1] + mem_req > 1.0
                or self.node_features[i, 2] + disk_req > 1.0
            ):
                feasible_mask[i] = False

        # Compute suitability scores: weighted sum of the 3 action dimensions
        suitability_scores = (
            w_ft * actions[:, 0] + w_util * actions[:, 1] + w_cost * actions[:, 2]
        )

        selected_node: Optional[int] = None
        is_feasible = False

        feasible_indices = np.where(feasible_mask)[0]
        if len(feasible_indices) > 0:
            best_idx = np.argmax(suitability_scores[feasible_indices])
            selected_node = int(feasible_indices[best_idx])
            is_feasible = True
        else:
            # Infeasible: fallback to highest score anyway, marked infeasible
            selected_node = int(np.argmax(suitability_scores))
            is_feasible = False

        # Failure injection with dynamic health and recent_failures
        if self.scenario not in ["tiny", "overloaded_zone"]:
            # Randomly degrade a node
            if self.rng.random() < 0.05:
                deg_node = self.rng.integers(0, self.num_nodes)
                self.node_features[deg_node, 4] = max(0.0, self.node_features[deg_node, 4] - 0.2)
                self.node_features[deg_node, 7] = min(1.0, self.node_features[deg_node, 7] + 0.1)
                # Zone-correlated risk
                zone_mates = np.where(self.node_zones == self.node_zones[deg_node])[0]
                for zm in zone_mates:
                    if zm != deg_node:
                        self.node_features[zm, 7] = min(1.0, self.node_features[zm, 7] + 0.05)
                        self.node_features[zm, 4] = max(0.0, self.node_features[zm, 4] - 0.05)

        # Compute signals for all nodes
        node_signals = np.zeros((self.num_nodes, 3), dtype=np.float32)
        for i in range(self.num_nodes):
            node_health = self.node_features[i, 4]
            recent_fails = self.node_features[i, 7]
            same_zone_mask = self.node_zones == self.node_zones[i]
            zone_avg_health = float(np.mean(self.node_features[same_zone_mask, 4]))
            
            # FT Signal
            if self.scenario == "tiny":
                ft_i = 1.0 if i == 1 else -0.5
            elif self.scenario == "overloaded_zone":
                ft_i = -0.8 if self.node_zones[i] == 0 else 1.0
            else:
                base_ft = 0.5 * node_health - 0.3 * recent_fails + 0.2 * zone_avg_health
                if ft_required > 0.5 and (node_health < 0.6 or zone_avg_health < 0.6):
                    base_ft -= 0.5
                ft_i = float(np.clip(base_ft * 2.0 - 1.0, -1.0, 1.0))
                
            # Util Signal
            if self.scenario == "tiny":
                util_i = 1.0 if i == 1 else -0.5
            elif self.scenario == "overloaded_zone":
                util_i = 0.9 if self.node_zones[i] == 1 else -0.4
            else:
                # Post-placement balance
                sim_cpu = self.node_features[:, 0].copy()
                sim_cpu[i] += cpu_req
                mean_cpu = np.mean(sim_cpu)
                std_cpu = np.std(sim_cpu)
                util_i = float(np.clip(1.0 - 2.0 * (mean_cpu + std_cpu), -1.0, 1.0))
                
            # Cost Signal
            cost_per_hour = self.node_features[i, 5]
            if self.scenario == "tiny":
                cost_i = 1.0 if i == 1 else -0.5
            elif self.scenario == "overloaded_zone":
                cost_i = 0.5
            else:
                cost_norm = (cost_per_hour - 0.2) / 0.7
                cost_i = float(np.clip(1.0 - 2.0 * cost_norm, -1.0, 1.0))
                
            node_signals[i] = [ft_i, util_i, cost_i]

        if not is_feasible:
            ft_signal, util_signal, cost_signal = -1.0, -1.0, -1.0
        else:
            sel = selected_node
            ft_signal, util_signal, cost_signal = node_signals[sel]
            
            if self.scenario not in ["tiny", "overloaded_zone"]:
                self.node_features[sel, 0] = min(1.0, self.node_features[sel, 0] + cpu_req)
                self.node_features[sel, 1] = min(1.0, self.node_features[sel, 1] + mem_req)
                self.node_features[sel, 2] = min(1.0, self.node_features[sel, 2] + disk_req)
                self.node_features[sel, 3] = min(1.0, self.node_features[sel, 3] + 1.0 / self.max_pods_per_node)
                self.active_allocations[sel].append({"cpu": cpu_req, "mem": mem_req, "disk": disk_req, "ttl": 10})

        total_reward = float(
            w_ft * ft_signal + w_util * util_signal + w_cost * cost_signal
        )

        done = bool(self.step_count >= self.max_steps_per_episode)
        self.current_pod = self._generate_pod_request()
        next_obs = self._get_obs()

        info = {
            "fault_tolerance_signal": float(ft_signal),
            "utilization_signal": float(util_signal),
            "cost_signal": float(cost_signal),
            "selected_node": selected_node,
            "feasible": is_feasible,
            "node_signals": node_signals,
        }

        return next_obs, total_reward, done, info

    def _progress_allocations(self):
        """Decrement pod allocation timers and release resources when pods complete."""
        for i in range(self.num_nodes):
            remaining = []
            for alloc in self.active_allocations[i]:
                alloc["ttl"] -= 1
                if alloc["ttl"] <= 0:
                    # Release resources
                    self.node_features[i, 0] = max(0.05, self.node_features[i, 0] - alloc["cpu"])
                    self.node_features[i, 1] = max(0.05, self.node_features[i, 1] - alloc["mem"])
                    self.node_features[i, 2] = max(0.05, self.node_features[i, 2] - alloc["disk"])
                    self.node_features[i, 3] = max(0.0, self.node_features[i, 3] - 1.0 / self.max_pods_per_node)
                else:
                    remaining.append(alloc)
            self.active_allocations[i] = remaining

    def observation_spec(self) -> Dict[str, Any]:
        return {
            "x": {"shape": (self.num_nodes, 8), "dtype": np.float32},
            "edge_index": {"shape": (2, self.edge_index.shape[1]), "dtype": np.int64},
            "edge_attr": {"shape": (self.edge_index.shape[1], 1), "dtype": np.float32},
            "pod_features": {"shape": (6,), "dtype": np.float32},
            "num_nodes": self.num_nodes,
        }

    def action_spec(self) -> Dict[str, Any]:
        return {
            "actions": {"shape": (self.num_nodes, 3), "dtype": np.float32, "range": [0.0, 1.0]}
        }
