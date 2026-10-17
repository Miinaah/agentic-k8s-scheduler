import numpy as np
from typing import Dict, Any, Tuple, Optional
from env.cluster_env import ClusterEnv, NodeStatus, QoSClass
from env.mock_cluster_env import MockClusterEnv

class WrappedClusterEnv:
    def __init__(self, num_nodes=10, num_zones=3, reward_weights=None, max_steps_per_episode=50, tiny=False, scenario="default", seed=None):
        self.num_nodes = 2 if tiny else num_nodes
        self.num_zones = 1 if tiny else num_zones
        self.max_steps_per_episode = max_steps_per_episode
        self.scenario = "tiny" if tiny else scenario
        self.env = ClusterEnv(num_nodes=self.num_nodes, max_steps=max_steps_per_episode, seed=seed)
        self.reward_weights = reward_weights or {"fault_tolerance": 0.35, "utilization": 0.35, "cost": 0.3}
        
        # Build graph
        self.node_zones = np.array([i % self.num_zones for i in range(self.num_nodes)], dtype=np.int64)
        src, dst = [], []
        same_zone_attrs = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
                    same_zone_attrs.append(1.0 if self.node_zones[i] == self.node_zones[j] else 0.0)
        
        if len(src) == 0:
            self.edge_index = np.zeros((2, 1), dtype=np.int64)
            self.edge_attr = np.ones((1, 1), dtype=np.float32)
        else:
            self.edge_index = np.array([src, dst], dtype=np.int64)
            self.edge_attr = np.array(same_zone_attrs, dtype=np.float32).reshape(-1, 1)

    def _get_obs(self):
        nodes = list(self.env.nodes.values())
        node_features = np.zeros((self.num_nodes, 8), dtype=np.float32)
        for i, n in enumerate(nodes):
            running_pods = min(1.0, len(n.hosted_pods) / 50.0)
            zone_id_norm = self.node_zones[i] / max(1, self.num_zones)
            node_features[i] = [
                n.utilization("cpu"),
                n.utilization("memory"),
                n.utilization("disk"),
                running_pods,
                n.health_score,
                n.cost_per_hour,
                zone_id_norm,
                n.recent_failures
            ]
            
        pod_features = np.zeros(6, dtype=np.float32)
        if self.env.pending_queue:
            p = self.env.pending_queue[0]
            priority = 1.0 if p.priority_class == "high" else 0.5
            ft_req = 1.0 if p.qos_class == QoSClass.GUARANTEED else 0.0
            pod_features = np.array([p.cpu_request, p.memory_request, 0.05, priority, ft_req, p.duration], dtype=np.float32)
            
        return {
            "x": node_features,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "pod_features": pod_features,
            "num_nodes": self.num_nodes
        }

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        return self._get_obs(), {"step": 0}

    def step(self, actions: np.ndarray):
        w_ft, w_util, w_cost = self.reward_weights["fault_tolerance"], self.reward_weights["utilization"], self.reward_weights["cost"]
        suitability_scores = w_ft * actions[:, 0] + w_util * actions[:, 1] + w_cost * actions[:, 2]
        
        # Determine feasibility based on the actual node objects
        feasible_mask = np.zeros(self.num_nodes, dtype=bool)
        if self.env.pending_queue:
            p = self.env.pending_queue[0]
            nodes = list(self.env.nodes.values())
            for i, n in enumerate(nodes):
                if n.can_fit(p.cpu_request, p.memory_request):
                    feasible_mask[i] = True
                    
        selected_node = None
        is_feasible = False
        feasible_indices = np.where(feasible_mask)[0]
        if len(feasible_indices) > 0:
            best_idx = np.argmax(suitability_scores[feasible_indices])
            selected_node = int(feasible_indices[best_idx])
            is_feasible = True
        elif self.env.pending_queue:
            selected_node = int(np.argmax(suitability_scores))
            
        # Formulate PettingZoo actions for ClusterEnv
        env_actions = {}
        for i, agent in enumerate(self.env.agents):
            if is_feasible and i == selected_node:
                env_actions[agent] = 1 # bid on the first pod in pending_queue
            else:
                env_actions[agent] = 0 # pass
                
        # To compute node_signals, we just use the MockClusterEnv logic but adapted
        node_signals = np.zeros((self.num_nodes, 3), dtype=np.float32)
        nodes = list(self.env.nodes.values())
        for i, n in enumerate(nodes):
            node_health = n.health_score
            recent_fails = n.recent_failures
            same_zone_mask = self.node_zones == self.node_zones[i]
            zone_avg_health = float(np.mean([nodes[j].health_score for j in range(self.num_nodes) if same_zone_mask[j]]))
            
            ft_required = 1.0 if self.env.pending_queue and self.env.pending_queue[0].qos_class == QoSClass.GUARANTEED else 0.0
            
            base_ft = 0.5 * node_health - 0.3 * recent_fails + 0.2 * zone_avg_health
            if ft_required > 0.5 and (node_health < 0.6 or zone_avg_health < 0.6):
                base_ft -= 0.5
            ft_i = float(np.clip(base_ft * 2.0 - 1.0, -1.0, 1.0))
            
            sim_cpu = np.array([nodes[j].utilization("cpu") for j in range(self.num_nodes)])
            if self.env.pending_queue:
                sim_cpu[i] = min(1.0, sim_cpu[i] + self.env.pending_queue[0].cpu_request)
            util_i = float(np.clip(1.0 - 2.0 * (np.mean(sim_cpu) + np.std(sim_cpu)), -1.0, 1.0))
            
            min_cost, max_cost = getattr(self.env, "cost_range", (0.5, 2.0))
            cost_norm = (n.cost_per_hour - min_cost) / max(1e-6, max_cost - min_cost)
            cost_i = float(np.clip(1.0 - 2.0 * cost_norm, -1.0, 1.0))
            node_signals[i] = [ft_i, util_i, cost_i]
            
        if not is_feasible:
            ft_signal, util_signal, cost_signal = -1.0, -1.0, -1.0
        else:
            ft_signal, util_signal, cost_signal = node_signals[selected_node]

        obs_dict, rewards_dict, term_dict, trunc_dict, infos_dict = self.env.step(env_actions)
        
        # Calculate total reward from Centralized scheduler point of view
        # We can just sum the rewards from all nodes (ClusterEnv gives shaped reward)
        total_reward = sum(rewards_dict.values()) if rewards_dict else 0.0
        done = any(term_dict.values()) if term_dict else True
        
        info = {
            "fault_tolerance_signal": float(ft_signal),
            "utilization_signal": float(util_signal),
            "cost_signal": float(cost_signal),
            "selected_node": selected_node,
            "feasible": is_feasible,
            "node_signals": node_signals,
        }
        
        return self._get_obs(), total_reward, done, info
