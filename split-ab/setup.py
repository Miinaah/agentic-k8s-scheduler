import os
import re

# 1. configs/default.yaml
yaml_path = 'configs/default.yaml'
with open(yaml_path, 'r') as f:
    config = f.read()
config = re.sub(r'gamma:\s*0\.95', 'gamma: 0.5', config)
config = re.sub(r'n_step_return:\s*3', 'n_step_return: 1\nselection_temp: 0.1', config)
config = re.sub(r'gnn_layers:\s*3', 'gnn_layers: 2', config)
config = re.sub(r'dropout:\s*0\.05', 'dropout: 0.0', config)
config = re.sub(r'node_feat_dim:\s*12', 'node_feat_dim: 8', config)
with open(yaml_path, 'w') as f:
    f.write(config)

# 2. env/mock_cluster_env.py
env_path = 'env/mock_cluster_env.py'
with open(env_path, 'r') as f:
    env_code = f.read()

# Replace node_features dim
env_code = re.sub(r'self\.node_features = np\.zeros\(\(self\.num_nodes, 12\), dtype=np\.float32\)', 'self.node_features = np.zeros((self.num_nodes, 8), dtype=np.float32)', env_code)
env_code = re.sub(r'\(self\.num_nodes, 12\)', '(self.num_nodes, 8)', env_code)
env_code = env_code.replace('12-dimensional node state vector', '8-dimensional node state vector')

# Fix initialization of node features
init_old = """            self.node_features[0] = np.array(
                [0.85, 0.85, 0.80, 0.70, 0.8, 0.50, 0.20, 0.80, 0.0, 0.7, 0.5, 0.8],
                dtype=np.float32,
            )
            self.node_features[1] = np.array(
                [0.10, 0.10, 0.10, 0.10, 0.1, 0.99, 0.95, 0.10, 0.0, 0.0, 0.1, 0.1],
                dtype=np.float32,
            )"""
init_new = """            self.node_features[0] = np.array([0.85, 0.85, 0.80, 0.8, 0.50, 0.80, 0.0, 0.20], dtype=np.float32)
            self.node_features[1] = np.array([0.10, 0.10, 0.10, 0.1, 0.99, 0.10, 0.0, 0.95], dtype=np.float32)"""
env_code = env_code.replace(init_old, init_new)

init_old2 = """            self.node_features[0] = np.array(
                [0.20, 0.20, 0.20, 0.20, 0.2, 0.90, 0.70, 0.40, 0.0, 0.2, 0.2, 0.3],
                dtype=np.float32,
            )
            # Node 1 (Zone 0 neighbor) — catastrophically failing
            self.node_features[1] = np.array(
                [0.99, 0.99, 0.95, 0.95, 0.9, 0.05, 0.01, 0.40, 0.0, 0.9, 0.9, 0.9],
                dtype=np.float32,
            )
            # Node 2 (Zone 1 target) — IDENTICAL raw features to Node 0
            self.node_features[2] = np.array(
                [0.20, 0.20, 0.20, 0.20, 0.2, 0.90, 0.70, 0.40, 0.0, 0.2, 0.2, 0.3],
                dtype=np.float32,
            )
            # Node 3 (Zone 1 neighbor) — healthy and balanced
            self.node_features[3] = np.array(
                [0.20, 0.20, 0.20, 0.20, 0.2, 0.95, 0.90, 0.40, 0.5, 0.1, 0.2, 0.2],
                dtype=np.float32,
            )"""
init_new2 = """            self.node_features[0] = np.array([0.20, 0.20, 0.20, 0.2, 0.90, 0.40, 0.0, 0.70], dtype=np.float32)
            self.node_features[1] = np.array([0.99, 0.99, 0.95, 0.9, 0.05, 0.40, 0.0, 0.01], dtype=np.float32)
            self.node_features[2] = np.array([0.20, 0.20, 0.20, 0.2, 0.90, 0.40, 0.5, 0.70], dtype=np.float32)
            self.node_features[3] = np.array([0.20, 0.20, 0.20, 0.2, 0.95, 0.40, 0.5, 0.90], dtype=np.float32)"""
env_code = env_code.replace(init_old2, init_new2)

init_old3 = """            cpu_util = float(self.rng.uniform(0.1, 0.6))
            mem_util = float(self.rng.uniform(0.1, 0.6))
            disk_util = float(self.rng.uniform(0.1, 0.5))
            net_util = float(self.rng.uniform(0.05, 0.4))
            running_pods = float(self.rng.integers(1, 20) / self.max_pods_per_node)
            health_score = float(self.rng.uniform(0.8, 1.0))
            time_since_failure = float(np.log1p(self.rng.uniform(10, 720)) / np.log1p(720))
            cost_per_hour = float(self.rng.uniform(0.2, 0.9))
            zone_id_norm = float(self.node_zones[i] / max(1, self.num_zones))
            queue_len = float(self.rng.uniform(0.0, 0.3))
            io_util = float(self.rng.uniform(0.1, 0.4))
            thermal_score = float(self.rng.uniform(0.2, 0.5))

            self.node_features[i] = np.array(
                [
                    cpu_util,
                    mem_util,
                    disk_util,
                    net_util,
                    running_pods,
                    health_score,
                    time_since_failure,
                    cost_per_hour,
                    zone_id_norm,
                    queue_len,
                    io_util,
                    thermal_score,
                ],
                dtype=np.float32,
            )"""
init_new3 = """            cpu_util = float(self.rng.uniform(0.1, 0.6))
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
            )"""
env_code = env_code.replace(init_old3, init_new3)

step_replace_old = """        # Compute the 3 independent reward signals
        if not is_feasible:
            ft_signal = -1.0
            util_signal = -1.0
            cost_signal = -1.0
        else:
            sel = selected_node
            # 1. Fault Tolerance Signal
            node_health = self.node_features[sel, 5]
            time_since_fail = self.node_features[sel, 6]

            # Zone health context (average health in the chosen node's zone)
            same_zone_mask = self.node_zones == self.node_zones[sel]
            zone_avg_health = float(np.mean(self.node_features[same_zone_mask, 5]))
            zone_avg_util = float(np.mean(self.node_features[same_zone_mask, 0]))

            if self.scenario == "tiny":
                # Tiny mode: Node 1 has 0.99 health, Node 0 has 0.5 health
                ft_signal = 1.0 if sel == 1 else -0.5
            elif self.scenario == "overloaded_zone":
                # If chosen node is in Zone 0 (overloaded/failing), penalize heavily
                if self.node_zones[sel] == 0:
                    ft_signal = -0.8
                else:
                    ft_signal = 1.0
            else:
                base_ft = 0.5 * node_health + 0.3 * time_since_fail + 0.2 * zone_avg_health
                if ft_required > 0.5 and (node_health < 0.6 or zone_avg_health < 0.6):
                    base_ft -= 0.5
                ft_signal = float(np.clip(base_ft * 2.0 - 1.0, -1.0, 1.0))

            # 2. Utilization Signal
            curr_cpu_util = self.node_features[sel, 0]
            curr_mem_util = self.node_features[sel, 1]
            if self.scenario == "tiny":
                # Node 1 has 0.1 util (vast free capacity), Node 0 has 0.85 util
                util_signal = 1.0 if sel == 1 else -0.5
            elif self.scenario == "overloaded_zone":
                # Zone 1 is balanced, Zone 0 has severe imbalance
                util_signal = 0.9 if self.node_zones[sel] == 1 else -0.4
            else:
                # Reward choosing nodes with ample headroom and avoid high utilization bottlenecks
                avg_util = 0.5 * (curr_cpu_util + curr_mem_util)
                util_signal = float(np.clip(1.0 - 2.0 * avg_util, -1.0, 1.0))

            # 3. Cost Signal
            cost_per_hour = self.node_features[sel, 7]
            if self.scenario == "tiny":
                cost_signal = 1.0 if sel == 1 else -0.5
            elif self.scenario == "overloaded_zone":
                cost_signal = 0.5
            else:
                # Cheaper node yields higher cost signal
                cost_signal = float(np.clip(1.0 - 2.0 * cost_per_hour, -1.0, 1.0))

            # Update selected node resource state
            if self.scenario not in ["tiny", "overloaded_zone"]:
                self.node_features[sel, 0] = min(1.0, self.node_features[sel, 0] + cpu_req)
                self.node_features[sel, 1] = min(1.0, self.node_features[sel, 1] + mem_req)
                self.node_features[sel, 2] = min(1.0, self.node_features[sel, 2] + disk_req)
                self.node_features[sel, 4] = min(
                    1.0, self.node_features[sel, 4] + 1.0 / self.max_pods_per_node
                )
                self.active_allocations[sel].append(
                    {"cpu": cpu_req, "mem": mem_req, "disk": disk_req, "ttl": 10}
                )"""

step_replace_new = """        # Failure injection with dynamic health and recent_failures
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
                cost_i = float(np.clip(1.0 - 2.0 * cost_per_hour, -1.0, 1.0))
                
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
                self.active_allocations[sel].append({"cpu": cpu_req, "mem": mem_req, "disk": disk_req, "ttl": 10})"""

env_code = env_code.replace(step_replace_old, step_replace_new)

env_code = re.sub(r'self\.node_features\[i, 4\] = max\(\n?\s*0\.0,\s*self\.node_features\[i, 4\] - 1\.0 / self\.max_pods_per_node\n?\s*\)', 'self.node_features[i, 3] = max(0.0, self.node_features[i, 3] - 1.0 / self.max_pods_per_node)', env_code)

info_old = """        info = {
            "fault_tolerance_signal": float(ft_signal),
            "utilization_signal": float(util_signal),
            "cost_signal": float(cost_signal),
            "selected_node": selected_node,
            "feasible": is_feasible,
        }"""
info_new = """        info = {
            "fault_tolerance_signal": float(ft_signal),
            "utilization_signal": float(util_signal),
            "cost_signal": float(cost_signal),
            "selected_node": selected_node,
            "feasible": is_feasible,
            "node_signals": node_signals,
        }"""
env_code = env_code.replace(info_old, info_new)

with open(env_path, 'w') as f:
    f.write(env_code)

# 3. training/maddpg_trainer.py
trainer_path = 'training/maddpg_trainer.py'
with open(trainer_path, 'r') as f:
    tr_code = f.read()

tr_code = tr_code.replace('self.actor(embeddings, pod)', 'self.actor(torch.cat([embeddings, self.gnn_encoder.input_proj(x)], dim=-1), pod)')

update_old = """        current_embeddings = self.gnn_encoder(obs_batch).view(B, N, -1)
        pred_actions = self.actor(current_embeddings, pod_features)  # (B, N, 3)

        # Q-values under predicted actions (detach pred_actions so actor updates focus on node selection softmax)
        actor_q_all = self.critic(current_embeddings, pred_actions.detach(), pod_features).squeeze(-1)  # (B, N)"""

update_new = """        current_embeddings = self.gnn_encoder(obs_batch).view(B, N, -1)
        current_h0 = self.gnn_encoder.input_proj(obs_batch.x).view(B, N, -1)
        cat_embeddings = torch.cat([current_embeddings, current_h0], dim=-1)
        pred_actions = self.actor(cat_embeddings, pod_features)  # (B, N, 3)

        # Q-values under predicted actions (detach pred_actions so actor updates focus on node selection softmax)
        actor_q_all = self.critic(current_embeddings.detach(), pred_actions.detach(), pod_features).squeeze(-1)  # (B, N)"""

tr_code = tr_code.replace(update_old, update_new)

q_eval_old = """            # Target GNN embeddings
            next_embeddings = self.gnn_encoder_target(next_obs_batch).view(B, N, -1)

            # Target decentralized actor actions
            next_actions = self.actor_target(next_embeddings, next_pod_features)"""
q_eval_new = """            self.gnn_encoder_target.eval(); self.actor_target.eval(); self.critic_target.eval()
            # Target GNN embeddings
            next_embeddings = self.gnn_encoder_target(next_obs_batch).view(B, N, -1)
            next_h0 = self.gnn_encoder_target.input_proj(next_obs_batch.x).view(B, N, -1)

            # Target decentralized actor actions
            next_actions = self.actor_target(torch.cat([next_embeddings, next_h0], dim=-1), next_pod_features)"""
tr_code = tr_code.replace(q_eval_old, q_eval_new)

# Aux loss
aux_old = """        # Expected Q-value of the cluster scheduling decision
        expected_cluster_q = (selection_weights * actor_q_all).sum(dim=1)  # (B,)
        actor_loss = -expected_cluster_q.mean()"""
aux_new = """        # Expected Q-value of the cluster scheduling decision
        expected_cluster_q = (selection_weights * actor_q_all).sum(dim=1)  # (B,)
        actor_loss = -expected_cluster_q.mean()
        
        # Auxiliary loss for meaningful heads
        node_signals = replay_buffer_sample[-1] # This will be the last returned item from sample()
        aux = F.mse_loss(pred_actions, (node_signals + 1) / 2)
        actor_loss = actor_loss + 0.5 * aux"""
tr_code = tr_code.replace(aux_old, aux_new)

sample_unpack_old = """            obs_batch,
            actions,
            rewards,
            next_obs_batch,
            dones,
            pod_features,
            next_pod_features,
            selected_nodes,
        ) = replay_buffer.sample(batch_size, device=self.device)"""
sample_unpack_new = """            obs_batch,
            actions,
            rewards,
            next_obs_batch,
            dones,
            pod_features,
            next_pod_features,
            selected_nodes,
            node_signals,
        ) = replay_buffer.sample(batch_size, device=self.device)
        replay_buffer_sample = [node_signals]"""
tr_code = tr_code.replace(sample_unpack_old, sample_unpack_new)

with open(trainer_path, 'w') as f:
    f.write(tr_code)

# 4. training/train.py
train_path = 'training/train.py'
with open(train_path, 'r') as f:
    train_code = f.read()

train_code = train_code.replace('selection_temp: float = 0.1,', '')
train_code = re.sub(r'device=device,', 'device=device,\n        selection_temp=config.get("selection_temp", 0.1),', train_code)

actor_init_old = """    actor = NodeActor(
        embed_dim=config.get("hidden_dim", 64),"""
actor_init_new = """    actor = NodeActor(
        embed_dim=config.get("hidden_dim", 64) * 2,"""
train_code = train_code.replace(actor_init_old, actor_init_new)

push_old = """            replay_buffer.push(
                obs,
                actions,
                reward,
                next_obs,
                done,
                selected_node=info.get("selected_node", 0),
            )"""
push_new = """            replay_buffer.push(
                obs,
                actions,
                reward,
                next_obs,
                done,
                selected_node=info.get("selected_node", 0),
                node_signals=info.get("node_signals", np.zeros((env.num_nodes, 3), dtype=np.float32)),
            )"""
train_code = train_code.replace(push_old, push_new)
train_code = re.sub(r'# Set random seeds\n?', '', train_code)
train_code = re.sub(r'def train\(config: Dict\[str, Any\]\):', 'def train(config: Dict[str, Any], seed: int = 42):\n    import random\n    torch.manual_seed(seed)\n    np.random.seed(seed)\n    random.seed(seed)\n', train_code)
train_code = re.sub(r'env = MockClusterEnv\(', 'env = MockClusterEnv(\n        seed=seed,', train_code)
train_code = re.sub(r'--seed\s+and\s+seed\s+torch,\s+numpy\s+and\s+the\s+env\.', '', train_code)
with open(train_path, 'w') as f:
    f.write(train_code)

# 5. training/replay_buffer.py
buffer_path = 'training/replay_buffer.py'
with open(buffer_path, 'r') as f:
    buf_code = f.read()

push_buf_old = """        done: bool,
        selected_node: int = 0,
    ):
        \"\"\"Push a transition to the n-step buffer and commit when ready.\"\"\"
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "selected_node": selected_node,
        }"""
push_buf_new = """        done: bool,
        selected_node: int = 0,
        node_signals: Optional[np.ndarray] = None,
    ):
        \"\"\"Push a transition to the n-step buffer and commit when ready.\"\"\"
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "selected_node": selected_node,
            "node_signals": node_signals,
        }"""
buf_code = buf_code.replace(push_buf_old, push_buf_new)

stored_item_old = """            "selected_node": int(first_trans.get("selected_node", 0)),
        }"""
stored_item_new = """            "selected_node": int(first_trans.get("selected_node", 0)),
            "node_signals": first_trans.get("node_signals", None),
        }"""
buf_code = buf_code.replace(stored_item_old, stored_item_new)

sample_sig_old = """        return (
            obs_batch,
            actions,
            rewards,
            next_obs_batch,
            dones,
            pod_features,
            next_pod_features,
            selected_nodes,
        )"""
sample_sig_new = """        node_signals = torch.tensor(
            np.array([s.get("node_signals", np.zeros((s["action"].shape[0], 3))) for s in samples]),
            dtype=torch.float32,
            device=device,
        )
        return (
            obs_batch,
            actions,
            rewards,
            next_obs_batch,
            dones,
            pod_features,
            next_pod_features,
            selected_nodes,
            node_signals,
        )"""
buf_code = buf_code.replace(sample_sig_old, sample_sig_new)

sample_def_old = """    def sample(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor, Batch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:"""
sample_def_new = """    def sample(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor, Batch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:"""
buf_code = buf_code.replace(sample_def_old, sample_def_new)

with open(buffer_path, 'w') as f:
    f.write(buf_code)

print("Modifications done!")
