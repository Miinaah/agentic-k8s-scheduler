"""
test_cluster_env.py

Unit tests for Split A (Cluster Simulator & Workload Generator), matching
the "How to test" checklist in the project plan section 5A:
  - idle-cluster utilization test
  - Poisson arrival-rate statistical test
  - node-failure injection test
  - total-capacity overcommit property test

Run with: pytest test_cluster_env.py -v
"""

import pytest

from env.cluster_env import (
    ClusterEnv,
    WorkloadGenerator,
    NodeStatus,
    inject_node_failure,
)


# --------------------------------------------------------------------------
# 1. Idle cluster -> zero utilization
# --------------------------------------------------------------------------

def test_idle_cluster_has_zero_utilization():
    """With arrival_rate=0, no pods ever arrive, so utilization stays 0%
    on every node across many steps."""
    env = ClusterEnv(
        num_nodes=5,
        max_steps=50,
        seed=1,
        workload_generator=WorkloadGenerator(arrival_rate=0.0, rng=None, seed=1),
    )
    env.reset(seed=1)

    for _ in range(30):
        actions = {agent: 0 for agent in env.agents}  # everyone passes
        env.step(actions)

    for node in env.nodes.values():
        assert node.cpu_used == 0.0
        assert node.memory_used == 0.0
        assert node.utilization("cpu") == 0.0
        assert node.utilization("memory") == 0.0


# --------------------------------------------------------------------------
# 2. Poisson arrival rate statistical test
# --------------------------------------------------------------------------

def test_poisson_arrival_rate_within_tolerance():
    """Over many timesteps, the observed mean arrival count per step
    should be within 5% of the configured arrival_rate (lambda)."""
    lam = 5.0
    n_steps = 10_000
    gen = WorkloadGenerator(arrival_rate=lam, seed=123)

    total_arrivals = 0
    for t in range(n_steps):
        pods = gen.next_arrivals(current_time=t)
        total_arrivals += len(pods)

    observed_rate = total_arrivals / n_steps
    relative_error = abs(observed_rate - lam) / lam

    assert relative_error < 0.05, (
        f"observed rate {observed_rate:.3f} deviates from lambda={lam} "
        f"by {relative_error:.1%}, exceeding 5% tolerance"
    )


# --------------------------------------------------------------------------
# 3. Node failure injection
# --------------------------------------------------------------------------

def test_node_failure_injection_stops_new_scheduling():
    """A failed node: health_score drops to 0, status becomes FAILED, and
    it can no longer accept new pods (can_fit always False) until it
    recovers."""
    env = ClusterEnv(num_nodes=3, max_steps=50, seed=7)
    env.reset(seed=7)

    failed_id = inject_node_failure(env, node_id="node_0")
    failed_node = env.nodes[failed_id]

    assert failed_node.status == NodeStatus.FAILED
    assert failed_node.health_score == 0.0
    assert not failed_node.can_fit(cpu_req=0.01, mem_req=0.01)

    # Even a trivially small pod request must not fit on a failed node.
    assert failed_node.can_fit(0.0, 0.0) is False


def test_node_failure_evicts_and_requeues_hosted_pods():
    """Pods hosted on a node that fails are marked EVICTED and pushed back
    into the pending queue rather than silently disappearing."""
    # A single-node cluster guarantees any scheduled pod lands on node_0 --
    # avoids depending on which node happens to win bids under a given seed.
    env = ClusterEnv(num_nodes=1, max_steps=50, seed=3,
                      workload_generator=WorkloadGenerator(arrival_rate=3.0, seed=3))
    env.reset(seed=3)

    for _ in range(5):
        actions = {agent: (1 if env.pending_queue else 0) for agent in env.agents}
        env.step(actions)

    hosted_before = list(env.nodes["node_0"].hosted_pods)
    assert hosted_before, "expected at least one pod scheduled onto the only node"

    inject_node_failure(env, node_id="node_0")

    for pod_id in hosted_before:
        pod = env.pods[pod_id]
        assert pod.status.value == "evicted"
        assert pod.node_id is None
        assert pod_id in [p.pod_id for p in env.pending_queue]


# --------------------------------------------------------------------------
# 4. Overcommit property test
# --------------------------------------------------------------------------

def test_cpu_and_memory_never_overcommitted():
    """At no point should any node's used resources exceed its capacity,
    across a long run with a busy workload and random bidding."""
    import random

    env = ClusterEnv(
        num_nodes=6,
        max_steps=200,
        seed=99,
        workload_generator=WorkloadGenerator(arrival_rate=4.0, seed=99),
    )
    env.reset(seed=99)
    policy_rng = random.Random(99)

    for _ in range(150):
        actions = {}
        for agent in env.agents:
            queue_len = len(env.pending_queue)
            actions[agent] = policy_rng.randint(0, queue_len) if queue_len else 0
        env.step(actions)

        for node in env.nodes.values():
            assert node.cpu_used <= node.cpu_capacity + 1e-6, (
                f"{node.node_id} overcommitted CPU: {node.cpu_used} > {node.cpu_capacity}"
            )
            assert node.memory_used <= node.memory_capacity + 1e-6, (
                f"{node.node_id} overcommitted memory: {node.memory_used} > {node.memory_capacity}"
            )


# --------------------------------------------------------------------------
# Bonus: determinism (same seed -> same trajectory)
# --------------------------------------------------------------------------

def test_same_seed_produces_identical_arrivals():
    """Two WorkloadGenerators with the same seed must produce identical
    pod counts and request sizes -- this is what makes experiments
    reproducible across runs."""
    gen_a = WorkloadGenerator(arrival_rate=3.0, seed=55)
    gen_b = WorkloadGenerator(arrival_rate=3.0, seed=55)

    for t in range(20):
        pods_a = gen_a.next_arrivals(t)
        pods_b = gen_b.next_arrivals(t)
        assert len(pods_a) == len(pods_b)
        for pa, pb in zip(pods_a, pods_b):
            assert pa.cpu_request == pb.cpu_request
            assert pa.memory_request == pb.memory_request
            assert pa.qos_class == pb.qos_class
