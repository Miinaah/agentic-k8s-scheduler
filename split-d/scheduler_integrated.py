import os
import logging
import time
from datetime import datetime, timezone

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.utils.quantity import parse_quantity

from decision_layer.pipeline import decide
from resource_utils import cpu_to_cores, memory_to_gb


SCHEDULER_NAME = os.getenv("SCHEDULER_NAME", "split-d-scheduler")
POLL_INTERVAL_SECONDS = 3

# Explicit fallback assumptions, not measured telemetry.
DEFAULT_DISK_CAPACITY_GB = 10.0
DEFAULT_NODE_COST_PER_HOUR = 0.0
DEFAULT_ESTIMATED_DURATION_SEC = 60.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def quantity(value):
    """Parse a Kubernetes quantity into a numeric value."""
    return float(parse_quantity(str(value)))


def container_requests(pod):
    """Sum regular-container CPU and memory requests."""
    cpu = 0.0
    memory_bytes = 0.0

    for container in pod.spec.containers or []:
        requests = (container.resources.requests or {}) if container.resources else {}
        cpu += quantity(requests.get("cpu", "0"))
        memory_bytes += quantity(requests.get("memory", "0"))

    # Account for init containers using the maximum init request.
    init_cpu = 0.0
    init_memory_bytes = 0.0
    for container in pod.spec.init_containers or []:
        requests = (container.resources.requests or {}) if container.resources else {}
        init_cpu = max(init_cpu, quantity(requests.get("cpu", "0")))
        init_memory_bytes = max(
            init_memory_bytes, quantity(requests.get("memory", "0"))
        )

    return max(cpu, init_cpu), max(memory_bytes, init_memory_bytes)


def pod_qos_class(pod):
    """Return the QoS class assigned by Kubernetes, if available."""
    if pod.status and pod.status.qos_class:
        return pod.status.qos_class
    return "BestEffort"


def pod_priority(pod):
    """Map Kubernetes priority information to the decision-layer categories."""
    if pod.spec.priority_class_name:
        name = pod.spec.priority_class_name.lower()
        if "high" in name or "critical" in name:
            return "high"
        return pod.spec.priority_class_name
    return "default"


def node_is_ready_and_schedulable(node):
    if node.spec.unschedulable:
        return False

    return any(
        condition.type == "Ready" and condition.status == "True"
        for condition in (node.status.conditions or [])
    )


def node_allocated_requests(core_api, node_name):
    """Sum requests for pods already assigned to this node.

    This is scheduling reservation data, not measured CPU/RAM usage.
    """
    cpu_total = 0.0
    memory_total_bytes = 0.0
    pod_count = 0

    for pod in core_api.list_pod_for_all_namespaces().items:
        if pod.spec.node_name != node_name:
            continue
        if pod.metadata.deletion_timestamp is not None:
            continue
        if pod.status and pod.status.phase in ("Succeeded", "Failed"):
            continue

        cpu, memory_bytes = container_requests(pod)
        cpu_total += cpu
        memory_total_bytes += memory_bytes
        pod_count += 1

    return cpu_total, memory_total_bytes, pod_count


def make_cluster_state(core_api, pending_pod):
    """Build Split C's expected ClusterState dictionary."""
    nodes = core_api.list_node().items
    candidate_nodes = []
    ready_node_names = set()

    pod_cpu, pod_memory_bytes = container_requests(pending_pod)
    if pod_cpu <= 0:
        pod_cpu = 0.001
    if pod_memory_bytes <= 0:
        pod_memory_bytes = 1_000_000

    for node in nodes:
        if not node_is_ready_and_schedulable(node):
            continue

        allocatable = node.status.allocatable or {}
        cpu_capacity = quantity(allocatable.get("cpu", "0"))
        memory_capacity_bytes = quantity(allocatable.get("memory", "0"))

        if cpu_capacity <= 0 or memory_capacity_bytes <= 0:
            continue

        cpu_used, memory_used_bytes, pod_count = node_allocated_requests(
            core_api, node.metadata.name
        )

        # Split C's pipeline filters infeasible nodes using capacity minus used.
        # Exclude the pending pod itself if it has already been counted.
        if pending_pod.spec.node_name == node.metadata.name:
            own_cpu, own_mem_bytes = container_requests(pending_pod)
            cpu_used = max(0.0, cpu_used - own_cpu)
            memory_used_bytes = max(0.0, memory_used_bytes - own_mem_bytes)

        labels = node.metadata.labels or {}
        zone = (
            labels.get("topology.kubernetes.io/zone")
            or labels.get("failure-domain.beta.kubernetes.io/zone")
            or "unknown"
        )

        candidate_nodes.append({
            "node_id": node.metadata.name,
            "cpu_capacity": cpu_capacity,
            "cpu_used": cpu_used,
            "mem_capacity_gb": memory_to_gb(memory_capacity_bytes),
            "mem_used_gb": memory_to_gb(memory_used_bytes),
            "disk_capacity_gb": DEFAULT_DISK_CAPACITY_GB,
            "disk_used_gb": 0.0,
            "health_score": 1.0,
            "cost_per_hour": DEFAULT_NODE_COST_PER_HOUR,
            "zone": zone,
            "pod_count": pod_count,
            "recent_failures": 0,
        })
        ready_node_names.add(node.metadata.name)

    if not candidate_nodes:
        return None, ready_node_names

    priority = pod_priority(pending_pod)
    if priority == "default":
        priority = "default"

    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nodes": candidate_nodes,
        "pending_pod": {
            "pod_id": pending_pod.metadata.name,
            "cpu_request": pod_cpu,
            "mem_request_gb": memory_to_gb(pod_memory_bytes),
            "priority_class": priority,
            "qos_class": pod_qos_class(pending_pod),
            "estimated_duration_sec": DEFAULT_ESTIMATED_DURATION_SEC,
        },
    }
    return state, ready_node_names


def bind_pod(core_api, pod, node_name):
    target = client.V1ObjectReference(
        api_version="v1",
        kind="Node",
        name=node_name,
    )
    binding = client.V1Binding(
        api_version="v1",
        kind="Binding",
        metadata=client.V1ObjectMeta(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
        ),
        target=target,
    )

    core_api.create_namespaced_pod_binding(
        name=pod.metadata.name,
        namespace=pod.metadata.namespace,
        body=binding,
        _preload_content=False,
    )


def main():
    config.load_incluster_config()
    core_api = client.CoreV1Api()
    logger.info("%s started with Split C decision layer", SCHEDULER_NAME)

    while True:
        try:
            pods = core_api.list_pod_for_all_namespaces().items

            for pod in pods:
                if pod.spec.scheduler_name != SCHEDULER_NAME:
                    continue
                if pod.spec.node_name is not None:
                    continue
                if pod.status and pod.status.phase in ("Succeeded", "Failed"):
                    continue

                cluster_state, ready_names = make_cluster_state(core_api, pod)
                if not cluster_state:
                    logger.warning("No Ready, schedulable nodes for pod %s", pod.metadata.name)
                    continue

                decision = decide(
                    cluster_state=cluster_state,
                    use_llm=False,
                    scorer="heuristic",
                )

                if decision is None:
                    logger.info("No decision for pod %s", pod.metadata.name)
                    continue

                selected_node = decision.selected_node_id
                logger.info(
                    "Split C decision: pod=%s selected_node=%s latency_ms=%.2f",
                    pod.metadata.name,
                    selected_node,
                    decision.decision_latency_ms,
                )

                if selected_node is None:
                    logger.warning(
                        "No feasible node for pod %s; leaving it unbound",
                        pod.metadata.name,
                    )
                    continue

                if selected_node not in ready_names:
                    logger.error(
                        "Decision returned unknown/ineligible node %s; not binding",
                        selected_node,
                    )
                    continue

                try:
                    bind_pod(core_api, pod, selected_node)
                    logger.info(
                        "Scheduled %s/%s onto %s",
                        pod.metadata.namespace,
                        pod.metadata.name,
                        selected_node,
                    )
                except ApiException as exc:
                    # A pod may have been handled by another scheduler instance.
                    logger.error(
                        "Binding failed for %s/%s: %s %s",
                        pod.metadata.namespace,
                        pod.metadata.name,
                        exc.status,
                        exc.body,
                    )

        except ApiException as exc:
            logger.error("Kubernetes API error: %s %s", exc.status, exc.body)
        except Exception:
            logger.exception("Unexpected scheduler error")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
