
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
import time

SCHEDULER_NAME = "split-d-scheduler"


def main():
    # Connect Python to the existing Kubernetes cluster
    config.load_incluster_config()

    core_api = client.CoreV1Api()

    print(f"{SCHEDULER_NAME} started.")

    while True:
        try:
            # Find Pods that have no node assigned and request our scheduler
            pods = core_api.list_pod_for_all_namespaces().items
            print(f"DEBUG: Found {len(pods)} pods")
            for pod in pods:
                if pod.spec.scheduler_name != SCHEDULER_NAME:
                    continue
                print(f"DEBUG: Found custom-scheduler pod {pod.metadata.name}, node={pod.spec.node_name}")
                if pod.spec.node_name is not None:
                    continue

                # Get nodes that are Ready and schedulable
                nodes = core_api.list_node().items

                ready_nodes = []
                for node in nodes:
                    if node.spec.unschedulable:
                        continue

                    for condition in node.status.conditions or []:
                        if condition.type == "Ready" and condition.status == "True":
                            ready_nodes.append(node)
                            break

                if not ready_nodes:
                    print("No Ready nodes available.")
                    continue

                # Temporary mock policy: choose the first Ready node
                selected_node = ready_nodes[0].metadata.name

                # Bind the Pod to the selected node
                print(f"DEBUG: Pod={pod.metadata.name}, selected_node={selected_node}")
                print(f"DEBUG: target type={type(client.V1ObjectReference(api_version='v1', kind='Node', name=selected_node))}")
                try:
                    target = client.V1ObjectReference(
                        api_version="v1",
                        kind="Node",
                        name=selected_node
                    )

                    print(f"DEBUG: target={target!r}")

                    binding = client.V1Binding(
                        api_version="v1",
                        kind="Binding",
                        metadata=client.V1ObjectMeta(
                            name=pod.metadata.name,
                            namespace=pod.metadata.namespace
                        ),
                        target=target
                    )

                    print(f"DEBUG: binding target={binding.target!r}")

                except Exception as exc:
                    print(f"Binding construction failed: {exc!r}")
                    raise
		
                try:
                    core_api.create_namespaced_pod_binding(
                        name=pod.metadata.name,
                        namespace=pod.metadata.namespace,
                        body=binding,
                        _preload_content=False
                    )
                    print(
                        f"Scheduled {pod.metadata.namespace}/"
                        f"{pod.metadata.name} onto {selected_node}"
                    )
                except ApiException as exc:
                    print(f"Binding failed: {exc.status} {exc.body}")


        except ApiException as exc:
            print(f"Kubernetes API error: {exc.status} {exc.body}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")

        time.sleep(3)


if __name__ == "__main__":
    main()


