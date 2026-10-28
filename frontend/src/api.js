/**
 * frontend/src/api.js
 * API client to interact with the backend FastAPI service.
 */

const API_BASE = '/api';

export async function fetchClusterState() {
  const res = await fetch(`${API_BASE}/cluster/state`);
  if (!res.ok) throw new Error(`Failed to fetch cluster state: ${res.statusText}`);
  return res.json();
}

export async function resetCluster(scenario = 'calm') {
  const res = await fetch(`${API_BASE}/cluster/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario }),
  });
  if (!res.ok) throw new Error(`Failed to reset cluster: ${res.statusText}`);
  return res.json();
}

export async function updateNodeMetrics(nodeId, updates) {
  const res = await fetch(`${API_BASE}/cluster/node/update`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ node_id: nodeId, ...updates }),
  });
  if (!res.ok) throw new Error(`Failed to update node: ${res.statusText}`);
  return res.json();
}

export async function evaluateSchedule({ use_llm = false, scorer = 'heuristic', epsilon = 0.05, pod = null }) {
  const res = await fetch(`${API_BASE}/schedule/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_llm, scorer, epsilon, pod }),
  });
  if (!res.ok) throw new Error(`Failed to evaluate schedule: ${res.statusText}`);
  return res.json();
}

export async function applySchedule({ use_llm = false, scorer = 'heuristic', epsilon = 0.05, pod = null }) {
  const res = await fetch(`${API_BASE}/schedule/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_llm, scorer, epsilon, pod }),
  });
  if (!res.ok) throw new Error(`Failed to apply schedule: ${res.statusText}`);
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error(`Failed to fetch history: ${res.statusText}`);
  return res.json();
}

export async function clearHistory() {
  const res = await fetch(`${API_BASE}/history/clear`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to clear history: ${res.statusText}`);
  return res.json();
}

export async function fetchScenarios() {
  const res = await fetch(`${API_BASE}/scenarios`);
  if (!res.ok) throw new Error(`Failed to fetch scenarios: ${res.statusText}`);
  return res.json();
}

export async function fetchK8sStatus() {
  const res = await fetch(`${API_BASE}/k8s/status`);
  if (!res.ok) throw new Error(`Failed to fetch K8s status: ${res.statusText}`);
  return res.json();
}
