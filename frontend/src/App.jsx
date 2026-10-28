import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import MetricsOverview from './components/MetricsOverview';
import ClusterTopology from './components/ClusterTopology';
import PodStudio from './components/PodStudio';
import DecisionPipeline from './components/DecisionPipeline';
import WhatIfScenarios from './components/WhatIfScenarios';
import SchedulingHistory from './components/SchedulingHistory';
import {
  fetchClusterState,
  resetCluster,
  updateNodeMetrics,
  evaluateSchedule,
  applySchedule,
  fetchHistory,
  clearHistory,
  fetchScenarios,
  fetchK8sStatus
} from './api';
import { CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';

export default function App() {
  const [clusterState, setClusterState] = useState(null);
  const [activeScenario, setActiveScenario] = useState('calm');
  const [scenarios, setScenarios] = useState([]);
  const [podTemplates, setPodTemplates] = useState([]);
  const [decisionData, setDecisionData] = useState(null);
  const [history, setHistory] = useState([]);
  const [k8sStatus, setK8sStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  // Load initial data
  const loadInitialData = useCallback(async () => {
    try {
      setLoading(true);
      const [stateRes, scenariosRes, historyRes, k8sRes] = await Promise.all([
        fetchClusterState(),
        fetchScenarios(),
        fetchHistory(),
        fetchK8sStatus(),
      ]);

      setClusterState(stateRes);
      setActiveScenario(stateRes.scenario || 'calm');
      setScenarios(scenariosRes.scenarios || []);
      setPodTemplates(scenariosRes.pod_templates || []);
      setHistory(historyRes || []);
      setK8sStatus(k8sRes);
    } catch (err) {
      console.error('Failed to load initial data:', err);
      showToast('Error connecting to backend API. Please make sure the server is running.', 'error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Scenario selection
  const handleSelectScenario = async (scenarioId) => {
    try {
      setLoading(true);
      const res = await resetCluster(scenarioId);
      setClusterState(res.state);
      setActiveScenario(scenarioId);
      setDecisionData(null);
      showToast(`Cluster reset to scenario: ${scenarioId.toUpperCase()}`);
    } catch (err) {
      showToast(`Failed to reset scenario: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Node metric updates
  const handleUpdateMetrics = async (nodeId, updates) => {
    try {
      await updateNodeMetrics(nodeId, updates);
      const updated = await fetchClusterState();
      setClusterState(updated);
      showToast(`Metrics updated for ${nodeId}`);

      // If decision data is present, re-evaluate automatically with same pod
      if (decisionData && clusterState?.pending_pod) {
        handleEvaluate({
          use_llm: false,
          scorer: decisionData.scorer_used || 'heuristic',
          epsilon: 0.05,
          pod: clusterState.pending_pod,
        });
      }
    } catch (err) {
      showToast(`Failed to update metrics: ${err.message}`, 'error');
    }
  };

  // Dry-run evaluate
  const handleEvaluate = async (params) => {
    try {
      setEvaluating(true);
      const trace = await evaluateSchedule(params);
      setDecisionData(trace);
      if (trace.selected_node_id) {
        showToast(`Simulation complete: ${trace.selected_node_id} selected in ${trace.latency_ms.toFixed(1)}ms`);
      } else {
        showToast('Simulation complete: No feasible node found', 'error');
      }
    } catch (err) {
      showToast(`Evaluation failed: ${err.message}`, 'error');
    } finally {
      setEvaluating(false);
    }
  };

  // Commit apply
  const handleApply = async (params) => {
    try {
      setEvaluating(true);
      const res = await applySchedule(params);
      setDecisionData(res.decision);
      setClusterState(res.new_cluster_state);

      const newHist = await fetchHistory();
      setHistory(newHist);

      if (res.bound && res.decision.selected_node_id) {
        showToast(`Pod successfully bound to ${res.decision.selected_node_id}!`);
      } else {
        showToast('Pod could not be bound (no feasible nodes)', 'error');
      }
    } catch (err) {
      showToast(`Failed to schedule pod: ${err.message}`, 'error');
    } finally {
      setEvaluating(false);
    }
  };

  // Chaos triggers
  const handleTriggerChaos = async (type) => {
    if (!clusterState) return;

    if (type === 'cpu_surge') {
      try {
        setLoading(true);
        for (const node of clusterState.nodes) {
          const surgedCpu = Math.min(node.cpu_capacity, Math.round((node.cpu_used + (node.cpu_capacity * 0.4)) * 10) / 10);
          await updateNodeMetrics(node.node_id, { cpu_used: surgedCpu });
        }
        const updated = await fetchClusterState();
        setClusterState(updated);
        showToast('Cluster-wide CPU surge injected across all nodes!');
      } catch (err) {
        showToast('Failed to inject CPU surge', 'error');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleClearHistory = async () => {
    try {
      await clearHistory();
      setHistory([]);
      showToast('Placement history cleared');
    } catch (err) {
      showToast('Failed to clear history', 'error');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Toast Notification */}
      {toast && (
        <div className={`fixed bottom-6 right-6 z-50 px-4 py-3 rounded-xl border flex items-center gap-2.5 shadow-2xl backdrop-blur-md transition-all duration-300 ${
          toast.type === 'error'
            ? 'bg-rose-950/90 border-rose-500/80 text-rose-200 glow-red'
            : 'bg-teal-950/90 border-teal-500/80 text-teal-200 glow-teal'
        }`}>
          {toast.type === 'error' ? (
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-teal-400 shrink-0" />
          )}
          <span className="text-xs font-medium">{toast.message}</span>
        </div>
      )}

      {/* Top Navigation */}
      <Navbar
        scenarios={scenarios}
        activeScenario={activeScenario}
        onSelectScenario={handleSelectScenario}
        onResetCluster={() => handleSelectScenario(activeScenario)}
        onRefresh={loadInitialData}
        stressLevel={decisionData?.stress?.level || 'low'}
        k8sStatus={k8sStatus}
        loading={loading}
      />

      {/* Main Studio Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* Cluster High-Level KPIs */}
        <MetricsOverview
          clusterState={clusterState}
          stressData={decisionData?.stress}
        />

        {/* Studio Controls: Pod Submitter & Policy Controls */}
        <PodStudio
          podTemplates={podTemplates}
          onEvaluate={handleEvaluate}
          onApply={handleApply}
          loading={evaluating || loading}
        />

        {/* Visual Multi-Objective Decision Pipeline Walkthrough */}
        <DecisionPipeline
          decisionData={decisionData}
          loading={evaluating}
        />

        {/* Chaos & What-If Sandbox Scenarios */}
        <WhatIfScenarios
          onTriggerScenario={handleTriggerChaos}
          onResetToScenario={handleSelectScenario}
          loading={loading}
        />

        {/* Cluster Nodes & Availability Zones Topology Grid */}
        <ClusterTopology
          nodes={clusterState?.nodes || []}
          winnerNodeId={decisionData?.selected_node_id}
          candidateScores={decisionData?.scores || []}
          feasibilityResults={decisionData?.feasibility || []}
          onUpdateMetrics={handleUpdateMetrics}
        />

        {/* Real-time Scheduling History Event Stream */}
        <SchedulingHistory
          history={history}
          onClearHistory={handleClearHistory}
          loading={loading}
        />

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-teal-400 inline-block" />
            <span>Agentic Kubernetes Scheduler Studio</span>
          </div>
          <div className="text-slate-400">
            Powered by Graph Neural Networks (SAGEConv/GAT), MADDPG Multi-Agent Core &amp; LLM Decision Arbitration
          </div>
        </div>
      </footer>
    </div>
  );
}
