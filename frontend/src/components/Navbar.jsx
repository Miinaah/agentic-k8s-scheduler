import React from 'react';
import {
  Layers,
  RotateCcw,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Server,
  Sparkles,
  Cpu
} from 'lucide-react';

export default function Navbar({
  scenarios = [],
  activeScenario = 'calm',
  onSelectScenario,
  onResetCluster,
  onRefresh,
  stressLevel = 'low',
  k8sStatus = null,
  loading = false,
  autoRefresh = false,
  onToggleAutoRefresh,
}) {
  const getStressBadge = () => {
    switch (stressLevel) {
      case 'high':
        return (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-950/80 border border-red-500/50 text-red-300 text-xs font-semibold glow-red">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>High Stress</span>
          </div>
        );
      case 'medium':
        return (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-950/80 border border-amber-500/50 text-amber-300 text-xs font-semibold glow-amber">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            <span>Elevated Stress</span>
          </div>
        );
      case 'low':
      default:
        return (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-500/50 text-emerald-300 text-xs font-semibold">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Normal / Calm</span>
          </div>
        );
    }
  };

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand & Subtitle */}
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-tr from-teal-600 to-cyan-500 text-slate-950 shadow-lg shadow-teal-500/20">
            <Layers className="w-6 h-6 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
                Agentic K8s Scheduler
                <span className="text-xs font-medium px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-400 border border-teal-500/30">
                  GNN + Multi-Objective
                </span>
              </h1>
            </div>
            <p className="text-xs text-slate-400">
              Reinforcement Learning Core &bull; LLM Decision Arbiter &bull; Lexicographic Tolerance
            </p>
          </div>
        </div>

        {/* Center: Status indicators */}
        <div className="flex items-center gap-3">
          {getStressBadge()}

          {/* K8s Status Indicator */}
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-900 border border-slate-700/60 text-slate-300 text-xs">
            <Server className="w-3.5 h-3.5 text-teal-400" />
            <span>
              {k8sStatus?.available ? (
                <span className="text-emerald-400 font-medium">K8s Live ({k8sStatus.node_count} nodes)</span>
              ) : (
                <span className="text-slate-400">Sandbox Simulation</span>
              )}
            </span>
          </div>
        </div>

        {/* Right: Controls & Scenario Switcher */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <label htmlFor="scenario-select" className="text-xs text-slate-400 font-medium hidden lg:inline">
              Preset Scenario:
            </label>
            <select
              id="scenario-select"
              value={activeScenario}
              onChange={(e) => onSelectScenario(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer hover:border-slate-600"
            >
              {scenarios.map((sc) => (
                <option key={sc.id} value={sc.id}>
                  {sc.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={onResetCluster}
            disabled={loading}
            title="Reset active scenario to initial baseline"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white text-xs font-medium transition disabled:opacity-50"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>

          <button
            onClick={onRefresh}
            disabled={loading}
            title="Fetch latest telemetry"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600/20 hover:bg-teal-600/30 border border-teal-500/40 text-teal-300 hover:text-teal-200 text-xs font-medium transition disabled:opacity-50 ${loading ? 'animate-pulse' : ''}`}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Sync</span>
          </button>
        </div>
      </div>
    </header>
  );
}
