import React, { useState } from 'react';
import {
  Cpu,
  Database,
  Flame,
  Heart,
  AlertTriangle,
  DollarSign,
  CheckCircle2,
  Sliders,
  RotateCcw,
  Sparkles
} from 'lucide-react';

export default function NodeCard({
  node,
  isWinner = false,
  candidateScore = null,
  feasibility = null,
  onUpdateMetrics
}) {
  const [showEdit, setShowEdit] = useState(false);
  const [cpuUsed, setCpuUsed] = useState(node.cpu_used);
  const [memUsed, setMemUsed] = useState(node.mem_used_gb);
  const [health, setHealth] = useState(node.health_score);
  const [failures, setFailures] = useState(node.recent_failures);

  const cpuPercent = Math.min(100, Math.round((node.cpu_used / node.cpu_capacity) * 100));
  const memPercent = Math.min(100, Math.round((node.mem_used_gb / node.mem_capacity_gb) * 100));

  const handleSave = () => {
    onUpdateMetrics(node.node_id, {
      cpu_used: parseFloat(cpuUsed),
      mem_used_gb: parseFloat(memUsed),
      health_score: parseFloat(health),
      recent_failures: parseInt(failures, 10),
    });
    setShowEdit(false);
  };

  const handleQuickSpike = () => {
    const newCpu = Math.min(node.cpu_capacity, round2(node.cpu_used + (node.cpu_capacity * 0.35)));
    onUpdateMetrics(node.node_id, { cpu_used: newCpu });
  };

  const handleQuickOutage = () => {
    onUpdateMetrics(node.node_id, {
      health_score: 0.15,
      recent_failures: node.recent_failures + 3,
    });
  };

  const handleQuickHeal = () => {
    onUpdateMetrics(node.node_id, {
      health_score: 1.0,
      recent_failures: 0,
      cpu_used: Math.round(node.cpu_capacity * 0.25),
      mem_used_gb: Math.round(node.mem_capacity_gb * 0.25),
    });
  };

  function round2(v) {
    return Math.round(v * 100) / 100;
  }

  const getHealthColor = (h) => {
    if (h >= 0.95) return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
    if (h >= 0.70) return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
    return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
  };

  return (
    <div
      className={`relative glass-card rounded-xl p-4 transition-all duration-300 ${
        isWinner
          ? 'ring-2 ring-teal-400/80 glow-teal bg-teal-950/20'
          : 'border-slate-800 hover:border-slate-700'
      }`}
    >
      {/* Winner Badge */}
      {isWinner && (
        <div className="absolute -top-3 left-4 px-3 py-0.5 rounded-full bg-gradient-to-r from-teal-500 to-emerald-500 text-slate-950 text-[11px] font-bold flex items-center gap-1 shadow-md">
          <Sparkles className="w-3 h-3 fill-current" />
          <span>SELECTED WINNER</span>
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-mono text-sm font-semibold text-white tracking-wide">
              {node.node_id}
            </h3>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700">
              {node.zone}
            </span>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-slate-400 flex items-center gap-1">
              <DollarSign className="w-3 h-3 text-amber-400" />
              ${node.cost_per_hour}/hr
            </span>
            <span className="text-xs text-slate-500">&bull;</span>
            <span className="text-xs text-slate-400">
              {node.pod_count} pods
            </span>
          </div>
        </div>

        {/* Health & Edit button */}
        <div className="flex items-center gap-1.5">
          <div className={`px-2 py-0.5 rounded-full border text-[11px] font-mono font-medium flex items-center gap-1 ${getHealthColor(node.health_score)}`}>
            <Heart className="w-3 h-3 fill-current" />
            <span>{(node.health_score * 100).toFixed(0)}%</span>
          </div>

          <button
            onClick={() => setShowEdit(!showEdit)}
            title="Adjust metrics"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition"
          >
            <Sliders className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Resource Bars */}
      <div className="space-y-2.5 my-3">
        {/* CPU */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400 flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" /> CPU Allocation
            </span>
            <span className="font-mono text-slate-200">
              {node.cpu_used.toFixed(1)} / {node.cpu_capacity.toFixed(1)} cores ({cpuPercent}%)
            </span>
          </div>
          <div className="w-full bg-slate-800/80 h-2 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                cpuPercent >= 90 ? 'bg-rose-500' : cpuPercent >= 75 ? 'bg-amber-400' : 'bg-cyan-400'
              }`}
              style={{ width: `${cpuPercent}%` }}
            />
          </div>
        </div>

        {/* Memory */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400 flex items-center gap-1">
              <Database className="w-3.5 h-3.5 text-indigo-400" /> RAM Allocation
            </span>
            <span className="font-mono text-slate-200">
              {node.mem_used_gb.toFixed(1)} / {node.mem_capacity_gb.toFixed(1)} GB ({memPercent}%)
            </span>
          </div>
          <div className="w-full bg-slate-800/80 h-2 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                memPercent >= 90 ? 'bg-rose-500' : memPercent >= 75 ? 'bg-amber-400' : 'bg-indigo-400'
              }`}
              style={{ width: `${memPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Failure alert if any */}
      {node.recent_failures > 0 && (
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-rose-950/60 border border-rose-500/30 text-rose-300 text-xs mb-3">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
          <span>{node.recent_failures} recent failure event{node.recent_failures > 1 ? 's' : ''}</span>
        </div>
      )}

      {/* Candidate Scores Snapshot (if available) */}
      {candidateScore && (
        <div className="pt-2 border-t border-slate-800/80 grid grid-cols-3 gap-1.5 text-center my-2 text-xs">
          <div className="bg-slate-900/60 rounded p-1 border border-slate-800">
            <div className="text-[10px] text-slate-400">FT Score</div>
            <div className="font-mono font-semibold text-emerald-300">
              {candidateScore.fault_tolerance.toFixed(2)}
            </div>
          </div>
          <div className="bg-slate-900/60 rounded p-1 border border-slate-800">
            <div className="text-[10px] text-slate-400">Util Score</div>
            <div className="font-mono font-semibold text-cyan-300">
              {candidateScore.utilisation.toFixed(2)}
            </div>
          </div>
          <div className="bg-slate-900/60 rounded p-1 border border-slate-800">
            <div className="text-[10px] text-slate-400">Cost Score</div>
            <div className="font-mono font-semibold text-amber-300">
              {candidateScore.cost.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Feasibility note if infeasible */}
      {feasibility && !feasibility.feasible && (
        <div className="text-[11px] text-rose-400 bg-rose-950/30 border border-rose-900/50 rounded p-1.5 mt-1 font-mono">
          🚫 Infeasible: {feasibility.reason}
        </div>
      )}

      {/* Quick Fault Injection Actions */}
      <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between gap-1 text-[11px]">
        <span className="text-slate-500 text-[10px]">What-if:</span>
        <div className="flex items-center gap-1">
          <button
            onClick={handleQuickSpike}
            title="Spike CPU load (+35%)"
            className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition flex items-center gap-1"
          >
            <Flame className="w-2.5 h-2.5 text-orange-400" />
            +Load
          </button>
          <button
            onClick={handleQuickOutage}
            title="Simulate node failure / degradation"
            className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-rose-300 transition flex items-center gap-1"
          >
            <AlertTriangle className="w-2.5 h-2.5 text-rose-400" />
            Fail
          </button>
          <button
            onClick={handleQuickHeal}
            title="Heal node to 100% health & 0 failures"
            className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-emerald-300 transition flex items-center gap-1"
          >
            <RotateCcw className="w-2.5 h-2.5 text-emerald-400" />
            Heal
          </button>
        </div>
      </div>

      {/* Metric Customizer Modal/Drawer */}
      {showEdit && (
        <div className="mt-3 p-3 rounded-lg bg-slate-900 border border-slate-700 space-y-2 text-xs">
          <div className="font-semibold text-slate-200 text-[11px] uppercase tracking-wider mb-2">
            Configure {node.node_id} Metrics
          </div>

          <div className="flex items-center justify-between gap-2">
            <label className="text-slate-400">CPU Used ({node.cpu_capacity} cap):</label>
            <input
              type="number"
              step="0.5"
              min="0"
              max={node.cpu_capacity}
              value={cpuUsed}
              onChange={(e) => setCpuUsed(e.target.value)}
              className="w-20 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-right font-mono"
            />
          </div>

          <div className="flex items-center justify-between gap-2">
            <label className="text-slate-400">RAM Used ({node.mem_capacity_gb} GB):</label>
            <input
              type="number"
              step="1"
              min="0"
              max={node.mem_capacity_gb}
              value={memUsed}
              onChange={(e) => setMemUsed(e.target.value)}
              className="w-20 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-right font-mono"
            />
          </div>

          <div className="flex items-center justify-between gap-2">
            <label className="text-slate-400">Health Score (0.0 - 1.0):</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={health}
              onChange={(e) => setHealth(e.target.value)}
              className="w-20 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-right font-mono"
            />
          </div>

          <div className="flex items-center justify-between gap-2">
            <label className="text-slate-400">Recent Failures:</label>
            <input
              type="number"
              step="1"
              min="0"
              value={failures}
              onChange={(e) => setFailures(e.target.value)}
              className="w-20 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-right font-mono"
            />
          </div>

          <div className="flex justify-end gap-2 pt-2">
            <button
              onClick={() => setShowEdit(false)}
              className="px-2.5 py-1 rounded bg-slate-800 text-slate-400 hover:text-white"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-2.5 py-1 rounded bg-teal-600 hover:bg-teal-500 text-white font-medium"
            >
              Apply Updates
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
