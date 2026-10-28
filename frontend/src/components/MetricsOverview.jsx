import React from 'react';
import { Cpu, Database, DollarSign, Activity, HardDrive, ShieldCheck } from 'lucide-react';

export default function MetricsOverview({ clusterState, stressData }) {
  const nodes = clusterState?.nodes || [];
  const nodeCount = nodes.length;

  const totalCpuCap = nodes.reduce((sum, n) => sum + (n.cpu_capacity || 0), 0);
  const totalCpuUsed = nodes.reduce((sum, n) => sum + (n.cpu_used || 0), 0);
  const avgCpuUtil = totalCpuCap > 0 ? (totalCpuUsed / totalCpuCap) * 100 : 0;

  const totalMemCap = nodes.reduce((sum, n) => sum + (n.mem_capacity_gb || 0), 0);
  const totalMemUsed = nodes.reduce((sum, n) => sum + (n.mem_used_gb || 0), 0);
  const avgMemUtil = totalMemCap > 0 ? (totalMemUsed / totalMemCap) * 100 : 0;

  const totalPods = nodes.reduce((sum, n) => sum + (n.pod_count || 0), 0);
  const totalCost = nodes.reduce((sum, n) => sum + (n.cost_per_hour || 0), 0);
  const totalFailures = nodes.reduce((sum, n) => sum + (n.recent_failures || 0), 0);

  const zones = Array.from(new Set(nodes.map(n => n.zone)));

  const stressVal = stressData?.value ?? 0.0;
  const stressPercent = Math.min(100, Math.round(stressVal * 100));

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
      {/* Node / Zone Count */}
      <div className="glass-card p-3.5 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="text-xs font-medium">Nodes & Zones</span>
          <HardDrive className="w-4 h-4 text-teal-400" />
        </div>
        <div className="text-xl font-bold text-white tracking-tight">
          {nodeCount} <span className="text-xs font-normal text-slate-400">nodes</span>
        </div>
        <div className="text-[11px] text-slate-400 mt-1">
          Across {zones.length} zones
        </div>
      </div>

      {/* Cluster CPU Util */}
      <div className="glass-card p-3.5 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="text-xs font-medium">Avg CPU Load</span>
          <Cpu className="w-4 h-4 text-cyan-400" />
        </div>
        <div className="text-xl font-bold text-white tracking-tight">
          {avgCpuUtil.toFixed(1)}%
        </div>
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1.5">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              avgCpuUtil > 80 ? 'bg-red-500' : avgCpuUtil > 60 ? 'bg-amber-400' : 'bg-teal-400'
            }`}
            style={{ width: `${Math.min(100, avgCpuUtil)}%` }}
          />
        </div>
        <div className="text-[10px] text-slate-400 mt-1">
          {totalCpuUsed.toFixed(1)} / {totalCpuCap.toFixed(1)} cores
        </div>
      </div>

      {/* Cluster Memory Util */}
      <div className="glass-card p-3.5 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="text-xs font-medium">Avg RAM Load</span>
          <Database className="w-4 h-4 text-indigo-400" />
        </div>
        <div className="text-xl font-bold text-white tracking-tight">
          {avgMemUtil.toFixed(1)}%
        </div>
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1.5">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              avgMemUtil > 80 ? 'bg-red-500' : avgMemUtil > 60 ? 'bg-amber-400' : 'bg-indigo-400'
            }`}
            style={{ width: `${Math.min(100, avgMemUtil)}%` }}
          />
        </div>
        <div className="text-[10px] text-slate-400 mt-1">
          {totalMemUsed.toFixed(1)} / {totalMemCap.toFixed(1)} GB
        </div>
      </div>

      {/* Total Pods */}
      <div className="glass-card p-3.5 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="text-xs font-medium">Active Pods</span>
          <Activity className="w-4 h-4 text-emerald-400" />
        </div>
        <div className="text-xl font-bold text-white tracking-tight">
          {totalPods}
        </div>
        <div className="text-[11px] text-slate-400 mt-1">
          {totalFailures > 0 ? (
            <span className="text-red-400 font-medium">{totalFailures} failures</span>
          ) : (
            <span className="text-emerald-400">0 failure events</span>
          )}
        </div>
      </div>

      {/* Hourly Cost */}
      <div className="glass-card p-3.5 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="text-xs font-medium">Hourly Cost</span>
          <DollarSign className="w-4 h-4 text-amber-400" />
        </div>
        <div className="text-xl font-bold text-white tracking-tight">
          ${totalCost.toFixed(2)}
          <span className="text-xs font-normal text-slate-400">/hr</span>
        </div>
        <div className="text-[11px] text-slate-400 mt-1">
          ~${(totalCost * 730).toFixed(0)}/month
        </div>
      </div>

      {/* Cluster Stress Index */}
      <div className="glass-card p-3.5 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between text-slate-400 mb-1">
          <span className="text-xs font-medium">Stress Index</span>
          <ShieldCheck className="w-4 h-4 text-rose-400" />
        </div>
        <div className="text-xl font-bold text-white tracking-tight flex items-baseline gap-1">
          {stressVal.toFixed(2)}
          <span className="text-xs font-normal text-slate-400">/ 1.0</span>
        </div>
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1.5">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              stressPercent >= 50 ? 'bg-red-500' : stressPercent >= 25 ? 'bg-amber-400' : 'bg-emerald-400'
            }`}
            style={{ width: `${stressPercent}%` }}
          />
        </div>
        <div className="text-[10px] text-slate-400 mt-1 capitalize">
          Status: {stressData?.level || 'low'}
        </div>
      </div>
    </div>
  );
}
