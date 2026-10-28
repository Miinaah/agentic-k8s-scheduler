import React from 'react';
import {
  Flame,
  AlertTriangle,
  RotateCcw,
  TrendingUp,
  Zap,
  DollarSign,
  CloudLightning
} from 'lucide-react';

export default function WhatIfScenarios({ onTriggerScenario, onResetToScenario, loading = false }) {
  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <CloudLightning className="w-5 h-5 text-amber-400" />
            What-If Chaos & Stress Sandbox
          </h2>
          <p className="text-xs text-slate-400">
            One-click cluster disaster simulations to test agentic resilience, LLM arbitration, and failover
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Scenario 1: CPU Surge */}
        <button
          onClick={() => onTriggerScenario('cpu_surge')}
          disabled={loading}
          className="p-3.5 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-amber-500/50 text-left transition group disabled:opacity-50"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 group-hover:scale-110 transition">
              <Flame className="w-4 h-4" />
            </span>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800/60">
              Contention
            </span>
          </div>
          <div className="text-xs font-bold text-slate-200">
            Cluster-Wide CPU Surge
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            Injects +40% CPU utilization across all nodes to push cluster stress into high tier.
          </p>
        </button>

        {/* Scenario 2: Zone Outage */}
        <button
          onClick={() => onResetToScenario('zone_outage')}
          disabled={loading}
          className="p-3.5 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-rose-500/50 text-left transition group disabled:opacity-50"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="p-1.5 rounded-lg bg-rose-500/10 text-rose-400 group-hover:scale-110 transition">
              <AlertTriangle className="w-4 h-4" />
            </span>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800/60">
              Failure Domain
            </span>
          </div>
          <div className="text-xs font-bold text-slate-200">
            Zone us-central1-c Outage
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            Drops zone health to 20% and injects failures to test fault-tolerant re-routing.
          </p>
        </button>

        {/* Scenario 3: High Contention */}
        <button
          onClick={() => onResetToScenario('stressed')}
          disabled={loading}
          className="p-3.5 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-red-500/50 text-left transition group disabled:opacity-50"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="p-1.5 rounded-lg bg-red-500/10 text-red-400 group-hover:scale-110 transition">
              <Zap className="w-4 h-4" />
            </span>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-red-950 text-red-300 border border-red-800/60">
              Critical
            </span>
          </div>
          <div className="text-xs font-bold text-slate-200">
            Memory & Pod Contention
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            Nears maximum memory capacity with flapping node events.
          </p>
        </button>

        {/* Scenario 4: Calm Recovery */}
        <button
          onClick={() => onResetToScenario('calm')}
          disabled={loading}
          className="p-3.5 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-teal-500/50 text-left transition group disabled:opacity-50"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="p-1.5 rounded-lg bg-teal-500/10 text-teal-400 group-hover:scale-110 transition">
              <RotateCcw className="w-4 h-4" />
            </span>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-800/60">
              Healthy
            </span>
          </div>
          <div className="text-xs font-bold text-slate-200">
            Restore Calm Cluster
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            Resets all nodes to healthy baseline capacities, 100% health, and 0 failures.
          </p>
        </button>
      </div>
    </div>
  );
}
