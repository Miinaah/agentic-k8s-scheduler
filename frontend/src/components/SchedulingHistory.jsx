import React from 'react';
import { History, Trash2, Cpu, Database, CheckCircle, Clock } from 'lucide-react';

export default function SchedulingHistory({ history = [], onClearHistory, loading = false }) {
  if (history.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 text-center space-y-2">
        <History className="w-6 h-6 text-slate-600 mx-auto" />
        <div className="text-xs font-semibold text-slate-400">
          No Pods Bound Yet
        </div>
        <p className="text-[11px] text-slate-500">
          Click "Schedule & Bind Pod" to commit workload placements and watch them appear here.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <History className="w-5 h-5 text-teal-400" />
          <h2 className="text-base font-bold text-white">
            Placement Event Feed ({history.length})
          </h2>
        </div>
        <button
          onClick={onClearHistory}
          disabled={loading}
          className="text-xs text-slate-400 hover:text-rose-400 flex items-center gap-1 transition"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Clear Feed</span>
        </button>
      </div>

      <div className="space-y-2.5 max-h-[360px] overflow-y-auto pr-1">
        {history.map((event) => (
          <div
            key={event.event_id}
            className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-2 hover:border-slate-700 transition"
          >
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono font-bold text-white">
                  {event.pod_id}
                </span>
                <span className="px-2 py-0.5 rounded bg-teal-950 border border-teal-500/40 text-teal-300 font-mono text-[11px] font-semibold">
                  ➔ {event.node_id}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-1 text-[11px] text-slate-400">
                <span className="flex items-center gap-0.5">
                  <Cpu className="w-3 h-3 text-cyan-400" /> {event.cpu_request} CPU
                </span>
                <span>&bull;</span>
                <span className="flex items-center gap-0.5">
                  <Database className="w-3 h-3 text-indigo-400" /> {event.mem_request_gb} GB
                </span>
                <span>&bull;</span>
                <span className="text-slate-300">
                  {event.qos_class} ({event.priority_class})
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono shrink-0">
              {event.decision?.latency_ms !== undefined && (
                <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 flex items-center gap-1">
                  <Clock className="w-3 h-3 text-teal-400" />
                  {event.decision.latency_ms.toFixed(1)}ms
                </span>
              )}
              <span className="text-slate-500">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
