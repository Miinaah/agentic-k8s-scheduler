import React from 'react';
import {
  Sparkles,
  ShieldCheck,
  ShieldAlert,
  Zap,
  Clock,
  CheckCircle2,
  XCircle,
  ChevronRight,
  Filter,
  BrainCircuit,
  BarChart3,
  Scale,
  Award
} from 'lucide-react';

export default function DecisionPipeline({ decisionData, loading = false }) {
  if (loading) {
    return (
      <div className="glass-panel rounded-2xl p-8 border border-slate-800 text-center space-y-3">
        <div className="inline-block p-3 rounded-full bg-teal-500/10 text-teal-400 animate-spin">
          <BrainCircuit className="w-8 h-8" />
        </div>
        <div className="text-base font-semibold text-white">
          Executing Multi-Objective Decision Pipeline...
        </div>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          Evaluating cluster stress, running GNN / heuristic node scoring, arbitrating priority objectives, and filtering lexicographically with epsilon tolerance.
        </p>
      </div>
    );
  }

  if (!decisionData) {
    return (
      <div className="glass-panel rounded-2xl p-8 border border-slate-800 text-center space-y-2">
        <BrainCircuit className="w-8 h-8 text-slate-600 mx-auto" />
        <div className="text-sm font-medium text-slate-300">
          No Decision Evaluated Yet
        </div>
        <p className="text-xs text-slate-500 max-w-sm mx-auto">
          Configure a pod workload above and click "Simulate Decision" or "Schedule & Bind Pod" to watch the agentic decision lifecycle.
        </p>
      </div>
    );
  }

  const {
    pod_id,
    selected_node_id,
    decision_reason,
    latency_ms,
    scorer_used,
    scorer_warning,
    stress,
    priority,
    scores = [],
    feasibility = [],
    elimination_trace = [],
  } = decisionData;

  const stressVal = stress?.value ?? 0;
  const stressPercent = Math.min(100, Math.round(stressVal * 100));

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Award className="w-5 h-5 text-teal-400" />
              Scheduling Decision Trace
            </h2>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-teal-950/80 text-teal-300 border border-teal-500/40 font-medium">
              Pod: {pod_id}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Full audit log of cluster stress analysis, multi-objective scoring, agentic priority ranking, and node elimination.
          </p>
        </div>

        {/* Latency & Scorer Badges */}
        <div className="flex items-center gap-2">
          <div className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-700/80 text-xs font-mono text-slate-300 flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-teal-400" />
            <span>{latency_ms.toFixed(1)} ms</span>
          </div>
          <div className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-700/80 text-xs font-mono text-teal-300">
            Engine: {scorer_used?.toUpperCase() || 'HEURISTIC'}
          </div>
        </div>
      </div>

      {/* Warning alert if GNN fell back to heuristic */}
      {scorer_warning && (
        <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-200 text-xs flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400 shrink-0" />
          <span>{scorer_warning}</span>
        </div>
      )}

      {/* Selected Winner Banner */}
      <div className={`p-4 rounded-xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-3 ${
        selected_node_id
          ? 'bg-gradient-to-r from-teal-950/60 to-slate-900 border-teal-500/60 glow-teal'
          : 'bg-rose-950/40 border-rose-800 text-rose-200'
      }`}>
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-teal-500 text-slate-950">
            <CheckCircle2 className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <div className="text-xs uppercase font-semibold tracking-wider text-teal-400">
              Selected Target Node
            </div>
            <div className="text-lg font-bold font-mono text-white">
              {selected_node_id || 'NO FEASIBLE NODE'}
            </div>
            <div className="text-xs text-slate-300 mt-0.5">
              {decision_reason}
            </div>
          </div>
        </div>
      </div>

      {/* 5-Stage Step Visualizer Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Stage 1: Cluster Stress Assessment */}
        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-teal-500/20 text-teal-400 inline-flex items-center justify-center text-xs font-mono font-bold">1</span>
              Cluster Stress Metric
            </h3>
            <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full uppercase ${
              stress?.level === 'high'
                ? 'bg-red-950 text-red-300 border border-red-800'
                : stress?.level === 'medium'
                ? 'bg-amber-950 text-amber-300 border border-amber-800'
                : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
            }`}>
              {stress?.level || 'low'} Stress
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">Composite Stress Score:</span>
              <span className="text-white font-bold">{stressVal.toFixed(3)} / 1.000</span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  stressPercent >= 50 ? 'bg-red-500' : stressPercent >= 25 ? 'bg-amber-400' : 'bg-emerald-400'
                }`}
                style={{ width: `${stressPercent}%` }}
              />
            </div>
          </div>

          {/* Factor Breakdown */}
          {stress?.breakdown && (
            <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
              <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                <div className="text-slate-400">Avg Resource Load (40%)</div>
                <div className="font-mono font-semibold text-slate-200">
                  {(stress.breakdown.avg_utilisation * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                <div className="text-slate-400">Recent Failures (35%)</div>
                <div className="font-mono font-semibold text-slate-200">
                  {stress.breakdown.recent_failures_total} events
                </div>
              </div>
              <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                <div className="text-slate-400">Queue Backlog (15%)</div>
                <div className="font-mono font-semibold text-slate-200">
                  {stress.breakdown.queue_length} pending
                </div>
              </div>
              <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                <div className="text-slate-400">Load Variance (10%)</div>
                <div className="font-mono font-semibold text-slate-200">
                  {stress.breakdown.utilisation_variance.toFixed(4)}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Stage 2: Priority Ordering & Agentic Arbiter */}
        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-teal-500/20 text-teal-400 inline-flex items-center justify-center text-xs font-mono font-bold">2</span>
              Priority Order Arbiter
            </h3>
            <span className="text-[11px] px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-700/50 font-mono">
              Source: {priority?.source?.toUpperCase() || 'RULE'}
            </span>
          </div>

          {/* Priority Ranking Badges */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {priority?.order?.map((obj, idx) => (
              <React.Fragment key={obj}>
                <div className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center gap-1 ${
                  idx === 0
                    ? 'bg-teal-500/20 text-teal-300 border border-teal-500/50'
                    : idx === 1
                    ? 'bg-slate-800 text-slate-300 border border-slate-700'
                    : 'bg-slate-900 text-slate-400 border border-slate-800'
                }`}>
                  <span className="font-mono text-[10px] text-teal-400">#{idx + 1}</span>
                  <span className="capitalize">{obj.replace('_', ' ')}</span>
                </div>
                {idx < (priority?.order?.length - 1) && (
                  <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Justification Box */}
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/60 text-xs text-slate-300 space-y-1">
            <div className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
              Reasoning Rationale & Confidence: {Math.round((priority?.confidence || 0.9) * 100)}%
            </div>
            <p className="italic text-slate-300 leading-relaxed">
              "{priority?.justification}"
            </p>
          </div>
        </div>

      </div>

      {/* Stage 3: Multi-Objective Candidate Scoring Table */}
      <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
          <span className="w-5 h-5 rounded-full bg-teal-500/20 text-teal-400 inline-flex items-center justify-center text-xs font-mono font-bold">3</span>
          Multi-Objective Candidate Scores (Range: [0, 1])
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                <th className="py-2 px-3">Node ID</th>
                <th className="py-2 px-3">Fault Tolerance (FT)</th>
                <th className="py-2 px-3">Utilisation (Util)</th>
                <th className="py-2 px-3">Cost Efficiency</th>
                <th className="py-2 px-3 text-right">Feasibility</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {scores.map((s) => {
                const isSelected = selected_node_id === s.node_id;
                const nodeFeas = feasibility.find(f => f.node_id === s.node_id);
                return (
                  <tr
                    key={s.node_id}
                    className={`transition ${isSelected ? 'bg-teal-950/40 text-teal-200' : 'hover:bg-slate-800/40'}`}
                  >
                    <td className="py-2 px-3 font-semibold text-white flex items-center gap-1.5">
                      {isSelected && <Sparkles className="w-3.5 h-3.5 text-teal-400 fill-current" />}
                      {s.node_id}
                    </td>
                    <td className="py-2 px-3 text-emerald-400">
                      {s.fault_tolerance.toFixed(4)}
                    </td>
                    <td className="py-2 px-3 text-cyan-400">
                      {s.utilisation.toFixed(4)}
                    </td>
                    <td className="py-2 px-3 text-amber-400">
                      {s.cost.toFixed(4)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {nodeFeas?.feasible ? (
                        <span className="text-emerald-400 font-sans inline-flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" /> Eligible
                        </span>
                      ) : (
                        <span className="text-rose-400 font-sans inline-flex items-center gap-1">
                          <XCircle className="w-3 h-3" /> Ineligible
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Stage 4: Lexicographic Elimination Trace */}
      {elimination_trace && elimination_trace.length > 0 && (
        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <span className="w-5 h-5 rounded-full bg-teal-500/20 text-teal-400 inline-flex items-center justify-center text-xs font-mono font-bold">4</span>
            Lexicographic Filtering with Tolerance (ε) Trace
          </h3>

          <div className="space-y-3">
            {elimination_trace.map((step) => (
              <div
                key={step.step}
                className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60 text-xs space-y-2"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-slate-300">
                  <div className="font-semibold flex items-center gap-2">
                    <span className="px-1.5 py-0.5 rounded bg-slate-700 text-teal-300 font-mono text-[10px]">
                      Step {step.step}
                    </span>
                    <span>Objective: <strong className="text-teal-300 capitalize">{step.objective.replace('_', ' ')}</strong></span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-400">
                    Max Score: <span className="text-white font-bold">{step.max_score}</span> &bull;
                    Cutoff (Max - ε): <span className="text-amber-300 font-bold">{step.cutoff_threshold}</span>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <span className="text-[11px] text-slate-400">Surviving Nodes ({step.surviving_count}):</span>
                  {step.survivor_node_ids.map((id) => (
                    <span key={id} className="px-2 py-0.5 rounded bg-teal-950/80 border border-teal-500/50 text-teal-200 font-mono text-[11px]">
                      {id}
                    </span>
                  ))}

                  {step.eliminated && step.eliminated.length > 0 && (
                    <>
                      <span className="text-[11px] text-slate-400 ml-2">Eliminated:</span>
                      {step.eliminated.map((el) => (
                        <span key={el.node_id} className="px-2 py-0.5 rounded bg-rose-950/60 border border-rose-800 text-rose-300 font-mono text-[11px] line-through">
                          {el.node_id} ({el.score})
                        </span>
                      ))}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
