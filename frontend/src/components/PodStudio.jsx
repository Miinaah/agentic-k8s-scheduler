import React, { useState } from 'react';
import {
  Sparkles,
  Send,
  Play,
  Sliders,
  Cpu,
  Database,
  Clock,
  ShieldAlert,
  CheckCircle,
  BrainCircuit,
  Bot
} from 'lucide-react';

export default function PodStudio({
  podTemplates = [],
  onEvaluate,
  onApply,
  loading = false
}) {
  const [selectedTemplateId, setSelectedTemplateId] = useState(podTemplates[0]?.id || 'critical-db');

  // Custom form state
  const [podId, setPodId] = useState('pod-payment-service-critical');
  const [cpuReq, setCpuReq] = useState(2.0);
  const [memReq, setMemReq] = useState(6.0);
  const [priorityClass, setPriorityClass] = useState('high');
  const [qosClass, setQosClass] = useState('Guaranteed');
  const [durationSec, setDurationSec] = useState(3600);

  // Engine controls
  const [scorer, setScorer] = useState('heuristic'); // 'heuristic' or 'gnn'
  const [useLlm, setUseLlm] = useState(false);
  const [epsilon, setEpsilon] = useState(0.05);

  const handleSelectTemplate = (tmpl) => {
    setSelectedTemplateId(tmpl.id);
    setPodId(`pod-${tmpl.id}-${Math.floor(Math.random() * 900 + 100)}`);
    setCpuReq(tmpl.cpu_request);
    setMemReq(tmpl.mem_request_gb);
    setPriorityClass(tmpl.priority_class);
    setQosClass(tmpl.qos_class);
    setDurationSec(tmpl.estimated_duration_sec);
  };

  const getPodPayload = () => ({
    pod_id: podId.trim() || `pod-${Date.now()}`,
    cpu_request: parseFloat(cpuReq),
    mem_request_gb: parseFloat(memReq),
    priority_class: priorityClass,
    qos_class: qosClass,
    estimated_duration_sec: parseFloat(durationSec),
  });

  const handleRunEvaluation = () => {
    onEvaluate({
      use_llm: useLlm,
      scorer,
      epsilon: parseFloat(epsilon),
      pod: getPodPayload(),
    });
  };

  const handleRunApply = () => {
    onApply({
      use_llm: useLlm,
      scorer,
      epsilon: parseFloat(epsilon),
      pod: getPodPayload(),
    });
  };

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-5">
      {/* Title */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-teal-400" />
            Pod Workload & Decision Studio
          </h2>
          <p className="text-xs text-slate-400">
            Configure incoming workload requirements, select scoring model, and invoke agentic scheduling
          </p>
        </div>
      </div>

      {/* Preset Workload Templates */}
      <div>
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
          Preset Workload Templates
        </label>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
          {podTemplates.map((tmpl) => {
            const isSelected = selectedTemplateId === tmpl.id;
            return (
              <button
                key={tmpl.id}
                onClick={() => handleSelectTemplate(tmpl)}
                className={`p-2.5 rounded-xl text-left border transition-all ${
                  isSelected
                    ? 'bg-teal-950/60 border-teal-500/80 text-teal-200 glow-teal'
                    : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="text-xs font-semibold truncate">{tmpl.name}</div>
                <div className="text-[10px] text-slate-400 mt-1 font-mono">
                  {tmpl.cpu_request} CPU &bull; {tmpl.mem_request_gb} GB
                </div>
                <div className="text-[10px] text-teal-400/80 mt-0.5 font-medium">
                  {tmpl.qos_class}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Pod Specification Form */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3 p-4 rounded-xl bg-slate-900/50 border border-slate-800">
        {/* Pod ID */}
        <div className="lg:col-span-2">
          <label className="block text-xs text-slate-400 mb-1">Pod Identifier</label>
          <input
            type="text"
            value={podId}
            onChange={(e) => setPodId(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono focus:ring-1 focus:ring-teal-500 focus:outline-none"
          />
        </div>

        {/* CPU Request */}
        <div>
          <label className="block text-xs text-slate-400 mb-1 flex items-center gap-1">
            <Cpu className="w-3 h-3 text-cyan-400" /> CPU (cores)
          </label>
          <input
            type="number"
            step="0.5"
            min="0.1"
            value={cpuReq}
            onChange={(e) => setCpuReq(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono focus:ring-1 focus:ring-teal-500 focus:outline-none"
          />
        </div>

        {/* Mem Request */}
        <div>
          <label className="block text-xs text-slate-400 mb-1 flex items-center gap-1">
            <Database className="w-3 h-3 text-indigo-400" /> RAM (GB)
          </label>
          <input
            type="number"
            step="0.5"
            min="0.1"
            value={memReq}
            onChange={(e) => setMemReq(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono focus:ring-1 focus:ring-teal-500 focus:outline-none"
          />
        </div>

        {/* Priority Class */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Priority Class</label>
          <select
            value={priorityClass}
            onChange={(e) => setPriorityClass(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:ring-1 focus:ring-teal-500 focus:outline-none cursor-pointer"
          >
            <option value="high">High</option>
            <option value="standard">Standard</option>
            <option value="low">Low</option>
          </select>
        </div>

        {/* QoS Class */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">QoS Class</label>
          <select
            value={qosClass}
            onChange={(e) => setQosClass(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:ring-1 focus:ring-teal-500 focus:outline-none cursor-pointer"
          >
            <option value="Guaranteed">Guaranteed</option>
            <option value="Burstable">Burstable</option>
            <option value="BestEffort">BestEffort</option>
          </select>
        </div>
      </div>

      {/* Scheduler Policy & Engine Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
        {/* Scorer Selection */}
        <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
          <label className="block text-xs font-semibold text-slate-300 mb-2">
            1. Candidate Scoring Engine
          </label>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <button
              type="button"
              onClick={() => setScorer('heuristic')}
              className={`p-2 rounded-lg border text-center font-medium transition ${
                scorer === 'heuristic'
                  ? 'bg-teal-950 border-teal-500 text-teal-300'
                  : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              Heuristic Scorer
              <div className="text-[10px] text-slate-400 font-normal mt-0.5">Deterministic weights</div>
            </button>
            <button
              type="button"
              onClick={() => setScorer('gnn')}
              className={`p-2 rounded-lg border text-center font-medium transition ${
                scorer === 'gnn'
                  ? 'bg-teal-950 border-teal-500 text-teal-300 glow-teal'
                  : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              GNN SAGE Model
              <div className="text-[10px] text-slate-400 font-normal mt-0.5">Graph Neural Network</div>
            </button>
          </div>
        </div>

        {/* Priority Arbiter */}
        <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
          <label className="block text-xs font-semibold text-slate-300 mb-2 flex items-center justify-between">
            <span>2. Priority Order Arbiter</span>
            {useLlm && <span className="text-[10px] text-teal-400 font-mono">LLM Enabled</span>}
          </label>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <button
              type="button"
              onClick={() => setUseLlm(false)}
              className={`p-2 rounded-lg border text-center font-medium transition ${
                !useLlm
                  ? 'bg-teal-950 border-teal-500 text-teal-300'
                  : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              Rule-Based
              <div className="text-[10px] text-slate-400 font-normal mt-0.5">Stress tier mapping</div>
            </button>
            <button
              type="button"
              onClick={() => setUseLlm(true)}
              className={`p-2 rounded-lg border text-center font-medium transition ${
                useLlm
                  ? 'bg-indigo-950 border-indigo-500 text-indigo-300 glow-indigo'
                  : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              LLM Agentic
              <div className="text-[10px] text-slate-400 font-normal mt-0.5">Anthropic Reasoning</div>
            </button>
          </div>
        </div>

        {/* Tolerance Epsilon */}
        <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300 mb-2">
            <span>3. Lexicographic Tolerance (ε)</span>
            <span className="font-mono text-teal-400">{epsilon}</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.20"
            step="0.01"
            value={epsilon}
            onChange={(e) => setEpsilon(e.target.value)}
            className="w-full accent-teal-500 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>0.01 (Strict)</span>
            <span>0.05 (Standard)</span>
            <span>0.20 (Broad)</span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-end gap-3 pt-2">
        <button
          onClick={handleRunEvaluation}
          disabled={loading}
          className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 hover:text-white text-xs font-semibold flex items-center justify-center gap-2 transition disabled:opacity-50"
        >
          <Play className="w-4 h-4 text-teal-400" />
          <span>Simulate Decision (Dry Run)</span>
        </button>

        <button
          onClick={handleRunApply}
          disabled={loading}
          className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-gradient-to-r from-teal-500 to-emerald-600 hover:from-teal-400 hover:to-emerald-500 text-slate-950 text-xs font-bold flex items-center justify-center gap-2 shadow-lg shadow-teal-500/25 transition disabled:opacity-50"
        >
          <Send className="w-4 h-4 fill-current" />
          <span>Schedule & Bind Pod</span>
        </button>
      </div>
    </div>
  );
}
