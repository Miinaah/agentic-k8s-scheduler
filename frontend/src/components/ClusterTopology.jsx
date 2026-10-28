import React from 'react';
import NodeCard from './NodeCard';
import { Globe, Server, CheckCircle2, AlertOctagon } from 'lucide-react';

export default function ClusterTopology({
  nodes = [],
  winnerNodeId = null,
  candidateScores = [],
  feasibilityResults = [],
  onUpdateMetrics
}) {
  // Group nodes by availability zone
  const zonesMap = nodes.reduce((acc, node) => {
    const z = node.zone || 'unknown-zone';
    if (!acc[z]) acc[z] = [];
    acc[z].push(node);
    return acc;
  }, {});

  const scoresById = candidateScores.reduce((acc, s) => {
    acc[s.node_id] = s;
    return acc;
  }, {});

  const feasibilityById = feasibilityResults.reduce((acc, f) => {
    acc[f.node_id] = f;
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Server className="w-5 h-5 text-teal-400" />
            Cluster Node Topology & Health
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time multi-zone nodes, capacity reservations, health scores, and dynamic fault injection
          </p>
        </div>
      </div>

      {Object.entries(zonesMap).map(([zoneName, zoneNodes]) => {
        const zoneCapCpu = zoneNodes.reduce((sum, n) => sum + n.cpu_capacity, 0);
        const zoneUsedCpu = zoneNodes.reduce((sum, n) => sum + n.cpu_used, 0);
        const zoneCapMem = zoneNodes.reduce((sum, n) => sum + n.mem_capacity_gb, 0);
        const zoneUsedMem = zoneNodes.reduce((sum, n) => sum + n.mem_used_gb, 0);
        const avgHealth = zoneNodes.reduce((sum, n) => sum + n.health_score, 0) / zoneNodes.length;

        const isDegraded = avgHealth < 0.85;

        return (
          <div
            key={zoneName}
            className={`glass-panel rounded-2xl p-4 border transition-all ${
              isDegraded ? 'border-rose-900/40 bg-rose-950/10' : 'border-slate-800'
            }`}
          >
            {/* Zone Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-3 mb-4 border-b border-slate-800/80 gap-2">
              <div className="flex items-center gap-2.5">
                <div className={`p-1.5 rounded-lg ${isDegraded ? 'bg-rose-500/20 text-rose-400' : 'bg-teal-500/10 text-teal-400'}`}>
                  <Globe className="w-4 h-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-bold text-slate-100">
                      Zone: {zoneName}
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                      {zoneNodes.length} node{zoneNodes.length > 1 ? 's' : ''}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Allocated: {zoneUsedCpu.toFixed(1)}/{zoneCapCpu.toFixed(1)} CPU &bull; {zoneUsedMem.toFixed(1)}/{zoneCapMem.toFixed(1)} GB RAM
                  </div>
                </div>
              </div>

              {/* Zone health status */}
              <div className="flex items-center gap-2 text-xs">
                {isDegraded ? (
                  <span className="flex items-center gap-1 text-rose-400 font-medium px-2.5 py-1 rounded-full bg-rose-950 border border-rose-800">
                    <AlertOctagon className="w-3.5 h-3.5" /> Degraded Zone
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-emerald-400 font-medium px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/60">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Healthy Domain
                  </span>
                )}
              </div>
            </div>

            {/* Nodes in this Zone */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {zoneNodes.map((node) => (
                <NodeCard
                  key={node.node_id}
                  node={node}
                  isWinner={winnerNodeId === node.node_id}
                  candidateScore={scoresById[node.node_id]}
                  feasibility={feasibilityById[node.node_id]}
                  onUpdateMetrics={onUpdateMetrics}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
