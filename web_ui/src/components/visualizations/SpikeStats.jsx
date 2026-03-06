import { useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { fetchHistory, fetchModelInfo } from '../../services/api';
import { registerVisualization } from './VisualizationRegistry';

export default function SpikeStats() {
  const [layer, setLayer] = useState(null);

  const { data: modelInfo } = useQuery({ queryKey: ['modelInfo'], queryFn: fetchModelInfo });
  const { data: history, isLoading } = useQuery({
    queryKey: ['history'],
    queryFn: fetchHistory,
    refetchInterval: 5000,
    retry: false,
  });

  // All layers except input (index 0) — those have full spike recording
  const layerNames  = modelInfo?.layers?.slice(1).map(l => l.name) || [];
  const activeLayer = layer || layerNames[0] || null;

  const key    = activeLayer ? `${activeLayer}/num_spikes` : null;
  const values = (key && history?.metrics?.[key]) || [];
  const chartData = values.map((v, i) => ({ sample: i, spikes: v }));

  const noDataYet = !isLoading && (!history || values.length === 0);

  return (
    <div>
      <div style={{ marginBottom: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
        <div>
          <label style={{ fontSize: 13 }}>Layer: </label>
          <select value={activeLayer || ''} onChange={e => setLayer(e.target.value)} style={{ fontSize: 13 }}>
            {layerNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        {history && (
          <span style={{ fontSize: 12, color: '#888' }}>
            {history.num_samples} samples trained
          </span>
        )}
      </div>

      {isLoading && <p style={{ color: '#94a3b8', fontSize: 13 }}>Loading...</p>}

      {noDataYet && (
        <div style={{
          padding: 20, textAlign: 'center', color: '#94a3b8',
          border: '1px dashed #e5e7eb', borderRadius: 8,
        }}>
          <p style={{ margin: 0, fontSize: 14 }}>No spike data yet.</p>
          <p style={{ margin: '6px 0 0', fontSize: 12 }}>
            Run training first:<br />
            <code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 3 }}>
              python scripts/train.py --config config/experiments/exp_stdp_baseline.yaml --dataset datasets/vox1_small
            </code>
          </p>
        </div>
      )}

      {chartData.length > 0 && (
        <>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="sample" label={{ value: 'Sample', position: 'insideBottom', offset: -5 }} />
              <YAxis tickFormatter={v => v.toLocaleString()} />
              <Tooltip formatter={v => v.toLocaleString()} />
              <Area type="monotone" dataKey="spikes" stroke="#f97316" fill="#fed7aa" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 12, color: '#555', marginTop: 6, display: 'flex', gap: 20 }}>
            <span>Mean: {Math.round(values.reduce((a, b) => a + b, 0) / values.length).toLocaleString()}</span>
            <span>Max: {Math.max(...values).toLocaleString()}</span>
            <span>Min: {Math.min(...values).toLocaleString()}</span>
          </div>
        </>
      )}
    </div>
  );
}

registerVisualization('spike_stats', {
  component: SpikeStats,
  label: 'Spike Statistics',
  description: 'Spike count per sample per layer. Auto-refreshes every 5s.',
});
