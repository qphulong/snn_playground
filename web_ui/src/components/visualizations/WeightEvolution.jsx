import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { fetchHistory, fetchSpecificWeight, fetchModelInfo } from '../../services/api';
import { registerVisualization } from './VisualizationRegistry';

const AGG_METRICS = ['mean_w', 'std_w', 'delta_w'];
const AGG_COLORS  = { mean_w: '#4e9af1', std_w: '#f4a24d', delta_w: '#e05c5c' };

export default function WeightEvolution() {
  const [synapse,  setSynapse]  = useState(null);
  const [mode,     setMode]     = useState('aggregate'); // 'aggregate' | 'specific'
  const [srcIdx,   setSrcIdx]   = useState(0);
  const [tgtIdx,   setTgtIdx]   = useState(0);
  const [queried,  setQueried]  = useState(null); // {src, tgt} — last fetched pair

  const { data: modelInfo } = useQuery({ queryKey: ['modelInfo'], queryFn: fetchModelInfo });
  const { data: history }   = useQuery({ queryKey: ['history'],   queryFn: fetchHistory, refetchInterval: 5000 });

  const synapseNames  = modelInfo?.synapses?.map(s => s.name) || [];
  const activeSynapse = synapse || synapseNames[0] || null;

  // ── Aggregate chart data ────────────────────────────────────────────────
  const aggMetrics = activeSynapse && history?.metrics
    ? AGG_METRICS.map(m => ({
        key: `${activeSynapse}/${m}`,
        label: m,
        values: history.metrics[`${activeSynapse}/${m}`] || [],
      }))
    : [];

  const maxLen = Math.max(...aggMetrics.map(m => m.values.length), 0);
  const aggData = Array.from({ length: maxLen }, (_, i) => {
    const pt = { sample: i };
    aggMetrics.forEach(m => { pt[m.label] = m.values[i]; });
    return pt;
  });

  // ── Specific weight query ────────────────────────────────────────────────
  const { data: specificData, isFetching: specificLoading, error: specificError, refetch: refetchSpecific } =
    useQuery({
      queryKey: ['specificWeight', activeSynapse, queried?.src, queried?.tgt],
      queryFn: () => fetchSpecificWeight(activeSynapse, queried.src, queried.tgt),
      enabled: !!queried && !!activeSynapse,
      retry: false,
    });

  const specificChartData = specificData
    ? specificData.sample_indices.map((s, i) => ({ sample: s, weight: specificData.values[i] }))
    : [];

  const handleQuery = () => setQueried({ src: Number(srcIdx), tgt: Number(tgtIdx) });

  const maxSrc = (activeSynapse && modelInfo)
    ? (modelInfo.layers.find(l => l.name === modelInfo.synapses.find(s => s.name === activeSynapse)?.src)?.N ?? 899)
    : 899;
  const maxTgt = (activeSynapse && modelInfo)
    ? (modelInfo.layers.find(l => l.name === modelInfo.synapses.find(s => s.name === activeSynapse)?.tgt)?.N ?? 899)
    : 899;

  return (
    <div>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <label style={{ fontSize: 13 }}>Synapse: </label>
          <select value={activeSynapse || ''} onChange={e => setSynapse(e.target.value)} style={{ fontSize: 13 }}>
            {synapseNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>

        <div style={{ display: 'flex', gap: 4 }}>
          {['aggregate', 'specific'].map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                padding: '4px 12px', fontSize: 12,
                background: mode === m ? '#2563eb' : '#e5e7eb',
                color: mode === m ? '#fff' : '#374151',
                border: 'none', borderRadius: 4, cursor: 'pointer',
              }}
            >
              {m === 'aggregate' ? 'Aggregate stats' : 'Specific weight'}
            </button>
          ))}
        </div>
      </div>

      {/* Aggregate mode */}
      {mode === 'aggregate' && (
        <>
          <p style={{ fontSize: 12, color: '#888', margin: '0 0 8px' }}>
            {history ? `${history.num_samples} samples trained` : 'loading...'}
          </p>
          {aggData.length === 0
            ? <p style={{ color: '#94a3b8' }}>No training data yet. Run training first.</p>
            : (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={aggData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sample" label={{ value: 'Sample', position: 'insideBottom', offset: -5 }} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {AGG_METRICS.map(m => (
                    <Line key={m} type="monotone" dataKey={m} stroke={AGG_COLORS[m]} dot={false} strokeWidth={1.5} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )
          }
        </>
      )}

      {/* Specific weight mode */}
      {mode === 'specific' && (
        <div>
          <p style={{ fontSize: 12, color: '#888', margin: '0 0 10px' }}>
            Requires <code>save_weights_history: true</code> in experiment config.
          </p>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
            <label style={{ fontSize: 13 }}>src neuron:</label>
            <input
              type="number" min={0} max={maxSrc - 1} value={srcIdx}
              onChange={e => setSrcIdx(e.target.value)}
              style={{ width: 70, padding: '3px 6px', fontSize: 13 }}
            />
            <label style={{ fontSize: 13 }}>tgt neuron:</label>
            <input
              type="number" min={0} max={maxTgt - 1} value={tgtIdx}
              onChange={e => setTgtIdx(e.target.value)}
              style={{ width: 70, padding: '3px 6px', fontSize: 13 }}
            />
            <button onClick={handleQuery} style={{ padding: '4px 14px', fontSize: 13 }}>
              {specificLoading ? 'Loading…' : 'Plot'}
            </button>
          </div>

          {specificError && (
            <p style={{ color: '#e05c5c', fontSize: 13 }}>
              {specificError.response?.data?.detail || specificError.message}
            </p>
          )}

          {specificChartData.length > 0 && (
            <>
              <p style={{ fontSize: 12, color: '#555', margin: '0 0 6px' }}>
                w[{queried?.src}, {queried?.tgt}] over {specificChartData.length} samples
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={specificChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sample" label={{ value: 'Sample', position: 'insideBottom', offset: -5 }} />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip formatter={v => v.toFixed(5)} />
                  <Line type="monotone" dataKey="weight" stroke="#7c3aed" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </>
          )}

          {queried && !specificLoading && specificChartData.length === 0 && !specificError && (
            <p style={{ color: '#94a3b8', fontSize: 13 }}>No data returned.</p>
          )}
        </div>
      )}
    </div>
  );
}

registerVisualization('weight_evolution', {
  component: WeightEvolution,
  label: 'Weight Evolution',
  description: 'Aggregate training curves or plot a specific weight w[i,j] over time.',
});
