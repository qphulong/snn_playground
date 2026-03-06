import { useQuery } from '@tanstack/react-query';
import { fetchModelInfo } from '../../services/api';

export default function ModelPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['modelInfo'],
    queryFn: fetchModelInfo,
    refetchInterval: 10000,
  });

  if (isLoading) return <p>Loading model info...</p>;
  if (error)     return <p style={{ color: 'red' }}>Error: {error.message}</p>;
  if (!data)     return <p>No experiment loaded.</p>;

  return (
    <div>
      <h3 style={{ margin: '0 0 8px' }}>{data.config_name}</h3>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 12 }}>{data.description}</p>

      <h4>Layers</h4>
      <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#f5f5f5' }}>
            <th style={th}>Name</th><th style={th}>Type</th><th style={th}>Neurons</th>
          </tr>
        </thead>
        <tbody>
          {data.layers.map(l => (
            <tr key={l.name}>
              <td style={td}>{l.name}</td>
              <td style={td}><code>{l.type}</code></td>
              <td style={td}>{l.N.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4 style={{ marginTop: 16 }}>Synapses</h4>
      <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#f5f5f5' }}>
            <th style={th}>Name</th><th style={th}>Src → Tgt</th>
            <th style={th}>Connectivity</th><th style={th}>Learning Rule</th>
          </tr>
        </thead>
        <tbody>
          {data.synapses.map(s => (
            <tr key={s.name}>
              <td style={td}>{s.name}</td>
              <td style={td}>{s.src} → {s.tgt}</td>
              <td style={td}><code>{s.connectivity}</code></td>
              <td style={td}><code>{s.learning_rule}</code></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const th = { padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ddd', fontWeight: 600 };
const td = { padding: '5px 8px', borderBottom: '1px solid #eee' };
