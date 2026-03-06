import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchTrainingStatus, fetchCheckpoints, loadCheckpoint, loadExperiment } from '../../services/api';

export default function TrainingPanel() {
  const qc = useQueryClient();
  const [configPath, setConfigPath] = useState('config/experiments/exp_stdp_baseline.yaml');

  const { data: status }    = useQuery({ queryKey: ['trainingStatus'], queryFn: fetchTrainingStatus, refetchInterval: 3000 });
  const { data: ckptList }  = useQuery({ queryKey: ['checkpoints'],   queryFn: fetchCheckpoints });

  const loadExpMut = useMutation({
    mutationFn: loadExperiment,
    onSuccess: () => qc.invalidateQueries(),
  });

  const loadCkptMut = useMutation({
    mutationFn: loadCheckpoint,
    onSuccess: () => qc.invalidateQueries(),
  });

  return (
    <div>
      {/* Status */}
      <div style={{ marginBottom: 16 }}>
        <strong>Status: </strong>
        <span style={{ color: status?.is_training ? '#22c55e' : '#94a3b8' }}>
          {status?.is_training ? 'Training' : 'Idle'}
        </span>
        {status?.samples_trained != null && (
          <span style={{ marginLeft: 12, fontSize: 13, color: '#666' }}>
            {status.samples_trained} samples trained
          </span>
        )}
      </div>

      {/* Load experiment */}
      <div style={{ marginBottom: 16 }}>
        <h4 style={{ margin: '0 0 6px' }}>Load Experiment Config</h4>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            value={configPath}
            onChange={e => setConfigPath(e.target.value)}
            style={{ flex: 1, padding: '4px 8px', fontSize: 13 }}
          />
          <button onClick={() => loadExpMut.mutate(configPath)} disabled={loadExpMut.isPending}>
            {loadExpMut.isPending ? 'Loading...' : 'Load'}
          </button>
        </div>
        {loadExpMut.isSuccess && <p style={{ color: '#22c55e', fontSize: 13 }}>Loaded!</p>}
        {loadExpMut.isError   && <p style={{ color: 'red', fontSize: 13 }}>{loadExpMut.error.message}</p>}
      </div>

      {/* Checkpoint list */}
      <div>
        <h4 style={{ margin: '0 0 6px' }}>Checkpoints</h4>
        {!ckptList?.checkpoints?.length && <p style={{ color: '#888', fontSize: 13 }}>No checkpoints yet.</p>}
        <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#f5f5f5' }}>
              <th style={th}>ID</th><th style={th}>Samples</th><th style={th}>Time</th><th style={th} />
            </tr>
          </thead>
          <tbody>
            {(ckptList?.checkpoints || []).map(c => (
              <tr key={c.id} style={{ background: status?.current_checkpoint === c.id ? '#eff6ff' : '' }}>
                <td style={td}><code style={{ fontSize: 11 }}>{c.id}</code></td>
                <td style={td}>{c.sample_idx}</td>
                <td style={td}>{c.timestamp ? new Date(c.timestamp * 1000).toLocaleString() : '—'}</td>
                <td style={td}>
                  <button style={{ fontSize: 12 }} onClick={() => loadCkptMut.mutate(c.id)}>
                    Load
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const th = { padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #ddd', fontWeight: 600 };
const td = { padding: '5px 8px', borderBottom: '1px solid #eee' };
