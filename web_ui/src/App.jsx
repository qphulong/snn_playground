import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

import ModelPanel    from './components/panels/ModelPanel';
import TrainingPanel from './components/panels/TrainingPanel';

// Side-effect imports register all built-in visualizations
import { listVisualizations } from './components/visualizations/VisualizationRegistry';
import './components/visualizations/ArchDiagram';
import './components/visualizations/WeightHeatmap';
import './components/visualizations/WeightEvolution';
import './components/visualizations/SpikeStats';

const queryClient = new QueryClient();

const TABS = ['Model Info', 'Training', 'Visualizations'];

function AppInner() {
  const [tab, setTab] = useState('Model Info');
  const [activeVizName, setActiveVizName] = useState(null);

  const vizList = listVisualizations();
  const currentViz = vizList.find(v => v.name === activeVizName) || vizList[0];
  // Must be capitalized for React to treat it as a component
  const VizComponent = currentViz?.component || null;

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto', padding: 16 }}>
      <header style={{ borderBottom: '2px solid #2563eb', paddingBottom: 8, marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: 22, color: '#1e3a5f' }}>SNN Research Playground</h1>
      </header>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '6px 16px',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
              background: tab === t ? '#2563eb' : '#e5e7eb',
              color:  tab === t ? '#fff' : '#374151',
              fontWeight: tab === t ? 600 : 400,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Model Info' && (
        <Card title="Network Architecture">
          <ModelPanel />
        </Card>
      )}

      {tab === 'Training' && (
        <Card title="Training Control">
          <TrainingPanel />
        </Card>
      )}

      {tab === 'Visualizations' && (
        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 16 }}>
          {/* Viz selector */}
          <Card title="Visualizations">
            {vizList.map(v => (
              <button
                key={v.name}
                onClick={() => setActiveVizName(v.name)}
                style={{
                  display: 'block', width: '100%', textAlign: 'left',
                  padding: '8px 10px', marginBottom: 4,
                  background: (activeVizName || vizList[0]?.name) === v.name ? '#eff6ff' : 'transparent',
                  border: '1px solid #ddd', borderRadius: 4, cursor: 'pointer',
                }}
              >
                <strong style={{ fontSize: 13 }}>{v.label}</strong>
                <br />
                <span style={{ fontSize: 11, color: '#666' }}>{v.description}</span>
              </button>
            ))}
          </Card>

          {/* Active visualization — must use a capitalized variable */}
          <Card title={currentViz?.label || 'Select a visualization'}>
            {VizComponent
              ? <VizComponent />
              : <p style={{ color: '#888' }}>Select a visualization from the left.</p>}
          </Card>
        </div>
      )}
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div style={{
      border: '1px solid #e5e7eb', borderRadius: 8, padding: 16,
      background: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,.06)',
    }}>
      {title && <h2 style={{ margin: '0 0 12px', fontSize: 16, color: '#1e3a5f' }}>{title}</h2>}
      {children}
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppInner />
    </QueryClientProvider>
  );
}
