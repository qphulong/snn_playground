import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useQuery } from '@tanstack/react-query';
import { fetchWeightRegion, fetchModelInfo } from '../../services/api';
import { registerVisualization } from './VisualizationRegistry';

const PAGE = 200; // neurons per viewport tile

export default function WeightHeatmap() {
  const svgRef = useRef(null);
  const [synapse, setSynapse] = useState(null);
  const [viewport, setViewport] = useState({ r0: 0, r1: PAGE, c0: 0, c1: PAGE });

  const { data: modelInfo } = useQuery({ queryKey: ['modelInfo'], queryFn: fetchModelInfo });

  const synapseNames = modelInfo?.synapses?.map(s => s.name) || [];

  useEffect(() => {
    if (synapseNames.length > 0 && !synapse) setSynapse(synapseNames[0]);
  }, [synapseNames, synapse]);

  const { data: region } = useQuery({
    queryKey: ['weightRegion', synapse, viewport],
    queryFn: () => fetchWeightRegion(synapse, viewport.r0, viewport.r1, viewport.c0, viewport.c1),
    enabled: !!synapse,
  });

  useEffect(() => {
    if (!region || !svgRef.current) return;
    const matrix = region.region;
    const rows = matrix.length;
    const cols = matrix[0]?.length || 0;
    if (rows === 0 || cols === 0) return;

    const W = 600, H = 500;
    const cellW = W / cols, cellH = H / rows;

    const flat = matrix.flat();
    const colorScale = d3.scaleSequential(d3.interpolateInferno)
      .domain([d3.min(flat), d3.max(flat)]);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', W).attr('height', H);

    matrix.forEach((row, ri) => {
      row.forEach((val, ci) => {
        svg.append('rect')
          .attr('x', ci * cellW)
          .attr('y', ri * cellH)
          .attr('width', cellW)
          .attr('height', cellH)
          .attr('fill', colorScale(val));
      });
    });
  }, [region]);

  const move = (dRow, dCol) => {
    const shape = region?.full_shape || [900, 900];
    setViewport(v => ({
      r0: Math.max(0, Math.min(v.r0 + dRow * PAGE, shape[0] - PAGE)),
      r1: Math.max(PAGE, Math.min(v.r1 + dRow * PAGE, shape[0])),
      c0: Math.max(0, Math.min(v.c0 + dCol * PAGE, shape[1] - PAGE)),
      c1: Math.max(PAGE, Math.min(v.c1 + dCol * PAGE, shape[1])),
    }));
  };

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <label>Synapse: </label>
        <select value={synapse || ''} onChange={e => setSynapse(e.target.value)}>
          {synapseNames.map(n => <option key={n} value={n}>{n}</option>)}
        </select>
        {region && (
          <span style={{ marginLeft: 16, color: '#666', fontSize: 12 }}>
            rows {viewport.r0}–{viewport.r1} / cols {viewport.c0}–{viewport.c1}
            {' '}(full shape: {region.full_shape?.join('×')})
          </span>
        )}
      </div>

      <svg ref={svgRef} style={{ border: '1px solid #ccc' }} />

      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button onClick={() => move(-1, 0)}>↑ rows</button>
        <button onClick={() => move(1, 0)}>↓ rows</button>
        <button onClick={() => move(0, -1)}>← cols</button>
        <button onClick={() => move(0, 1)}>→ cols</button>
      </div>

      {region?.stats && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#555' }}>
          Region stats — mean: {region.stats.mean?.toFixed(4)} |
          std: {region.stats.std?.toFixed(4)} |
          min: {region.stats.min?.toFixed(4)} |
          max: {region.stats.max?.toFixed(4)}
        </div>
      )}
    </div>
  );
}

registerVisualization('weight_heatmap', {
  component: WeightHeatmap,
  label: 'Weight Heatmap',
  description: 'Paged view of synapse weight matrix. Navigate to load different regions.',
});
