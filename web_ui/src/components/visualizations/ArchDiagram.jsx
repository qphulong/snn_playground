import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useQuery } from '@tanstack/react-query';
import { fetchModelInfo } from '../../services/api';
import { registerVisualization } from './VisualizationRegistry';

const W = 800, H = 480;
const NODE_W = 140, NODE_H = 60;

const LAYER_COLORS = {
  adaptive_lif: '#3b82f6',
  simple_lif:   '#10b981',
  default:      '#8b5cf6',
};

export default function ArchDiagram() {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  const { data: modelInfo, isLoading, error } = useQuery({
    queryKey: ['modelInfo'],
    queryFn: fetchModelInfo,
  });

  useEffect(() => {
    if (!modelInfo || !svgRef.current) return;

    const layers   = modelInfo.layers   || [];
    const synapses = modelInfo.synapses || [];

    // Position layers evenly left→right
    const spacing = W / (layers.length + 1);
    const nodePos = {};
    layers.forEach((l, i) => {
      nodePos[l.name] = { x: spacing * (i + 1), y: H / 2 };
    });

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // ── Arrow marker ────────────────────────────────────────────────
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#94a3b8');

    // ── Synapse edges ────────────────────────────────────────────────
    synapses.forEach(syn => {
      const src = nodePos[syn.src];
      const tgt = nodePos[syn.tgt];
      if (!src || !tgt) return;

      const x1 = src.x + NODE_W / 2;
      const x2 = tgt.x - NODE_W / 2;
      const y  = src.y;

      const edgeG = svg.append('g').style('cursor', 'default');

      edgeG.append('line')
        .attr('x1', x1).attr('y1', y)
        .attr('x2', x2).attr('y2', y)
        .attr('stroke', '#94a3b8')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrow)');

      // Edge label
      edgeG.append('text')
        .attr('x', (x1 + x2) / 2)
        .attr('y', y - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('fill', '#64748b')
        .text(`${syn.connectivity} · ${syn.learning_rule}`);
    });

    // ── Layer nodes (draggable) ──────────────────────────────────────
    layers.forEach(layer => {
      const pos = nodePos[layer.name];
      const color = LAYER_COLORS[layer.type] || LAYER_COLORS.default;

      const g = svg.append('g')
        .attr('transform', `translate(${pos.x - NODE_W / 2}, ${pos.y - NODE_H / 2})`)
        .style('cursor', 'grab');

      // Shadow
      g.append('rect')
        .attr('x', 3).attr('y', 3)
        .attr('width', NODE_W).attr('height', NODE_H)
        .attr('rx', 8).attr('fill', '#00000018');

      // Box
      g.append('rect')
        .attr('width', NODE_W).attr('height', NODE_H)
        .attr('rx', 8)
        .attr('fill', color)
        .attr('fill-opacity', 0.15)
        .attr('stroke', color)
        .attr('stroke-width', 2);

      // Layer name
      g.append('text')
        .attr('x', NODE_W / 2).attr('y', 22)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14).attr('font-weight', 600)
        .attr('fill', color)
        .text(layer.name);

      // Type + N
      g.append('text')
        .attr('x', NODE_W / 2).attr('y', 38)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('fill', '#374151')
        .text(layer.type);

      g.append('text')
        .attr('x', NODE_W / 2).attr('y', 52)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('fill', '#6b7280')
        .text(`${layer.N.toLocaleString()} neurons`);

      // Drag behaviour
      const drag = d3.drag()
        .on('start', function () {
          d3.select(this).style('cursor', 'grabbing');
        })
        .on('drag', function (event) {
          const newX = event.x;
          const newY = event.y;
          d3.select(this).attr('transform', `translate(${newX - NODE_W / 2}, ${newY - NODE_H / 2})`);
          // Redraw edges
          nodePos[layer.name] = { x: newX, y: newY };
          redrawEdges();
        })
        .on('end', function () {
          d3.select(this).style('cursor', 'grab');
        });

      g.call(drag);
    });

    function redrawEdges() {
      svg.selectAll('.dyn-edge').remove();
      synapses.forEach(syn => {
        const src = nodePos[syn.src];
        const tgt = nodePos[syn.tgt];
        if (!src || !tgt) return;
        const x1 = src.x + NODE_W / 2;
        const x2 = tgt.x - NODE_W / 2;
        svg.insert('line', ':first-child')
          .attr('class', 'dyn-edge')
          .attr('x1', x1).attr('y1', src.y)
          .attr('x2', x2).attr('y2', tgt.y)
          .attr('stroke', '#94a3b8')
          .attr('stroke-width', 2)
          .attr('marker-end', 'url(#arrow)');
      });
    }

  }, [modelInfo]);

  if (isLoading) return <p>Loading architecture...</p>;
  if (error)     return <p style={{ color: 'red' }}>API error: {error.message}</p>;
  if (!modelInfo) return <p style={{ color: '#888' }}>No experiment loaded.</p>;

  return (
    <div>
      <p style={{ fontSize: 12, color: '#888', margin: '0 0 8px' }}>
        Drag layer nodes to rearrange. Edges follow automatically.
      </p>
      <svg
        ref={svgRef}
        width={W}
        height={H}
        style={{ border: '1px solid #e5e7eb', borderRadius: 8, background: '#fafafa', width: '100%' }}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
      />
    </div>
  );
}

registerVisualization('arch_diagram', {
  component: ArchDiagram,
  label: 'Architecture Diagram',
  description: 'Network topology. Drag layer nodes to rearrange.',
});
