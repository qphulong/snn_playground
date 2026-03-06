/**
 * Visualization plugin registry.
 *
 * To add a new visualization:
 *   1. Create a component in this directory
 *   2. Import it in App.jsx (side-effect import)
 *   3. Call registerVisualization() at the bottom of that component file
 *   4. It will automatically appear in the Visualizations tab
 */

const registry = {};

export function registerVisualization(name, config) {
  registry[name] = config;
}

export function getVisualization(name) {
  return registry[name] || null;
}

export function listVisualizations() {
  return Object.entries(registry).map(([name, cfg]) => ({ name, ...cfg }));
}
