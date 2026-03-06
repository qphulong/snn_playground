import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: BASE_URL });

// ── Model ─────────────────────────────────────────────────────────────────

export const fetchModelInfo = () =>
  api.get('/api/model/info').then(r => r.data);

export const fetchWeightRegion = (synapseName, rowStart, rowEnd, colStart, colEnd) =>
  api.get('/api/model/weights', {
    params: { synapse_name: synapseName, row_start: rowStart, row_end: rowEnd, col_start: colStart, col_end: colEnd },
  }).then(r => r.data);

export const fetchWeightStats = (synapseName) =>
  api.get('/api/model/weights/stats', { params: { synapse_name: synapseName } }).then(r => r.data);

export const fetchConnectivity = (synapseName) =>
  api.get('/api/model/connectivity', { params: { synapse_name: synapseName } }).then(r => r.data);

// ── Metrics ───────────────────────────────────────────────────────────────

export const fetchHistory = (checkpointId) =>
  api.get('/api/metrics/history', { params: checkpointId ? { checkpoint_id: checkpointId } : {} }).then(r => r.data);

export const fetchWeightEvolution = (synapseName, metric = 'mean_w') =>
  api.get('/api/metrics/weight-evolution', { params: { synapse_name: synapseName, metric } }).then(r => r.data);

export const fetchSpikeStats = (layerName) =>
  api.get('/api/metrics/spike-stats', { params: { layer_name: layerName } }).then(r => r.data);

export const fetchSpecificWeight = (synapseName, srcIdx, tgtIdx) =>
  api.get('/api/metrics/specific-weight', {
    params: { synapse_name: synapseName, src_idx: srcIdx, tgt_idx: tgtIdx },
  }).then(r => r.data);

// ── Training ──────────────────────────────────────────────────────────────

export const fetchTrainingStatus = () =>
  api.get('/api/training/status').then(r => r.data);

export const fetchCheckpoints = () =>
  api.get('/api/training/checkpoints').then(r => r.data);

export const loadCheckpoint = (checkpointId) =>
  api.post('/api/training/load-checkpoint', null, { params: { checkpoint_id: checkpointId } }).then(r => r.data);

export const loadExperiment = (configPath) =>
  api.post('/api/training/load-experiment', null, { params: { config_path: configPath } }).then(r => r.data);
