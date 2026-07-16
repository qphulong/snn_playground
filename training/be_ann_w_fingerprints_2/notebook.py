# CELL 1
# ============================== CONFIG ==============================
import os
os.environ["MALLOC_ARENA_MAX"] = "2"
import os, math, glob, time, random, gc, ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

SEED = 1234
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ---- Ablation flags: all 8 combos expressible (all-False invalid) ----
USE_WEIGHTS     = True     # keep the (3,23,192) STDP weight type-images
USE_INPUT_RATE  = True     # keep the gathered input_activity planes
USE_HIDDEN_RATE = True     # keep the broadcast hidden_activity plane
assert USE_WEIGHTS or USE_INPUT_RATE or USE_HIDDEN_RATE, "at least one channel group must be on"
MODE_TAG = f"w{int(USE_WEIGHTS)}_ir{int(USE_INPUT_RATE)}_hr{int(USE_HIDDEN_RATE)}"

# ---- Geometry ----
N_AUDITORY_CH = 64
N_PER_CHANNEL = 3                      # 192 hidden = 64 channels x 3
N_HIDDEN      = 192
N_INPUT_TYPES = 3                      # sustained / onset / phase
R_EXC_CHANNEL = 11
H_OFF         = 2 * R_EXC_CHANNEL + 1  # 23 tonotopic offsets (-11..+11)
IN_CH         = 7                      # 3 weights + 3 input-rate + 1 hidden-rate
EMBED_DIM     = 256
NEUTRAL_FILL  = math.exp(-30)          # zero-variance fill, neutralized by input BatchNorm

# ---- Kaggle paths ----
INPUT_DIR = "/kaggle/input/datasets/qphulong/vox1and2-fingerprints-tripletstdp"
DEV_GLOB  = os.path.join(INPUT_DIR, "vox2_*_fingerprints.npz")   # train (closed-set)
TEST_GLOB = os.path.join(INPUT_DIR, "vox1_*_fingerprints.npz")   # eval  (open-set)

# ---- PK sampler / batch ----
P_CLASSES  = 32
K_SAMPLES  = 4
BATCH_SIZE = P_CLASSES * K_SAMPLES     # 128
VAL_BATCH  = 64

# ---- Optim / schedule ----
EPOCHS               = 80
LR                   = 0.08
WEIGHT_DECAY         = 5e-4
WARMUP_EPOCHS        = 3
MARGIN_WARMUP_EPOCHS = 15
AAM_M                = 0.2
AAM_S                = 30
GRAD_CLIP            = 5.0
EARLY_STOP_PATIENCE  = 0               # disabled: always run full EPOCHS
NUM_WORKERS          = 0

# ---- Checkpointing ----
RESUME    = True
CKPT_DIR  = "/kaggle/working"
LAST_CKPT = os.path.join(CKPT_DIR, f"last_{MODE_TAG}.pt")
BEST_CKPT = os.path.join(CKPT_DIR, f"best_{MODE_TAG}.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU  = torch.cuda.device_count()
USE_DP = N_GPU > 1
print(f"device={device}  N_GPU={N_GPU}  DataParallel={USE_DP}  MODE_TAG={MODE_TAG}")



# Cell 2
# ================= IN_IDX gather (VERIFY vs _fingerprint_core_2.py) =================
def build_in_idx():
    """
    Fixed gather index, shape (3, 23, 192), co-registering the per-input-neuron rate vector
    input_activity(192,) onto the (type, offset, hidden) layout of the weight type-images.

        IN_IDX[t, o, h] = position in input_activity(192,) for the input neuron of type t
        feeding hidden neuron h at tonotopic offset (o - R_EXC_CHANNEL).

    >>> RECONSTRUCTED FROM THE SPEC PROSE. This is the ONE place to reconcile with the real
    >>> helper. Two assumptions to check:
    >>>   (a) hidden h sits at channel  h // N_PER_CHANNEL
    >>>   (b) input_activity is CHANNEL-MAJOR:  index = channel * N_PER_CHANNEL + type
    >>> If it is type-major instead, use:       index = type * N_AUDITORY_CH + channel
    """
    idx = np.empty((N_INPUT_TYPES, H_OFF, N_HIDDEN), dtype=np.int64)
    for t in range(N_INPUT_TYPES):
        for o in range(H_OFF):
            off = o - R_EXC_CHANNEL                      # -11 .. +11
            for h in range(N_HIDDEN):
                base_ch = h // N_PER_CHANNEL             # hidden -> its own channel (0..63)
                src_ch  = (base_ch + off) % N_AUDITORY_CH
                idx[t, o, h] = src_ch * N_PER_CHANNEL + t    # channel-major
    return idx

IN_IDX = build_in_idx()
assert IN_IDX.shape == (3, 23, 192), IN_IDX.shape
assert 0 <= IN_IDX.min() and IN_IDX.max() < N_HIDDEN
IN_IDX_FLAT = torch.from_numpy(IN_IDX.reshape(-1))       # (3*23*192,)
print("IN_IDX ok:", IN_IDX.shape, "range", int(IN_IDX.min()), int(IN_IDX.max()))



# Cell 3
# ===================== EAGER IN-RAM LOADING (no memmap) =====================
# Preallocate exact-size fp16 buffers, fill shard-by-shard, close each handle before the
# next shard. Nothing cached or memory-mapped across shards. Members read individually so
# metadata scans never touch the big weight arrays.

def list_shards(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no shards match {pattern}")
    return files

def scan_counts(files):
    total, per_shard = 0, []
    for f in files:
        n = len(np.load(f)["person_ids"])   # touches only the small id member
        per_shard.append(n); total += n
    return total, per_shard

def load_split(files, name):
    total, per_shard = scan_counts(files)
    W  = np.empty((total, N_INPUT_TYPES, H_OFF, N_HIDDEN), dtype=np.float16)
    IA = np.empty((total, N_HIDDEN), dtype=np.float16)
    HA = np.empty((total, N_HIDDEN), dtype=np.float16)
    PID = np.empty(total, dtype=object)
    RID = np.empty(total, dtype=object)
    off = 0
    for f, n in zip(files, per_shard):
        z = np.load(f)                                   # lazy zip handle
        W[off:off+n]  = z["weights"].astype(np.float16, copy=False)
        IA[off:off+n] = z["input_activity"].astype(np.float16, copy=False)
        HA[off:off+n] = z["hidden_activity"].astype(np.float16, copy=False)
        PID[off:off+n] = z["person_ids"].astype(str)
        RID[off:off+n] = z["record_ids"].astype(str)
        z.close()                                        # release before next shard
        off += n
    assert off == total
    print(f"[{name}] {total} samples / {len(files)} shards | "
          f"W {W.nbytes/1e9:.2f}GB  IA {IA.nbytes/1e9:.3f}GB  HA {HA.nbytes/1e9:.3f}GB")
    return W, IA, HA, PID, RID

def assert_disjoint(dev_pid, test_pid):
    d, t = set(np.unique(dev_pid)), set(np.unique(test_pid))
    inter = d & t
    assert not inter, f"dev/test speaker overlap: {sorted(inter)[:10]}"
    print(f"[disjoint] dev speakers={len(d)}  test speakers={len(t)}  overlap=0")

def subsample_one_per_session(PID, RID, seed=SEED):
    """<=1 fingerprint per (person_id, record_id): seeded random draw within each session."""
    rng = np.random.default_rng(seed)
    sessions = {}
    for i, (p, r) in enumerate(zip(PID, RID)):
        sessions.setdefault((p, r), []).append(i)
    keep = [rows[rng.integers(len(rows))] if len(rows) > 1 else rows[0]
            for rows in sessions.values()]
    return np.array(sorted(keep), dtype=np.int64)

def encode_labels(pid):
    classes = sorted(set(pid.tolist()))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.fromiter((c2i[p] for p in pid), dtype=np.int64, count=len(pid))
    return y, len(classes), c2i




# Cell 4
# ===================== BANK / AUGMENT / PK SAMPLER =====================
class FingerprintAugment:
    """Applied once per BATCH. Roll is shared across w/ia/ha (gather is relative)."""
    def __call__(self, w, ia, ha):
        if random.random() < 0.5:                            # circular tonotopic roll
            k = random.randint(-2, 2)                         # inclusive both ends
            shift = N_PER_CHANNEL * k
            if shift:
                w  = torch.roll(w,  shifts=shift, dims=-1)
                ia = torch.roll(ia, shifts=shift, dims=-1)
                ha = torch.roll(ha, shifts=shift, dims=-1)
        if random.random() < 0.5:                            # additive gaussian, 1% of own std
            w  = w  + torch.randn_like(w)  * (0.01 * w.std())
            ia = ia + torch.randn_like(ia) * (0.01 * ia.std())
            ha = ha + torch.randn_like(ha) * (0.01 * ha.std())
        if random.random() < 0.5:                            # global amplitude scale
            s = random.uniform(0.95, 1.05)
            w, ia, ha = w * s, ia * s, ha * s
        return w, ia, ha

class FingerprintBank:
    """Batch-vectorized: one index_select + .float() cast per BATCH (not per sample)."""
    def __init__(self, W, IA, HA, y, augment=False):
        self.W  = torch.from_numpy(W)                        # fp16 zero-copy views
        self.IA = torch.from_numpy(IA)
        self.HA = torch.from_numpy(HA)
        self.y  = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.aug = FingerprintAugment() if augment else None

    def batch(self, idx):
        idx_t = idx if torch.is_tensor(idx) else torch.as_tensor(idx, dtype=torch.long)
        w  = self.W.index_select(0, idx_t).float()
        ia = self.IA.index_select(0, idx_t).float()
        ha = self.HA.index_select(0, idx_t).float()
        y  = self.y.index_select(0, idx_t)
        if self.aug is not None:
            w, ia, ha = self.aug(w, ia, ha)
        return w, ia, ha, y

class PKSampler:
    """P classes x K samples per batch, drawn from the FULL class set every batch.
       Sample K with replacement for classes with fewer than K fingerprints."""
    def __init__(self, y, p_classes, k_samples, seed=SEED):
        self.p, self.k = p_classes, k_samples
        self.rng = np.random.default_rng(seed)               # persists across epochs
        self.cls2idx = {}
        for i, c in enumerate(y):
            self.cls2idx.setdefault(int(c), []).append(i)
        self.cls2idx = {c: np.asarray(v) for c, v in self.cls2idx.items()}
        self.classes = np.fromiter(self.cls2idx.keys(), dtype=np.int64)
        self.n = len(y)

    def num_batches(self):
        return max(1, self.n // (self.p * self.k))

    def __iter__(self):
        for _ in range(self.num_batches()):
            chosen = self.rng.choice(self.classes, size=self.p, replace=False)
            batch = []
            for c in chosen:
                pool = self.cls2idx[c]
                batch.extend(self.rng.choice(pool, size=self.k,
                                             replace=len(pool) < self.k).tolist())
            yield torch.as_tensor(batch, dtype=torch.long)

def sequential_batches(n, bs):
    for s in range(0, n, bs):
        yield torch.arange(s, min(s + bs, n), dtype=torch.long)



# Cell 5
# ===================== MODEL =====================
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        st = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=st, padding=1, bias=False)
        self.bn1, self.act1 = nn.BatchNorm2d(cout), nn.PReLU(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if cin != cout or st != (1, 1):
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=st, bias=False),
                                       nn.BatchNorm2d(cout))
        else:
            self.short = nn.Identity()
        self.act2 = nn.PReLU(cout)

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act2(y + self.short(x))

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, att_dim=128):
        super().__init__()
        self.att1 = nn.Conv1d(in_dim, att_dim, 1)
        self.att2 = nn.Conv1d(att_dim, in_dim, 1)

    def forward(self, x):                                    # x: (B, C, L)
        a = torch.softmax(self.att2(torch.tanh(self.att1(x))), dim=-1)
        mu  = (a * x).sum(-1)
        sig = torch.sqrt(((a * x ** 2).sum(-1) - mu ** 2).clamp_min(1e-9))
        return torch.cat([mu, sig], dim=1)                   # (B, 2*C)

class FingerprintCNN(nn.Module):
    def __init__(self, in_ch=7, embed_dim=256,
                 use_weights=True, use_input_rate=True, use_hidden_rate=True,
                 in_idx_flat=None):
        super().__init__()
        self.use_weights, self.use_input_rate, self.use_hidden_rate = \
            use_weights, use_input_rate, use_hidden_rate
        self.register_buffer("in_idx_flat", in_idx_flat.long(), persistent=False)
        self.in_bn = nn.BatchNorm2d(in_ch)                   # standardizes weights vs rates
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(32), nn.PReLU(32))
        self.block1 = ResBlock(32,  64,  (1, 2))             # (B, 64, 23, 96)
        self.block2 = ResBlock(64,  128, (2, 2))             # (B,128, 12, 48)
        self.block3 = ResBlock(128, 256, (2, 2))             # (B,256,  6, 24)
        self.block4 = ResBlock(256, 256, (2, 2))             # (B,256,  3, 12)
        self.pool    = AttentiveStatsPool(256)               # (B,512)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(512, embed_dim, bias=False)
        self.fc_bn   = nn.BatchNorm1d(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def expand(self, w, ia, ha):
        """Compact (w, ia, ha) -> (B,7,23,192). Flag-off groups -> NEUTRAL_FILL constant."""
        B = w.shape[0]
        x = torch.empty(B, IN_CH, H_OFF, N_HIDDEN, device=w.device, dtype=w.dtype)
        x[:, 0:3] = w if self.use_weights else NEUTRAL_FILL
        if self.use_input_rate:
            x[:, 3:6] = ia.index_select(1, self.in_idx_flat).view(B, 3, H_OFF, N_HIDDEN)
        else:
            x[:, 3:6] = NEUTRAL_FILL
        if self.use_hidden_rate:
            x[:, 6:7] = ha.view(B, 1, 1, N_HIDDEN).expand(B, 1, H_OFF, N_HIDDEN)
        else:
            x[:, 6:7] = NEUTRAL_FILL
        return x

    def forward(self, w, ia, ha):
        x = self.in_bn(self.expand(w, ia, ha))
        x = self.stem(x)
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = self.pool(x.flatten(2))                          # (B,512)
        x = self.fc_bn(self.fc(self.dropout(x)))
        return x                                             # (B,256), L2-normalize at eval


# Cell 6
# ===================== AAMSoftmax (ArcFace) =====================
class AAMSoftmax(nn.Module):
    def __init__(self, embed_dim, num_classes, m=0.2, s=30):
        super().__init__()
        self.s = s
        self.W = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_normal_(self.W)
        self.set_margin(m)

    def set_margin(self, m):
        self.m = m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)                      # easy-margin threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt((1 - cos ** 2).clamp_min(1e-9))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)  # easy-margin guard
        one_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1 - one_hot) * cos) * self.s
        return F.cross_entropy(logits, labels)


# Cell 7
# ===================== OPTIMIZER + SCHEDULE =====================
def build_optimizer(model, aam, lr, wd):
    """Weight decay on conv/linear weights (incl. AAM W) only; not BN/PReLU/biases."""
    decay, no_decay = [], []
    for module in (model, aam):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return SGD([{"params": decay,    "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}],
               lr=lr, momentum=0.9, nesterov=True)

def build_scheduler(opt, epochs, warmup):
    warm = LinearLR(opt, start_factor=0.1, total_iters=warmup)
    cos  = CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=1e-5)
    return SequentialLR(opt, [warm, cos], milestones=[warmup])


# Cell 8
# ===================== EVAL: session-free vs leaky, block-wise =====================
@torch.no_grad()
def extract_embeddings(net, bank, n):
    net.eval()
    out = torch.empty(n, EMBED_DIM, dtype=torch.float32)
    pos = 0
    for idx in sequential_batches(n, VAL_BATCH):
        w, ia, ha, _ = bank.batch(idx)
        with torch.autocast("cuda", dtype=torch.float16):
            e = net(w.to(device), ia.to(device), ha.to(device))
        e = F.normalize(e.float(), dim=1).cpu()
        out[pos:pos + e.shape[0]] = e
        pos += e.shape[0]
    return out

def _eer_from_hist(gen_hist, imp_hist):
    g  = gen_hist / max(gen_hist.sum(), 1)
    im = imp_hist / max(imp_hist.sum(), 1)
    frr = np.cumsum(g)               # genuine in bins <= thr -> rejected
    far = 1.0 - np.cumsum(im)        # impostor in bins  > thr -> accepted
    k = int(np.argmin(np.abs(frr - far)))
    return float((frr[k] + far[k]) / 2)

@torch.no_grad()
def retrieval_metrics(embs, labels, record_ids, session_free, block=512, n_bins=20000):
    N = embs.shape[0]
    E   = embs.to(device)
    lab = torch.as_tensor(labels, device=device)
    rec = torch.as_tensor(np.unique(record_ids, return_inverse=True)[1], device=device)
    gen_hist = np.zeros(n_bins); imp_hist = np.zeros(n_bins)
    r1 = r5 = 0; ap_sum = 0.0; valid = 0
    for s in range(0, N, block):
        e    = E[s:s + block]
        sims = e @ E.t()                                     # (b, N)
        b    = sims.shape[0]
        rows = torch.arange(s, s + b, device=device)
        if session_free:
            excl = rec[rows][:, None] == rec[None, :]        # same recording (incl. self)
        else:
            excl = rows[:, None] == torch.arange(N, device=device)[None, :]   # self only
        sims = sims.masked_fill(excl, -2.0)
        same = (lab[rows][:, None] == lab[None, :]) & (~excl)
        order = torch.argsort(sims, dim=1, descending=True)
        same_sorted = torch.gather(same, 1, order)
        r1 += same_sorted[:, 0].sum().item()
        r5 += same_sorted[:, :5].any(dim=1).sum().item()
        rel = same.sum(dim=1)
        for i in range(b):
            rc = int(rel[i])
            if rc == 0:
                continue
            hit_ranks = same_sorted[i].nonzero(as_tuple=False).squeeze(1).float() + 1
            precs = torch.arange(1, rc + 1, device=device).float() / hit_ranks
            ap_sum += float(precs.sum() / rc); valid += 1
        gen_hist += torch.histc(sims[same],              bins=n_bins, min=-1, max=1).cpu().numpy()
        imp_hist += torch.histc(sims[(~same) & (~excl)], bins=n_bins, min=-1, max=1).cpu().numpy()
    return dict(rank1=r1 / max(valid, 1), rank5=r5 / max(valid, 1),
                mAP=ap_sum / max(valid, 1), eer=_eer_from_hist(gen_hist, imp_hist))

def evaluate(net, val_bank, labels, records, has_dup):
    embs = extract_embeddings(net, val_bank, len(labels))
    sf = retrieval_metrics(embs, labels, records, session_free=True)
    lk = retrieval_metrics(embs, labels, records, session_free=False) if has_dup else sf
    return sf, lk


# Cell 9
def mem_report():
    rss = rss_gb()
    if torch.cuda.is_available():
        return (f"RSS={rss:.2f}GB "
                f"gpu_resv={torch.cuda.memory_reserved()/1e9:.2f} "
                f"gpu_alloc={torch.cuda.memory_allocated()/1e9:.2f}")
    return f"RSS={rss:.2f}GB"

# ===================== TRAINING =====================
def rss_gb():
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e9

class _Mallinfo2(ctypes.Structure):
    _fields_ = [(f, ctypes.c_size_t) for f in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd",
                 "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]

try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.mallinfo2.restype = _Mallinfo2
    _libc.mallinfo2.argtypes = []
    def _heap_stats():
        m = _libc.mallinfo2()
        return dict(arena_mb=m.arena / 1e6, in_use_mb=m.uordblks / 1e6,
                    free_retained_mb=m.fordblks / 1e6, mmap_mb=m.hblkhd / 1e6)
except AttributeError:
    _heap_stats = None   # glibc too old for mallinfo2 (pre-2.33)

_prev_heap = {}

def leak_probe(tag):
    """tracemalloc showed only noise (it can't see ATen's CPU tensor storage - that's
    allocated via raw malloc, bypassing Python's pymalloc entirely). mallinfo2() reads
    the real glibc heap: if `in_use_mb` (uordblks, bytes glibc considers actually
    allocated) grows ~1GB/epoch, that's a genuine native leak, not fragmentation
    (fragmentation would show up as `free_retained_mb` growing while `in_use_mb` stays
    flat - and malloc_trim(0) already proved trim doesn't reclaim anything here)."""
    n_tensors = sum(1 for o in gc.get_objects() if torch.is_tensor(o))
    n_arrays  = sum(1 for o in gc.get_objects() if isinstance(o, np.ndarray))
    msg = f"    [leak-probe:{tag}] live_tensors={n_tensors}  live_ndarrays={n_arrays}"
    if _heap_stats is not None:
        h = _heap_stats()
        prev = _prev_heap.get(tag)
        if prev is not None:
            d = {k: h[k] - prev[k] for k in h}
            msg += (f"  heap(MB) arena={h['arena_mb']:.1f}({d['arena_mb']:+.1f}) "
                    f"in_use={h['in_use_mb']:.1f}({d['in_use_mb']:+.1f}) "
                    f"free_retained={h['free_retained_mb']:.1f}({d['free_retained_mb']:+.1f}) "
                    f"mmap={h['mmap_mb']:.1f}({d['mmap_mb']:+.1f})")
        else:
            msg += (f"  heap(MB) arena={h['arena_mb']:.1f} in_use={h['in_use_mb']:.1f} "
                    f"free_retained={h['free_retained_mb']:.1f} mmap={h['mmap_mb']:.1f}")
        _prev_heap[tag] = h
    else:
        msg += "  (mallinfo2 unavailable - glibc < 2.33)"
    print(msg)

def _core(net):
    return net.module if isinstance(net, nn.DataParallel) else net

def save_last(path, net, aam, opt, sched, scaler, history, epoch, best_eer):
    torch.save(dict(mode_tag=MODE_TAG, epoch=epoch, best_eer=best_eer, history=history,
                    model=_core(net).state_dict(), aam=aam.state_dict(),
                    opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict()), path)

def save_best(path, net, aam, epoch, metrics):
    torch.save(dict(mode_tag=MODE_TAG, epoch=epoch, metrics=metrics,
                    model=_core(net).state_dict(), aam=aam.state_dict()), path)

def banner(where):
    print(f"===== RUN CONFIG ({where}) =====")
    print(f"  MODE_TAG={MODE_TAG}  W={USE_WEIGHTS} IR={USE_INPUT_RATE} HR={USE_HIDDEN_RATE}")
    print(f"  EPOCHS={EPOCHS}  EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE} (0=disabled)")
    print(f"  DataParallel={USE_DP} (N_GPU={N_GPU})  AMP=fp16 autocast + GradScaler")
    print("=================================")

def train():
    # ---- data ----
    Wd, IAd, HAd, PIDd, RIDd = load_split(list_shards(DEV_GLOB),  "dev")
    Wt, IAt, HAt, PIDt, RIDt = load_split(list_shards(TEST_GLOB), "test")
    assert_disjoint(PIDd, PIDt)

    kd = subsample_one_per_session(PIDd, RIDd)
    kt = subsample_one_per_session(PIDt, RIDt)
    Wd, IAd, HAd, PIDd, RIDd = Wd[kd], IAd[kd], HAd[kd], PIDd[kd], RIDd[kd]
    Wt, IAt, HAt, PIDt, RIDt = Wt[kt], IAt[kt], HAt[kt], PIDt[kt], RIDt[kt]
    print(f"[subsample] dev {len(kd)}  test {len(kt)} (one fingerprint per session)")

    yd, num_classes, _ = encode_labels(PIDd)
    yt, _, _           = encode_labels(PIDt)                 # test labels: retrieval grouping only

    HAS_DUP = len(np.unique(RIDt)) < len(RIDt)
    print(f"[eval] HAS_DUP_SESSIONS={HAS_DUP} "
          f"(session-free vs leaky {'differ' if HAS_DUP else 'collapse -> single pass'})")

    train_bank = FingerprintBank(Wd, IAd, HAd, yd, augment=True)
    val_bank   = FingerprintBank(Wt, IAt, HAt, yt, augment=False)
    sampler    = PKSampler(yd, P_CLASSES, K_SAMPLES)

    # ---- model / loss / optim ----
    net = FingerprintCNN(IN_CH, EMBED_DIM, USE_WEIGHTS, USE_INPUT_RATE, USE_HIDDEN_RATE,
                         IN_IDX_FLAT).to(device)
    aam = AAMSoftmax(EMBED_DIM, num_classes, m=AAM_M, s=AAM_S).to(device)
    if USE_DP:
        net = nn.DataParallel(net)
    opt    = build_optimizer(net, aam, LR, WEIGHT_DECAY)
    sched  = build_scheduler(opt, EPOCHS, WARMUP_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    history, best_eer, start_epoch = [], float("inf"), 0
    banner("after setup")

    # ---- sanity forward (eval + no_grad so BN running stats stay clean) ----
    net.eval()
    with torch.no_grad():
        w0, ia0, ha0, _ = train_bank.batch(torch.arange(min(8, len(yd))))
        _ = net(w0.to(device), ia0.to(device), ha0.to(device))
    net.train()
    print("[sanity] forward pass OK")

    # ---- resume ----
    if RESUME and os.path.exists(LAST_CKPT):
        ck = torch.load(LAST_CKPT, map_location=device)
        if ck.get("mode_tag") == MODE_TAG:
            _core(net).load_state_dict(ck["model"]); aam.load_state_dict(ck["aam"])
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            history, best_eer, start_epoch = ck["history"], ck["best_eer"], ck["epoch"] + 1
            print(f"[resume] epoch {start_epoch}  best_eer={best_eer:.4f}")
        else:
            print(f"[resume] MODE_TAG mismatch ({ck.get('mode_tag')} != {MODE_TAG}); fresh start")

    banner("before training loop")

    for epoch in range(start_epoch, EPOCHS):
        aam.set_margin(AAM_M * min(1.0, epoch / max(1, MARGIN_WARMUP_EPOCHS)))  # margin warmup
        net.train()
        t0, running, nb = time.time(), 0.0, 0

        r0 = rss_gb()
        for idx in sampler:
            w, ia, ha, y = train_bank.batch(idx)
            w, ia, ha, y = w.to(device), ia.to(device), ha.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                emb = net(w, ia, ha)
            loss = aam(emb.float(), y)                        # ArcFace in fp32
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(aam.parameters()),
                                           GRAD_CLIP)
            scaler.step(opt); scaler.update()
            running += loss.item(); nb += 1
        sched.step()
        r1 = rss_gb()
        leak_probe("train")

        sf, lk = evaluate(net, val_bank, yt, RIDt, HAS_DUP)
        r2 = rss_gb()
        leak_probe("eval")
        print(f"    dRSS train={r1-r0:+.2f}GB  eval={r2-r1:+.2f}GB")
        
        improved = sf["eer"] < best_eer
        if improved:
            best_eer = sf["eer"]
        history.append(dict(epoch=epoch, loss=running / max(nb, 1), sf=sf, leaky=lk,
                            rss=rss_gb(), lr=opt.param_groups[0]["lr"]))

        print(f"[e{epoch:02d}] loss={history[-1]['loss']:.4f} "
              f"m={aam.m:.3f} lr={history[-1]['lr']:.2e} | "
              f"SF eer={sf['eer']:.4f} r1={sf['rank1']:.3f} r5={sf['rank5']:.3f} mAP={sf['mAP']:.3f} | "
              f"LK eer={lk['eer']:.4f} r1={lk['rank1']:.3f} | "
              f"{mem_report()} {'*best' if improved else ''} "
              f"({time.time() - t0:.0f}s)")

        save_last(LAST_CKPT, net, aam, opt, sched, scaler, history, epoch, best_eer)
        if improved:
            save_best(BEST_CKPT, net, aam, epoch, sf)

    print(f"[done] best session-free EER={best_eer:.4f}  ->  {BEST_CKPT}")
    return history


# Cell 10
# ===================== RUN =====================
history = train()
