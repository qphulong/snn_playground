import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import os

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_epoch_001.npz")
SYNAPSE     = "in->hid"
K           = 5       # used for the final scatter plot
K_RANGE     = range(1, 21)   # elbow curve sweep
MAX_ITER    = 300
RANDOM_SEED = 42

# ── Load weights ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

pfx  = SYNAPSE.replace("->", "_to_")
data = np.load(NPZ_PATH, allow_pickle=True)
W    = data[f"{pfx}__final_weights"]   # (N_pre, N_post)
X    = W.T.astype(np.float64)          # (N_post, N_pre) — each row = one neuron
N    = X.shape[0]
print(f"Data: {N} neurons × {X.shape[1]}-dim weight vectors")

# ── K-medians (L1) with K-means++ init ───────────────────────────────────────

def kmedians(X, k, max_iter, rng):
    # K-means++ seeding
    centroids = [X[rng.integers(N)]]
    for _ in range(k - 1):
        dists = cdist(X, np.array(centroids), metric="cityblock").min(axis=1)
        probs = dists / dists.sum()
        centroids.append(X[rng.choice(N, p=probs)])
    centroids = np.array(centroids)

    labels = np.zeros(N, dtype=np.int32)
    for it in range(max_iter):
        dists  = cdist(X, centroids, metric="cityblock")   # (N, k)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and it > 0:
            print(f"  Converged at iteration {it}")
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                centroids[c] = np.median(members, axis=0)
    else:
        print(f"  Reached max_iter={max_iter}")
    return labels, centroids

# ── Elbow curve (K = 1 … 20) ─────────────────────────────────────────────────

def inertia_l1(X, labels, centroids):
    dists = cdist(X, centroids, metric="cityblock")
    return float(dists[np.arange(len(X)), labels].sum())

print("Running elbow sweep ...")
inertias = []
for k in K_RANGE:
    lbl, cen = kmedians(X, k, MAX_ITER, np.random.default_rng(RANDOM_SEED))
    inertias.append(inertia_l1(X, lbl, cen))
    print(f"  K={k:2d}  inertia={inertias[-1]:.2f}")

fig_elbow, ax_elbow = plt.subplots(figsize=(9, 4))
ks = list(K_RANGE)
ax_elbow.plot(ks, inertias, marker="o", color="steelblue", linewidth=1.8)
ax_elbow.set_title("Elbow curve — K-medians (L1)\nhidden neuron weight vectors",
                   fontsize=12, fontweight="bold")
ax_elbow.set_xlabel("K")
ax_elbow.set_ylabel("Total L1 inertia")
ax_elbow.set_xticks(ks)
ax_elbow.grid(True, alpha=0.3)
plt.tight_layout()
elbow_out = os.path.join(os.path.dirname(NPZ_PATH), "k_mean_elbow.png")
fig_elbow.savefig(elbow_out, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.relpath(elbow_out)}")

# ── Final clustering with chosen K ───────────────────────────────────────────

print(f"Running K-medians (L1), K={K} ...")
labels, centroids = kmedians(X, K, MAX_ITER, rng)

for c in range(K):
    print(f"  Cluster {c:2d}: {(labels == c).sum()} neurons")

# ── PCA → 3D ─────────────────────────────────────────────────────────────────
pca   = PCA(n_components=3, random_state=RANDOM_SEED)
X_3d  = pca.fit_transform(X)
C_3d  = pca.transform(centroids)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
cmap   = plt.get_cmap("tab10" if K <= 10 else "tab20")
colors = [cmap(c / max(K - 1, 1)) for c in range(K)]

fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection="3d")

for c in range(K):
    mask = labels == c
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
               s=18, color=colors[c], alpha=0.7, linewidths=0,
               label=f"C{c} (n={mask.sum()})")

for c in range(K):
    ax.scatter(C_3d[c, 0], C_3d[c, 1], C_3d[c, 2],
               s=200, color=colors[c], marker="X",
               edgecolors="black", linewidths=0.8, zorder=5)

ev = pca.explained_variance_ratio_
ax.set_title(
    f"K-medians (L1), K={K} — hidden neuron weight clusters\n"
    f"PCA 3D ({ev.sum()*100:.1f}% variance) — epoch 1 final weights",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)")
ax.legend(fontsize=8, ncol=2, loc="best")
plt.tight_layout()

plt.show()
