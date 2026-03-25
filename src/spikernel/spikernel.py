"""
spikernel.py
============
Core implementation of the Spikernel — a sequence kernel for comparing
neural spike-count recordings.

Algorithm reference
-------------------
The Spikernel compares two spike-count sequences by accumulating similarity
scores across all common sub-sequences of length up to n_max, using a
dynamic programming recurrence to avoid explicit enumeration of the
(infinite) feature space.

Key ideas:
  - Sequences are matrices of shape (n_neurons, T), where each column is
    one time bin's spike counts across all recorded neurons.
  - A "bin similarity" measures how alike two individual time bins are.
  - The decay parameter λ down-weights older time bins relative to recent
    ones, giving the kernel a causal, recency-sensitive character.
  - The DP runs in O(T * T' * n_max) time, which is linear in recording
    length and fast enough for real-time brain-machine interface use.

Module contents
---------------
  bin_similarity(x, y, gamma)          – RBF similarity between two bins
  spikernel(s, s_prime, n_max, lam, gamma)
                                       – kernel value for one pair
  spikernel_all_lengths(...)           – kernel values for every n = 1..n_max
  gram_matrix(sequences, ...)          – full symmetric Gram (kernel) matrix
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Similarity function
# ─────────────────────────────────────────────────────────────────────────────

def bin_similarity(x, y, gamma=0.5):
    """
    Compute the RBF (Gaussian) similarity between two spike-count bin vectors.

    The similarity is defined as:

        sim(x, y) = exp( -gamma * ||x - y||^2 )

    where ||·||^2 is the squared L2 norm (sum of squared differences across
    all neurons).  This maps:
      - identical bins  → 1.0  (distance = 0, e^0 = 1)
      - very different  → ~0.0 (large distance, e^{-large} ≈ 0)

    Parameters
    ----------
    x : array-like, shape (n_neurons,)
        Spike counts for all neurons in one time bin of sequence s.
    y : array-like, shape (n_neurons,)
        Spike counts for all neurons in the matching time bin of sequence s'.
    gamma : float, optional (default 0.5)
        Bandwidth parameter of the RBF kernel.
        - Larger gamma → similarity falls off faster with distance
          (the kernel is more "selective").
        - Smaller gamma → similarity stays high even for dissimilar bins
          (the kernel is more "permissive").

    Returns
    -------
    float in (0.0, 1.0]
        1.0 when x == y exactly; approaches 0.0 as ||x - y|| grows.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    squared_dist = np.sum((x - y) ** 2)
    return np.exp(-gamma * squared_dist)


# ─────────────────────────────────────────────────────────────────────────────
# Core Spikernel
# ─────────────────────────────────────────────────────────────────────────────

def spikernel(s, s_prime, n_max=3, lam=0.8, gamma=0.5, normalize=True):
    """
    Compute the Spikernel K(s, s') between two spike-count sequences.

    The kernel measures the total similarity of all common sub-sequences of
    length up to n_max, with an exponential time-decay bias toward recent bins.

    Algorithm — dynamic programming with two tables
    -----------------------------------------------
    Two 3-D tables are maintained, both of shape (n_max+1, T+1, T'+1):

      G[k, i, j]
        Accumulated kernel value when considering sub-sequences of length k,
        using the first i bins of s and the first j bins of s'.
        Intuitively: "how similar are s[:i] and s'[:j] at the level of
        length-k rhythm snippets?"

      GR[k, i, j]  (auxiliary / "cache" table)
        Stores a running sum that enables gaps (time warping) between the two
        sequences.  It accumulates, up to position j in s', all contributions
        from extending a length-(k-1) match at position i-1 in s with the
        current bin pair (i, *) — allowing the matching j-position to float.

    Recurrences (1-indexed, bins run from 1 to T / T')
    ---------------------------------------------------
    Base case:
        G[0, i, j] = 1   for all i, j
        (a sub-sequence of length 0 trivially matches everywhere)

    For k = 1 … n_max, i = 1 … T, j = 1 … T':

        sim = bin_similarity(s[:, i-1], s'[:, j-1], gamma)

        GR[k, i, j] = lam * GR[k, i, j-1]          # carry forward in j
                    + lam * sim * G[k-1, i-1, j-1]  # extend a shorter match

        G[k, i, j]  = lam * G[k, i-1, j]            # carry forward in i
                    + lam * GR[k, i, j]              # add new completions

    The lam multipliers implement exponential time-decay:
      - Each step forward in i or j multiplies by lam.
      - Older contributions accumulate more lam factors, so they are
        down-weighted relative to contributions near the end of the sequence.

    Final result:  G[n_max, T, T']

    Parameters
    ----------
    s : ndarray, shape (n_neurons, T)
        First spike-count sequence.  Each column s[:, t] is the spike-count
        vector across all neurons for time bin t.
    s_prime : ndarray, shape (n_neurons, T')
        Second spike-count sequence.  T' need not equal T.
    n_max : int, optional (default 3)
        Maximum sub-sequence length to consider.
        Higher values capture longer rhythmic patterns but increase runtime
        by a factor of n_max.
    lam : float in (0, 1), optional (default 0.8)
        Exponential decay parameter.
        - Values close to 1  → slow decay, all history contributes equally.
        - Values close to 0  → fast decay, only the most recent bins matter.
    gamma : float, optional (default 0.5)
        RBF bandwidth passed to bin_similarity (see that function).
    normalize : bool, optional (default True)
        If True, return the normalised kernel:

            K_norm(s, s') = K(s, s') / sqrt( K(s, s) * K(s', s') )

        This maps the result to (0, 1], removing the effect of sequence
        length and making scores comparable across pairs of different lengths.
        If False, return the raw kernel value (grows with sequence length).

    Returns
    -------
    float
        If normalize=True:  value in (0, 1], where 1.0 means identical.
        If normalize=False: raw kernel value, always positive.

    Complexity
    ----------
    Time : O(n_max * T * T')  for the cross kernel.
           When normalize=True, two additional self-kernel calls are made,
           each O(n_max * T^2), so total is O(n_max * (T*T' + T^2 + T'^2)).
    Space: O(n_max * T * T')  for the two DP tables.
    """
    s       = np.asarray(s, dtype=float)
    s_prime = np.asarray(s_prime, dtype=float)

    T  = s.shape[1]
    Tp = s_prime.shape[1]

    # Allocate DP tables — shape: (n_max+1) x (T+1) x (T'+1)
    # Index [k][i][j]; the +1 allows a clean 1-based loop with a 0 border.
    G  = np.zeros((n_max + 1, T + 1, Tp + 1))
    GR = np.zeros((n_max + 1, T + 1, Tp + 1))

    # Base case: length-0 sub-sequences always match
    G[0, :, :] = 1.0

    # Fill tables for each sub-sequence length k
    for k in range(1, n_max + 1):
        for i in range(1, T + 1):
            for j in range(1, Tp + 1):

                # Bin-level similarity at positions i, j (convert to 0-based)
                sim = bin_similarity(s[:, i - 1], s_prime[:, j - 1], gamma)

                # --- GR update (Eq. 6 / 23 in the paper) ---
                # Carry forward the running sum along j, then add a new
                # contribution: extend the best length-(k-1) alignment ending
                # at (i-1, j-1) with the current bin pair.
                GR[k, i, j] = (lam * GR[k, i, j - 1]
                              + lam * sim * G[k - 1, i - 1, j - 1])

                # --- G update (Eq. 21 in the paper) ---
                # Carry forward the kernel score along i, then absorb all new
                # length-k completions accumulated in GR at (i, j).
                G[k, i, j] = (lam * G[k, i - 1, j]
                            + lam * GR[k, i, j])

    raw = G[n_max, T, Tp]

    if not normalize:
        return raw

    # Normalise: K_norm = K(s, s') / sqrt( K(s,s) * K(s',s') )
    k_ss  = spikernel(s,       s,       n_max, lam, gamma, normalize=False)
    k_sps = spikernel(s_prime, s_prime, n_max, lam, gamma, normalize=False)
    denom = np.sqrt(k_ss * k_sps)
    return raw / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-length breakdown
# ─────────────────────────────────────────────────────────────────────────────

def spikernel_all_lengths(s, s_prime, n_max=3, lam=0.8, gamma=0.5):
    """
    Compute the Spikernel for every sub-sequence length n = 1 … n_max in a
    single DP pass (no extra cost over calling spikernel once).

    This is useful for:
      - Debugging / understanding how longer patterns contribute.
      - Building the "weighted-sum variant" described in the paper, where
        K_combined = sum_n( weight_n * K_n(s, s') ).

    Parameters
    ----------
    s, s_prime : ndarray, shape (n_neurons, T) and (n_neurons, T')
        Input spike-count sequences.
    n_max : int, optional (default 3)
        Maximum sub-sequence length.
    lam : float in (0, 1), optional (default 0.8)
        Exponential decay parameter.
    gamma : float, optional (default 0.5)
        RBF bandwidth for bin_similarity.

    Returns
    -------
    dict  {n: float}
        Mapping from sub-sequence length n to kernel value K_n(s, s').
        Keys are integers 1 through n_max.

    Notes
    -----
    The DP tables G and GR are built identically to spikernel().
    The result for length n is simply G[n, T, T'].
    """
    s       = np.asarray(s, dtype=float)
    s_prime = np.asarray(s_prime, dtype=float)

    T  = s.shape[1]
    Tp = s_prime.shape[1]

    G  = np.zeros((n_max + 1, T + 1, Tp + 1))
    GR = np.zeros((n_max + 1, T + 1, Tp + 1))
    G[0, :, :] = 1.0

    for k in range(1, n_max + 1):
        for i in range(1, T + 1):
            for j in range(1, Tp + 1):
                sim = bin_similarity(s[:, i - 1], s_prime[:, j - 1], gamma)
                GR[k, i, j] = (lam * GR[k, i, j - 1]
                              + lam * sim * G[k - 1, i - 1, j - 1])
                G[k, i, j]  = (lam * G[k, i - 1, j]
                             + lam * GR[k, i, j])

    return {n: G[n, T, Tp] for n in range(1, n_max + 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Gram matrix
# ─────────────────────────────────────────────────────────────────────────────

def gram_matrix(sequences, n_max=3, lam=0.8, gamma=0.5):
    """
    Compute the full symmetric Gram (kernel) matrix for a list of sequences.

    K[i, j] = spikernel(sequences[i], sequences[j])

    Because the Spikernel is symmetric (K(a,b) == K(b,a)) and positive
    semi-definite, only the upper triangle is computed; the lower triangle
    is filled by symmetry.

    Parameters
    ----------
    sequences : list of ndarray, each shape (n_neurons, T)
        Collection of spike-count sequences to compare pairwise.
    n_max : int, optional (default 3)
        Maximum sub-sequence length passed to spikernel.
    lam : float in (0, 1), optional (default 0.8)
        Decay parameter passed to spikernel.
    gamma : float, optional (default 0.5)
        RBF bandwidth passed to spikernel.

    Returns
    -------
    K : ndarray, shape (N, N)
        Symmetric Gram matrix.  K[i, i] is the self-similarity of sequence i.
    """
    N = len(sequences)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            val = spikernel(sequences[i], sequences[j], n_max, lam, gamma)
            K[i, j] = val
            K[j, i] = val  # symmetry
    return K


# ─────────────────────────────────────────────────────────────────────────────
# Sample data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_random_sequence(n_neurons=4, T=10, mean_rate=3.0, seed=None):
    """
    Generate a random spike-count sequence sampled from a Poisson distribution.

    Parameters
    ----------
    n_neurons : int
        Number of neurons (rows in the output matrix).
    T : int
        Number of time bins (columns).
    mean_rate : float
        Mean spike count per bin per neuron (Poisson lambda).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (n_neurons, T)
        Integer spike counts.
    """
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=mean_rate, size=(n_neurons, T)).astype(float)


def make_shifted_sequence(s, shift=2):
    """
    Shift a sequence along the time axis, padding with zeros.

    This simulates a recording with the same underlying neural rhythm but
    delayed in time — the Spikernel should still give a high similarity
    because the sub-sequence patterns are preserved.

    Parameters
    ----------
    s : ndarray, shape (n_neurons, T)
        Original spike-count sequence.
    shift : int
        Number of bins to shift to the right (positive = later).
        Bins that roll off the right edge wrap to the left; they are then
        zeroed so there is no circular leakage.

    Returns
    -------
    ndarray, shape (n_neurons, T)
        Shifted sequence with the first `shift` columns set to zero.
    """
    shifted = np.roll(s, shift, axis=1)
    shifted[:, :shift] = 0.0  # zero-pad instead of wrap-around
    return shifted


def make_noisy_sequence(s, noise_std=0.5, seed=None):
    """
    Add Gaussian noise to a sequence and clip to non-negative values.

    Simulates slight recording variability or trial-to-trial jitter while
    keeping the underlying firing-rate pattern intact.

    Parameters
    ----------
    s : ndarray, shape (n_neurons, T)
        Original spike-count sequence.
    noise_std : float
        Standard deviation of the Gaussian noise added to each bin.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (n_neurons, T)
        Noisy copy of s, clipped to >= 0.
    """
    rng = np.random.default_rng(seed)
    noisy = s + rng.normal(0.0, noise_std, s.shape)
    return np.clip(noisy, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Main: print results for manual inspection
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title):
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_matrix(mat, row_labels, col_labels, fmt=".4f"):
    """Print a matrix with row and column labels."""
    col_w = 10
    header = " " * 6 + "".join(f"{lbl:>{col_w}}" for lbl in col_labels)
    print(header)
    print(" " * 6 + "-" * (col_w * len(col_labels)))
    for lbl, row in zip(row_labels, mat):
        vals = "".join(f"{v:{col_w}{fmt}}" for v in row)
        print(f"  {lbl:<4}|{vals}")


def main():

    # ── Parameters ──────────────────────────────────────────────────────────
    N_NEURONS = 4    # neurons recorded simultaneously
    T         = 10   # number of 200 ms time bins per trial
    N_MAX     = 3    # maximum sub-sequence length
    LAM       = 0.8  # decay parameter λ
    GAMMA     = 0.5  # RBF bandwidth γ

    print_section("Parameters")
    print(f"  n_neurons = {N_NEURONS}")
    print(f"  T         = {T}   (time bins per sequence)")
    print(f"  n_max     = {N_MAX}   (max sub-sequence length)")
    print(f"  λ (lam)   = {LAM}  (time-decay; closer to 1 → slower decay)")
    print(f"  γ (gamma) = {GAMMA}  (RBF bandwidth; larger → stricter similarity)")

    # ── Build sample sequences ───────────────────────────────────────────────
    # s_A : base sequence (Poisson spike counts, seed fixed for reproducibility)
    # s_B : s_A shifted 2 bins — same rhythm, slightly delayed
    # s_C : s_A with small noise added — same rhythm, minor perturbation
    # s_D : completely independent random sequence — unrelated rhythm
    s_A = make_random_sequence(N_NEURONS, T, mean_rate=3.0, seed=42)
    s_B = make_shifted_sequence(s_A, shift=2)
    s_C = make_noisy_sequence(s_A, noise_std=0.5, seed=7)
    s_D = make_random_sequence(N_NEURONS, T, mean_rate=3.0, seed=99)

    print_section("Sample Sequences  (rows=neurons, cols=time bins)")
    for label, seq in [("A (base)", s_A), ("B (shifted +2)", s_B),
                       ("C (noisy A)", s_C), ("D (random)", s_D)]:
        print(f"\n  {label}:")
        for ni, row in enumerate(seq):
            vals = "  ".join(f"{v:4.1f}" for v in row)
            print(f"    neuron {ni}: [{vals}]")

    # ── Bin-level similarity spot checks ─────────────────────────────────────
    print_section("Bin Similarity Checks  (bin_similarity)")
    print("  Expected: identical bins → 1.0, dissimilar → near 0\n")

    bin0_A = s_A[:, 0]
    bin0_D = s_D[:, 0]
    print(f"  sim(A[:,0], A[:,0])  identical   = "
          f"{bin_similarity(bin0_A, bin0_A, GAMMA):.6f}")
    print(f"  sim(A[:,0], D[:,0])  unrelated   = "
          f"{bin_similarity(bin0_A, bin0_D, GAMMA):.6f}")
    print(f"  sim(A[:,0], A[:,0]+1) off-by-one = "
          f"{bin_similarity(bin0_A, bin0_A + 1, GAMMA):.6f}")
    print(f"  sim(A[:,0], A[:,0]+5) off-by-five= "
          f"{bin_similarity(bin0_A, bin0_A + 5, GAMMA):.6f}")

    # ── Pairwise kernel values ────────────────────────────────────────────────
    print_section("Pairwise Spikernel Values  K(x, y)")
    print("  Expected ordering:  K(A,A) ≥ K(A,B) ≥ K(A,C) >> K(A,D)\n")

    pairs = [
        ("A", "A", s_A, s_A, "self-similarity (upper bound)"),
        ("A", "B", s_A, s_B, "shifted copy     → should be high"),
        ("A", "C", s_A, s_C, "noisy copy       → should be high"),
        ("A", "D", s_A, s_D, "unrelated        → should be low"),
        ("B", "C", s_B, s_C, "shifted vs noisy → moderate"),
        ("C", "D", s_C, s_D, "noisy vs random  → should be low"),
    ]

    for la, lb, sa, sb, note in pairs:
        k = spikernel(sa, sb, N_MAX, LAM, GAMMA)
        print(f"  K({la},{lb}) = {k:9.4f}   # {note}")

    # ── Per-length breakdown ─────────────────────────────────────────────────
    print_section("Kernel Value per Sub-sequence Length  (spikernel_all_lengths)")
    print("  Longer n captures more complex rhythmic structure.\n")

    for la, sa, lb, sb in [("A", s_A, "A", s_A),
                            ("A", s_A, "B", s_B),
                            ("A", s_A, "D", s_D)]:
        vals = spikernel_all_lengths(sa, sb, N_MAX, LAM, GAMMA)
        print(f"  K(n, {la}, {lb}):")
        for n, v in vals.items():
            bar = "█" * min(int(v * 8), 50)   # scale bar for readability
            print(f"    n={n}: {v:9.4f}  {bar}")
        print()

    # ── Weighted-sum variant ─────────────────────────────────────────────────
    print_section("Weighted-Sum Spikernel Variant")
    print("  K_combined = 0.2*K_1 + 0.3*K_2 + 0.5*K_3\n")

    weights = {1: 0.2, 2: 0.3, 3: 0.5}
    for la, sa, lb, sb in [("A", s_A, "A", s_A),
                            ("A", s_A, "B", s_B),
                            ("A", s_A, "C", s_C),
                            ("A", s_A, "D", s_D)]:
        vals   = spikernel_all_lengths(sa, sb, N_MAX, LAM, GAMMA)
        k_comb = sum(weights[n] * vals[n] for n in weights)
        print(f"  K_combined({la},{lb}) = {k_comb:.4f}")

    # ── Gram matrix ───────────────────────────────────────────────────────────
    print_section("Gram (Kernel) Matrix  [A, B, C, D]")
    print("  Diagonal = self-similarity.  Off-diagonal = cross-similarity.\n")

    seqs   = [s_A, s_B, s_C, s_D]
    labels = ["A", "B", "C", "D"]
    K      = gram_matrix(seqs, N_MAX, LAM, GAMMA)
    print_matrix(K, labels, labels)

    # ── Symmetry check ───────────────────────────────────────────────────────
    print_section("Symmetry Check  K(x,y) == K(y,x)")
    print("  (Should be True for all pairs)\n")

    for la, sa, lb, sb in [("A", s_A, "B", s_B),
                            ("A", s_A, "D", s_D),
                            ("B", s_B, "C", s_C)]:
        kxy = spikernel(sa, sb, N_MAX, LAM, GAMMA)
        kyx = spikernel(sb, sa, N_MAX, LAM, GAMMA)
        ok  = np.isclose(kxy, kyx)
        print(f"  K({la},{lb})={kxy:.6f}  K({lb},{la})={kyx:.6f}  match={ok}")

    # ── Sensitivity to λ ─────────────────────────────────────────────────────
    print_section("Sensitivity to λ  (decay parameter)")
    print("  Pair: K(A,B) vs K(A,D) — ratio shows discriminability.\n")
    print(f"  {'λ':>6}  {'K(A,B)':>10}  {'K(A,D)':>10}  {'ratio':>8}")
    print("  " + "-" * 40)

    for lv in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        ab = spikernel(s_A, s_B, N_MAX, lv, GAMMA)
        ad = spikernel(s_A, s_D, N_MAX, lv, GAMMA)
        ratio = ab / ad if ad > 0 else float("inf")
        print(f"  {lv:>6.2f}  {ab:>10.4f}  {ad:>10.4f}  {ratio:>8.2f}x")

    # ── Sensitivity to γ ─────────────────────────────────────────────────────
    print_section("Sensitivity to γ  (RBF bandwidth)")
    print("  Pair: K(A,B) vs K(A,D).\n")
    print(f"  {'γ':>6}  {'K(A,B)':>10}  {'K(A,D)':>10}  {'ratio':>8}")
    print("  " + "-" * 40)

    for gv in [0.05, 0.1, 0.5, 1.0, 2.0]:
        ab = spikernel(s_A, s_B, N_MAX, LAM, gv)
        ad = spikernel(s_A, s_D, N_MAX, LAM, gv)
        ratio = ab / ad if ad > 0 else float("inf")
        print(f"  {gv:>6.2f}  {ab:>10.4f}  {ad:>10.4f}  {ratio:>8.2f}x")

    print("\n" + "=" * 62)
    print("  Done.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()