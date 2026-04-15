"""
Spikernel Algorithm Implementation
===================================
A kernel method for measuring similarity between neural spike sequences.
Each formula is annotated with its LaTeX source equation number.
"""

import numpy as np


# ===========================================================================
# 1. BIN-WISE DISTANCE
# ===========================================================================

def bin_wise_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the squared L2 (Euclidean) distance between two firing-rate
    snapshot vectors.

    Formula (Eq. 1):
        ||a - b||_2^2 = sum_{k=1}^{d} (a_k - b_k)^2

    Parameters
    ----------
    a, b : np.ndarray of shape (d,)
        Instantaneous firing-rate vectors for d cortical units.

    Returns
    -------
    float
        Squared L2 distance between a and b.
    """
    # ||a - b||_2^2 = sum_k (a_k - b_k)^2
    return float(np.sum((a - b) ** 2))


# ===========================================================================
# 2. SPIKERNEL FOR n-LONG SUBSEQUENCES  (dynamic programming)
# ===========================================================================

def spikernel_n(
    s: np.ndarray,
    t: np.ndarray,
    n: int,
    lam: float,
    mu: float,
) -> float:
    """
    Computes K_n(s, t): the Spikernel similarity summed over all n-long
    matching subsequences, using the DP recurrence.

    DP tables
    ---------
    K[i, j]  = K_n(s_{1:i}, t_{1:j})   — total similarity for n-patterns
    Kp[i, j] = K'_n(s_{1:i}, t_{1:j})  — auxiliary cached variable

    Base cases (Eq. 4):
        K_0(s, t) = 1,  K'_0(s, t) = 1   for all s, t
        K_n(s, t) = 0,  K'_n(s, t) = 0   if min(len(s), len(t)) < n

    Recurrence for K_n (Eq. 2):
        K_n(s_{1:i}, t_{1:j}) =
            lambda * K_n(s_{1:i-1}, t_{1:j})   +   K'_n(s_{1:i}, t_{1:j})

    Recurrence for K'_n (Eq. 3):
        K'_n(s_{1:i}, t_{1:j}) =
            lambda^2 * K_{n-1}(s_{1:i-1}, t_{1:j-1}) * mu^(0.5 * ||s_i - t_j||^2)
            + lambda * K'_n(s_{1:i}, t_{1:j-1})

    Parameters
    ----------
    s : np.ndarray, shape (d, len_s)   — sequence s
    t : np.ndarray, shape (d, len_t)   — sequence t
    n : int                            — subsequence length
    lam : float in (0, 1)              — temporal decay lambda
    mu  : float in (0, 1)              — similarity decay mu

    Returns
    -------
    float  — K_n(s, t)
    """
    d, len_s = s.shape
    _, len_t = t.shape

    # --- Base case: pattern length 0 -----------------------------------------
    # K_0 = 1 everywhere; K'_0 = 1 everywhere  (Eq. 4)
    if n == 0:
        return 1.0

    # --- Base case: sequence too short ----------------------------------------
    # K_n = 0 if min(len_s, len_t) < n  (Eq. 4)
    if min(len_s, len_t) < n:
        return 0.0

    # --- Allocate DP tables (1-indexed via shape len+1) -----------------------
    # K_prev[i, j]  = K_{n-1}(s_{1:i}, t_{1:j})
    # K_cur [i, j]  = K_n    (s_{1:i}, t_{1:j})
    # Kp    [i, j]  = K'_n   (s_{1:i}, t_{1:j})

    # We build K_{n-1} first with a recursive call so we can use it when
    # filling K'_n.  For efficiency the recursion bottoms out at n=0.
    K_prev = _build_K_table(s, t, n - 1, lam, mu)

    # Allocate K and K' tables; rows = i in [0..len_s], cols = j in [0..len_t]
    K  = np.zeros((len_s + 1, len_t + 1))
    Kp = np.zeros((len_s + 1, len_t + 1))

    # Row 0 and col 0 stay 0 (empty prefix cannot contain n>=1 pattern)

    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):

            # --- Eq. 3 : K'_n update -------------------------------------------
            # similarity of current snapshot pair
            dist_sq = bin_wise_distance(s[:, i - 1], t[:, j - 1])
            # mu^(0.5 * ||s_i - t_j||^2)
            sim = mu ** (0.5 * dist_sq)

            # K'_n(s_{1:i}, t_{1:j}) =
            #   lambda^2 * K_{n-1}(s_{1:i-1}, t_{1:j-1}) * sim
            #   + lambda  * K'_n(s_{1:i}, t_{1:j-1})
            Kp[i, j] = (lam ** 2) * K_prev[i - 1, j - 1] * sim \
                       + lam * Kp[i, j - 1]

            # --- Eq. 2 : K_n update --------------------------------------------
            # K_n(s_{1:i}, t_{1:j}) =
            #   lambda * K_n(s_{1:i-1}, t_{1:j})  +  K'_n(s_{1:i}, t_{1:j})
            K[i, j] = lam * K[i - 1, j] + Kp[i, j]

    return float(K[len_s, len_t])


def _build_K_table(
    s: np.ndarray,
    t: np.ndarray,
    n: int,
    lam: float,
    mu: float,
) -> np.ndarray:
    """
    Internal helper: returns the full DP table K_n[i, j] for 0<=i<=len_s,
    0<=j<=len_t.  Used by spikernel_n to access K_{n-1} values mid-loop.
    """
    d, len_s = s.shape
    _, len_t = t.shape

    # Base case n=0 : K_0 = 1 everywhere  (Eq. 4)
    if n == 0:
        return np.ones((len_s + 1, len_t + 1))

    # Base case: any sequence shorter than n gives 0  (Eq. 4)
    if min(len_s, len_t) < n:
        return np.zeros((len_s + 1, len_t + 1))

    K_prev = _build_K_table(s, t, n - 1, lam, mu)

    K  = np.zeros((len_s + 1, len_t + 1))
    Kp = np.zeros((len_s + 1, len_t + 1))

    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            dist_sq = bin_wise_distance(s[:, i - 1], t[:, j - 1])
            sim = mu ** (0.5 * dist_sq)

            # Eq. 3
            Kp[i, j] = (lam ** 2) * K_prev[i - 1, j - 1] * sim \
                       + lam * Kp[i, j - 1]
            # Eq. 2
            K[i, j] = lam * K[i - 1, j] + Kp[i, j]

    return K


# ===========================================================================
# 3. MAIN SPIKERNEL
# ===========================================================================

def spikernel(
    s: np.ndarray,
    t: np.ndarray,
    n_max: int,
    lam: float,
    mu: float,
    q: float,
) -> float:
    """
    Computes the main Spikernel K(s, t): a weighted sum of n-long
    subsequence kernels.

    Formula (Eq. 5):
        K(s, t) = sum_{i=1}^{n} q^i * K_i(s, t)

    where
        q      — length-scale weighting factor
        K_i    — Spikernel for patterns of length i  (Eq. 2–4)

    Parameters
    ----------
    s, t   : np.ndarray, shape (d, len)  — neural sequences
    n_max  : int                         — maximum pattern length
    lam    : float in (0, 1)             — temporal decay lambda
    mu     : float in (0, 1)             — similarity decay mu
    q      : float                       — pattern-length weight

    Returns
    -------
    float  — K(s, t)
    """
    total = 0.0
    for i in range(1, n_max + 1):
        # K_i(s, t) — similarity for patterns of length i
        Ki = spikernel_n(s, t, i, lam, mu)
        # Eq. 5 contribution: q^i * K_i(s, t)
        total += (q ** i) * Ki

    return total

# ===========================================================================
# 4. NORMALIZED SPIKERNEL
# ===========================================================================

def spikernel_normalized(
    s: np.ndarray,
    t: np.ndarray,
    n_max: int,
    lam: float,
    mu: float,
    q: float,
    eps: float = 1e-12,
) -> float:
    """
    Normalized Spikernel:

        K_norm(s,t) = K(s,t) / sqrt(K(s,s) * K(t,t))

    Prevents scale bias and ensures comparability.

    Parameters
    ----------
    eps : small constant to avoid division by zero

    Returns
    -------
    float in [0,1]
    """
    K_st = spikernel(s, t, n_max, lam, mu, q)
    K_ss = spikernel(s, s, n_max, lam, mu, q)
    K_tt = spikernel(t, t, n_max, lam, mu, q)

    denom = np.sqrt(K_ss * K_tt) + eps
    return K_st / denom

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # ============================================================
    # Fixed setup (as requested)
    # ============================================================
    n_max = 5
    lam = 0.7
    mu = 0.35
    q = 3

    def sim(a, b):
        return spikernel_normalized(a, b, n_max, lam, mu, q)

    def print_case(title, s, t, expected):
        print(f"\n=== {title} ===")
        print("s:", s)
        print("t:", t)
        print("Expected:", expected)
        print("Actual:", sim(s, t))

    # ============================================================
    # Base signal (1 neuron, length 10)
    # ============================================================
    s = np.array([[0,1,2,3,4,5,6,1,2,3]])

    # ============================================================
    # 1. Identity
    # ============================================================
    print_case(
        "Identity",
        s, s,
        "≈ 1.0"
    )

    # ============================================================
    # 2. Small noise (tolerance)
    # ============================================================
    t = np.array([[0,1,2,3,4,5,5,1,2,4]])  # slight change at end
    print_case(
        "Small noise",
        s, t,
        "High similarity"
    )

    # ============================================================
    # 3. Completely different
    # ============================================================
    t = np.array([[4,6,1,2,4,9,4,7,1,5]])
    print_case(
        "Completely different",
        s, t,
        "Low similarity"
    )

    # ============================================================
    # 4. Time shift (IMPORTANT)
    # ============================================================
    t = np.array([[0,0,1,2,3,4,5,6,1,2]])  # shifted right
    print_case(
        "Shift right",
        s, t,
        "Moderate to HIGH similarity"
    )

    # ============================================================
    # 5. Reverse (order matters!)
    # ============================================================
    t = np.array([[3,2,1,6,5,4,3,2,1,0]])
    print_case(
        "Reversed sequence",
        s, t,
        "Low similarity (order matters)"
    )

    # ============================================================
    # 6. Time stretch (warp)
    # ============================================================
    t = np.array([[0,1,2,3,3,4,5,5,6,2]])
    print_case(
        "Time stretch",
        s, t,
        "Moderate to HIGH similarity"
    )

    # ============================================================
    # 7. Recency bias (CRITICAL TEST)
    # ============================================================
    t_early = np.array([[4,5,2,3,4,5,6,1,2,3]])
    t_late  = np.array([[0,1,2,3,4,5,4,3,2,1]])

    sim_early = sim(s, t_early)
    sim_late  = sim(s, t_late)

    print("\n=== Recency Bias ===")
    print("Early mismatch:", sim_early)
    print("Late mismatch:", sim_late)
    print("Expected: early > late")

    # ============================================================
    # 8. Single spike alignment
    # ============================================================
    s_spike = np.array([[0,0,0,0,5,0,0,0,0,0]])
    t_spike_shift = np.array([[0,0,0,0,0,5,0,0,0,0]])

    print_case(
        "Single spike shift",
        s_spike, t_spike_shift,
        "Non-zero similarity (alignment works)"
    )

    # ============================================================
    # 9. Sparse vs dense
    # ============================================================
    sparse = np.array([[0,0,0,5,0,0,0,0,0,0]])
    dense  = np.array([[1,1,1,1,1,1,1,1,1,1]])

    print_case(
        "Sparse vs dense",
        sparse, dense,
        "Low similarity"
    )

    # # ============================================================
    # # 10. Scaling (normalization check)
    # # ============================================================
    # t_scaled = s * 3

    # print_case(
    #     "Scaling invariance",
    #     s, t_scaled,
    #     "High similarity (normalization works)"
    # )