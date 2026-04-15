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
# 2 & 3. SPIKERNEL FOR ALL n-LONG SUBSEQUENCES — single DP pass
# ===========================================================================
 
def spikernel_dp(
    s: np.ndarray,
    t: np.ndarray,
    n_max: int,
    lam: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fills both DP tables in one pass over all (i, j) positions.
 
    Tables (shape: (len_s+1, len_t+1, n_max+1)):
        K [i, j, n]  =  K_n (s_{1:i}, t_{1:j})
        Kp[i, j, n]  =  K'_n(s_{1:i}, t_{1:j})
 
    Base cases (Eq. 4):
        K [i, j, 0] = 1   for all i, j      (K_0 = 1 everywhere)
        Kp[i, j, 0] = 1   for all i, j      (K'_0 = 1 everywhere)
        K [0, j, n] = 0   for n >= 1        (empty prefix, no n-pattern)
        Kp[i, 0, n] = 0   for n >= 1        (empty prefix, no n-pattern)
 
    Inner recurrences for n = 1..n_max, i = 1..len_s, j = 1..len_t:
 
        Eq. 3 — K'_n update (auxiliary variable, completed by s_i & t_j):
            Kp[i, j, n] = lambda^2 * K[i-1, j-1, n-1] * mu^(0.5*||s_i-t_j||^2)
                        + lambda   * Kp[i, j-1, n]
 
        Eq. 2 — K_n update (total similarity for n-patterns):
            K[i, j, n]  = lambda * K[i-1, j, n]  +  Kp[i, j, n]
 
    Parameters
    ----------
    s, t   : np.ndarray, shape (d, len)
    n_max  : int    — maximum pattern length
    lam    : float  — temporal decay lambda  in (0, 1)
    mu     : float  — similarity decay mu    in (0, 1)
 
    Returns
    -------
    K, Kp  : np.ndarray, each shape (len_s+1, len_t+1, n_max+1)
    """
    _, len_s = s.shape
    _, len_t = t.shape
 
    # Allocate both tables — zeros by default, which handles:
    #   K[0, j, n>=1] = 0, K[i, 0, n>=1] = 0  (Eq. 4 boundary conditions)
    K  = np.zeros((len_s + 1, len_t + 1, n_max + 1))
    Kp = np.zeros((len_s + 1, len_t + 1, n_max + 1))
 
    # Base case (Eq. 4): K_0 = 1 and K'_0 = 1 everywhere
    K [:, :, 0] = 1.0
    Kp[:, :, 0] = 1.0
 
    # Single pass: fill all n simultaneously at each (i, j)
    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
 
            # Snapshot similarity at current position: mu^(0.5 * ||s_i - t_j||^2)
            # (shared across all n, computed once per (i,j))
            dist_sq = bin_wise_distance(s[:, i - 1], t[:, j - 1])
            sim = mu ** (0.5 * dist_sq)          # scalar, reused for all n
 
            # Fill all pattern lengths n = 1..n_max at this (i, j)
            for n in range(1, n_max + 1):
 
                # Eq. 3 — K'_n(s_{1:i}, t_{1:j}):
                #   lambda^2 * K_{n-1}(s_{1:i-1}, t_{1:j-1}) * sim
                #   + lambda * K'_n(s_{1:i}, t_{1:j-1})
                Kp[i, j, n] = (lam ** 2) * K[i - 1, j - 1, n - 1] * sim \
                            +  lam        * Kp[i, j - 1, n]
 
                # Eq. 2 — K_n(s_{1:i}, t_{1:j}):
                #   lambda * K_n(s_{1:i-1}, t_{1:j}) + K'_n(s_{1:i}, t_{1:j})
                K[i, j, n] = lam * K[i - 1, j, n] + Kp[i, j, n]
 
    return K, Kp
 
 
# ===========================================================================
# 4. MAIN SPIKERNEL
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
    Main Spikernel: weighted sum of all n-long subsequence kernels.
 
    Formula (Eq. 5):
        K(s, t) = sum_{i=1}^{n_max} q^i * K_i(s, t)
 
    Reads K_i directly from the precomputed DP table K[len_s, len_t, i].
 
    Parameters
    ----------
    s, t   : np.ndarray, shape (d, len)
    n_max  : int    — maximum pattern length
    lam    : float  — temporal decay lambda
    mu     : float  — similarity decay mu
    q      : float  — pattern-length weighting factor
 
    Returns
    -------
    float  — K(s, t)
    """
    _, len_s = s.shape
    _, len_t = t.shape
 
    # Fill both DP tables in a single pass (Eqs. 2–4)
    K, _ = spikernel_dp(s, t, n_max, lam, mu)
 
    # Eq. 5 — weighted sum over pattern lengths
    total = 0.0
    for i in range(1, n_max + 1):
        Ki = K[len_s, len_t, i]          # K_i(s, t) read from the table
        contribution = (q ** i) * Ki
        total += contribution

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
    lam = 0.99
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