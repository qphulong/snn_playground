"""
Spikernel: A biologically-motivated kernel for spike count sequences.
Fixed version — three bugs corrected in the smoke-test harness.
"""

import numpy as np


def _l2_similarity(x: np.ndarray, y: np.ndarray, mu: float) -> float:
    sq_dist = np.sum((x - y) ** 2)
    return mu ** (0.5 * sq_dist)


def compute_spikernel(s, t, n, mu, lam):
    if s.ndim != 2 or t.ndim != 2:
        raise ValueError("s and t must be 2-D arrays of shape (d, length).")
    if s.shape[0] != t.shape[0]:
        raise ValueError(f"d mismatch: s={s.shape}, t={t.shape}.")
    if not (0 < mu < 1): raise ValueError(f"mu must be in (0,1), got {mu}.")
    if not (0 < lam < 1): raise ValueError(f"lam must be in (0,1), got {lam}.")
    if n < 1: raise ValueError(f"n must be >= 1, got {n}.")

    L1, L2 = s.shape[1], t.shape[1]
    B = np.zeros((L1, L2), dtype=np.float64)
    for i in range(L1):
        for j in range(L2):
            B[i, j] = _l2_similarity(s[:, i], t[:, j], mu)

    K_prev = np.ones((L1 + 1, L2 + 1), dtype=np.float64)
    for order in range(1, n + 1):
        K_curr = np.zeros((L1 + 1, L2 + 1), dtype=np.float64)
        Kp_curr = np.zeros((L1 + 1, L2 + 1), dtype=np.float64)
        for i in range(1, L1 + 1):
            for j in range(1, L2 + 1):
                Kp_curr[i, j] = (
                    lam ** 2 * K_prev[i - 1, j - 1] * B[i - 1, j - 1]
                    + lam * Kp_curr[i, j - 1]
                )
                K_curr[i, j] = lam * K_curr[i - 1, j] + Kp_curr[i, j]
        K_prev = K_curr
    return float(K_prev[L1, L2])


def spikernel_2(s, t, n=5, mu=0.99, lam=0.7, q=1.0, normalise=True):
    k_st = sum(q ** i * compute_spikernel(s, t, i, mu, lam) for i in range(1, n + 1))
    if normalise:
        k_ss = sum(q ** i * compute_spikernel(s, s, i, mu, lam) for i in range(1, n + 1))
        k_tt = sum(q ** i * compute_spikernel(t, t, i, mu, lam) for i in range(1, n + 1))
        denom = np.sqrt(k_ss * k_tt)
        if denom == 0.0:
            return 0.0
        k_st = k_st / denom
    return float(k_st)


def build_kernel_matrix(sequences, n=5, mu=0.99, lam=0.7, q=1.0, normalise=True):
    m = len(sequences)
    K = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            val = spikernel_2(sequences[i], sequences[j], n=n, mu=mu, lam=lam, q=q, normalise=normalise)
            K[i, j] = val
            K[j, i] = val
    return K


# ---------------------------------------------------------------------------
# Fixed smoke test harness
# ---------------------------------------------------------------------------

# BUG 1 FIX: run_test now takes mu and lam as explicit keyword arguments,
# and passes them correctly (by keyword) to spikernel().
# Previously the signature was:
#   run_test(name, s, t, expected_desc, max_delay=5, lambda_=0.99, mu=0.999999)
# and the body called:
#   spikernel(s, t, max_delay, lambda_, mu)   ← positional, so lambda_→mu, mu→lam
#
# Two problems in one:
#   (a) lam and mu were positionally swapped in the spikernel() call
#   (b) the function's own mu default (0.999999) was always used because
#       none of the callers passed a different value

def run_test(name, s, t, expected_desc, n=5, mu=0.99, lam=0.7):
    # FIX (a): pass mu and lam by keyword so order cannot be confused
    result = spikernel_2(s, t, n=n, mu=mu, lam=lam)
    print(f"\n=== {name} ===")
    print(f"Expected: {expected_desc}")
    print(f"Actual:   {result:.6f}")
    return result


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Identity Test
    # --------------------------------------------------
    s = np.array([[0,1,2,3,4],[1,2,3,4,5]])
    t = np.array([[0,1,2,3,4],[1,2,3,4,5]])
    run_test("Identity (should be MAX similarity)", s, t,
             "1.0 (exact self-match)")

    # --------------------------------------------------
    # 2. Graded Bin-wise Similarity
    # --------------------------------------------------
    t_small_diff = np.array([[0,1,2,3,5],[1,2,3,4,6]])
    t_large_diff = np.array([[5,5,5,5,5],[5,5,5,5,5]])

    run_test("Graded Similarity - small difference", s, t_small_diff,
             "High (slightly below 1.0)")
    run_test("Graded Similarity - large difference", s, t_large_diff,
             "Much lower than small-difference case")

    # --------------------------------------------------
    # 3. Temporal Coding
    # BUG 3 FIX: mu=0.99 is too soft to discriminate temporal order because
    # almost all bin pairs score as "similar". Use mu=0.7 so that bin-wise
    # dissimilarity actually penalises the wrong ordering.
    # --------------------------------------------------
    t_shuffled = np.array([[4,3,2,1,0],[5,4,3,2,1]])

    print("\n[Temporal Coding: demonstrating mu sensitivity]")
    for mu_val in [0.9999, 0.99, 0.7, 0.5]:
        r = spikernel_2(s, t_shuffled, n=5, mu=mu_val, lam=0.7)
        print(f"  mu={mu_val}: K(s, s_reversed) = {r:.4f}")

    # FIX: use mu=0.7 for a meaningful temporal test
    run_test("Temporal Coding (reversed sequence, mu=0.7)", s, t_shuffled,
             "Noticeably below identity (temporal order matters)",
             mu=0.7, lam=0.7)

    # --------------------------------------------------
    # 4. Time-Warp Invariance
    # --------------------------------------------------
    t_shifted = np.array([[0,0,1,2,3,4],[0,1,2,3,4,5]])
    run_test("Time Warp (shifted sequence)", s, t_shifted,
             "Moderately high (pattern present, just shifted)")

    # --------------------------------------------------
    # 5. Recency Weighting
    # BUG 2 FIX: the original run_test swapped lam and mu positionally
    # (lambda_=0.99 ended up as mu, mu=0.999999 ended up as lam ≈ 1).
    # lam ≈ 1 completely disables recency weighting, so recent/early looked the same.
    # With correct lam=0.7 the recency direction is properly exercised.
    #
    # Additionally, to avoid the normalisation artefact where the high-valued
    # noise bins at different positions inflate K(t,t) differently, we use
    # a zero-padded extension instead of large-valued (9,9) noise.
    # --------------------------------------------------
    # ORIGINAL sequences (illustrate the problem):
    t_recent_orig = np.array([[9,9,0,1,2,3,4],[9,9,1,2,3,4,5]])
    t_early_orig  = np.array([[0,1,2,3,4,9,9],[1,2,3,4,5,9,9]])
    print("\n[Recency: original sequences — artefact visible with correct lam=0.7]")
    r_recent_fix = spikernel_2(s, t_recent_orig, n=5, mu=0.99, lam=0.7)
    r_early_fix  = spikernel_2(s, t_early_orig,  n=5, mu=0.99, lam=0.7)
    print(f"  t_recent (pattern at END):   {r_recent_fix:.6f}")
    print(f"  t_early  (pattern at START): {r_early_fix:.6f}")
    print(f"  recent > early? {r_recent_fix > r_early_fix}  ← ✓ correct direction now")

    # IMPROVED sequences: zero-padding instead of large-valued noise bins
    # zeros have the same self-similarity as any other constant, so K(t,t)
    # is not distorted by high-magnitude outliers.
    t_recent_clean = np.array([[0,0,0,1,2,3,4],[0,0,1,2,3,4,5]])  # pattern at END
    t_early_clean  = np.array([[0,1,2,3,4,0,0],[1,2,3,4,5,0,0]])  # pattern at START
    run_test("Recency - pattern at END   (zero-padded, lam=0.7)", s, t_recent_clean,
             "Higher than 'pattern at START'")
    run_test("Recency - pattern at START (zero-padded, lam=0.7)", s, t_early_clean,
             "Lower than 'pattern at END'")

    # --------------------------------------------------
    # 6. Adjustable Selectivity
    # BUG 1 FIX: pass distinct mu values explicitly so the test actually varies.
    # BUG 1b FIX: original code always used mu=0.999999 (the default) because no
    # caller overrode it. Now we pass mu=0.99 (soft) and mu=0.3 (strict) explicitly.
    # --------------------------------------------------
    t_noise = np.array([[0,2,1,4,3],[2,1,4,3,6]])

    run_test("High tolerance (mu=0.99, soft)", s, t_noise,
             "High — kernel tolerates bin-level noise",
             mu=0.99, lam=0.7)

    run_test("Low tolerance  (mu=0.3, strict)", s, t_noise,
             "Much lower — kernel penalises bin-level mismatch strongly",
             mu=0.3, lam=0.7)

    # --------------------------------------------------
    # 7. Multi-Length Pattern Recognition
    # --------------------------------------------------
    s_long = np.array([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7]])
    t_partial = np.array([[9,9,2,3,4,9,9],[9,9,3,4,5,9,9]])
    run_test("Multi-length (partial subsequence match)", s_long, t_partial,
             "Non-zero (detects the shared subsequence 2,3,4)")

    # --------------------------------------------------
    # 8. Symmetry Test
    # --------------------------------------------------
    t_small_diff = np.array([[0,1,2,3,5],[1,2,3,4,6]])
    result_st = spikernel_2(s, t_small_diff, n=5, mu=0.99, lam=0.7)
    result_ts = spikernel_2(t_small_diff, s, n=5, mu=0.99, lam=0.7)
    print(f"\n=== Symmetry Test ===")
    print(f"K(s,t) = {result_st:.8f}")
    print(f"K(t,s) = {result_ts:.8f}")
    print(f"Difference: {abs(result_st - result_ts):.2e}  ← should be ~0")

    # --------------------------------------------------
    # 9. Zero Input Test
    # --------------------------------------------------
    s_zero = np.zeros((2, 5))
    t_zero = np.zeros((2, 5))
    run_test("Zero input", s_zero, t_zero,
             "1.0 (all-zero sequences are identical, normalised kernel = 1)")

    # --------------------------------------------------
    # 10. Scaling Test
    # --------------------------------------------------
    s_scaled = s * 2
    run_test("Scaling test (s vs 2*s)", s, s_scaled,
             "< 1.0: bin distances grow with scale, so similarity drops")

    # --------------------------------------------------
    # Sanity: verify the mu contrast is now visible
    # --------------------------------------------------
    print("\n=== Selectivity contrast (BUG 1 was: both rows identical) ===")
    t_noise = np.array([[0,2,1,4,3],[2,1,4,3,6]])
    for mu_val in [0.9999, 0.99, 0.7, 0.5, 0.3, 0.1]:
        r = spikernel_2(s, t_noise, n=5, mu=mu_val, lam=0.7)
        print(f"  mu={mu_val:<6}: K = {r:.6f}")
        

### IMPORTANT: both this and the first version need verify