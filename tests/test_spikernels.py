import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spikernel import spikernel, spikernel_2, spikernel_3


# -----------------------
# Utilities
# -----------------------
def run_test(name, fn):
    print(f"\n=== {name} ===")
    fn()


def compare(k1, k2, k3=None):
    print(f"spikernel   : {k1:.6f}")
    print(f"spikernel_2 : {k2:.6f}")
    if k3 is not None:
        print(f"spikernel_3 : {k3:.6f}")
        print(f"diff (v1-v2): {abs(k1 - k2):.6f}")
        print(f"diff (v2-v3): {abs(k2 - k3):.6f}")
    else:
        print(f"diff        : {abs(k1 - k2):.6f}")


# -----------------------
# 1. Identity Test
# -----------------------
def test_identity():
    s = np.array([[0,1,2,3,4],
                  [1,2,3,4,5]])

    k1 = spikernel(s, s)
    k2 = spikernel_2(s, s)
    k3 = spikernel_3(s, s)

    print("Expected: maximum similarity")
    compare(k1, k2, k3)


# -----------------------
# 2. Graded Similarity (μ)
# -----------------------
def test_graded_similarity():
    base = np.array([[0,1,2,3,4]])

    small_noise = base + 0.1
    large_noise = base + 2.0

    print("Small difference:")
    compare(spikernel(base, small_noise),
            spikernel_2(base, small_noise),
            spikernel_3(base, small_noise))

    print("\nLarge difference:")
    compare(spikernel(base, large_noise),
            spikernel_2(base, large_noise),
            spikernel_3(base, large_noise))

    print("\nExpected: similarity decreases with noise")


# -----------------------
# 3. Temporal Coding Sensitivity
# -----------------------
def test_temporal_order():
    s = np.array([[0,1,2,3,4]])
    t = np.array([[4,3,2,1,0]])  # reversed

    k1 = spikernel(s, t)
    k2 = spikernel_2(s, t)
    k3 = spikernel_3(s, t)

    print("Expected: low similarity (order matters)")
    compare(k1, k2, k3)


# -----------------------
# 4. Time-Warp Invariance
# -----------------------
def test_time_warp():
    s = np.array([[0,1,2,3,4]])
    t = np.array([[0,0,1,2,3,4,4]])  # stretched

    k1 = spikernel(s, t)
    k2 = spikernel_2(s, t)
    k3 = spikernel_3(s, t)

    print("Expected: still relatively high similarity")
    compare(k1, k2, k3)


# -----------------------
# 5. Recency Weighting (λ)
# -----------------------
def test_recency():
    early = np.array([[1,1,5,5,1,1,1,1,1,1,1,1,1]])
    late  = np.array([[1,1,1,1,1,1,1,1,1,5,5,1,1]])

    ref   = np.array([[1,1,1,5,5,1,1,1,1,1,1,1,1]])

    print("Match in early region:")
    compare(spikernel(ref, early),
            spikernel_2(ref, early),
            spikernel_3(ref, early))

    print("\nMatch in late region:")
    compare(spikernel(ref, late),
            spikernel_2(ref, late),
            spikernel_3(ref, late))

    print("\nExpected: late match > early match")


# -----------------------
# 6. Noise Robustness
# -----------------------
def test_noise_robustness():
    base = np.random.rand(3, 50)
    noisy = base + np.random.normal(0, 0.1, base.shape)

    k1 = spikernel(base, noisy)
    k2 = spikernel_2(base, noisy)
    k3 = spikernel_3(base, noisy)

    print("Expected: still high similarity under noise")
    compare(k1, k2, k3)


# -----------------------
# 7. Multi-length Pattern
# -----------------------
def test_multiscale():
    short = np.array([[1,2,3]])
    long  = np.array([[0,1,2,3,0]])

    k1 = spikernel(short, long)
    k2 = spikernel_2(short, long)
    k3 = spikernel_3(short, long)

    print("Expected: detect embedded pattern")
    compare(k1, k2, k3)


# -----------------------
# 8. Performance Benchmark
# -----------------------
def test_performance():
    s = np.random.rand(5, 100)
    t = np.random.rand(5, 100)

    start = time.time()
    spikernel(s, t)
    t1 = time.time() - start

    start = time.time()
    spikernel_2(s, t)
    t2 = time.time() - start

    start = time.time()
    spikernel_3(s, t)
    t3 = time.time() - start

    print(f"spikernel   time: {t1:.6f}s")
    print(f"spikernel_2 time: {t2:.6f}s")
    print(f"spikernel_3 time: {t3:.6f}s")


# -----------------------
# Run All
# -----------------------
if __name__ == "__main__":
    run_test("Identity", test_identity)
    run_test("Graded Similarity", test_graded_similarity)
    run_test("Temporal Order", test_temporal_order)
    run_test("Time Warp", test_time_warp)
    run_test("Recency Weighting", test_recency)
    run_test("Noise Robustness", test_noise_robustness)
    run_test("Multi-scale Pattern", test_multiscale)
    run_test("Performance", test_performance)