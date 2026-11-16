"""Check of the correlated-ASSET structure (D3) against exhaustive search, K = 14 (unguarded, < 27 s):
  at the exhaustive optimum S*, with lambda = A/(2Q), C = sum_{l in S*} v_l and phi_k = a_k - lambda - 2 lambda v_k.C:
  phi_k >= 0 => k in S*;  phi_k <= -2 lambda |v_k|^2 => k not in S*;  otherwise 'gray'.
  Also: how often the prefix heuristic (+ single add/remove refinement) misses the exhaustive optimum."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import w3_core as wc
import asset_core as ac

wc.short_guard(27.0)
t0 = time.time()
K, n = 14, 150
M = ac.all_subsets(K)
w = np.ones(K)
for st in ("A", "B"):
    V, D, R, _ = wc.make_R(K, st, seed=5)
    rng = np.random.default_rng(8)
    viol = gray_tot = gray_max = miss_pre = miss_ref = 0
    gap = []
    for _ in range(n):
        mu = np.zeros(K); idx = rng.choice(K, rng.integers(1, K), replace=False); mu[idx] = rng.uniform(0.5, 2.5, len(idx))
        z = wc.sample_z(1, V, D, rng, mu)[0]
        v, m = ac.exhaustive_max(z, w, R, M)
        A = z[m].sum(); Q = R[np.ix_(m, m)].sum(); lam = A / (2 * Q)
        C = V[m].sum(0)
        phi = z - lam - 2 * lam * (V @ C)
        lo = -2 * lam * (V ** 2).sum(1)
        viol += int(np.any((phi >= 0) & ~m) or np.any((phi <= lo) & m))
        g = int(np.sum((phi > lo) & (phi < 0))); gray_tot += g; gray_max = max(gray_max, g)
        vp, mp = ac.prefix_max_corr(z, w, R)
        vr, mr = ac.local_refine(z, w, R, mp)
        miss_pre += int(vp < v - 1e-9); miss_ref += int(vr < v - 1e-9); gap.append(1 - vp / v)
    print(f"{st}: violations of the threshold-with-gray-zone rule {viol}/{n}; gray elements mean {gray_tot/n:.2f} max {gray_max}; "
          f"prefix misses {miss_pre}/{n} (max rel gap {max(gap):.4f}, mean {np.mean(gap):.5f}); after refinement {miss_ref}/{n}")
print(f"({time.time()-t0:.1f}s)")
