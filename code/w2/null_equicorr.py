"""Type-I inflation of the independence-model full search and fast version under weak equicorrelation
z = sqrt(rho) f + sqrt(1 - rho) e (all pairwise correlations rho), critical values from the independent null.
  null_equicorr.py K seed n"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asset_core as ac
K, seed, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
t0 = time.time(); rng = np.random.default_rng(seed)
E = rng.standard_normal((n, K)); f0, s0, _ = ac.tfull_eq(E)[0], ac.tfast_eq(E)[0], None
line = []
for q in (1e-2, 1e-3):
    cf = np.quantile(f0, 1 - q); fin = np.isfinite(s0); cs = np.quantile(s0[fin], 1 - q / fin.mean())
    for rho in (0.01, 0.03, 0.1):
        Zc = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, K))
        tf = ac.tfull_eq(Zc)[0]; ts = ac.tfast_eq(Zc)[0]
        line.append(f"alpha={q:g} rho={rho}: full {np.mean(tf > cf)/q:.2f}x, fast {np.mean(ts > cs)/q:.2f}x")
print(f"K={K} n={n}: " + "; ".join(line) + f" ({time.time()-t0:.1f}s)")
