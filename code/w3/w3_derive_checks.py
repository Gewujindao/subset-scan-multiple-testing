"""Short derivation checks (unguarded, < 27 s): structure statistics, FFT null tails vs Monte Carlo,
psi_s, inverse-Wishart correction.   w3_derive_checks.py part"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import w3_core as wc
from scipy.stats import norm

wc.short_guard(27.0)
part = sys.argv[1]
t0 = time.time()
if part == "struct":
    for K in (30, 116, 300):
        for st in ("A", "B"):
            V, D, R, blocks = wc.make_R(K, st)
            o = R[np.triu_indices(K, 1)]
            q = np.quantile(o, [0.1, 0.5, 0.9, 0.99])
            print(f"K={K} {st}: mean {o.mean():.4f} q10/50/90/99 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}/{q[3]:.3f} "
                  f"max {o.max():.3f} pos {np.mean(o > 0):.2f} |r|<0.05 {np.mean(np.abs(o) < 0.05):.3f} "
                  f">=0.2 {np.mean(o >= 0.2):.4f} blocks {[len(b) for b in blocks]}")
elif part == "fft":
    K = int(sys.argv[2]); n = int(sys.argv[3])
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((n, K))
    mc = wc.MixCalib(K)
    print(f"calib built {time.time()-t0:.1f}s")
    for s in mc.S:
        L = wc.ell(Z, s, K).sum(1)
        for a in (1e-2, 1e-3):
            c = mc.crit(s, a)
            print(f"K={K} s={s}: alpha {a:g} crit {c:.4f} MC tail {np.mean(L > c):.5f}")
        print(f"   crit at 5e-8/G: {mc.crit(s, 5e-8/len(mc.S)):.4f}  psi {wc.psi(s, K):.3f}")
    p = mc.minp(Z)
    print(f"Bonferroni minp: P(p<=1e-2) {np.mean(p <= 1e-2):.5f}, P(p<=1e-3) {np.mean(p <= 1e-3):.5f}")
elif part == "wish":
    K, M = int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(11)
    for st in ("A", "B"):
        V, D, R, _ = wc.make_R(K, st)
        Rh = wc.estimate_Rhat(V, D, M, rng)
        W = wc.inv_sqrt(Rh)
        C = W @ R @ W
        c2 = wc.wishart_c(K, M) ** 2
        off = C[np.triu_indices(K, 1)]
        ev = np.linalg.eigvalsh(C)
        e = np.abs(Rh - R)[np.triu_indices(K, 1)]
        print(f"K={K} M={M} {st}: mean|Rh-R| {e.mean():.4f}; tr(C)/K {np.trace(C)/K:.4f}, corrected {c2*np.trace(C)/K:.4f}; "
              f"residual offdiag mean {off.mean():.2e} sd {off.std():.4f}; eig C [{ev.min():.3f},{ev.max():.3f}]; "
              f"1'C1/K-1 {(np.ones(K)@C@np.ones(K))/K - 1:.4f}")
print(f"({time.time()-t0:.1f}s)")
