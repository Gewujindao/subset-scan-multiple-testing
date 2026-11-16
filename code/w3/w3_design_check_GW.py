"""Design-stage check (R known exactly, alpha 1e-3/1e-4 only, separate seeds from the evaluation):
whitened full-subset max T_W versus the subset GLRT T_G = max_S 1_S'R^{-1}z / sqrt(1_S'R^{-1}1_S)
(prefix heuristic on y = R^{-1} z).  Diffuse scenario information-matched as in w3_predict.
  w3_design_check_GW.py K struct n_null n_alt"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import w3_core as wc
import asset_core as ac

wc.short_guard(27.0)
K, st, n0, n1 = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t0 = time.time()
V, D, R, _ = wc.make_R(K, st)
P = np.linalg.inv(R)
W = wc.inv_sqrt(R)
rng = np.random.default_rng(31337)


def tG(Z, nb=200):
    Y = Z @ P
    out = np.empty(len(Z))
    for b in range(0, len(Z), nb):
        y = Y[b:b + nb]
        perm = np.argsort(-y, axis=1)
        A = np.cumsum(np.take_along_axis(y, perm, 1), 1)
        Pp = P[perm[:, :, None], perm[:, None, :]]
        Q = np.cumsum(np.cumsum(Pp, 1), 2)[:, np.arange(K), np.arange(K)]
        out[b:b + nb] = (A / np.sqrt(Q)).max(1)
    return out


rs = np.random.default_rng([2026, K, 0])
mu0 = np.zeros(K); s = int(round(0.3 * K)); idx = rs.choice(K, s, replace=False); mu0[idx] = 7 / np.sqrt(s)
mu = mu0 * np.linalg.norm(mu0) / np.sqrt(mu0 @ P @ mu0)
Z0 = wc.sample_z(n0, V, D, rng)
g0, w0 = tG(Z0), ac.tfull_eq(Z0 @ W)[0]
Z1 = wc.sample_z(n1, V, D, rng, mu)
g1, w1 = tG(Z1), ac.tfull_eq(Z1 @ W)[0]
for a in (1e-2, 1e-3, 1e-4):
    cg, cw = np.quantile(g0, 1 - a), np.quantile(w0, 1 - a)
    print(f"K={K} {st} alpha={a:g}: crit G {cg:.3f} W {cw:.3f} | diffuse power G {np.mean(g1 > cg):.3f} W {np.mean(w1 > cw):.3f}")
print(f"null medians G {np.median(g0):.3f} W {np.median(w0):.3f}; ({time.time()-t0:.1f}s)")
