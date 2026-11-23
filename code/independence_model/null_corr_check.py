"""Type-I error of the independence-model statistics when the true null has the weak dense correlation of the
GTEx data (vhat, all |rho| < 0.2, so fastASSET's block rule treats the 44 tissues as independent).
  null_corr_check.py sim seed n      chunk: T_full / T_fast under N(0, I) and N(0, vhat)
  null_corr_check.py summary
"""
import sys, os, glob, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import asset_core as ac
DATA = os.path.join(HERE, "..", "..", "downloads", "gtexresults_v6", "gtex_v6_z.npz")
RES = os.path.join(HERE, "..", "..", "results", "independence_model", "gtex"); os.makedirs(RES, exist_ok=True)
t0 = time.time()
if sys.argv[1] == "sim":
    seed, n = int(sys.argv[2]), int(sys.argv[3])
    V = np.load(DATA)["vhat"]; L = np.linalg.cholesky(V); rng = np.random.default_rng(seed)
    out = {"fI": [], "sI": [], "fV": [], "sV": []}
    for i in range(0, n, 20000):
        m = min(20000, n - i)
        U = rng.standard_normal((m, 44)); out["fI"].append(ac.tfull_eq(U)[0]); out["sI"].append(ac.tfast_eq(U)[0])
        Zc = rng.standard_normal((m, 44)) @ L.T; out["fV"].append(ac.tfull_eq(Zc)[0]); out["sV"].append(ac.tfast_eq(Zc)[0])
    np.savez(os.path.join(RES, f"nullcorr_s{seed}.npz"), **{k: np.concatenate(v) for k, v in out.items()})
    print(f"nullcorr seed {seed} n {n}: {time.time()-t0:.1f}s")
else:
    V = np.load(DATA)["vhat"]; off = V[np.triu_indices(44, 1)]
    print(f"vhat off-diagonal: mean {off.mean():+.4f}, share positive {np.mean(off>0):.3f}; implied Var(sum of all 44)/44 = {V.sum()/44:.2f}")
    fs = sorted(glob.glob(os.path.join(RES, "nullcorr_s*.npz")))
    d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
    n = len(d["fI"]); print(f"null draws per model: {n}")
    for name, a, b in (("T_full", "fI", "fV"), ("T_fast", "sI", "sV")):
        for q in (1e-2, 1e-3, 1e-4):
            xI = d[a]; fin = np.isfinite(xI)
            c = np.quantile(xI[fin], 1 - q / fin.mean())
            print(f"  {name} alpha={q:g} (crit from independent null {c:.3f}): type-I under vhat = {np.mean(d[b] > c):.2e} ({np.mean(d[b] > c)/q:.2f} x alpha)")
