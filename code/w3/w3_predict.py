"""Deterministic numbers for the pre-registered predictions (no simulation of the tests):
  - signal scenarios exactly as the simulation builds them (information-matched under R)
  - exact per-component power of the subset-mixture test via FFT (whitened coordinates, R-hat = R)
  - whitened noncentrality on the true set for the full-subset max; chi2 power (noncentral chi2)
  - type-I predictions from the variance-inflation law
  w3_predict.py K I,A,B   -> res/predict_K<K>_<structs>.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, chi2, ncx2
import w3_core as wc

if os.environ.get("W3_GUARDED") != "1":
    wc.short_guard(27.0)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res")
os.makedirs(RES, exist_ok=True)
K = int(sys.argv[1])
t0 = time.time()
ALPHA = 5e-8
S = wc.grid_s(K)
G = len(S)


def scenario(K, scen, struct, R):
    """Means for the scenario: week-2 shapes, positions fixed by seed, scaled so sqrt(mu' R^-1 mu) = ||mu0||."""
    rng = np.random.default_rng([2026, K, {"diffuse": 0, "sparse": 1, "mixed": 2}[scen]])
    mu0 = np.zeros(K)
    if scen == "diffuse":
        s = int(round(0.3 * K)); idx = rng.choice(K, s, replace=False); mu0[idx] = 7 / np.sqrt(s)
    elif scen == "sparse":
        idx = rng.choice(K, 2, replace=False); mu0[idx] = 7 / np.sqrt(2)
    else:
        idx = rng.choice(K, 2 + int(round(0.2 * K)), replace=False); mu0[idx[:2]] = 3.5; mu0[idx[2:]] = 0.8
    f = np.linalg.norm(mu0) / np.sqrt(mu0 @ np.linalg.solve(R, mu0))
    return mu0 * f, idx, f


NFFT = 2 ** 16
CRIT = {}


def mix_power(means):
    m = np.round(means, 2)
    out = {}
    for s in S:
        if s >= K:
            out[s] = float(norm.sf(norm.isf(ALPHA / G) - m.sum() / np.sqrt(K)))
        else:
            if s not in CRIT:
                g, sf0 = wc.comp_pmf_sum(K, s, nfft=NFFT)
                CRIT[s] = float(np.interp(-np.log(ALPHA / G), -np.log(np.maximum(sf0, 1e-300)), g))
            g1, sf1 = wc.comp_pmf_sum(K, s, means=m, nfft=NFFT)
            out[s] = float(np.interp(CRIT[s], g1, sf1))
    return out


res = {"K": K, "G": G, "S": S, "psi": {s: wc.psi(s, K) for s in S}}
for struct in sys.argv[2].split(","):
    V, D, R, blocks = wc.make_R(K, struct)
    W = wc.inv_sqrt(R)
    o = R[np.triu_indices(K, 1)]
    rr = {"rho_bar": float(o.mean())}
    # systematic under-estimate b = 0.003 of every off-diagonal entry: residual mean correlation after whitening
    if struct != "I":
        Wb = wc.inv_sqrt(R - 0.003 * (1 - np.eye(K)))
        C = Wb @ R @ Wb
        rr["rtilde_b003"] = float(C[np.triu_indices(K, 1)].mean())
        rr["diag_b003"] = float(np.mean(np.diag(C)) - 1)
    for scen in ("diffuse", "sparse", "mixed"):
        mu, idx, f = scenario(K, scen, struct, R)
        mt = W @ mu
        s_true = len(idx)
        eta_W = float(mt[idx].sum() / np.sqrt(s_true))
        top = np.sort(mt)[::-1]
        eta_W_best = float(np.max(np.cumsum(top) / np.sqrt(np.arange(1, K + 1))))
        eta_A = float(mu[idx].sum() / np.sqrt(R[np.ix_(idx, idx)].sum()))
        lam = float(mu @ np.linalg.solve(R, mu))
        rr[scen] = {"scale_f": float(f), "eta_W_trueset": eta_W, "eta_W_bestprefix_of_means": eta_W_best,
                    "eta_ASSET_R_trueset": eta_A, "lambda": lam,
                    "chi2_power": float(ncx2.sf(chi2.isf(ALPHA, K), K, lam)),
                    "mix_power": mix_power(mt), "mean_per_signal_trait": float(mu[idx].mean())}
    res[struct] = rr
    print(struct, f"{time.time()-t0:.1f}s", flush=True)
json.dump(res, open(os.path.join(RES, f"predict_K{K}_{sys.argv[2].replace(',', '')}.json"), "w"), indent=1)
print(json.dumps(res, indent=1)[:6000])
print(f"({time.time()-t0:.1f}s)")
