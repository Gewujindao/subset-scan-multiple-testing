"""Calibrated power comparison.
  run_power.py K scenario seed n [kind]     one chunk; appends to results/independence_model/pow_<kind>_K<K>_<scenario>_s<seed>.npz
  run_power.py summary K [kind]
Scenarios (non-centrality 7 on the optimal set): diffuse s=round(0.3K) x 7/sqrt(s); sparse 2 x 7/sqrt(2);
mixed 2 x 3.5 + round(0.2K) x 0.8.  Heterogeneous weights ('ln'): z means beta*w_k on the true set with
beta = 7/sqrt(sum_S w^2) (diffuse/sparse) so that the oracle subset has non-centrality 7.
Decisions at alpha = 5e-8: oracle-calibrated (IS critical values), each method's own DLM p-value, and the
Bonferroni combination (either test at alpha/2)."""
import sys, os, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asset_core as ac

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "independence_model")
ALPHA = 5e-8


def weights(K, kind):
    if kind == "eq":
        return np.ones(K)
    r = np.random.default_rng(777 + K)
    w = np.sqrt(r.lognormal(0.0, 0.8, K))
    return w / np.sqrt(np.mean(w ** 2))


def means(K, scen, w, kind, rng):
    mu = np.zeros(K); truth = np.zeros(K, bool)
    if scen == "diffuse":
        s = int(round(0.3 * K)); idx = rng.choice(K, s, replace=False)
        mu[idx] = 7 / np.sqrt(s) if kind == "eq" else w[idx] * 7 / np.sqrt(np.sum(w[idx] ** 2))
    elif scen == "sparse":
        idx = rng.choice(K, 2, replace=False)
        mu[idx] = 7 / np.sqrt(2) if kind == "eq" else w[idx] * 7 / np.sqrt(np.sum(w[idx] ** 2))
    else:
        idx = rng.choice(K, 2 + int(round(0.2 * K)), replace=False)
        mu[idx[:2]] = 3.5; mu[idx[2:]] = 0.8
    truth[idx] = True
    return mu, truth


def dlm_crit_cached(k):
    f = os.path.join(RES, "dlm_crit_power.json")
    cache = json.load(open(f)) if os.path.exists(f) else {}
    out = {}
    for a in (ALPHA, ALPHA / 2):
        key = f"{k}_{a:g}"
        if key not in cache:
            cache[key] = ac.dlm_crit(a, k) if k > 0 else float("inf")
        out[a] = cache[key]
    json.dump(cache, open(f, "w"))
    return out


mode = sys.argv[1]
t0 = time.time()
if mode != "summary":
    K, scen, seed, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    kind = sys.argv[5] if len(sys.argv) > 5 else "eq"
    rng = np.random.default_rng(seed); w = weights(K, kind)
    crit = json.load(open(os.path.join(RES, f"crit_{kind}_K{K}.json")))
    cF, cF2 = crit[f"T_full_{ALPHA:g}"], crit[f"T_full_{ALPHA/2:g}"]
    cS, cS2 = crit[f"T_fast_{ALPHA:g}"], crit[f"T_fast_{ALPHA/2:g}"]
    rec = {k: [] for k in ("of", "os", "oc", "df", "ds", "dc", "nf", "ns", "tpf", "tps")}
    dK = dlm_crit_cached(K)
    for i in range(0, n, 5000):
        m = min(5000, n - i)
        Z = rng.standard_normal((m, K)); truths = np.zeros((m, K), bool)
        for r_ in range(m):
            mu, tr = means(K, scen, w, kind, rng); Z[r_] += mu; truths[r_] = tr
        if kind == "eq":
            tf, mf = ac.tfull_eq(Z); ts, ks, ms = ac.tfast_eq(Z)
            order = np.argsort(-Z, axis=1)
            keep, zt = ac.screen_adjust(Z); order_s = np.argsort(-zt, axis=1)
        else:
            tf, mf = ac.tfull_w(Z, w); ts, ks = ac.tfast_w(Z, w)
            order = np.argsort(-(Z / w), axis=1)
            keep, zt = ac.screen_adjust(Z); order_s = np.argsort(-np.where(keep, zt / w, -np.inf), axis=1); ms = None
        sel_f = np.zeros_like(truths); sel_s = np.zeros_like(truths)
        for r_ in range(m):
            sel_f[r_, order[r_, :mf[r_]]] = True
            if kind == "eq" and ms[r_] > 0:
                sel_s[r_, order_s[r_, :ms[r_]]] = True
        rec["of"].append(tf > cF); rec["os"].append(ts > cS); rec["oc"].append((tf > cF2) | (ts > cS2))
        rec["df"].append(tf > dK[ALPHA])
        cmap = {int(k): dlm_crit_cached(int(k)) for k in np.unique(ks)}
        dks = np.array([cmap[int(k)][ALPHA] for k in ks]); dks2 = np.array([cmap[int(k)][ALPHA / 2] for k in ks])
        rec["ds"].append(ts > dks); rec["dc"].append((tf > dK[ALPHA / 2]) | (ts > dks2))
        rec["nf"].append(sel_f.sum(1)); rec["ns"].append(sel_s.sum(1))
        rec["tpf"].append((sel_f & truths).sum(1)); rec["tps"].append((sel_s & truths).sum(1))
    np.savez(os.path.join(RES, f"pow_{kind}_K{K}_{scen}_s{seed}.npz"), **{k: np.concatenate(v) for k, v in rec.items()})
    print(f"power K={K} {scen} {kind} seed={seed} n={n}: {time.time()-t0:.1f}s")
else:
    K = int(sys.argv[2]); kind = sys.argv[3] if len(sys.argv) > 3 else "eq"
    for scen in ("diffuse", "sparse", "mixed"):
        fs = sorted(glob.glob(os.path.join(RES, f"pow_{kind}_K{K}_{scen}_s*.npz")))
        if not fs:
            continue
        d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
        n = len(d["of"]); se = lambda x: np.sqrt(x.mean() * (1 - x.mean()) / n)
        print(f"K={K} {kind} {scen} (n={n}): oracle full {d['of'].mean():.3f}+-{se(d['of']):.3f} fast {d['os'].mean():.3f} "
              f"comb {d['oc'].mean():.3f} | own-DLM full {d['df'].mean():.3f} fast {d['ds'].mean():.3f} comb {d['dc'].mean():.3f} | "
              f"selected size median full {np.median(d['nf']):.0f} fast {np.median(d['ns']):.0f}; true in selected full {d['tpf'].mean():.1f} fast {d['tps'].mean():.1f}")
