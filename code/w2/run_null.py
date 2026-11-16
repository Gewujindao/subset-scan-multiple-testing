"""Null distribution work (week 2): plain MC, importance sampling, DLM comparison, type-I error.
Resumable: every call writes one chunk file under res/; 'analyze' combines whatever chunks exist.
  mc  K seed n          plain Monte Carlo chunk (equal weights)
  is  K seed n [w]      importance-sampling chunk aimed at alpha = 5e-8 (w: 'eq' or 'ln' weights)
  analyze K [w]
"""
import sys, time, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asset_core as ac

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res")
os.makedirs(RES, exist_ok=True)


def weights(K, kind):
    if kind == "eq":
        return np.ones(K)
    r = np.random.default_rng(777 + K)          # fixed heterogeneous weights sqrt(N), N lognormal(sd 0.8)
    w = np.sqrt(r.lognormal(0.0, 0.8, K))
    return w / np.sqrt(np.mean(w ** 2))


def crit_cached(K, alpha=5e-8):
    import json
    f = os.path.join(RES, "dlm_crit_cache.json")
    cache = json.load(open(f)) if os.path.exists(f) else {}
    key = f"{K}_{alpha:g}"
    if key not in cache:
        cache[key] = ac.dlm_crit(alpha, K)
        json.dump(cache, open(f, "w"))
    return cache[key]


def stats(Z, w, kind):
    if kind == "eq":
        tf, _ = ac.tfull_eq(Z); ts, ks, _ = ac.tfast_eq(Z)
    else:
        tf, _ = ac.tfull_w(Z, w); ts, ks = ac.tfast_w(Z, w)
    return tf, ts, ks


mode = sys.argv[1]
t0 = time.time()
if mode == "mc":
    K, seed, n = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    kind = sys.argv[5] if len(sys.argv) > 5 else "eq"
    rng = np.random.default_rng(seed); w = weights(K, kind)
    out_f, out_s, out_k = [], [], []
    for i in range(0, n, 20000):
        Z = rng.standard_normal((min(20000, n - i), K))
        tf, ts, ks = stats(Z, w, kind); out_f.append(tf); out_s.append(ts); out_k.append(ks)
    np.savez(os.path.join(RES, f"mc_{kind}_K{K}_s{seed}.npz"), tf=np.concatenate(out_f), ts=np.concatenate(out_s), ks=np.concatenate(out_k))
    print(f"mc K={K} {kind} seed={seed} n={n}: {time.time()-t0:.1f}s")

elif mode == "is":
    K, seed, n = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    kind = sys.argv[5] if len(sys.argv) > 5 else "eq"
    rng = np.random.default_rng(seed); w = weights(K, kind)
    c_target = 0.95 * crit_cached(K)       # DLM is conservative for large K; aim a little below its threshold
    sf = (lambda X: ac.tfull_eq(X)[0]) if kind == "eq" else (lambda X: ac.tfull_w(X, w)[0])
    comps = ac.make_components_calibrated(K, w, c_target, sf)
    out = {"tf": [], "ts": [], "ks": [], "lw": []}
    for i in range(0, n, 5000):
        Z, lw = ac.is_sample(comps, K, min(5000, n - i), rng)
        tf, ts, ks = stats(Z, w, kind)
        out["tf"].append(tf); out["ts"].append(ts); out["ks"].append(ks); out["lw"].append(lw)
    np.savez(os.path.join(RES, f"is_{kind}_K{K}_s{seed}.npz"), **{k: np.concatenate(v) for k, v in out.items()}, c_target=c_target)
    print(f"is K={K} {kind} seed={seed} n={n} c_target(DLM 5e-8)={c_target:.3f}: {time.time()-t0:.1f}s")

elif mode == "isk":
    # IS for T_k = full-search statistic on k iid N(0,1) (= fast version's statistic given K_sel = k)
    kmin, kmax, n, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rng = np.random.default_rng(seed)
    for k in range(kmin, kmax + 1):
        w = np.ones(k)
        comps = ac.make_components_calibrated(k, w, 0.97 * crit_cached(k), lambda X: ac.tfull_eq(X)[0])
        Z, lw = ac.is_sample(comps, k, n, rng)
        tf, _ = ac.tfull_eq(Z)
        np.savez(os.path.join(RES, f"isk_k{k}_s{seed}.npz"), tf=tf, lw=lw)
    print(f"isk k={kmin}..{kmax} n={n} seed={seed}: {time.time()-t0:.1f}s")

elif mode == "fastdlm2":
    # per-k ratios r_k = P(T_k > DLM crit_k(alpha)) / alpha computed once, then mixed over K_sel ~ Bin(K, 0.025)
    import json
    from scipy.stats import binom
    data = {}
    for k in range(1, 26):
        fs = sorted(glob.glob(os.path.join(RES, f"isk_k{k}_s*.npz")))
        data[k] = (np.concatenate([np.load(f)["tf"] for f in fs]), np.concatenate([np.load(f)["lw"] for f in fs]))
    out = {}
    for alpha in (1e-4, 5e-8):
        r = {}
        for k, (tf, lw) in data.items():
            p, se = ac.is_tail(tf, lw, [ac.dlm_crit(alpha, k)]); r[k] = (p[0] / alpha, se[0] / alpha)
        print(f"alpha={alpha:g}: r_k (true/nominal) for k=1..25: " + " ".join(f"{r[k][0]:.2f}" for k in range(1, 26)))
        for K in (30, 44, 116, 300):
            tot = sum(binom.pmf(k, K, ac.PTHR) * r[k][0] for k in r)
            out[f"{K}_{alpha:g}"] = tot
            print(f"  K={K}: type-I/alpha of fast version with conditional DLM = {tot:.3f}; P(K_sel=0)={binom.pmf(0,K,ac.PTHR):.3f}; P(K_sel>25)={binom.sf(25,K,ac.PTHR):.1e}")
    json.dump(out, open(os.path.join(RES, "fastdlm_typeI.json"), "w"))
    print(f"time {time.time()-t0:.1f}s")

elif mode == "fastdlm":
    # type-I error of the fast version with its own conditional DLM p-value, K traits screened at z > 1.96
    import json
    from scipy.stats import binom
    K = int(sys.argv[2])
    out = {}
    for alpha in (1e-4, 1e-6, 5e-8, 2.5e-8, 5e-9):
        tot, worst = 0.0, 0.0
        for k in range(1, 26):
            fs = sorted(glob.glob(os.path.join(RES, f"isk_k{k}_s*.npz")))
            if not fs:
                continue
            tf = np.concatenate([np.load(f)["tf"] for f in fs]); lw = np.concatenate([np.load(f)["lw"] for f in fs])
            c = ac.dlm_crit(alpha, k)
            p, se = ac.is_tail(tf, lw, [c])
            tot += binom.pmf(k, K, ac.PTHR) * p[0]
            worst = max(worst, p[0] / alpha)
            if alpha == 5e-8 and k in (1, 2, 3, 5, 8, 12, 16, 20, 25):
                print(f"  k={k}: true P(T_k > DLM crit) / alpha = {p[0]/alpha:.3f} (rel se {se[0]/max(p[0],1e-300):.2f})")
        out[f"{alpha:g}"] = tot / alpha
        print(f"K={K} fast version with conditional DLM: type-I / alpha at alpha={alpha:g} = {tot/alpha:.3f} (max over k {worst:.2f}); P(K_sel>25) = {binom.sf(25, K, ac.PTHR):.2e}")
    json.dump(out, open(os.path.join(RES, f"fastdlm_typeI_K{K}.json"), "w"))

elif mode == "analyze":
    K = int(sys.argv[2]); kind = sys.argv[3] if len(sys.argv) > 3 else "eq"
    mc = [np.load(f) for f in sorted(glob.glob(os.path.join(RES, f"mc_{kind}_K{K}_s*.npz")))]
    isf = [np.load(f) for f in sorted(glob.glob(os.path.join(RES, f"is_{kind}_K{K}_s*.npz")))]
    tfI = np.concatenate([d["tf"] for d in isf]); tsI = np.concatenate([d["ts"] for d in isf]); lw = np.concatenate([d["lw"] for d in isf])
    print(f"K={K} {kind}: IS samples {len(lw)}, ESS fraction {np.exp(2*np.log(np.exp(lw).sum()) - np.log(np.exp(2*lw).sum()))/len(lw):.3f}")
    if mc:
        tfM = np.concatenate([d["tf"] for d in mc]); tsM = np.concatenate([d["ts"] for d in mc])
        print(f"  plain MC samples {len(tfM)}")
        for name, M_, I_ in (("T_full", tfM, tfI), ("T_fast", tsM, tsI)):
            for q in (1e-3, 1e-4, 1e-5):
                if q * len(M_) < 20:
                    continue
                c = np.quantile(M_[np.isfinite(M_)] if name == "T_fast" else M_, 1 - q / np.mean(np.isfinite(M_)))
                pm = np.mean(M_ > c); sem = np.sqrt(pm * (1 - pm) / len(M_))
                pi, sei = ac.is_tail(I_, lw, [c])
                extra = f" DLM={ac.dlm_tail(c, K)[0]:.3e}" if name == "T_full" else ""
                print(f"  {name} c={c:.3f}: MC {pm:.3e}+-{sem:.1e}  IS {pi[0]:.3e}+-{sei[0]:.1e}  z={(pi[0]-pm)/np.hypot(sem, sei[0]):+.2f}{extra}")
    # tail curves and critical values from IS
    import json
    crit = {}
    for name, I_ in (("T_full", tfI), ("T_fast", tsI)):
        fin = I_[np.isfinite(I_)]
        cs = np.linspace(np.quantile(fin, 0.05), fin.max(), 1500)
        p, se = ac.is_tail(I_, lw, cs)
        for alpha in (1e-4, 5e-5, 1e-6, 5e-8, 2.5e-8, 5e-9, 1e-10):
            j = np.nonzero(p <= alpha)[0]
            if len(j) and j[0] > 0:
                crit[f"{name}_{alpha:g}"] = float(cs[j[0]]); crit[f"{name}_{alpha:g}_relse"] = float(se[j[0]] / max(p[j[0]], 1e-300))
        for alpha in (1e-4, 1e-6, 5e-8, 5e-9, 1e-10):
            j = np.nonzero(p <= alpha)[0]
            if len(j) == 0 or j[0] == 0:
                print(f"  {name} alpha={alpha:g}: out of IS range"); continue
            j = j[0]; c = cs[j]
            line = f"  {name} alpha={alpha:g}: oracle crit {c:.3f} (IS p {p[j]:.3e}, rel se {se[j]/max(p[j],1e-300):.2f})"
            if name == "T_full" and kind == "eq":
                cd = ac.dlm_crit(alpha, K); pt, st = ac.is_tail(I_, lw, [cd])
                line += f" | DLM crit {cd:.3f}, true type-I at DLM crit {pt[0]:.3e} (+-{st[0]:.1e}) = {pt[0]/alpha:.2f} x alpha; DLM/true at oracle crit {ac.dlm_tail(c, K)[0]/p[j]:.2f}"
            print(line)
    json.dump(crit, open(os.path.join(RES, f"crit_{kind}_K{K}.json"), "w"), indent=1)
    print(f"time {time.time()-t0:.1f}s")
