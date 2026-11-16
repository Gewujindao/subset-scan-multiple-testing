"""Week 3 evaluation simulation (run as batch jobs, each job < 5 min).
  w3_sim.py cal K n seed                         independent-null calibration of tfull_eq / tfast_eq (top 1.1%)
  w3_sim.py null K struct M b d0 d1 nrep seed    type-I counts per R-hat draw d in [d0, d1)
  w3_sim.py power K struct M scen d0 d1 nrep seed  rejections at 5e-8 per R-hat draw
Outputs in res/ (json; calibration as npz)."""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import chi2
import w3_core as wc
import asset_core as ac

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
W2RES = os.path.join(HERE, "..", "w2", "res")
os.makedirs(RES, exist_ok=True)
SID = {"I": 0, "A": 1, "B": 2}
ALPHAS = (1e-2, 1e-3, 1e-4)
NB = 10000
t0 = time.time()
if os.environ.get("W3_SHORT") == "1":
    wc.short_guard(27.0)


def cal_thresholds(K):
    """Quantiles of the independent null at the levels needed (merging all calibration chunks for K)."""
    files = sorted(glob.glob(os.path.join(RES, f"cal_K{K}_s*.npz")))
    n = 0; tf, ts = [], []
    for f in files:
        d = np.load(f); n += int(d["n"]); tf.append(d["full_top"]); ts.append(d["fast_top"])
    tf = np.sort(np.concatenate(tf))[::-1]; ts = np.sort(np.concatenate(ts))[::-1]
    out = {}
    for a in ALPHAS:
        for lev in (a, 0.8 * a, 0.2 * a):
            j = int(round(lev * n))
            out[("full", lev)] = 0.5 * (tf[j - 1] + tf[j])
            out[("fast", lev)] = 0.5 * (ts[j - 1] + ts[j])
    return out, n


def deep_thresholds(K):
    c = json.load(open(os.path.join(W2RES, f"crit_eq_K{K}.json")))

    def interp(name, a):
        lv = sorted((float(k.split("_")[-1]), v) for k, v in c.items() if k.startswith(name) and not k.endswith("relse"))
        return float(np.interp(np.log(a), np.log([x for x, _ in lv]), [v for _, v in lv]))
    return {"full": interp("T_full", 5e-8), "full08": interp("T_full", 4e-8),
            "fast": interp("T_fast", 5e-8), "fast02": interp("T_fast", 1e-8)}


def setup(K, struct, M, b, d, seed):
    V, D, R, _ = wc.make_R(K, struct)
    rngR = np.random.default_rng([seed, K, SID[struct], M, int(round(b * 1e6)), d, 1])
    Rh = wc.estimate_Rhat(V, D, M, rngR, b)
    return V, D, R, Rh, wc.inv_sqrt(Rh), wc.wishart_c(K, M), wc.block_whitener(Rh)


def stats_batch(Z, Wraw, c, Wb, mixcal):
    return wc.all_stats(Z, Wraw, c, Wb, mixcal)


mode = sys.argv[1]
if mode == "cal":
    K, n, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng([seed, K, 99])
    keep = int(np.ceil(0.011 * n))
    tf, ts = [], []
    for st in range(0, n, NB):
        Z = rng.standard_normal((min(NB, n - st), K))
        tf.append(ac.tfull_eq(Z)[0]); ts.append(ac.tfast_eq(Z)[0])
    tf = np.sort(np.concatenate(tf))[::-1][:keep]; ts = np.sort(np.concatenate(ts))[::-1][:keep]
    np.savez(os.path.join(RES, f"cal_K{K}_s{seed}.npz"), n=n, full_top=tf, fast_top=ts)
    print(f"cal K={K} n={n} seed={seed} ({time.time()-t0:.1f}s)")

elif mode == "null":
    K, struct, M, b = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
    d0, d1, nrep, seed = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
    thr, ncal = cal_thresholds(K)
    mixcal = wc.MixCalib(K)
    out = {"K": K, "struct": struct, "M": M, "b": b, "nrep": nrep, "ncal": ncal, "draws": {}}
    for d in range(d0, d1):
        V, D, R, Rh, Wraw, c, Wb = setup(K, struct, M, b, d, seed)
        rng = np.random.default_rng([seed, K, SID[struct], M, int(round(b * 1e6)), d, 2])
        cnt = {}
        for stt in range(0, nrep, NB):
            Z = wc.sample_z(min(NB, nrep - stt), V, D, rng)
            s = stats_batch(Z, Wraw, c, Wb, mixcal)
            for a in ALPHAS:
                rej = {"full_ind": s["full_ind"] > thr[("full", a)],
                       "fast": s["fast"] > thr[("fast", a)],
                       "chi2": s["chi2"] > chi2.isf(a, K),
                       "full_W": s["full_W"] > thr[("full", a)],
                       "mix": s["mix_p"] <= a,
                       "Wsplit": (s["full_W"] > thr[("full", 0.8 * a)]) | (s["fast_W"] > thr[("fast", 0.2 * a)])}
                for k, v in rej.items():
                    cnt[f"{k}@{a:g}"] = cnt.get(f"{k}@{a:g}", 0) + int(v.sum())
        C = Wraw @ R @ Wraw * c * c
        out["draws"][d] = {"counts": cnt, "trC_over_K": float(np.trace(C) / K),
                           "u1": float(np.ones(K) @ C @ np.ones(K) / K),
                           "n_clusters_multi": int(sum(len(x) > 1 for x in wc.clusters(Rh)))}
        print(f"draw {d} done ({time.time()-t0:.1f}s)", flush=True)
    fn = os.path.join(RES, f"null_K{K}_{struct}_M{M}_b{b:g}_d{d0}-{d1}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"wrote {fn} ({time.time()-t0:.1f}s)")

elif mode == "power":
    K, struct, M, scen = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    d0, d1, nrep, seed = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
    th = deep_thresholds(K)
    mixcal = wc.MixCalib(K)
    out = {"K": K, "struct": struct, "M": M, "scen": scen, "nrep": nrep, "thr": th, "draws": {}}
    for d in range(d0, d1):
        V, D, R, Rh, Wraw, c, Wb = setup(K, struct, M, 0.0, 1000 + d, seed)
        mu, idx, f = wc.scenario(K, scen, R)
        rng = np.random.default_rng([seed, K, SID[struct], M, d, 3, len(scen)])
        cnt = {}
        for stt in range(0, nrep, NB):
            Z = wc.sample_z(min(NB, nrep - stt), V, D, rng, mu)
            s = stats_batch(Z, Wraw, c, Wb, mixcal)
            rej = {"full_ind": s["full_ind"] > th["full"], "fast": s["fast"] > th["fast"],
                   "chi2": s["chi2"] > chi2.isf(5e-8, K), "full_W": s["full_W"] > th["full"],
                   "fast_W": s["fast_W"] > th["fast"], "mix": s["mix_p"] <= 5e-8,
                   "Wsplit": (s["full_W"] > th["full08"]) | (s["fast_W"] > th["fast02"])}
            for k, v in rej.items():
                cnt[k] = cnt.get(k, 0) + int(v.sum())
        out["draws"][d] = {"counts": cnt, "scale_f": float(f)}
        print(f"draw {d} done ({time.time()-t0:.1f}s)", flush=True)
    fn = os.path.join(RES, f"pow_K{K}_{struct}_{scen}_M{M}_d{d0}-{d1}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"wrote {fn} ({time.time()-t0:.1f}s)")
