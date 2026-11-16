"""Summaries of the week-3 evaluation (unguarded, < 27 s).  w3_summary.py  -> res/summary.txt and stdout
Type I: pooled ratio over R-hat draws (count / (n alpha)), 95% Poisson interval, between-draw sd of log ratio,
model part of that sd after removing Monte-Carlo noise, kappa from the pooled 1e-3 ratio and its 5e-8 extrapolation.
Power: mean over draws at 5e-8; diffuse gain ratio against the frozen week-2 independent gain; sparse ratio to fast."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import brentq

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res")
METH = ["full_ind", "fast", "chi2", "full_W", "mix", "Wsplit"]
GAIN_IND = {30: 0.459 - 0.088, 116: 0.136 - 0.001, 300: 0.051 - 0.000}
lines = []


def P(s=""):
    print(s); lines.append(s)


def ratio_of_kappa(k, a):
    x = norm.isf(a)
    return norm.sf(x / np.sqrt(1 + k)) / a


def kappa_of_ratio(r, a):
    if r <= 1:
        return 0.0
    return brentq(lambda k: ratio_of_kappa(k, a) - r, 0, 1e3)


null = {}
for f in glob.glob(os.path.join(RES, "null_K*.json")):
    d = json.load(open(f))
    key = (d["K"], d["struct"], d["M"], d["b"])
    null.setdefault(key, {"nrep": d["nrep"], "draws": {}, "ncal": d["ncal"]})["draws"].update(d["draws"])

P("== type I: pooled ratio [95% CI] | between-draw sd(log ratio) total/model | kappa(1e-3) -> 5e-8 ratio ==")
for key in sorted(null):
    K, st, M, b = key
    dd = null[key]; n = dd["nrep"]; draws = list(dd["draws"].values()); nd = len(draws)
    P(f"K={K} {st} M={M} b={b:g}  draws={nd} reps/draw={n} ncal={dd['ncal']}  "
      f"trC/K mean {np.mean([x['trC_over_K'] for x in draws]):.4f}  1'C1/K sd {np.std([x['u1'] for x in draws]):.4f}")
    for m in METH:
        parts = []
        for a in (1e-2, 1e-3, 1e-4):
            c = np.array([x["counts"][f"{m}@{a:g}"] for x in draws])
            tot = c.sum(); r = tot / (nd * n * a)
            lo = chi2.ppf(0.025, 2 * tot) / 2 / (nd * n * a) if tot > 0 else 0.0
            hi = chi2.ppf(0.975, 2 * tot + 2) / 2 / (nd * n * a)
            s = f"{a:g}: {r:.2f} [{lo:.2f},{hi:.2f}]"
            if a == 1e-3:
                lr = np.log(np.maximum(c, 0.5) / (n * a))
                sd_tot = lr.std(ddof=1)
                sd_mod = np.sqrt(max(sd_tot ** 2 - np.mean(1.0 / np.maximum(c, 1)), 0.0))
                k = kappa_of_ratio(r, a)
                khi = kappa_of_ratio(hi, a)
                s += f" sd {sd_tot:.2f}/{sd_mod:.2f} kappa {k:.3f} -> 5e-8 {ratio_of_kappa(k, 5e-8):.2f} (upper {ratio_of_kappa(khi, 5e-8):.2f})"
            parts.append(s)
        P(f"   {m:8s} " + " | ".join(parts))

pw = {}
for f in glob.glob(os.path.join(RES, "pow_K*.json")):
    d = json.load(open(f))
    key = (d["K"], d["struct"], d["scen"])
    e = pw.setdefault(key, {"n": 0, "c": {}})
    for x in d["draws"].values():
        e["n"] += d["nrep"]
        for m, v in x["counts"].items():
            e["c"][m] = e["c"].get(m, 0) + v
P("")
P("== power at 5e-8 (information-matched signals; R-hat from M=1e4) ==")
PM = ["full_ind", "fast", "chi2", "full_W", "fast_W", "mix", "Wsplit"]
P("K   struct scen     " + " ".join(f"{m:>8s}" for m in PM) + "   gainratio(Wsplit) gainratio(full_W) sparse Wsplit/fast")
for key in sorted(pw):
    K, st, sc = key
    e = pw[key]; p = {m: e["c"].get(m, 0) / e["n"] for m in PM}
    extra = ""
    if sc == "diffuse":
        extra = f"   {(p['Wsplit'] - p['fast']) / GAIN_IND[K]:.2f}  {(p['full_W'] - p['fast']) / GAIN_IND[K]:.2f}"
    if sc == "sparse":
        extra = f"   sparse ratio {p['Wsplit'] / p['fast']:.2f}"
    P(f"{K:<4d}{st:6s} {sc:8s} " + " ".join(f"{p[m]:8.3f}" for m in PM) + f"  (n={e['n']})" + extra)
open(os.path.join(RES, "summary.txt"), "w", encoding="utf8").write("\n".join(lines) + "\n")
