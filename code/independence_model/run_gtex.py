"""Real data: GTEx v6 'strong' eQTL z-scores, 16,069 genes x 44 tissues (Urbut et al.,
stephenslab/gtexresults, MatrixEQTLSumStats.Portable.Z.rds, parsed by read_rds.py).
Each gene is oriented so that its largest |z| is positive; one-sided ASSET search in that direction.
Correlation: the file's own null correlation 'vhat'; fastASSET's block rule (|rho| >= 0.2) gives 44 singleton
blocks, so under fastASSET's model the traits are independent and the full search is the exact prefix scan.
Weights: equal (main; one null distribution for all genes) and 1/s (sensitivity, subset sizes only).
  run      -> per-gene results to results/independence_model/gtex/genes.npz
  summary  -> needs results/independence_model/crit_eq_K44.json (IS) and results/independence_model/isk_* (fast version's conditional null)
"""
import sys, os, json, time, glob
import numpy as np
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asset_core as ac

DATA = os.path.join(HERE, "..", "..", "downloads", "gtexresults_v6", "gtex_v6_z.npz")
RES = os.path.join(HERE, "..", "..", "results", "independence_model"); GR = os.path.join(RES, "gtex"); os.makedirs(GR, exist_ok=True)
mode = sys.argv[1]
t0 = time.time()

if mode == "run":
    d = np.load(DATA)
    Z = d["strong.z"].copy(); S = d["strong.s"]; V = d["vhat"]; tissues = [str(x) for x in d["strong.z__cols"]]
    off = np.abs(V[np.triu_indices(44, 1)])
    print(f"vhat off-diagonal |rho|: median {np.median(off):.3f}, 90% {np.quantile(off,0.9):.3f}, max {off.max():.3f}; pairs >= 0.2: {np.sum(off>=0.2)}")
    G, K = Z.shape
    sgn = np.sign(Z[np.arange(G), np.argmax(np.abs(Z), 1)]); Z *= sgn[:, None]
    tf, mf = ac.tfull_eq(Z)
    ts, ks, ms = ac.tfast_eq(Z)
    rank = np.argsort(np.argsort(-Z, 1), 1)              # 0 = largest z
    keep, zt = ac.screen_adjust(Z)
    rank_s = np.argsort(np.argsort(-zt, 1), 1)
    selF = rank < mf[:, None]
    selS = (rank_s < ms[:, None]) & keep
    # sensitivity 1: weights 1/s (prefix by z/w), vectorised over genes
    W = 1.0 / S; W = W / np.sqrt(np.mean(W ** 2, 1, keepdims=True))
    o = np.argsort(-(Z / W), 1)
    cA = np.cumsum(np.take_along_axis(Z * W, o, 1), 1); cB = np.cumsum(np.take_along_axis(W * W, o, 1), 1)
    nW = (cA / np.sqrt(cB)).argmax(1) + 1
    np.savez(os.path.join(GR, "genes.npz"), tf=tf, mf=mf, ts=ts, ks=ks, ms=ms, selF=selF, selS=selS, keep=keep, nW=nW, sgn=sgn)
    json.dump({"tissues": tissues}, open(os.path.join(GR, "meta.json"), "w"))
    print(f"genes {G}; time {time.time()-t0:.1f}s")

elif mode == "dense":
    # sensitivity: dense vhat in the denominator (prefix order + single add/remove refinement) on a gene sample
    start, stop = int(sys.argv[2]), int(sys.argv[3])
    d = np.load(DATA); V = d["vhat"]; g = np.load(os.path.join(GR, "genes.npz"))
    rng = np.random.default_rng(1); sub = rng.choice(d["strong.z"].shape[0], 400, replace=False)[start:stop]
    same, jac, gain = 0, [], []
    for i in sub:
        z = d["strong.z"][i] * g["sgn"][i]; w = np.ones(44)
        v2, m2 = ac.local_refine(z, w, V, ac.prefix_max_corr(z, w, V)[1])
        m1 = g["selF"][i]
        v1 = (w[m1] @ z[m1]) / np.sqrt(w[m1] @ V[np.ix_(m1, m1)] @ w[m1])
        same += np.array_equal(m1, m2); jac.append((m1 & m2).sum() / (m1 | m2).sum()); gain.append((v2 - v1) / v1)
    np.savez(os.path.join(GR, f"dense_{start:03d}.npz"), same=same, jac=np.array(jac), gain=np.array(gain))
    print(f"dense {start}-{stop}: identical {same}/{len(sub)}, mean Jaccard {np.mean(jac):.3f}; {time.time()-t0:.1f}s")

elif mode == "summary":
    g = np.load(os.path.join(GR, "genes.npz")); meta = json.load(open(os.path.join(GR, "meta.json")))
    K = 44; alpha = 5e-8
    crit = json.load(open(os.path.join(RES, "crit_eq_K44.json")))
    cF = crit["T_full_5e-08"]
    # fast version oracle critical value from the conditional nulls T_k (k = 1..25) mixed over K_sel ~ Bin(44, 0.025)
    tails = {}
    for k in range(1, 26):
        fs = sorted(glob.glob(os.path.join(RES, f"isk_k{k}_s*.npz")))
        tf_ = np.concatenate([np.load(f)["tf"] for f in fs]); lw = np.concatenate([np.load(f)["lw"] for f in fs])
        tails[k] = (tf_, np.exp(lw))
    def p_fast(c):
        return sum(binom.pmf(k, K, ac.PTHR) * np.mean(w * (t > c)) for k, (t, w) in tails.items())
    lo, hi = 3.0, 9.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if p_fast(mid) > alpha else (lo, mid)
    cS = 0.5 * (lo + hi)
    tf, ts, ks, mf, ms = g["tf"], g["ts"], g["ks"], g["mf"], g["ms"]
    sigF, sigS = tf > cF, ts > cS
    print(f"oracle critical values at 5e-8: full {cF:.3f}, fast {cS:.3f}; DLM critical (full, K=44) {ac.dlm_crit(alpha, K):.3f}")
    print(f"significant at 5e-8 (oracle): full {sigF.mean():.4f}, fast {sigS.mean():.4f}; full only {np.mean(sigF & ~sigS):.4f}, fast only {np.mean(sigS & ~sigF):.4f}")
    for lab, x in (("full subset size", mf), ("fast subset size", ms), ("traits passing screen", ks), ("full subset size, 1/s weights", g["nW"])):
        print(f"  {lab}: median {np.median(x):.0f}, mean {np.mean(x):.1f}, quartiles {np.quantile(x,[0.25,0.75]).tolist()}")
    out_tissues = (g["selF"] & ~g["keep"]).sum(1)
    print(f"  full subset tissues that fail the 1.96 screen: median {np.median(out_tissues):.0f}, mean {out_tissues.mean():.1f}; genes with >=5 such tissues {np.mean(out_tissues>=5):.3f}")
    print(f"  genes where fast subset is not contained in full subset: {np.mean((g['selS'] & ~g['selF']).any(1)):.3f}")
    print(f"  ratio full/fast subset size (genes with fast>0): median {np.median(mf[ms>0]/ms[ms>0]):.2f}")
    for lo_, hi_ in ((1, 3), (4, 10), (11, 25), (26, 44)):
        sel = (ks >= lo_) & (ks <= hi_)
        print(f"  K_sel in [{lo_},{hi_}] ({sel.mean():.3f} of genes): median full {np.median(mf[sel]):.0f}, fast {np.median(ms[sel]):.0f}")
    fs = sorted(glob.glob(os.path.join(GR, "dense_*.npz")))
    if fs:
        same = sum(int(np.load(f)["same"]) for f in fs); jac = np.concatenate([np.load(f)["jac"] for f in fs]); gain = np.concatenate([np.load(f)["gain"] for f in fs])
        print(f"  dense vhat sensitivity ({len(jac)} genes): identical subset {same/len(jac):.3f}, mean Jaccard {jac.mean():.3f}, statistic gain from refinement median {np.median(gain):.4f} max {gain.max():.4f}")
    print(f"time {time.time()-t0:.1f}s")
