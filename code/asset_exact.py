"""ASSET all-subset search without enumeration (claim checked: the 2024 Nat Commun multi-trait
pleiotropy atlas (PMC11324957) says the original ASSET, "which searches through all subsets, is
computationally intractable", and fastASSET therefore searches only traits passing a p<0.05 pre-screen).

ASSET subset statistic (one-sided, fixed effect): Z(S) = a(S)/sqrt(b(S)), a(S)=sum_{i in S} w_i z_i,
b(S)= w_S^T R_SS w_S.  Lemma (independent traits, R=I): every local maximum under single add/remove
with a(S)>0 is S = {i: a_i>0, z_i/w_i > a(S)/(2 b(S))}; hence the global max over all 2^K-1 subsets is
attained on one of the K prefixes of the traits sorted by z_i/w_i.  Proof: adding j does not help iff
a_j(2a+a_j) <= (a^2/b) b_j, removing i does not help iff a_i(2a-a_i) >= (a^2/b) b_i; with r=a_i/b_i these
give r_j < a/(2b) < r_i for a_i, a_j > 0, and a_i <= 0 is never in an optimal set when a>0.
Block-diagonal R (small clusters, as fastASSET assumes between clusters): a and b are additive over
blocks; at the optimum each block's option maximises a_B - lam*b_B with lam = a/(2b) (same algebra), so
sweeping lam over the breakpoints of each block's upper hull of (b_B, a_B) gives the exact optimum.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, itertools, time
import numpy as np
from scipy.stats import norm


def exhaustive(z, w, R=None):
    K = len(z)
    best, bestS = -np.inf, None
    for m in range(1, K + 1):
        for S in itertools.combinations(range(K), m):
            S = list(S)
            a = np.dot(w[S], z[S])
            b = w[S] @ (R[np.ix_(S, S)] if R is not None else np.eye(m)) @ w[S]
            v = a / np.sqrt(b)
            if v > best:
                best, bestS = v, S
    return best, sorted(bestS)


def prefix_scan(z, w):
    """Exact max over all nonempty subsets for independent traits, O(K log K)."""
    a = w * z
    pos = np.nonzero(a > 0)[0]
    if len(pos) == 0:
        i = int(np.argmax(z)); return z[i], [i]
    order = pos[np.argsort(-(z[pos] / w[pos]))]
    ca, cb = np.cumsum(a[order]), np.cumsum(w[order] ** 2)
    v = ca / np.sqrt(cb)
    j = int(np.argmax(v))
    return v[j], sorted(order[:j + 1].tolist())


def block_sweep(z, w, R, blocks):
    """Exact max for block-diagonal R via per-block upper hulls and a lambda sweep."""
    hulls = []
    for B in blocks:
        opts = [(0.0, 0.0, ())]
        for m in range(1, len(B) + 1):
            for S in itertools.combinations(B, m):
                S = list(S)
                opts.append((float(w[S] @ R[np.ix_(S, S)] @ w[S]), float(np.dot(w[S], z[S])), tuple(S)))
        # upper hull for maximising a - lam*b, lam >= 0: start at empty option, then greedy by slope
        cur = opts[0]; hull = [cur]
        while True:
            cand = [(o[1] - cur[1]) / (o[0] - cur[0]) for o in opts if o[0] > cur[0] + 1e-15]
            if not cand:
                break
            k = int(np.argmax(cand)); slope = cand[k]
            if slope <= 0:
                break
            nxt = [o for o in opts if o[0] > cur[0] + 1e-15][k]
            hull.append((nxt[0], nxt[1], nxt[2], slope)); cur = nxt
        hulls.append(hull)
    # events: (slope, block index, hull position); process in decreasing slope
    events = sorted([(h[p][3], bi, p) for bi, h in enumerate(hulls) for p in range(1, len(h))], reverse=True)
    pos = [0] * len(blocks)
    A = B_ = 0.0
    best, bestS = -np.inf, None
    for slope, bi, p in events:
        old = hulls[bi][pos[bi]]; new = hulls[bi][p]
        A += new[1] - old[1]; B_ += new[0] - old[0]; pos[bi] = p
        v = A / np.sqrt(B_)
        if v > best:
            best = v
            bestS = sorted(i for bj, q in enumerate(pos) for i in hulls[bj][q][2])
    if bestS is None:
        i = int(np.argmax(z)); return z[i], [i]
    return best, bestS


def verify(rng, reps=80):
    bad = 0
    for _ in range(reps):
        K = 12
        z = rng.normal(0.5, 1.3, K); w = np.sqrt(rng.lognormal(9, 0.8, K))
        v1, S1 = exhaustive(z, w); v2, S2 = prefix_scan(z, w)
        bad += (abs(v1 - v2) > 1e-9 * abs(v1)) or (S1 != S2)
    print(f"independent K=12: {reps} random cases, mismatches vs exhaustive = {bad}")
    bad = 0
    for _ in range(reps):
        K = 12
        blocks = [[0, 1, 2], [3, 4], [5, 6, 7], [8], [9, 10, 11]]
        R = np.eye(K)
        for Bk in blocks:
            if len(Bk) > 1:
                X = rng.normal(size=(len(Bk), 3)); C = X @ X.T + 0.5 * np.eye(len(Bk))
                d = np.sqrt(np.diag(C)); R[np.ix_(Bk, Bk)] = C / np.outer(d, d)
        z = rng.normal(0.5, 1.3, K); w = np.sqrt(rng.lognormal(9, 0.8, K))
        v1, S1 = exhaustive(z, w, R); v2, S2 = block_sweep(z, w, R, blocks)
        bad += (abs(v1 - v2) > 1e-9 * abs(v1)) or (S1 != S2)
    print(f"block-correlated K=12 (blocks 3,2,3,1,3): {reps} random cases, mismatches vs exhaustive = {bad}")


def stat_full(Z):
    """Vectorised T_full = max_m (sum of top-m z)/sqrt(m) for equal weights; rows = replicates."""
    s = -np.sort(-Z, axis=1)
    c = np.cumsum(s, axis=1) / np.sqrt(np.arange(1, Z.shape[1] + 1))
    return c.max(1), c.argmax(1) + 1


def stat_fast(Z, p_thr=0.05):
    """fastASSET-like: keep |z| > z_thr, adjust p~ = P(|Z|>|z|)/p_thr, z~ = sign*Phi^-1(1-p~/2);
    one-sided search over retained positive traits (equal weights). -inf if none retained."""
    zthr = norm.ppf(1 - p_thr / 2)
    keep = Z > zthr
    pt = 2 * norm.sf(np.abs(Z)) / p_thr
    zt = np.where(keep, norm.isf(np.clip(pt, 1e-300, 1) / 2), -np.inf)
    s = -np.sort(-zt, axis=1)
    s = np.where(np.isfinite(s), s, 0.0)
    cnt = np.cumsum(np.isfinite(-np.sort(-zt, axis=1)), axis=1)
    c = np.cumsum(s, axis=1) / np.sqrt(np.maximum(cnt, 1))
    c = np.where(cnt == np.arange(1, Z.shape[1] + 1), c, -np.inf)
    out = c.max(1)
    return out, np.where(np.isfinite(out), c.argmax(1) + 1, 0)


if __name__ == "__main__":
    rng = np.random.default_rng(20261110)
    if sys.argv[1] == "verify":
        verify(rng)
    elif sys.argv[1] == "power":
        K = 116
        t0 = time.time()
        nul_f, nul_s = [], []
        nchunk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        for _ in range(nchunk):
            Z = rng.normal(size=(20000, K))
            nul_f.append(stat_full(Z)[0]); nul_s.append(stat_fast(Z)[0])
        nul_f, nul_s = np.concatenate(nul_f), np.concatenate(nul_s)
        fin = np.mean(np.isfinite(nul_s))
        print(f"null reps {len(nul_f)} in {time.time()-t0:.1f}s; P(fast stat = -inf | null) = {1-fin:.3f}")
        crit = {al: (np.quantile(nul_f, 1 - al), np.quantile(nul_s[np.isfinite(nul_s)], 1 - al / fin)) for al in (1e-3, 1e-4)}
        for al, (cf, cs) in crit.items():
            print(f"alpha={al:g}: critical T_full={cf:.3f}, T_fast={cs:.3f}")
        # combined test: min of the two empirical p-values (CDFs from first half of null, min calibrated on second half)
        h = len(nul_f) // 2
        sf_sorted, ss_sorted = np.sort(nul_f[:h]), np.sort(np.where(np.isfinite(nul_s[:h]), nul_s[:h], -1e9))
        def emp_p(x, srt):
            return (len(srt) - np.searchsorted(srt, x, side="left") + 1) / (len(srt) + 1)
        pmin_null = np.minimum(emp_p(nul_f[h:], sf_sorted), emp_p(np.where(np.isfinite(nul_s[h:]), nul_s[h:], -1e9), ss_sorted))
        ccomb = {al: np.quantile(pmin_null, al) for al in crit}
        for s_, mu in [(2, 4.0), (5, 2.5), (10, 1.8), (20, 1.3), (40, 0.9), (80, 0.6)]:
            Z = rng.normal(size=(10000, K)); Z[:, :s_] += mu
            tf, mf = stat_full(Z); ts, ms = stat_fast(Z)
            pm = np.minimum(emp_p(tf, sf_sorted), emp_p(np.where(np.isfinite(ts), ts, -1e9), ss_sorted))
            pw = " ".join(f"a={al:g}: full {np.mean(tf>cf):.3f} fast {np.mean(ts>cs):.3f} comb {np.mean(pm<ccomb[al]):.3f};" for al, (cf, cs) in crit.items())
            print(f"   s={s_:3d} mu={mu:.1f}: {pw} median subset full={np.median(mf):.0f} fast={np.median(ms):.0f};"
                  f" mean #true with z>1.96={np.mean((Z[:, :s_]>1.96).sum(1)):.1f}")
        print(f"total {time.time()-t0:.1f}s")
