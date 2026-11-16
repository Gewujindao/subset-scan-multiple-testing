"""Exact posterior marginals for array (matrix) group testing with imperfect assays.

Claim checked: Bilder & Zhang (2009) / binGroup gtreg.mp / groupTesting glm.gt use a Gibbs
sampler in the EM E-step because the posterior P(Y_ij=1 | row, column, retest results) has
"no closed form". Here: inclusion-exclusion over rows gives the exact marginals in
O(2^R * R * C) per array (R = the shorter side).

Model: individual statuses Y_ij ~ Bernoulli(p_ij) independent (p_ij from a regression);
row pool i tested once (Se, Sp), column pool j tested once, individuals at positive
row x positive column intersections retested individually (any local evidence works).
Usage: python array_exact.py
"""
import itertools, time
from fractions import Fraction
import numpy as np


def pool_lik(res, Se, Sp):
    # returns (f(0), f(1)) = P(result | pool truly negative / positive)
    return ((1 - Sp) if res else Sp, Se if res else (1 - Se))


def exact_Z(l0, l1, f, g, one=1.0):
    """Z = sum_x prod_ij l_{x_ij} prod_i f_i(OR row) prod_j g_j(OR col).
    l0,l1: R x C nested lists/arrays of local weights; f: list of (f0,f1) per row; g per column.
    Expand f_i(s) = f_i1 + (f_i0 - f_i1) [row i all zero] over subsets A of rows."""
    R, C = len(l0), len(l0[0])
    Z = 0 * one
    for mask in range(1 << R):
        coef = one
        for i in range(R):
            coef *= (f[i][0] - f[i][1]) if (mask >> i) & 1 else f[i][1]
        if coef == 0:
            continue
        prod = one
        for j in range(C):
            all0 = one; free = one
            for i in range(R):
                all0 *= l0[i][j]
                free *= l0[i][j] if (mask >> i) & 1 else (l0[i][j] + l1[i][j])
            prod *= g[j][0] * all0 + g[j][1] * (free - all0)
        Z += coef * prod
    return Z


def exact_Z_np(l0, l1, f, g):
    """Vectorised over all 2^R row subsets (float64)."""
    R, C = l0.shape
    masks = ((np.arange(1 << R)[:, None] >> np.arange(R)[None, :]) & 1).astype(bool)  # 2^R x R
    f0 = np.array([a for a, b in f]); f1 = np.array([b for a, b in f])
    coef = np.where(masks, f0 - f1, f1).prod(1)                       # 2^R
    all0 = l0.prod(0)                                                  # C
    lf = np.log(l0 + l1); ll0 = np.log(l0)
    free = np.exp(masks.astype(float) @ ll0 + (~masks).astype(float) @ lf)  # 2^R x C
    g0 = np.array([a for a, b in g]); g1 = np.array([b for a, b in g])
    col = g0[None, :] * all0[None, :] + g1[None, :] * (free - all0[None, :])
    return float((coef * col.prod(1)).sum())


def marginals_np(l0, l1, f, g):
    Z = exact_Z_np(l0, l1, f, g)
    R, C = l0.shape
    out = np.empty((R, C))
    for i in range(R):
        for j in range(C):
            m0 = l0.copy(); m0[i, j] = 1e-300  # clamp Y_ij = 1 (l0 -> 0)
            out[i, j] = exact_Z_np(m0, l1, f, g) / Z
    return out, Z


def brute(l0, l1, f, g):
    R, C = l0.shape
    Z = 0.0; M = np.zeros((R, C))
    for bits in itertools.product([0, 1], repeat=R * C):
        x = np.array(bits).reshape(R, C)
        w = np.where(x == 1, l1, l0).prod()
        w *= np.prod([f[i][int(x[i].any())] for i in range(R)])
        w *= np.prod([g[j][int(x[:, j].any())] for j in range(C)])
        Z += w; M += w * x
    return M / Z, Z


def simulate(R, C, rng, Se=0.95, Sp=0.98, prev=0.05):
    p = np.clip(rng.lognormal(np.log(prev), 0.7, size=(R, C)), 1e-4, 0.5)
    y = rng.random((R, C)) < p
    rowres = [(rng.random() < Se) if y[i].any() else (rng.random() > Sp) for i in range(R)]
    colres = [(rng.random() < Se) if y[:, j].any() else (rng.random() > Sp) for j in range(C)]
    l0 = 1 - p; l1 = p.copy()
    for i in range(R):
        for j in range(C):
            if rowres[i] and colres[j]:  # individual retest
                t = (rng.random() < Se) if y[i, j] else (rng.random() > Sp)
                l1[i, j] *= Se if t else (1 - Se)
                l0[i, j] *= (1 - Sp) if t else Sp
    f = [pool_lik(r, Se, Sp) for r in rowres]; g = [pool_lik(c, Se, Sp) for c in colres]
    return l0, l1, f, g, y


def gibbs(l0, l1, f, g, n_iter, rng, burn=200):
    R, C = l0.shape
    x = np.zeros((R, C), dtype=int); acc = np.zeros((R, C)); kept = 0
    for it in range(n_iter + burn):
        for i in range(R):
            for j in range(C):
                w = []
                for v in (0, 1):
                    x[i, j] = v
                    w.append((l1 if v else l0)[i, j] * f[i][int(x[i].any())] * g[j][int(x[:, j].any())])
                x[i, j] = int(rng.random() < w[1] / (w[0] + w[1]))
        if it >= burn:
            acc += x; kept += 1
    return acc / kept


if __name__ == "__main__":
    rng = np.random.default_rng(20261110)
    # 1. brute-force check on 3x4 arrays (4096 configs), 30 random arrays, forced informative results
    worst = 0.0
    for t in range(30):
        l0, l1, f, g, _ = simulate(3, 4, rng, prev=0.25)
        Me, Ze = marginals_np(l0, l1, f, g)
        Mb, Zb = brute(l0, l1, f, g)
        worst = max(worst, np.abs(Me - Mb).max(), abs(Ze / Zb - 1))
    print(f"[1] 3x4 brute force vs exact, 30 arrays: max abs diff = {worst:.2e}")
    # 2. cancellation check: float64 vs exact rationals on an 8x12 array with many positives
    l0, l1, f, g, y = simulate(8, 12, rng, prev=0.15)
    q = lambda v: Fraction(v).limit_denominator(10**6)
    L0 = [[q(v) for v in row] for row in l0]; L1 = [[q(v) for v in row] for row in l1]
    F = [(q(a), q(b)) for a, b in f]; G = [(q(a), q(b)) for a, b in g]
    Zq = exact_Z(L0, L1, F, G, one=Fraction(1))
    Zf = exact_Z_np(np.array(L0, float), np.array(L1, float), [(float(a), float(b)) for a, b in F],
                    [(float(a), float(b)) for a, b in G])
    print(f"[2] 8x12, {sum(r[0] < r[1] for r in f)} pos rows / {sum(c[0] < c[1] for c in g)} pos cols: "
          f"float64 Z rel.err vs rational = {abs(Zf / float(Zq) - 1):.2e}")
    # 3. timing of all 96 (or 144) marginals, and Gibbs comparison on 8x12
    for (R, C) in [(8, 12), (12, 12)]:
        l0, l1, f, g, y = simulate(R, C, rng, prev=0.10)
        t0 = time.perf_counter(); M, Z = marginals_np(l0, l1, f, g); dt = time.perf_counter() - t0
        print(f"[3] {R}x{C}: all marginals exact in {dt*1000:.0f} ms; sum of marginals {M.sum():.4f}")
        if R == 8:
            t0 = time.perf_counter(); Mg = gibbs(l0, l1, f, g, 1000, rng); dg = time.perf_counter() - t0
            print(f"    Gibbs 1000 sweeps (python): {dg:.1f} s; max |Gibbs - exact| = {np.abs(Mg - M).max():.4f}")
