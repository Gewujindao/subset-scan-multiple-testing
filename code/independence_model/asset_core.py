"""Core routines: exact ASSET subset search, DLM approximation, importance sampling.

ASSET (Bhattacharjee et al. 2012): Z(S) = sum_{k in S} w_k Z_k / sd(sum_{k in S} w_k Z_k), w_k = sqrt(n_k).
One-sided statistic T = max_S Z(S).  Prefix lemma (independent traits): the maximiser is a prefix of the
traits sorted by z_k / w_k (descending).  Block-diagonal correlation: exact lambda sweep over block hulls.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import itertools
import numpy as np
from scipy.special import gammaln, logsumexp, log_ndtr, ndtri
from scipy.stats import norm

ZTHR = 1.959963984540054          # fastASSET screen |z| > 1.96 (two-sided p < 0.05); one-sided analogue z > 1.96
PTHR = 0.025


# ----------------------------------------------------------------- exact statistics
def all_subsets(K):
    idx = np.arange(1, 2 ** K)
    return ((idx[:, None] >> np.arange(K)) & 1).astype(bool)


def exhaustive_max(z, w, R=None, M=None):
    """Vectorised exhaustive max over all nonempty subsets; returns (value, subset mask)."""
    K = len(z)
    if M is None:
        M = all_subsets(K)
    A = M * w
    a = A @ z
    b = np.einsum("ij,ij->i", A @ (np.eye(K) if R is None else R), A)
    v = a / np.sqrt(b)
    j = int(np.argmax(v))
    return v[j], M[j]


def prefix_max(z, w):
    """Exact max for independent traits, O(K log K); returns (value, subset mask)."""
    a = w * z
    order = np.argsort(-(z / w), kind="stable")
    v = np.cumsum(a[order]) / np.sqrt(np.cumsum((w ** 2)[order]))
    j = int(np.argmax(v))
    best, mask = v[j], np.zeros(len(z), bool)
    mask[order[:j + 1]] = True
    i = int(np.argmax(z))
    if z[i] > best:                       # only when no a_k > 0 (all-negative row)
        mask = np.zeros(len(z), bool); mask[i] = True; best = z[i]
    return best, mask


def prefix_max_corr(z, w, R):
    """Heuristic for dense R: prefix order by z/w, true correlated denominator."""
    a = w * z
    order = np.argsort(-(z / w), kind="stable")
    best, bm = -np.inf, None
    for j in range(1, len(z) + 1):
        S = order[:j]
        v = a[S].sum() / np.sqrt(w[S] @ R[np.ix_(S, S)] @ w[S])
        if v > best:
            best, bm = v, S
    mask = np.zeros(len(z), bool); mask[bm] = True
    return best, mask


def local_refine(z, w, R, mask):
    """Single add/remove hill climbing from a starting subset (dense R)."""
    def val(m):
        S = np.nonzero(m)[0]
        return (w[S] @ z[S]) / np.sqrt(w[S] @ R[np.ix_(S, S)] @ w[S])
    cur, v = mask.copy(), val(mask)
    improved = True
    while improved:
        improved = False
        for k in range(len(z)):
            m = cur.copy(); m[k] = ~m[k]
            if m.sum() == 0:
                continue
            vv = val(m)
            if vv > v + 1e-12:
                cur, v, improved = m, vv, True
    return v, cur


def block_sweep(z, w, R, blocks):
    """Exact max for block-diagonal R: per-block upper hull of (b_B, a_B), sweep lambda."""
    hulls = []
    for B in blocks:
        B = list(B)
        opts = [(0.0, 0.0, ())]
        for m in range(1, len(B) + 1):
            for S in itertools.combinations(B, m):
                S = list(S)
                opts.append((float(w[S] @ R[np.ix_(S, S)] @ w[S]), float(w[S] @ z[S]), tuple(S)))
        cur, hull = opts[0], [opts[0] + (np.inf,)]
        while True:
            later = [o for o in opts if o[0] > cur[0] + 1e-15]
            if not later:
                break
            sl = [(o[1] - cur[1]) / (o[0] - cur[0]) for o in later]
            k = int(np.argmax(sl))
            if sl[k] <= 0:
                break
            cur = later[k]; hull.append(cur + (sl[k],))
        hulls.append(hull)
    events = sorted([(h[p][3], bi, p) for bi, h in enumerate(hulls) for p in range(1, len(h))], reverse=True)
    pos = [0] * len(blocks); A = Bv = 0.0; best, bestS = -np.inf, None
    for _, bi, p in events:
        old, new = hulls[bi][pos[bi]], hulls[bi][p]
        A += new[1] - old[1]; Bv += new[0] - old[0]; pos[bi] = p
        v = A / np.sqrt(Bv)
        if v > best:
            best, bestS = v, sorted(i for bj, q in enumerate(pos) for i in hulls[bj][q][2])
    mask = np.zeros(len(z), bool)
    i = int(np.argmax(z))
    if bestS is None or z[i] > best:      # all-negative case: best single trait
        mask[i] = True; return z[i], mask
    mask[bestS] = True
    return best, mask


def block_sweep_vec(z, w, R, blocks, masks):
    """Same as block_sweep, vectorised: masks[i] = all_subsets(len(blocks[i])) precomputed (includes no empty row)."""
    hulls = []
    for B, M in zip(blocks, masks):
        B = np.asarray(B)
        A = M * w[B]
        a = A @ z[B]
        b = np.einsum("ij,ij->i", A @ R[np.ix_(B, B)], A)
        a = np.concatenate([[0.0], a]); b = np.concatenate([[0.0], b])
        order = np.lexsort((-a, b))                       # by b ascending, a descending
        hull = [0]
        cur = 0
        while True:                                       # gift wrapping from the empty option
            later = b > b[cur] + 1e-15
            if not later.any():
                break
            sl = np.where(later, (a - a[cur]) / np.where(later, b - b[cur], 1.0), -np.inf)
            k = int(np.argmax(sl))
            if sl[k] <= 0:
                break
            hull.append(k); cur = k
        hulls.append((B, M, a, b, hull))
    events = []
    for bi, (B, M, a, b, hull) in enumerate(hulls):
        for p in range(1, len(hull)):
            i0, i1 = hull[p - 1], hull[p]
            events.append(((a[i1] - a[i0]) / (b[i1] - b[i0]), bi, p))
    events.sort(reverse=True)
    pos = [0] * len(hulls); Atot = Btot = 0.0; best, bestpos = -np.inf, None
    for _, bi, p in events:
        B, M, a, b, hull = hulls[bi]
        Atot += a[hull[p]] - a[hull[pos[bi]]]; Btot += b[hull[p]] - b[hull[pos[bi]]]; pos[bi] = p
        v = Atot / np.sqrt(Btot)
        if v > best:
            best, bestpos = v, list(pos)
    mask = np.zeros(len(z), bool)
    i = int(np.argmax(z))
    if bestpos is None or z[i] > best:
        mask[i] = True; return z[i], mask
    for (B, M, a, b, hull), p in zip(hulls, bestpos):
        idx = hull[p]
        if idx > 0:
            mask[B[M[idx - 1]]] = True
    return best, mask


def two_sided_sets(z, w):
    """ASSET two-sided: best subset within positive-sign traits and within negative-sign traits."""
    out = []
    for sgn in (1, -1):
        idx = np.nonzero(sgn * z > 0)[0]
        if len(idx) == 0:
            out.append((None, np.zeros(len(z), bool))); continue
        v, m = prefix_max(sgn * z[idx], w[idx])
        mask = np.zeros(len(z), bool); mask[idx[m]] = True
        out.append((v, mask))
    return out


# ----------------------------------------------------------------- vectorised statistics (rows = replicates)
def tfull_eq(Z):
    s = -np.sort(-Z, axis=1)
    c = np.cumsum(s, axis=1) / np.sqrt(np.arange(1, Z.shape[1] + 1))
    return c.max(1), c.argmax(1) + 1


def tfull_w(Z, w):
    order = np.argsort(-(Z / w), axis=1)
    A = np.take_along_axis(Z * w, order, 1)
    B = np.take_along_axis(np.broadcast_to(w ** 2, Z.shape), order, 1)
    c = np.cumsum(A, 1) / np.sqrt(np.cumsum(B, 1))
    return np.maximum(c.max(1), Z.max(1)), c.argmax(1) + 1


def screen_adjust(Z):
    """fastASSET-like one-sided screen and adjustment: keep z > 1.96, z~ = Phi^-1(1 - P(Z>z)/0.025)."""
    keep = Z > ZTHR
    zt = np.full(Z.shape, -np.inf)
    zt[keep] = -ndtri(norm.sf(Z[keep]) / PTHR)
    return keep, zt


def tfast_eq(Z):
    keep, zt = screen_adjust(Z)
    ksel = keep.sum(1)
    s = -np.sort(-zt, axis=1)
    m = np.arange(1, Z.shape[1] + 1)
    c = np.cumsum(np.where(np.isfinite(s), s, 0.0), 1) / np.sqrt(m)
    c = np.where(m[None, :] <= ksel[:, None], c, -np.inf)
    return c.max(1), ksel, np.where(ksel > 0, c.argmax(1) + 1, 0)


def tfast_w(Z, w):
    keep, zt = screen_adjust(Z)
    ksel = keep.sum(1)
    r = np.where(keep, zt / w, -np.inf)
    order = np.argsort(-r, axis=1)
    A = np.take_along_axis(np.where(keep, zt * w, 0.0), order, 1)
    B = np.take_along_axis(np.broadcast_to(w ** 2, Z.shape), order, 1)
    m = np.arange(1, Z.shape[1] + 1)
    c = np.cumsum(A, 1) / np.sqrt(np.cumsum(B, 1))
    c = np.where(m[None, :] <= ksel[:, None], c, -np.inf)
    zmax = np.where(keep, zt, -np.inf).max(1)
    return np.maximum(c.max(1), zmax), ksel


# ----------------------------------------------------------------- DLM (Bhattacharjee 2012), one-sided, equal weights
_GL_X, _GL_W = np.polynomial.legendre.leggauss(80)


def dlm_tail(T, K, span=12.0):
    """One-sided DLM: sum over all nonempty subsets of int_T^inf prod_k P(Z_{S+-k} < z | Z_S = z) phi(z) dz.
    Equal weights => subsets of equal size are exchangeable, so the 2^K - 1 terms collapse to K."""
    T = np.atleast_1d(np.asarray(T, float))
    if K <= 0:
        return np.ones_like(T)
    out = np.empty_like(T)
    m = np.arange(1, K + 1)[:, None]
    logC = gammaln(K + 1) - gammaln(m + 1) - gammaln(K - m + 1)
    for t_i, T0 in enumerate(T):
        z = 0.5 * span * _GL_X + T0 + 0.5 * span
        wz = 0.5 * span * _GL_W
        zz = z[None, :]
        lf_out = log_ndtr(zz * (np.sqrt(m + 1) - np.sqrt(m)))
        arg_in = (zz / np.sqrt(m) - zz * (np.sqrt(m) - np.sqrt(np.maximum(m - 1, 0)))) / np.sqrt(np.maximum(1 - 1 / m, 1e-300))
        lf_in = np.where(m >= 2, log_ndtr(arg_in), 0.0)
        lterm = logC + (K - m) * lf_out + m * lf_in + norm.logpdf(zz)
        out[t_i] = np.exp(logsumexp(lterm + np.log(wz)[None, :]))
    return out


def dlm_crit(alpha, K, lo=0.0, hi=40.0):
    """Threshold c with dlm_tail(c, K) = alpha (bisection)."""
    if K <= 0:
        return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dlm_tail(mid, K)[0] > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------- importance sampling
def make_components(K, w, c_full, sparse_shifts=(), sparse_rhos=()):
    """Product spike proposals: trait k shifted by delta_gk with prob rho_g, independently.
    Family A (dense..sparse, aimed at T_full = c_full): rho on a geometric grid, delta = kappa w with
    kappa = c_full / sqrt(rho * sum w^2).  Family B (aimed at sparse large z): given rhos and shifts (in z units)."""
    comps = []
    sw2 = np.sum(w ** 2)
    for rho in np.unique(np.clip(np.geomspace(1.0 / K, 1.0, 8), 1.0 / K, 1.0)):
        comps.append((rho, (c_full / np.sqrt(rho * sw2)) * w))
    for rho in sparse_rhos:
        for sh in sparse_shifts:
            comps.append((rho, np.full(K, sh) * (w / np.sqrt(np.mean(w ** 2)))))
    return comps


def make_components_calibrated(K, w, c_target, statfun, sparse_shifts=(4.5, 6.0), npilot=300, seed=4242):
    """Family A with the shift kappa per rho tuned so that the median of T_full under that component is
    about c_target (pilot draws with a fixed seed, so the proposal is deterministic); family B as before."""
    rng = np.random.default_rng(seed)
    comps = []
    for rho in np.unique(np.clip(np.geomspace(1.0 / K, 1.0, 8), 1.0 / K, 1.0)):
        lo, hi = 0.0, 3.0 * c_target / np.sqrt(rho * np.sum(w ** 2))
        X = rng.standard_normal((npilot, K)); inc = rng.random((npilot, K)) < rho
        for _ in range(12):
            kap = 0.5 * (lo + hi)
            med = np.median(statfun(X + inc * (kap * w)))
            if med < c_target:
                lo = kap
            else:
                hi = kap
        comps.append((float(rho), 0.5 * (lo + hi) * w))
    for rho in sorted(set([1.0 / K, min(1.0, 3.0 / K)])):
        for sh in sparse_shifts:
            comps.append((rho, np.full(K, sh) * (w / np.sqrt(np.mean(w ** 2)))))
    return comps


def is_sample(comps, K, n, rng):
    """Draw n samples from the mixture; return Z and log importance weight log(phi/q)."""
    G = len(comps)
    g = rng.integers(0, G, n)
    Z = rng.standard_normal((n, K))
    for gi in range(G):
        sel = np.nonzero(g == gi)[0]
        if len(sel):
            rho, d = comps[gi]
            inc = rng.random((len(sel), K)) < rho
            Z[sel] += inc * d
    lq = np.empty((n, G))
    for gi, (rho, d) in enumerate(comps):
        t = d * Z - 0.5 * d * d                        # log phi(z-d)/phi(z)
        if rho >= 1.0:
            lq[:, gi] = t.sum(1)
        else:
            lq[:, gi] = np.logaddexp(np.log1p(-rho), np.log(rho) + t).sum(1)
    logw = -(logsumexp(lq, axis=1) - np.log(G))
    return Z, logw


def is_tail(stat_vals, logw, cs):
    """IS estimates of P(T > c) for each c, with standard errors."""
    wts = np.exp(logw)
    est, se = [], []
    for c in cs:
        x = wts * (stat_vals > c)
        est.append(x.mean()); se.append(x.std(ddof=1) / np.sqrt(len(x)))
    return np.array(est), np.array(se)
