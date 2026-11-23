"""Whole-subset tests for multi-trait association under weak correlation (core routines).

Correlation structures (factor form R = V V^T + diag(D), unit diagonal):
  I  independent
  A  pervasive weak correlation: three 'shared-cohort' factors (one all-positive, one mixed-sign, one on half
     the traits); pairwise correlations mostly 0.01-0.05, median about 0.025, ~90% positive
  B  several small strong blocks (sizes 2-10, ~40% of traits, within-block 0.2-0.8) on top of A at half strength
R-hat: sample correlation of M null z-vectors (M plays the role of the effective number of independent null SNPs;
pairwise standard error about 1/sqrt(M)), optionally with every off-diagonal entry shrunk by a systematic bias b.

Tests (one-sided, positive direction, equal weights, as in code/independence_model):
  full_ind   ASSET full-subset max on raw z, independence calibration
  fast       fastASSET-like: ZCA-decorrelate inside clusters |R-hat| >= 0.2, screen z > 1.96, ASSET on the rest
  chi2       z' R-hat^{-1} z against chi2_K (usual global test)
  full_W     ASSET full-subset max on whitened z~ = c R-hat^{-1/2} z, c = sqrt((M-K-1)/M) (inverse-Wishart
             correction); exact by the prefix property; independence calibration is exact when R-hat = R
  mix        sum over all subsets: for a grid of sparsity levels s, the subset-mixture likelihood ratio
             L_s = sum_k log(1 - s/K + (s/K) exp(mu_s z~_k - mu_s^2/2)), mu_s = eta0/sqrt(s), on the same z~;
             null tail of each L_s by exact numerical convolution (FFT); Bonferroni over the grid
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
import numpy as np
from scipy.stats import norm, chi2
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "independence_model"))
import asset_core as ac  # noqa: E402

ETA0 = 6.0


# ----------------------------------------------------------------- correlation structures
def make_R(K, struct, seed=97):
    rng = np.random.default_rng([seed, K, {"I": 0, "A": 1, "B": 2}[struct]])
    blocks = []
    if struct == "I":
        V = np.zeros((K, 0))
    else:
        g1 = rng.uniform(0.08, 0.22, K)
        g2 = rng.uniform(0.0, 0.2, K) * rng.choice([-1.0, 1.0], K)
        g3 = rng.uniform(0.0, 0.15, K) * (rng.random(K) < 0.5)
        V = np.column_stack([g1, g2, g3])
        if struct == "B":
            V = V / np.sqrt(2.0)
            perm = rng.permutation(K)
            target, i, cols = int(round(0.4 * K)), 0, []
            while target - i >= 2:
                b = min(int(rng.integers(2, 11)), target - i)
                col = np.zeros(K)
                col[perm[i:i + b]] = rng.uniform(0.45, 0.9, b)
                cols.append(col)
                blocks.append(np.sort(perm[i:i + b]))
                i += b
            V = np.column_stack([V] + cols)
    D = 1.0 - (V ** 2).sum(1)
    R = V @ V.T + np.diag(D)
    return V, D, R, blocks


def sample_z(n, V, D, rng, mu=None):
    Z = rng.standard_normal((n, V.shape[1])) @ V.T + rng.standard_normal((n, len(D))) * np.sqrt(D)
    return Z if mu is None else Z + mu


def estimate_Rhat(V, D, M, rng, b=0.0, chunk=4000):
    K = len(D)
    S = np.zeros((K, K))
    for st in range(0, M, chunk):
        Z = sample_z(min(chunk, M - st), V, D, rng)
        S += Z.T @ Z
    d = np.sqrt(np.diag(S))
    Rh = S / np.outer(d, d)
    if b:
        Rh = Rh - b * (1.0 - np.eye(K))
    return Rh


def inv_sqrt(Rh):
    lam, U = np.linalg.eigh(Rh)
    return (U / np.sqrt(lam)) @ U.T


def clusters(Rh, thr=0.2):
    A = np.abs(Rh) >= thr
    np.fill_diagonal(A, False)
    n, lab = connected_components(A, directed=False)
    return [np.nonzero(lab == c)[0] for c in range(n)]


def block_whitener(Rh, thr=0.2):
    W = np.eye(len(Rh))
    for B in clusters(Rh, thr):
        if len(B) > 1:
            W[np.ix_(B, B)] = inv_sqrt(Rh[np.ix_(B, B)])
    return W


# ----------------------------------------------------------------- subset-mixture (sum over subsets) statistic
def grid_s(K):
    s, out = 1, []
    while s < K:
        out.append(s)
        s *= 2
    return out + [K]


def ell(Zt, s, K, eta0=ETA0):
    mu = eta0 / np.sqrt(s)
    if s >= K:
        return mu * Zt - 0.5 * mu * mu
    pi = s / K
    return np.logaddexp(np.log1p(-pi), np.log(pi) + mu * Zt - 0.5 * mu * mu)


def _zinv(y, pi, mu):
    """z with ell(z) - log(1-pi) = y (y > 0)."""
    return (y + np.log1p(-np.exp(-y)) + np.log((1.0 - pi) / pi) + 0.5 * mu * mu) / mu


def comp_pmf_sum(K, s, means=None, eta0=ETA0, nfft=2 ** 17):
    """Exact (up to grid rounding) distribution of S = sum_k [ell(z_k) - log(1-pi)], z_k ~ N(m_k, 1) independent.
    Returns grid of L values (L = S + K log(1-pi)) and survival P(L >= grid).  pi < 1 only."""
    pi, mu = s / K, eta0 / np.sqrt(s)
    m = np.zeros(K) if means is None else np.asarray(means, float)
    # range: null mean + many sd, and the single-term value at z = 8 + max mean
    zq = np.linspace(-9, 9, 4001)
    y0 = np.logaddexp(np.log1p(-pi), np.log(pi) + mu * zq - 0.5 * mu * mu) - np.log1p(-pi)
    w0 = norm.pdf(zq) * (zq[1] - zq[0])
    mean0, var0 = (w0 * y0).sum(), (w0 * y0 ** 2).sum() - (w0 * y0).sum() ** 2
    ytop = np.log(pi) + mu * (8.0 + m.max()) - 0.5 * mu * mu - np.log1p(-pi)
    smax = max(K * mean0 + 40 * np.sqrt(K * var0), ytop + 30.0) + mu * m.sum()
    half = nfft // 2
    h = smax / half
    edges = (np.arange(half + 1) - 0.5) * h
    edges[0] = 0.0
    ze = np.full(half + 1, -np.inf)
    ze[1:] = _zinv(edges[1:], pi, mu)
    ftot = np.ones(nfft // 2 + 1, complex)
    for mk, cnt in zip(*np.unique(m, return_counts=True)):
        cdf = norm.cdf(ze - mk)
        p = np.diff(cdf)
        p[-1] += norm.sf(ze[-1] - mk)
        ftot *= np.fft.rfft(p, nfft) ** cnt
    pmf = np.fft.irfft(ftot, nfft)[:half]
    pmf = np.clip(pmf, 0.0, None)
    sf = np.cumsum(pmf[::-1])[::-1]
    grid = np.arange(half) * h + K * np.log1p(-pi)
    return grid, sf


class MixCalib:
    """Null survival functions of every component L_s (independent N(0,1) coordinates)."""

    def __init__(self, K, eta0=ETA0):
        self.K, self.eta0, self.S = K, eta0, grid_s(K)
        self.tab = {}
        for s in self.S:
            if s < K:
                g, sf = comp_pmf_sum(K, s, eta0=eta0)
                keep = sf > 1e-300
                self.tab[s] = (g[keep], np.log(sf[keep]))

    def logsf(self, s, L):
        if s >= self.K:                      # exact normal: L = mu sum z - K mu^2/2
            mu = self.eta0 / np.sqrt(s)
            return norm.logsf((L + 0.5 * self.K * mu * mu) / (mu * np.sqrt(self.K)))
        g, lsf = self.tab[s]
        return np.interp(L, g, lsf, left=0.0, right=lsf[-1] - 50.0)

    def crit(self, s, a):
        if s >= self.K:
            mu = self.eta0 / np.sqrt(s)
            return norm.isf(a) * mu * np.sqrt(self.K) - 0.5 * self.K * mu * mu
        g, lsf = self.tab[s]
        return float(np.interp(-np.log(a), -lsf, g))

    def minp(self, Zt):
        """Bonferroni p-value of the min over components: G * min_s p_s (capped at 1)."""
        lp = np.full(Zt.shape[0], 0.0)
        for s in self.S:
            lp = np.minimum(lp, self.logsf(s, ell(Zt, s, self.K, self.eta0).sum(1)))
        return np.minimum(1.0, len(self.S) * np.exp(lp))


# ----------------------------------------------------------------- all statistics at once
def all_stats(Z, Wraw, c, Wb, mixcal):
    """Z: n x K raw z.  Wraw: R-hat^{-1/2}; c: inverse-Wishart correction; Wb: block whitener (fastASSET-like)."""
    out = {}
    out["full_ind"] = ac.tfull_eq(Z)[0]
    out["fast"] = ac.tfast_eq(Z @ Wb)[0]
    Zw = Z @ Wraw
    out["chi2"] = (Zw ** 2).sum(1)
    Zc = c * Zw
    out["full_W"] = ac.tfull_eq(Zc)[0]
    out["fast_W"] = ac.tfast_eq(Zc)[0]
    out["mix_p"] = mixcal.minp(Zc)
    return out


def wishart_c(K, M):
    return np.sqrt((M - K - 1.0) / M) if M else 1.0


def scenario(K, scen, R):
    """Means for the scenario (identical to predict_reference.scenario): signal shapes of run_power.py, positions fixed by seed,
    scaled so that sqrt(mu' R^-1 mu) = ||mu0|| (same oracle information as the independent case)."""
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


def psi(s, K, eta0=ETA0):
    """corr(ell_s(Z), Z)^2: share of the component's null variance that responds to a common mean shift."""
    zq = np.linspace(-10, 10, 8001)
    w = norm.pdf(zq) * (zq[1] - zq[0])
    l = ell(zq, s, K, eta0)
    m = (w * l).sum()
    v = (w * l * l).sum() - m * m
    c1 = (w * l * zq).sum()
    return c1 * c1 / v
