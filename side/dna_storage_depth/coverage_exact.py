"""Exact expected sequencing depth for DNA-storage retrieval with non-uniform strand
abundances (claim checked: Cao, Zhou, Chen, Brief Bioinform 27(3) bbag303, 2026, which says
Flajolet's formula "lacks a closed form and is computationally costly" and uses CQT/CCP
approximations instead).

Quantity: K_a(n, m) = number of reads until at least m of n strands have >= a reads, reads
i.i.d. over strands with probabilities p_i.  Poissonize: reads arrive at rate 1, strand i gets
Poisson(p_i t) reads by time t, independently.  tau = time of success; E[K_a] = E[tau] (Wald).
P(tau > t) = P(N(t) < m), N(t) ~ Poisson-binomial(q_i(t)), q_i(t) = P(Poisson(p_i t) >= a)
= gammainc(a, p_i t).  Also tau = m-th smallest of independent Gamma(a, rate p_i) (used for MC).

E[tau] = integral_0^inf P(N(t) < m) dt, evaluated with the full Poisson-binomial pmf at each
node (vectorised divide-and-conquer FFT product), so all m are obtained at once.
Single thread, small memory, each call < 30 s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time
import numpy as np
from scipy.special import gammainc
from scipy.stats import norm


def pb_pmf(q):
    """Poisson-binomial pmf of sum of Bernoulli(q_i); vectorised D&C, FFT for long levels."""
    q = np.asarray(q, float)
    n0 = len(q)
    # strip near-deterministic strands (exact shift / no-op up to 1e-300 effects)
    sure = q >= 1.0
    never = q <= 0.0
    shift = int(sure.sum())
    q = q[~(sure | never)]
    if len(q) == 0:
        out = np.zeros(n0 + 1); out[shift] = 1.0; return out
    size = 1
    while size < len(q):
        size *= 2
    polys = np.zeros((size, 2))
    polys[:len(q), 0] = 1 - q
    polys[:len(q), 1] = q
    polys[len(q):, 0] = 1.0           # identity polynomials
    L = 2
    while polys.shape[0] > 1:
        A = polys[0::2]; B = polys[1::2]
        newL = 2 * L - 1
        if L <= 32:
            C = np.zeros((A.shape[0], newL))
            for k in range(L):
                C[:, k:k + L] += A[:, k:k + 1] * B
        else:
            nfft = 1
            while nfft < newL:
                nfft *= 2
            C = np.fft.irfft(np.fft.rfft(A, nfft, axis=1) * np.fft.rfft(B, nfft, axis=1), nfft, axis=1)[:, :newL]
            np.maximum(C, 0.0, out=C)
        polys = C
        L = newL
    pmf = polys[0]
    # trailing degrees beyond len(q) are zero (identity padding)
    pmf = pmf[:len(q) + 1]
    pmf = pmf / pmf.sum()
    out = np.zeros(n0 + 1)
    out[shift:shift + len(pmf)] = pmf
    return out


def exact_expected_reads(p, a, m_list, nodes=96):
    """E[K_a] for each m in m_list, by Gauss-Legendre on log t over the transition region."""
    p = np.asarray(p, float)
    m_list = np.asarray(m_list)
    # transition region: use gamma means a/p_i; tau between order stats; bracket generously
    t_lo = 0.05 * np.min(a / np.sort(p)[::-1][:1])  # tiny lower start
    # find t where P(N<min m) ~ 1 and t where P(N<max m) ~ 0 by scanning on log grid
    def cdf_all(t):
        pmf = pb_pmf(gammainc(a, p * t))
        return np.cumsum(pmf)  # cdf[k] = P(N <= k)
    grid = np.exp(np.linspace(np.log(a / p.max()) - 3, np.log(a / p.min()) + 3, 60))
    surv_lo = None
    lo, hi = grid[0], grid[-1]
    for t in grid:
        c = cdf_all(t)
        if c[m_list.min() - 1] > 1 - 1e-14:
            lo = t
        if c[m_list.max() - 1] < 1e-14:
            hi = t
            break
    # integrate P(N(t) < m) over [lo, hi] in u = log t: dt = t du
    x, w = np.polynomial.legendre.leggauss(nodes)
    ul, uh = np.log(lo), np.log(hi)
    us = 0.5 * (uh - ul) * x + 0.5 * (uh + ul)
    ws = 0.5 * (uh - ul) * w
    acc = np.zeros(len(m_list))
    for u, wt in zip(us, ws):
        t = np.exp(u)
        c = cdf_all(t)
        acc += wt * t * c[m_list - 1]
    return lo + acc  # P = 1 below lo (error < 1e-14 * lo)


def mc_expected_reads(p, a, m, reps, rng):
    """Monte Carlo of E[tau]: m-th smallest of Gamma(a, scale 1/p_i)."""
    n = len(p)
    vals = np.empty(reps)
    chunk = max(1, 2_000_000 // n)
    i = 0
    while i < reps:
        r = min(chunk, reps - i)
        g = rng.gamma(a, 1.0, size=(r, n)) / p
        vals[i:i + r] = np.partition(g, m - 1, axis=1)[:, m - 1]
        i += r
    return vals.mean(), vals.std(ddof=1) / np.sqrt(reps)


def lognormal_p(n, mu, sigma):
    """Deterministic quantile sample of LN(mu, sigma^2), normalised to sum 1."""
    z = norm.ppf((np.arange(n) + 0.5) / n)
    p = np.exp(mu + sigma * z)
    return p / p.sum()


def cqt(a, R, mu, sigma, z=1.96):
    base = np.exp(-mu - sigma * norm.ppf(1 - R))
    return max(a - z * np.sqrt(a), 0) * base, a * base, (a + z * np.sqrt(a)) * base


def ccp_k2(a, R, mu, sigma):
    L = np.exp(-mu + sigma ** 2 / 2)
    lg = -np.log(1 - R)
    return L * ((a - 1) + np.log(2) * lg + (a - 1) * np.sqrt(2 * np.log(2) * lg / (a - 1)))


if __name__ == "__main__":
    mode = sys.argv[1]
    n = 11520
    sets = {"PCR10": (-9.71, 0.86), "PCR30": (-9.91, 0.98), "PCR60": (-10.38, 1.11)}
    if mode == "mc":
        # validation: exact vs Monte Carlo on three configurations
        rng = np.random.default_rng(20261110)
        for name, a, R in [("PCR30", 5, 0.9), ("PCR60", 2, 0.5), ("PCR10", 10, 0.95)]:
            mu, sg = sets[name]
            p = lognormal_p(n, mu, sg)
            m = int(np.ceil(R * n))
            t0 = time.time(); ex = exact_expected_reads(p, a, [m])[0]; te = time.time() - t0
            mc, se = mc_expected_reads(p, a, m, 1500, rng)
            print(f"{name} a={a} R={R} m={m}: exact={ex:.1f} ({te:.2f}s)  MC={mc:.1f} +- {se:.1f}  z={(mc-ex)/se:+.2f}")
    elif mode == "grid":
        name = sys.argv[2]
        a_vals = [int(x) for x in sys.argv[3].split(",")]
        mu, sg = sets[name]
        p = lognormal_p(n, mu, sg)
        lp = np.log(p); muh, sgh = lp.mean(), lp.std()
        Rs = np.round(np.arange(0.5, 0.951, 0.05), 2)
        ms = np.ceil(Rs * n).astype(int)
        nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 384   # 96 is not enough for a shared range (0.2% error)
        for a in a_vals:
            t0 = time.time()
            ex = exact_expected_reads(p, a, ms, nodes=nodes)
            for R, m, e in zip(Rs, ms, ex):
                kl, kc, ku = cqt(a, R, muh, sgh)
                k2 = ccp_k2(a, R, muh, sgh)
                print(f"{name}\ta={a}\tR={R:.2f}\texact={e:.0f}\tK_C={kc:.0f}\trelC={(kc-e)/e:+.4f}"
                      f"\tK_L={kl:.0f}\tK_U={ku:.0f}\tinside={int(kl<=e<=ku)}\tK2={k2:.0f}\trel2={(k2-e)/e:+.4f}")
            print(f"# {name} a={a} time {time.time()-t0:.1f}s mu_hat={muh:.3f} sigma_hat={sgh:.3f}", flush=True)
