"""Exhaustive checks of the prefix / block-sweep algorithms.  Output labels: P1 prefix scan (independent traits),
P2 two-sided subsets, P3 block sweep (block-diagonal R), P4 prefix heuristic under dense correlation."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import asset_core as ac

rng = np.random.default_rng(20261110)
part = sys.argv[1]


def rand_w(K):
    return np.sqrt(rng.lognormal(9.0, 0.8, K))


def rand_z(K, R=None):
    mu = np.zeros(K)
    kind = rng.integers(3)
    if kind == 0:
        mu[rng.choice(K, 2, replace=False)] = rng.uniform(2, 4)
    elif kind == 1:
        mu[rng.choice(K, K // 2, replace=False)] = rng.uniform(0.5, 1.5)
    x = rng.standard_normal(K) if R is None else np.linalg.cholesky(R) @ rng.standard_normal(K)
    return mu + x


if part == "p123":
    t0 = time.time()
    for K in (12, 14):
        M = ac.all_subsets(K)
        bad1 = bad2 = 0
        n = 400 if K == 12 else 150
        for _ in range(n):
            w, z = rand_w(K), rand_z(K)
            v1, m1 = ac.exhaustive_max(z, w, M=M); v2, m2 = ac.prefix_max(z, w)
            bad1 += (abs(v1 - v2) > 1e-9 * max(1, abs(v1))) or (not np.array_equal(m1, m2))
            for sgn, (v, msk) in zip((1, -1), ac.two_sided_sets(z, w)):
                idx = np.nonzero(sgn * z > 0)[0]
                if len(idx) == 0:
                    continue
                ve, me = ac.exhaustive_max(sgn * z[idx], w[idx])
                bad2 += (abs(ve - v) > 1e-9 * max(1, abs(ve))) or (not np.array_equal(idx[me], np.nonzero(msk)[0]))
        print(f"P1 K={K}: {n} cases, prefix vs exhaustive mismatches = {bad1};  P2 two-sided sets mismatches = {bad2}")
    K = 12; M = ac.all_subsets(K); bad3 = 0; n = 300
    for _ in range(n):
        sizes = []
        while sum(sizes) < K:
            sizes.append(int(min(rng.integers(1, 5), K - sum(sizes))))
        blocks, start, R = [], 0, np.eye(K)
        for s in sizes:
            B = list(range(start, start + s)); start += s; blocks.append(B)
            if s > 1:
                X = rng.standard_normal((s, 2)); C = X @ X.T + 0.3 * np.eye(s); d = np.sqrt(np.diag(C))
                R[np.ix_(B, B)] = C / np.outer(d, d)
        w, z = rand_w(K), rand_z(K, R)
        v1, m1 = ac.exhaustive_max(z, w, R, M); v2, m2 = ac.block_sweep(z, w, R, blocks)
        bad3 += (abs(v1 - v2) > 1e-9 * max(1, abs(v1))) or (not np.array_equal(m1, m2))
    print(f"P3 K=12 random blocks (sizes 1-4), {n} cases, block sweep vs exhaustive mismatches = {bad3}")
    print(f"time {time.time()-t0:.1f}s")

elif part == "p4":
    t0 = time.time()
    K = 12; M = ac.all_subsets(K); n = 200
    for label in ["equi0.1", "equi0.3", "equi0.5", "fac0.1", "fac0.3", "fac0.5"]:
        r = float(label[-3:])
        miss = miss_ref = 0; gaps = []; gaps_ref = []
        for _ in range(n):
            if label.startswith("equi"):
                R = np.full((K, K), r); np.fill_diagonal(R, 1.0)
            else:   # random signed one-factor-plus-noise structure with typical |rho| ~ r
                l = rng.choice([-1, 1], K) * np.sqrt(r) * rng.uniform(0.5, 1.5, K) ** 0.5
                R = np.outer(l, l); np.fill_diagonal(R, 1.0)
                ev = np.linalg.eigvalsh(R)
                if ev.min() <= 0.05:
                    R = R + (0.05 - ev.min()) * np.eye(K); d = np.sqrt(np.diag(R)); R = R / np.outer(d, d)
            w, z = rand_w(K), rand_z(K, R)
            v1, m1 = ac.exhaustive_max(z, w, R, M)
            v2, m2 = ac.prefix_max_corr(z, w, R)
            v3, m3 = ac.local_refine(z, w, R, m2)
            miss += not np.array_equal(m1, m2); miss_ref += not np.array_equal(m1, m3)
            gaps.append((v1 - v2) / abs(v1)); gaps_ref.append((v1 - v3) / abs(v1))
        gaps, gaps_ref = np.array(gaps), np.array(gaps_ref)
        print(f"P4 {label}: prefix miss {miss}/{n}, rel gap mean {gaps.mean():.4f} max {gaps.max():.4f} | "
              f"prefix+local refine miss {miss_ref}/{n}, gap max {gaps_ref.max():.4f}")
    print(f"time {time.time()-t0:.1f}s")

elif part == "vec":
    K = 12; M = ac.all_subsets(K); bad = 0; n = 300
    for _ in range(n):
        sizes = []
        while sum(sizes) < K:
            sizes.append(int(min(rng.integers(1, 6), K - sum(sizes))))
        blocks, start, R = [], 0, np.eye(K)
        for s in sizes:
            B = list(range(start, start + s)); start += s; blocks.append(B)
            if s > 1:
                X = rng.standard_normal((s, 2)); C = X @ X.T + 0.3 * np.eye(s); d = np.sqrt(np.diag(C))
                R[np.ix_(B, B)] = C / np.outer(d, d)
        perm = rng.permutation(K); Rp = R[np.ix_(perm, perm)]; inv = np.argsort(perm)
        blocks_p = [sorted(inv[B].tolist()) for B in blocks]
        w, z = rand_w(K), rand_z(K, Rp)
        v1, m1 = ac.exhaustive_max(z, w, Rp, M)
        v2, m2 = ac.block_sweep_vec(z, w, Rp, blocks_p, [ac.all_subsets(len(B)) for B in blocks_p])
        bad += (abs(v1 - v2) > 1e-9 * max(1, abs(v1))) or (not np.array_equal(m1, m2))
    print(f"vectorised block sweep (blocks 1-5, permuted order) vs exhaustive: {n} cases, mismatches = {bad}")

elif part == "cost":
    for B in (4, 8, 12, 15):
        K = B + 20
        blocks = [list(range(B))] + [[B + i] for i in range(20)]
        R = np.eye(K); X = rng.standard_normal((B, 2)); C = X @ X.T + 0.3 * np.eye(B); d = np.sqrt(np.diag(C))
        R[:B, :B] = C / np.outer(d, d)
        w, z = rand_w(K), rand_z(K, R)
        t = time.time(); ac.block_sweep(z, w, R, blocks); print(f"block size {B} (+20 singletons): {time.time()-t:.3f}s")
