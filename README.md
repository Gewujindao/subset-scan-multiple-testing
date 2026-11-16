# Exact All-Subset Scanning for Multi-Trait Association Tests

**中文简介**：本仓库研究多性状关联分析中的全子集检验（ASSET 一类）。在性状独立或成块相关时，全部 2^K−1 个子集上的最大统计量必定落在按 z/w 排序的前缀里，因此可以在 O(K log K) 内精确求出，不需要枚举；块对角相关时可按块做凸包扫描精确求解。研究同时量化了按独立模型定临界值时弱相关带来的一类错误膨胀（模拟与 GTEx v6 多组织 eQTL 实测相关），并设计、检验了先白化再做前缀扫描的全子集检验。另含两个独立的小题：DNA 存储检索所需测序深度的精确期望值，以及阵列群检 EM 算法 E 步的精确后验边际。原始数据不随仓库分发。

---

## 1. Research question

ASSET (Bhattacharjee et al., 2012) tests a variant against K traits by maximising a weighted fixed-effect statistic over all non-empty trait subsets,

Z(S) = Σ_{k∈S} w_k z_k / sqrt(w_Sᵀ R_SS w_S),  T = max_S Z(S),

with w_k = √n_k (or 1/s_k). Because the search space has 2^K − 1 subsets, ASSET and fastASSET (Qi et al., 2024) treat the full search as intractable for many traits; fastASSET therefore searches only traits that pass a univariate pre-screen.

1. Can the maximum over all subsets be computed exactly without enumeration, and under which correlation structures?
2. How should the null distribution of the full-search statistic be calibrated for K = 30 to 300 at genome-wide levels (5e-8)?
3. How sensitive is the full search to weak between-trait correlation that is ignored, as in the independence model or in fastASSET's block rule (|ρ| < 0.2 treated as zero)?
4. Can a whitened full-subset test keep the nominal type-I error when the correlation matrix is estimated, and what power does it retain?

## 2. Methods

### 2.1 Exact maximisation (`code/asset_exact.py`, `code/w2/asset_core.py`, `code/w2/run_props.py`)
- **Prefix property (independent traits).** Write a_k = w_k z_k, b_k = w_k², r_k = z_k / w_k. Any subset that is a local maximum under single additions and removals with a(S) > 0 has the form S = {k : a_k > 0, r_k > a(S) / (2 b(S))}. Hence the global maximum over all 2^K − 1 subsets is attained on one of the K prefixes of the traits sorted by z_k / w_k, computable in O(K log K). This holds for the published weights (√n_k or 1/s_k).
- **Two-sided version.** ASSET's two-sided statistic searches positive-sign and negative-sign traits separately and combines them by Fisher's method; the prefix property applies within each sign.
- **Block-diagonal correlation.** With independent blocks, a and b are additive over blocks; at the optimum each block's chosen subset maximises a_B − λ b_B with λ = a / (2b). Sweeping λ over the vertices of each block's upper convex hull of (b_B, a_B) gives the exact optimum at cost O(Σ_B 2^{|B|}).
- **Scope.** Under dense correlation the prefix property does not hold (numerical counterexamples in §3.1). With correlation in the denominator, the optimum satisfies a threshold-plus-grey-zone fixed-point rule (`code/w3/w3_check_fixedpoint.py`); no exact polynomial algorithm is given for the dense case.

### 2.2 Calibration (`code/w2/run_null.py`)
- The original DLM (discrete local maximum) approximation is evaluated against simulation; for equal weights the 2^K − 1 terms are collapsed into K terms by subset size.
- Tail probabilities down to 1e-10 are computed by importance sampling with a product "sparse-to-dense" mixture proposal, checked against plain Monte Carlo at 1e-3 and 1e-4.
- The pre-screened ("fast") version is calibrated by a DLM conditional on the number of traits passing the screen.

### 2.3 Weak correlation (`code/w2/null_equicorr.py`, `code/w2/null_corr_check.py`)
Critical values from the independence model are applied to null z-vectors drawn under (i) equicorrelation ρ = 0.01, 0.03, 0.1 and (ii) the 44 × 44 null correlation matrix of GTEx v6 (`vhat`).

### 2.4 Whitened full-subset test (`code/w3/`)
z̃ = c · R̂^{-1/2} z with the full estimated correlation matrix (no truncation) and the inverse-Wishart correction c = sqrt((M − K − 1)/M). On z̃ the prefix property is exact again. The main design ("W-split") rejects if the whitened full search exceeds its critical value at 0.8α or the whitened pre-screened statistic exceeds its critical value at 0.2α; critical values come from a one-off independent-null table. Simulations cover K = 30/116/300, correlation structures A (diffuse weak positive), B (small strong blocks on a weak background) and I (independent), R̂ from M = 1e4 or 4e4 null vectors, and a systematic under-estimation b = 0.003 of every off-diagonal entry. Criteria and predictions were written before the evaluation runs (`week2/判据与预测.txt`, `week3/判据与预测.txt`).

## 3. Main results

### 3.1 Exact search
- Prefix scan vs. exhaustive search: 0 mismatches in 1,700 checks — 550 independent cases (K = 12: 400, K = 14: 150), the same 550 cases for the two-sided version, and 600 random block-diagonal cases (300 block sweep + 300 vectorised sweep) (`code/w2/out_props_p123.txt`, lines 1–3; `code/w2/out_props_vec.txt`, line 1).
- Block sweep cost per SNP (Python, block plus 20 singletons): block size 12, 0.919 s; block size 15, 6.045 s (`code/w2/out_props_cost.txt`, lines 3–4).
- Dense correlation, K = 12, 200 cases each: the prefix scan misses the optimum in 14/40/61 cases at equicorrelation 0.1/0.3/0.5 and in 22/96/123 cases under single-factor mixed-sign correlation 0.1/0.3/0.5, with maximum relative gap up to 0.3792 (`code/w2/out_props_p4.txt`, lines 1–6).

### 3.2 Calibration
- At 5e-8 the DLM approximation is conservative for the full-search statistic and degrades with K: DLM/true tail ratio 1.68 (K = 30), 156.92 (K = 116), about 6.9 × 10^7 (K = 300) (`code/w2/res/log_165731_658.txt`, `log_165743_932.txt`, `log_165753_611.txt`). Importance sampling gives relative standard errors at 5e-8 of 0.03, 0.07 and 0.15 for K = 30, 116, 300 (same files).
- The conditional DLM of the pre-screened version gives type-I/α at 5e-8 of 0.574, 0.721, 0.974, 0.986 for K = 30, 44, 116, 300 (`code/w2/res/fastdlm_typeI.json`).

### 3.3 Sensitivity to ignored weak correlation
- Equicorrelation, K = 300, α = 1e-3, critical value from the independence model: type-I error inflated **23.58×** for the full search at ρ = 0.01 and 86.08× at ρ = 0.03; the pre-screened version 1.67× and 4.17× (`code/w2/res/o_eq.txt`, line 1).
- GTEx v6 null correlation (44 tissues; off-diagonal |ρ| median 0.024, max 0.198, no pair ≥ 0.2, so fastASSET's block rule treats all tissues as independent; `code/w2/res/o_gtex_run.txt`, line 1): full-search type-I inflation **3.21×, 7.87×, 21.25×** at α = 1e-2, 1e-3, 1e-4; pre-screened version 1.10×, 1.17×, 1.35× (`code/w2/res/o_nc.txt`, lines 3–8).

### 3.4 Power under the independence model (α = 5e-8, calibrated critical values)
Diffuse signal (optimal-subset non-centrality 7): full search 0.459 / 0.136 / 0.051 vs. pre-screened 0.088 / 0.001 / 0.000 for K = 30 / 116 / 300. Sparse strong signal (2 traits): full search 0.281 / 0.001 / 0.000 vs. pre-screened 0.684 / 0.553 / 0.371 (`code/w2/res/log_170506_862.txt`, `log_170509_799.txt`, `log_170513_036.txt`).

### 3.5 Whitened full-subset test
- With only random estimation error in R̂, the observed/nominal type-I ratio of the whitened full search and of W-split at 1e-3 and 1e-4 lies in **0.90–1.13** across all completed configurations (K = 30 and 116 with structures A/B/I and M = 1e4/4e4; K = 300 with A, M = 1e4) (`code/w3/res/summary.txt`, minimum at line 64, maximum at line 36). Under the same conditions the unwhitened full search is inflated, e.g. 4.38× at 1e-3 for K = 30, structure A (`summary.txt`, line 3).
- Power at 5e-8 with information-matched signals (`code/w3/res/summary.txt`, power table): for K = 116, diffuse signal, the whitened full search reaches 0.148 under independence and 0.032 / 0.037 under structures A / B; for sparse signals W-split retains 0.80–0.87 of the pre-screened test's power (K = 30 and 116). The loss in the diffuse case comes from the correlation itself: whitening subtracts the common component, so non-signal traits receive a small negative mean and part of the diffuse signal lies in the high-variance common direction (`week3/对弱相关稳健的全子集检验.txt`, conclusion 4).

## 4. Relation to existing work

- **Neill (2012, JRSS-B)** introduced the linear-time subset scanning (LTSS) property; the prefix property in §2.1 is its application to the ASSET statistic. The block-diagonal sweep is a small extension.
- **ASSET** (Bhattacharjee et al., 2012, AJHG) defines the subset statistic and the DLM approximation and enumerates subsets.
- **fastASSET** (Qi et al., 2024, Nat Commun) pre-screens traits, decorrelates within clusters with |ρ| ≥ 0.2 and ignores correlation between clusters.
- **TraitScan** (Cao et al., 2024, Bioinformatics) applies LTSS with ZCA whitening of the full correlation matrix to higher-criticism and truncated chi-square statistics in multi-trait GWAS, and describes ASSET as requiring exhaustive search. This project applies the same whitening-plus-scan principle to the ASSET statistic and adds the inverse-Wishart correction, a one-off importance-sampling null table at 5e-8, and an error budget for the estimated correlation matrix.

## 5. Data (not distributed)

- GTEx v6 multi-tissue eQTL summary statistics from the `stephenslab/gtexresults` repository (file `data/MatrixEQTLSumStats.Portable.Z.rds`, 66,560,266 bytes, used in Urbut et al.'s mash paper): strong set 16,069 genes × 44 tissues and the 44 × 44 null correlation `vhat`.
- `code/w2/read_rds.py` converts the RDS file to `downloads/gtexresults_v6/gtex_v6_z.npz`, the path expected by `run_gtex.py` and `null_corr_check.py`. The `downloads/` folder is not part of the repository.
- All other results use simulated data. Simulation arrays (`.npz`) are not included; the text summaries derived from them are.

## 6. Side problems (`side/`)

### 6.1 DNA-storage sequencing depth (`side/dna_storage_depth/`)
`coverage_exact.py` computes the exact expected number of reads until at least m of n strands have at least a reads each, with non-uniform strand abundances. After Poissonisation the stopping time has P(τ > t) = P(N(t) < m), where N(t) is Poisson-binomial with q_i(t) = P(Poisson(p_i t) ≥ a); E[τ] is obtained by quadrature using the full Poisson-binomial pmf (divide-and-conquer FFT), giving all m at once. The `coverage_grid_PCR*.tsv` files compare the exact value with the CQT and CCP approximations (columns `K_C`, `K_L`/`K_U`, `K2`) over a grid of a and R for three abundance profiles (PCR10/30/60).

### 6.2 Array group testing E-step (`side/array_group_testing/`)
`array_exact.py` computes exact posterior marginals P(Y_ij = 1 | row, column and retest results) for matrix pooling with imperfect assays by inclusion–exclusion over rows, in O(2^R · R · C) per array (R the shorter side). The script checks the result against brute force on 3 × 4 arrays and exact rationals on an 8 × 12 array, and compares it with a Gibbs sampler.

## 7. Repository layout

```
week2/
  判据与预测.txt                      criteria and predictions, written before the week-2 runs
  全子集搜索的完整陈述与检验.txt        propositions, proofs, calibration, power, GTEx results
week3/
  判据与预测.txt                      frozen design and predictions, written before the evaluation runs
  对弱相关稳健的全子集检验.txt          whitened test: type-I, error budget, power, derivations
code/
  asset_exact.py, asset_power_out.txt first exact-search prototype and power check
  w2/
    asset_core.py                     statistics, prefix and block sweeps, DLM, importance sampling
    run_props.py, out_props_*.txt     proposition checks against exhaustive search
    run_null.py, run_power.py         null calibration and power
    null_equicorr.py, null_corr_check.py   type-I under ignored correlation
    read_rds.py, run_gtex.py          GTEx v6 parser and analysis
    jobs_*.txt                        argument lists of the runs
    res/                              summaries: o_*.txt, *.json, per-job result logs (log_*.txt)
  w3/
    w3_core.py, w3_sim.py, w3_batch.py     whitened test, simulation driver
    w3_predict.py, w3_derive_checks.py, w3_design_check_GW.py, w3_check_fixedpoint.py
    w3_summary.py                     -> res/summary.txt
    jobs_w3.txt, jobs_w3b.txt         job lists
    res/                              null_*.json, pow_*.json, predict_*.json, summary.txt
side/
  dna_storage_depth/                  coverage_exact.py, coverage_grid_PCR10/30/60.tsv
  array_group_testing/                array_exact.py
```

Scripts are written for Python 3.10 with NumPy and SciPy; run them from their own directory, e.g. `python run_props.py p123` in `code/w2/`.

## License

MIT, see `LICENSE`.
