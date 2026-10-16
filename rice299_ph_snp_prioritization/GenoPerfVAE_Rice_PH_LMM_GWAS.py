#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rice299-PH LOCO linear mixed model (LMM) GWAS and GenoPerfVAE SNP concordance.

Purpose
-------
Conduct an independent-method association analysis of *observed* rice PH,
NOT an analysis of GenoPerfVAE predictions. The final post-hoc GenoPerfVAE checkpoint
is used ONLY to enforce sample/SNP order; no neural network is run or trained.

Statistical method
------------------
LOCO EMMAX/P3D-type approximation: estimate null-model variance ratio by
REML once for each chromosome with a genomic relationship matrix (GRM)
computed from all OTHER chromosomes, then test each SNP by generalized least
squares with that variance ratio held fixed. P values are approximate LMM
association P values, NOT permutation-exact P values. Includes intercept and
optional PCs obtained from genome-wide GRM (default 3).

IMPORTANT genotype semantics
----------------------------
The finalized numerical CSV codes HapMap-called alleles as exactly -1 or +1.
Fractional values mostly correspond to HapMap NA/uncertain calls and are NOT
additional alleles or dosage. In GWAS only, fractional entries are treated as
missing and imputed to the SNP's called-state mean. The effect allele is the
NUMERIC '+1' state, with -1 coded as 0, +1 as 1. This does NOT change the
original GenoPerfVAE benchmark preprocessing or imply +1 is the reference allele.

Requirements: numpy, pandas, scipy, matplotlib, torch (checkpoint metadata).
Windows command (existing folder paths are used by default):
    python GenoPerfVAE_Rice_PH_LMM_GWAS.py
To specify paths explicitly:
    python GenoPerfVAE_Rice_PH_LMM_GWAS.py --checkpoint "...pth" \\
        --genotype "...csv" --phenotype "...csv" --candidate-file "...csv"

Outputs: AllSNP_GWAS.csv, Candidate_GWAS_Comparison.csv,
CandidateControl_Pairs.csv, Candidate_LD_Pairs.csv, Candidate_LD_Clusters.csv,
NullModels.csv, PCA.csv, Summary.csv, Config.json, Manhattan.png, QQ.png.

Caveats: same 299 accessions as the model; cross-method corroboration rather
than external-cohort validation. No causal, breedability, or functional claim.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GENO_DEFAULT = Path(r"E:\VGMP\Extended Comparison\data\MET_crfilt_.90_outliers_removed_for_RRBlup_line_corrected.csv")
PHENO_DEFAULT = Path(r"E:\VGMP\Extended Comparison\data\rice299_PH.csv")
CKPT_DEFAULT = Path(r"E:\VGMP\Extended Comparison\PerVAE\results_Rice_PH_Final_PostHoc\GenoPerfVAE_Rice_PH_Final_PostHoc.pth")
PERT_DIR_DEFAULT = Path(r"E:\VGMP\Extended Comparison\PerVAE\results_Rice_PH_ObservedGenotype_SNP_Perturbation_All299")
OUT_DEFAULT = Path(r"E:\VGMP\Extended Comparison\PerVAE\results_Rice_PH_LMM_GWAS")
PREFIX = "GenoPerfVAE_Rice_PH_LMM_GWAS"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--genotype", type=Path, default=GENO_DEFAULT)
    p.add_argument("--phenotype", type=Path, default=PHENO_DEFAULT)
    p.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    p.add_argument("--candidate-file", type=Path, default=PERT_DIR_DEFAULT / "GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation_ExactPerturbations.csv")
    p.add_argument("--matching-file", type=Path, default=PERT_DIR_DEFAULT / "GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation_Matching.csv")
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--min-maf", type=float, default=0.05)
    p.add_argument("--min-call-rate", type=float, default=0.90)
    p.add_argument("--min-minor-count", type=int, default=10)
    p.add_argument("--pcs", type=int, default=3, help="Top GRM PCs included as fixed covariates, default 3")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--ld-window-bp", type=int, default=1000000)
    p.add_argument("--ld-clump-r2", type=float, default=0.8)
    p.add_argument("--pair-permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-loco", action="store_true", help="Use whole-genome GRM (not recommended: proximal contamination)")
    args = p.parse_args(argv)
    if not 0 < args.min_maf < 0.5 or not 0 < args.min_call_rate <= 1:
        p.error("--min-maf must be in (0,0.5) and --min-call-rate in (0,1]")
    if args.chunk_size < 1 or args.pcs < 0 or args.pair_permutations < 0:
        p.error("chunk-size >= 1, pcs >= 0, pair-permutations >= 0 required")
    return args


def read_checkpoint_order(checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required only for checkpoint sample/SNP metadata") from exc
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Please use a PyTorch version supporting torch.load(weights_only=True)") from exc
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dictionary")
    entries = [str(x).strip() for x in ckpt.get("aligned_entries", [])]
    snps = [str(x).strip() for x in ckpt.get("snp_names", [])]
    if not entries or not snps:
        raise KeyError("Checkpoint lacks aligned_entries and/or snp_names")
    if len(set(entries)) != len(entries) or len(set(snps)) != len(snps):
        raise ValueError("Repeated sample/SNP identifiers in checkpoint")
    return entries, snps, ckpt.get("training_config", {})


def load_aligned_data(geno_path, pheno_path, entries, snps):
    print("Loading numeric genotype CSV; no SNP scaling is applied.", flush=True)
    geno = pd.read_csv(geno_path, low_memory=False)
    if "Entry" not in geno:
        raise KeyError("Numeric genotype CSV lacks Entry")
    geno["Entry"] = geno["Entry"].astype(str).str.strip()
    if geno["Entry"].duplicated().any():
        raise ValueError("Duplicate genotype Entry IDs")
    missing_ids = set(entries).difference(geno["Entry"])
    missing_snps = set(snps).difference(geno.columns)
    if missing_ids or missing_snps:
        raise ValueError(f"Checkpoint alignment failed: missing IDs {list(missing_ids)[:4]}, SNPs {list(missing_snps)[:4]}")
    geno = geno.set_index("Entry").loc[entries, snps]
    try:
        X = geno.to_numpy(dtype=np.float32)
    except (ValueError, TypeError):
        # Fall back for mixed-type CSVs; prevent accidental object arrays.
        X = geno.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    del geno
    phe = pd.read_csv(pheno_path)
    if "Entry" not in phe or "PH" not in phe:
        raise KeyError("Phenotype must contain Entry and PH")
    phe["Entry"] = phe["Entry"].astype(str).str.strip()
    if phe["Entry"].duplicated().any():
        raise ValueError("Duplicate PH Entry IDs")
    phe = phe.set_index("Entry")
    if not set(entries).issubset(phe.index):
        raise ValueError("Some checkpoint samples are missing PH")
    y = pd.to_numeric(phe.loc[entries, "PH"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("PH contains missing or non-finite values after alignment")
    print(f"Aligned: {X.shape[0]} samples x {X.shape[1]} SNPs; PH mean={y.mean():.6f}, Q95={np.quantile(y, .95):.6f}", flush=True)
    return X, y


def parse_snp(s):
    match = re.fullmatch(r"S(?:Chr)?(\d+)_(\d+)", str(s), flags=re.IGNORECASE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def prepare_genotypes(X, names, min_maf, min_call_rate, min_minor_count):
    """Create called-allele 0/1, missing=mean, genomic standardization.

    Returns float32 arrays of dosage and standardization for ALL markers, and
    boolean test mask. The imputation is GWAS-specific, not GenoPerfVAE imputation.
    """
    called_plus = X == np.float32(1)
    called_minus = X == np.float32(-1)
    called = called_plus | called_minus
    n = X.shape[0]
    plus_n = called_plus.sum(axis=0).astype(np.int32)
    minus_n = called_minus.sum(axis=0).astype(np.int32)
    n_called = plus_n + minus_n
    call_rate = n_called.astype(np.float64) / n
    p = np.divide(plus_n, n_called, out=np.full(len(names), np.nan, dtype=np.float64), where=n_called > 0)
    maf = np.minimum(p, 1.0 - p)
    chrom_pos = [parse_snp(v) for v in names]
    chrom = np.asarray([x[0] if x[0] is not None else -1 for x in chrom_pos], dtype=np.int32)
    pos = np.asarray([x[1] if x[1] is not None else -1 for x in chrom_pos], dtype=np.int64)
    eligible = ((call_rate >= min_call_rate) & (maf >= min_maf) &
                (np.minimum(plus_n, minus_n) >= min_minor_count) & (chrom > 0))
    if not np.any(eligible):
        raise RuntimeError("No GWAS-eligible SNPs. Check MAF/call-rate thresholds and marker IDs.")
    # NaN is used ONLY as input missing-state marker here; do not use numeric fractions.
    allele = np.where(called_plus, 1.0, 0.0).astype(np.float32)
    allele = np.where(called, allele, np.where(np.isfinite(p), p, 0.0).astype(np.float32)[None, :])
    # Center and scale using Bernoulli called-state frequency.
    sd = np.sqrt(np.maximum(p * (1.0 - p), 0.0))
    Z = np.zeros_like(allele, dtype=np.float32)
    valid = eligible & (sd > 0)
    Z[:, valid] = ((allele[:, valid] - p[valid]) / sd[valid]).astype(np.float32)
    print(f"Called-state GWAS QC: {eligible.sum()}/{len(names)} eligible; "
          f"fractional/uncertain entries treated as missing ({int((~called).sum())} entries)", flush=True)
    summary = pd.DataFrame({
        "SNP": names, "Chromosome": chrom, "Position": pos,
        "N_Called": n_called, "N_PlusOne": plus_n, "N_MinusOne": minus_n,
        "Call_Rate": call_rate, "PlusOne_Frequency": p,
        "Minor_State_Frequency": maf, "GWAS_Eligible": eligible,
        "N_Fractional_or_Unknown": n - n_called,
    })
    return allele, Z, eligible, chrom, pos, summary


def kinship_by_chr(Z, chrom, eligible, chunk_size):
    """Return numerator sums of standardized marker GRMs, by chromosome."""
    n = Z.shape[0]
    K_by_chr = {}
    count = {}
    for c in sorted(np.unique(chrom[eligible])):
        ix = np.flatnonzero(eligible & (chrom == c))
        accum = np.zeros((n, n), dtype=np.float64)
        for start in range(0, len(ix), chunk_size):
            zz = Z[:, ix[start:start + chunk_size]].astype(np.float64, copy=False)
            accum += zz @ zz.T
        K_by_chr[int(c)] = accum
        count[int(c)] = len(ix)
        print(f"  Kinship markers chr {c}: {len(ix)}", flush=True)
    K_total = sum(K_by_chr.values(), np.zeros((n, n), dtype=np.float64))
    return K_total, K_by_chr, count


def make_global_pcs(K, n_pcs):
    eigenvalues, eigvecs = linalg.eigh((K + K.T) / 2.0, check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigvecs = eigenvalues[order], eigvecs[:, order]
    n_pcs = min(n_pcs, K.shape[0] - 3)
    pcs = eigvecs[:, :n_pcs] * np.sqrt(K.shape[0])
    return pcs, eigenvalues


def reml_log_delta(logdelta, eigenvalues, Uy, UW):
    """REML negative profile log-likelihood, additive constant omitted."""
    delta = np.exp(logdelta)
    v = np.maximum(eigenvalues, 0) + delta
    weights = np.sqrt(1.0 / v)
    y_w, W_w = Uy * weights, UW * weights[:, None]
    try:
        Q, R = np.linalg.qr(W_w, mode="reduced")
        if np.min(np.abs(np.diag(R))) < 1e-10:
            return np.inf
        residual = y_w - Q @ (Q.T @ y_w)
        sse = max(float(residual @ residual), 1e-30)
        n, q = W_w.shape
        return float((n-q) * np.log(sse / (n-q)) + np.sum(np.log(v)) +
                     2.0 * np.sum(np.log(np.abs(np.diag(R)))))
    except np.linalg.LinAlgError:
        return np.inf


def fit_null_model(K, y, W, chrom):
    n = len(y)
    K = (K + K.T) / 2.0
    # Tiny numerical negative eigenvalues from subtraction are clipped.
    eig, U = linalg.eigh(K, check_finite=True)
    if float(np.min(eig)) < -1e-4:
        raise RuntimeError(f"LOCO GRM for chr{chrom} has sizeable negative eigenvalues: {np.min(eig):g}")
    eig = np.maximum(eig, 0.0)
    Uy, UW = U.T @ y, U.T @ W
    f = lambda t: reml_log_delta(t, eig, Uy, UW)
    grid = np.linspace(-12.0, 12.0, 41)
    fvals = np.array([f(t) for t in grid])
    k = int(np.argmin(fvals))
    lo, hi = grid[max(k-1,0)], grid[min(k+1,len(grid)-1)]
    if lo >= hi:
        lo, hi = -12.0, 12.0
    res = optimize.minimize_scalar(f, bounds=(lo,hi), method="bounded",
                                   options={"xatol": 1e-5})
    logdelta = float(res.x if res.success else grid[k])
    if f(grid[k]) < f(logdelta):
        logdelta = float(grid[k])
    delta = float(np.exp(logdelta))
    w = 1.0 / np.sqrt(eig + delta)
    Ww = UW * w[:, None]
    yw = Uy * w
    Q, _ = np.linalg.qr(Ww, mode="reduced")
    yres = yw - Q @ (Q.T @ yw)
    sse_null = float(yres @ yres)
    if sse_null <= 0:
        raise ValueError("Null residual PH variance is zero; cannot run GWAS")
    # Projected whiten matrix maps original sample genotypes to GLS-residualized space.
    projector = ((np.eye(n) - Q @ Q.T) * w[None, :]) @ U.T
    # w scales spectral-coordinate columns of the residual projection.
    meta = {
        "Chromosome": chrom, "N_Samples": n, "FixedCovariates": W.shape[1],
        "REML_Delta_ResidualOverGenetic": delta, "REML_LogDelta": logdelta,
        "NullResidualSumSquares_Whitened": sse_null,
        "Kinship_Eigen_Min": float(eig.min()), "Kinship_Eigen_Max": float(eig.max()),
        "REML_GridBoundary": bool(k == 0 or k == len(grid)-1),
    }
    return projector, yres, sse_null, meta


def gwas_test_batch(projector, yres, sse_null, dosage, n_fixed_cov):
    """P3D/EMMAX approximate Wald t-test given estimated null variance ratio."""
    n = dosage.shape[0]
    df = n - n_fixed_cov - 1
    if df < 2:
        raise RuntimeError("Not enough residual degrees of freedom")
    xres = projector @ dosage.astype(np.float64, copy=False)
    xx = np.sum(xres * xres, axis=0)
    xy = xres.T @ yres
    beta = np.divide(xy, xx, out=np.full_like(xy, np.nan), where=xx > 1e-10)
    rss = np.maximum(sse_null - beta * xy, 1e-14)
    se = np.sqrt(np.divide(rss / df, xx, out=np.full_like(xx, np.nan), where=xx > 1e-10))
    tval = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = np.minimum(1.0, 2 * stats.t.sf(np.abs(tval), df))
    pvalue[xx <= 1e-10] = np.nan
    return beta, se, tval, pvalue, df


def run_gwas(allele, Z, elig, chrom, y, pcs, kinship_data, args):
    Ktotal, Kchr, counts = kinship_data
    chromosomes = sorted(counts)
    W = np.column_stack([np.ones(len(y), dtype=np.float64), pcs])
    if np.linalg.matrix_rank(W) != W.shape[1]:
        raise ValueError("Fixed-effect covariates are collinear")
    all_records = []
    null_rows = []
    mtotal = sum(counts.values())
    K_global = Ktotal / mtotal
    for c in chromosomes:
        idx = np.flatnonzero(elig & (chrom == c))
        if args.no_loco:
            K = K_global
            used_count = mtotal
        else:
            used_count = mtotal - counts[c]
            if used_count < 2:
                raise ValueError("LOCO requires markers outside each tested chromosome")
            K = (Ktotal - Kchr[c]) / used_count
        print(f"GWAS chr {c}: {len(idx)} SNP tests; null GRM from {used_count} markers ...", flush=True)
        projector, yres, sse, meta = fit_null_model(K, y, W, c)
        meta.update({"KinshipMarkers": used_count, "NSNP_Tested": len(idx),
                     "LOCO": not args.no_loco})
        null_rows.append(meta)
        for k in range(0, len(idx), args.chunk_size):
            j = idx[k:k + args.chunk_size]
            beta, se, tval, pv, df = gwas_test_batch(projector, yres, sse, allele[:, j], W.shape[1])
            all_records.append(pd.DataFrame({"SNPIndex0": j, "Beta_PlusOne_vs_MinusOne_PH": beta,
                                             "SE_PH": se, "T_stat": tval, "PValue": pv, "DF": df,
                                             "KinshipLOCOChr": c}))
    if not all_records:
        raise RuntimeError("No valid association tests")
    return pd.concat(all_records, ignore_index=True).sort_values("SNPIndex0"), pd.DataFrame(null_rows)


def bh_fdr(p):
    p = np.asarray(p, dtype=np.float64)
    q = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    if len(order):
        raw = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.minimum.accumulate(raw[::-1])[::-1].clip(0, 1)
    return q


def join_and_save_gwas(tests, metadata, output):
    full = metadata.join(tests.set_index("SNPIndex0"), on="SNPIndex0", how="left", validate="one_to_one")
    m = int(np.isfinite(full["PValue"]).sum())
    if m < 1:
        raise RuntimeError("No finite GWAS p values")
    full["PValue"] = full["PValue"].clip(lower=np.finfo(float).tiny)
    full["MinusLog10P"] = -np.log10(full["PValue"])
    full["FDR_BH"] = bh_fdr(full["PValue"].values)
    full["Bonferroni_Passed"] = full["PValue"] < 0.05 / m
    full["FDR05_Passed"] = full["FDR_BH"] < 0.05
    full["GWAS_PRank"] = full["PValue"].rank(method="min", ascending=True)
    full["GWAS_PPercentile_StrongerIsHigher"] = 100.0 * (m - full["GWAS_PRank"] + 1) / m
    full.to_csv(output / f"{PREFIX}_AllSNP_GWAS.csv", index=False)
    tested = full.loc[np.isfinite(full["PValue"])]
    chisq = stats.chi2.isf(np.clip(tested["PValue"], 1e-300, 1), df=1)
    lambda_gc = float(np.median(chisq) / stats.chi2.median(df=1))
    print(f"GWAS tested SNPs={m}, Bonferroni sig={int(tested.Bonferroni_Passed.sum())}, "
          f"FDR <0.05={int(tested.FDR05_Passed.sum())}, lambda_GC={lambda_gc:.4f}", flush=True)
    return full, m, lambda_gc


def candidate_cross_comparison(gwas, cand_path, match_path, outdir, args):
    report = {"CandidateComparisonAvailable": False}
    if not cand_path.is_file():
        print(f"Candidate file not found; GWAS still runs: {cand_path}", flush=True)
        return report, set()
    cand = pd.read_csv(cand_path)
    if "SNP" not in cand or "Role" not in cand:
        raise ValueError("Candidate file must contain SNP and Role columns")
    cand["SNP"] = cand["SNP"].astype(str)
    if cand["SNP"].duplicated().any():
        raise ValueError("Duplicate SNP ID in perturbation file")
    cols = ["SNP", "Chromosome", "Position", "GWAS_Eligible", "N_Called", "Call_Rate",
            "Minor_State_Frequency", "Beta_PlusOne_vs_MinusOne_PH", "PValue", "MinusLog10P",
            "FDR_BH", "Bonferroni_Passed", "GWAS_PRank", "GWAS_PPercentile_StrongerIsHigher"]
    lookup = gwas[cols]
    merged = cand.merge(lookup, on="SNP", how="left", validate="one_to_one", suffixes=("_Perturb", "_GWAS"), indicator=True)
    merged["GWAS_Status"] = np.where(merged["_merge"] == "left_only", "SNP_NOT_IN_GWAS_TABLE",
                                     np.where(merged["GWAS_Eligible"].fillna(False), "TESTED", "QC_FILTERED"))
    merged.drop(columns=["_merge"], inplace=True)
    merged.to_csv(outdir / f"{PREFIX}_Candidate_GWAS_Comparison.csv", index=False)
    known_top = set(cand.loc[cand["Role"] == "TopGradientCandidate", "SNP"])
    eligible_top = merged[(merged["Role"] == "TopGradientCandidate") & (merged["GWAS_Status"] == "TESTED")]
    report.update({"CandidateComparisonAvailable": True, "CandidateCount": len(known_top),
                   "CandidateGWASTested": len(eligible_top), "CandidateBonferroniHits": int(eligible_top.Bonferroni_Passed.fillna(False).sum()),
                   "CandidateMedianMinusLog10P": float(eligible_top.MinusLog10P.median()) if len(eligible_top) else None})
    if len(eligible_top) > 2 and "MeanAbsDeltaPH" in eligible_top:
        tmp = eligible_top[["MeanAbsDeltaPH", "MinusLog10P"]].dropna()
        if len(tmp) > 2:
            rho, p = stats.spearmanr(tmp["MeanAbsDeltaPH"], tmp["MinusLog10P"])
            report["Perturb_vs_GWAS_Spearman"] = float(rho)
            report["Perturb_vs_GWAS_SpearmanP_Descriptive"] = float(p)
    if match_path.is_file():
        pairs = pd.read_csv(match_path)
        if not {"CandidateSNP", "ControlSNP"}.issubset(pairs.columns):
            raise ValueError("Matching file missing CandidateSNP or ControlSNP")
        gv = gwas.set_index("SNP")
        for role, suffix in (("CandidateSNP", "Candidate"), ("ControlSNP", "Control")):
            pairs[f"GWAS_P_{suffix}"] = pairs[role].map(gv["PValue"])
            pairs[f"GWAS_LogP_{suffix}"] = pairs[role].map(gv["MinusLog10P"])
            pairs[f"GWAS_PRank_{suffix}"] = pairs[role].map(gv["GWAS_PRank"])
        pairs["LogP_Difference_CandidateMinusControl"] = pairs.GWAS_LogP_Candidate - pairs.GWAS_LogP_Control
        pairs.to_csv(outdir / f"{PREFIX}_CandidateControl_Pairs.csv", index=False)
        d = pairs["LogP_Difference_CandidateMinusControl"].dropna().to_numpy(dtype=float)
        report["MatchedPairsWithGWAS"] = len(d)
        if len(d) > 0:
            report["MeanMatchedLogP_Difference"] = float(np.mean(d))
            report["PairsCandidateStrongerP"] = int(np.sum(d > 0))
            # Conditional randomization by independently swapping labels within
            # each predetermined pair. Descriptive; model-preselection remains.
            rng = np.random.default_rng(args.seed)
            observed = abs(float(np.mean(d)))
            extreme = 0
            for start in range(0, args.pair_permutations, 2000):
                num = min(2000, args.pair_permutations-start)
                signs = rng.choice([-1.,1.], size=(num,len(d)))
                extreme += int(np.count_nonzero(np.abs((signs*d).mean(axis=1)) >= observed - 1e-12))
            report["PairSwap_PermutationP_TwoSided_Descriptive"] = ((extreme+1)/(args.pair_permutations+1)
                        if args.pair_permutations > 0 else None)
    return report, known_top


def ld_and_clumping(candidate_names, X, gwas, outdir, window_bp, r2_threshold):
    """Pairwise complete called-state LD among TOP candidate SNPs only."""
    idx = gwas.set_index("SNP", drop=False)
    top = sorted(set(candidate_names).intersection(idx.index))
    records = []
    parent = {s:s for s in top}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra,rb=root(a),root(b)
        if ra != rb: parent[rb]=ra
    for i,s1 in enumerate(top):
        m1 = idx.loc[s1]
        for s2 in top[i+1:]:
            m2 = idx.loc[s2]
            if int(m1.Chromosome) != int(m2.Chromosome): continue
            dist = abs(int(m1.Position)-int(m2.Position))
            if dist > window_bp: continue
            a = X[:,int(m1.SNPIndex0)]
            b = X[:,int(m2.SNPIndex0)]
            ok = ((a == 1)|(a == -1)) & ((b == 1)|(b == -1))
            r2 = np.nan
            n = int(ok.sum())
            if n >= 10 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                r2 = float(np.corrcoef(a[ok], b[ok])[0,1]**2)
            if np.isfinite(r2) and r2 >= r2_threshold:
                union(s1,s2)
            records.append({"Candidate1":s1,"Candidate2":s2,"Chromosome":int(m1.Chromosome),
                            "Distance_bp":dist,"N_PairwiseCalled":n,"R2_PairwiseCalled":r2,
                            "SameLDCluster":bool(np.isfinite(r2) and r2 >= r2_threshold)})
    pd.DataFrame(records, columns=["Candidate1","Candidate2","Chromosome","Distance_bp",
                                  "N_PairwiseCalled","R2_PairwiseCalled","SameLDCluster"]).to_csv(
        outdir / f"{PREFIX}_Candidate_LD_Pairs.csv",index=False)
    groups={}
    for s in top: groups.setdefault(root(s), []).append(s)
    clust=[]
    for no, members in enumerate(sorted(groups.values(),key=lambda m: min(m)),1):
        sig = idx.loc[members].sort_values("PValue", na_position="last")
        for snp in members:
            clust.append({"SNP":snp,"ClusterID":f"LD{no:02d}","ClusterSize":len(members),
                          "LeadSNP_MinGWASP":str(sig.iloc[0].SNP),
                          "LeadSNP_P":float(sig.iloc[0].PValue) if np.isfinite(sig.iloc[0].PValue) else np.nan,
                          "Definition":f"connected pairs within {window_bp} bp with r2 >= {r2_threshold}; candidate SNPs only"})
    pd.DataFrame(clust, columns=["SNP","ClusterID","ClusterSize","LeadSNP_MinGWASP","LeadSNP_P","Definition"]).to_csv(
        outdir / f"{PREFIX}_Candidate_LD_Clusters.csv",index=False)
    return len(groups),len(records)


def make_plots(full, candidate_names, outdir, m):
    tested = full.loc[full["PValue"].notna()].copy()
    plotdf = tested.loc[(tested["Chromosome"] > 0) & (tested["Position"] >= 0)].sort_values(["Chromosome","Position"])
    offsets = {}
    offset=0
    ticks=[]; ticklabels=[]
    for c,group in plotdf.groupby("Chromosome",sort=True):
        offsets[int(c)] = offset
        ticks.append(offset+(int(group.Position.min())+int(group.Position.max()))/2)
        ticklabels.append(str(int(c)))
        offset += int(group.Position.max()) + 1000000
    xx = np.array([offsets[int(c)] + int(p) for c,p in zip(plotdf.Chromosome,plotdf.Position)])
    yy = plotdf.MinusLog10P.to_numpy()
    fig,ax=plt.subplots(figsize=(13,5.1),dpi=150)
    colors=["#376A8C","#9AABBD"]
    for i,(c,g) in enumerate(plotdf.groupby("Chromosome",sort=True)):
        mask=plotdf.Chromosome.to_numpy()==c
        ax.scatter(xx[mask],yy[mask],s=4,alpha=.72,c=colors[i%2],linewidths=0,rasterized=True)
    top = plotdf.SNP.isin(candidate_names).to_numpy()
    if top.any():
        ax.scatter(xx[top],yy[top],s=34,facecolors="none",edgecolors="#C85D42",linewidths=.9,
                   label="GenoPerfVAE candidates",zorder=3)
        ax.legend(loc="upper right",fontsize=9)
    ax.axhline(-np.log10(.05/m),color="#B0413E",ls="--",lw=1,label="Bonferroni 0.05")
    ax.set_xticks(ticks,ticklabels); ax.set_xlabel("Chromosome"); ax.set_ylabel("-log10(LMM P)")
    ax.set_title("Rice299 PH | LOCO LMM GWAS (approximate EMMAX/P3D)")
    ax.grid(axis="y",alpha=.15)
    fig.tight_layout(); fig.savefig(outdir / f"{PREFIX}_Manhattan.png",dpi=210)
    plt.close(fig)
    p=np.sort(tested.PValue.to_numpy())
    expected=-np.log10((np.arange(1,len(p)+1)-0.5)/len(p))
    observed=-np.log10(np.clip(p,1e-300,1))
    fig,ax=plt.subplots(figsize=(5.8,5.8),dpi=150)
    ax.scatter(expected,observed,s=5,alpha=.5,c="#356C91",rasterized=True)
    hi=max(1,float(expected.max()),float(observed.max()))
    ax.plot([0,hi],[0,hi],color="gray",lw=1,ls="--")
    ax.set_xlim(0,hi*1.02); ax.set_ylim(0,hi*1.02)
    ax.set_xlabel("Expected -log10(P)"); ax.set_ylabel("Observed -log10(P)")
    ax.set_title("PH LMM GWAS | QQ plot")
    fig.tight_layout(); fig.savefig(outdir / f"{PREFIX}_QQ.png",dpi=210)
    plt.close(fig)


def main(argv=None):
    args=parse_args(argv)
    t0=time.perf_counter()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    print("="*84)
    print("Rice299-PH | LMM GWAS | observed phenotype | LOCO GRM")
    print("="*84)
    entries, names, train_conf=read_checkpoint_order(args.checkpoint)
    X,y=load_aligned_data(args.genotype,args.phenotype,entries,names)
    print(f"Checkpoint training recon weight: {train_conf.get('recon_weight', 'unknown')} "
          "(metadata ONLY; no neural network or decoder is used)", flush=True)
    if len(entries)!=299 or len(names)!=73147:
        warnings.warn(f"Nonstandard data size: {X.shape}; expected Rice299-PH 299x73147")
    allele,Z,elig,chrom,pos,metadata=prepare_genotypes(X,names,args.min_maf,args.min_call_rate,args.min_minor_count)
    metadata["SNPIndex0"] = np.arange(len(names),dtype=int)
    Ktotal,Kchr,count=kinship_by_chr(Z,chrom,elig,args.chunk_size)
    Kglobal=Ktotal/sum(count.values())
    PCs,eigenvalues=make_global_pcs(Kglobal,args.pcs)
    pcs_df=pd.DataFrame(PCs,columns=[f"PC{i+1}" for i in range(PCs.shape[1])])
    pcs_df.insert(0,"Entry",entries); pcs_df["PH"]=y
    pcs_df.to_csv(args.out_dir / f"{PREFIX}_PCA.csv",index=False)
    gwas,null_models=run_gwas(allele,Z,elig,chrom,y,PCs,(Ktotal,Kchr,count),args)
    null_models.to_csv(args.out_dir / f"{PREFIX}_NullModels.csv",index=False)
    full,m,lambda_gc=join_and_save_gwas(gwas,metadata,args.out_dir)
    report,cands=candidate_cross_comparison(full,args.candidate_file,args.matching_file,args.out_dir,args)
    ncluster,npairs=ld_and_clumping(cands,X,full,args.out_dir,args.ld_window_bp,args.ld_clump_r2)
    make_plots(full,cands,args.out_dir,m)
    summary={"Dataset":"Rice299","Trait":"PH","N_Samples":len(y),"N_All_SNPs":len(names),
             "N_GWAS_Eligible":int(elig.sum()),"N_GWAS_Tested":m,
             "Bonferroni_Threshold":float(.05/m),"Bonferroni_Hits":int(full.Bonferroni_Passed.sum()),
             "FDR_BH_05_Hits":int(full.FDR05_Passed.sum()),"Lambda_GC":lambda_gc,
             "LMM":"EMMAX/P3D-type approximate LMM fixed null REML variance ratio",
             "Kinship":"LOCO standardized called-only SNP GRM" if not args.no_loco else "whole-genome GRM",
             "PC_Count":PCs.shape[1],"Effect_Allele":"numeric +1 vs -1; not nucleotide annotated",
             "FractionalValues":"treated as missing for GWAS; called-state mean-imputed",
             "GWAS_Phenotype":"observed unstandardized PH, not GenoPerfVAE prediction",
             "MatchedCandidateLDPairsWithWindow":npairs,"CandidateLDClusters":ncluster,
             "AnalysisRole":"cross-method association analysis on the same 299 individuals; not independent cohort / causal",
             "Seconds":float(time.perf_counter()-t0), **report}
    pd.DataFrame([summary]).to_csv(args.out_dir / f"{PREFIX}_Summary.csv",index=False)
    conf={"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          "checkpoint_training_config":train_conf,
          "method":"LOCO EMMAX/P3D approximate LMM: per-chromosome REML null, conditional GLS Wald t test",
          "GWAS_mapping":"-1 -> 0, +1 -> 1; other numerical -> missing -> marker-specific called mean",
          "MAF":"minor-state frequency based on called -1/+1 only",
          "limitations":["approximate fixed null variance ratio P values", "same 299 PH accessions as model training",
                         "+1/-1 numeric direction is not a reference nucleotide", "paired perturbation controls selected within trained model",
                         "single-marker GWAS and LD do not prove causal effects", "candidate LD clumping covers selected candidates only"]}
    with open(args.out_dir/f"{PREFIX}_Config.json","w",encoding="utf-8") as f:
        json.dump(conf,f,ensure_ascii=False,indent=2,default=str)
    print("="*84)
    print("LMM GWAS DONE")
    for k,v in summary.items():
        if k in ("Seconds","N_GWAS_Tested","Bonferroni_Hits","FDR_BH_05_Hits","Lambda_GC",
                 "CandidateGWASTested","CandidateBonferroniHits","MeanMatchedLogP_Difference",
                 "PairSwap_PermutationP_TwoSided_Descriptive"):
            print(f"{k}: {v}")
    print(f"Saved to: {args.out_dir}")
    print("Do not interpret same-cohort GWAS as external validation or as evidence of causality.")


if __name__ == "__main__":
    main()
