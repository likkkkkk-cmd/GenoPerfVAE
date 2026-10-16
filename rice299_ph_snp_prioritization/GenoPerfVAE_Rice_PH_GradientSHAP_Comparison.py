#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GenoPerfVAE Rice299-PH | GradientSHAP vs exact observed-genotype perturbation
=======================================================================
Reviewer #7 baseline study.  Uses the SAME frozen full-data GenoPerfVAE checkpoint,
299 checkpoint-aligned rice accessions, and exactly the same called-state SNP QC.
NO training, NO latent optimization, NO decoder, NO fabricated genotype states.

Primary GradientSHAP baseline:
  * Captum GradientShap (sampled expected-gradients approximation; NOT exact SHAP)
  * 32 real observed genotype backgrounds drawn per seed; n_samples=16
  * stdevs=0 (no Gaussian noise; interpolation may still be non-genotypic)
  * Mean absolute attribution over valid +/-1 called observations per SNP
  * Ranking only among 38,796 eligible SNPs (by the original QC rule)
  * 299 query accessions; optionally run seeds 42,43,44 for stability

Fair comparison:
  * Independent global SHAP Top 150 vs existing gradient-screened Top 150
  * Calculate the same exact +/-1 flip score for SHAP-only markers, using
    the pre-existing perturbation function on all 299 observed accessions.
  * GWAS comparison uses the PREVIOUS PC10 observed-PH GWAS, not retrained.
  * Summary makes no claim of superiority, causal effects, independence, or
    preserved haplotype/LD structure.

Requirements: numpy, pandas, torch, performer-pytorch, captum; scipy optional.
Put this file next to:
  GenoPerfVAE_Rice_PH_Target_RoundTrip_Validation.py
  GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation.py

Usage on Windows (from E:\\VGMP\\Extended Comparison\\GenoPerfVAE):
  pip install captum
  python GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py --stage all --device cuda
  python GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py --stage shap --device cuda
  python GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py --stage compare --device cuda
  python GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py --stage all --seeds 42,43,44
  python GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py --self-test

For a fast unreportable preliminary run:
  python GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py --stage shap --smoke-test

The whole-cohort experiment may be computationally intensive on 73,147 SNPs.
If GPU OOM occurs, use --batch-size 1 or --n-samples 8 and clearly report
that change. Captum internally expands the effective batch approximately by
n_samples. Start with one seed before a 3-seed stability analysis.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PREFIX = "GenoPerfVAE_Rice_PH_GradientSHAP"
SNP_QC_EXPECTED = 38796
TOL = 1e-6


class FrozenPHModel(nn.Module):
    """Deterministic ORIGINAL-PH-scale output, identical to prior perturbation."""

    def __init__(self, model: nn.Module, ymean: float, yscale: float):
        super().__init__()
        self.model = model
        self.ymean = float(ymean)
        self.yscale = float(yscale)

    def forward(self, x):
        mu, _logvar = self.model.encode(x)
        return self.model.predictor(mu).reshape(-1) * self.yscale + self.ymean


def load_local_module(path: Path, name: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing required reference script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_json(obj):
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    return obj


def dump_json(path: Path, d: dict):
    path.write_text(json.dumps(safe_json(d), ensure_ascii=False, indent=2), encoding="utf-8")


def rank_desc(values, eligible):
    values = np.asarray(values, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    valid = np.flatnonzero(eligible & np.isfinite(values))
    ranked = valid[np.lexsort((valid, -values[valid]))]
    ranks = np.full(len(values), np.nan, dtype=np.float64)
    ranks[ranked] = np.arange(1, len(ranked) + 1, dtype=np.float64)
    return ranked, ranks


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    aa = pd.Series(a[ok]).rank(method="average").to_numpy()
    bb = pd.Series(b[ok]).rank(method="average").to_numpy()
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def validate_ids(snps):
    snps = np.asarray(snps, dtype=str)
    if len(set(snps.tolist())) != len(snps):
        raise ValueError("Duplicate SNP identifiers in aligned checkpoint input")
    return snps


def get_previous_exact(root: Path, override: Path | None, expected_k: int):
    name = "GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation_ExactPerturbations.csv"
    candidates = ([override] if override is not None else [
        root / "results_Rice_PH_Top150_FinalExpansion" / "01_ObservedGenotype_ExactPerturbation" / name,
        root / "results_Rice_PH_Top150_FinalExpansion" / "01_ObservedGenotype_ExactPerturbations" / name,
        root / name,
    ])
    found = next((x for x in candidates if x is not None and x.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            "Cannot locate ORIGINAL Top150 perturbation CSV. Specify --original-exact.\n"
            "Checked: " + "; ".join(map(str, candidates)))
    df = pd.read_csv(found)
    required = {"SNP", "SNPIndex0", "MeanAbsDeltaPH", "N_Evaluable"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Original perturbation CSV missing: {sorted(missing)}")
    if "Role" in df:
        df = df[df["Role"].eq("TopGradientCandidate")].copy()
    if len(df) != expected_k or df.SNP.nunique() != expected_k:
        raise ValueError(f"Expected {expected_k} distinct original candidates, got {len(df)}")
    if df.N_Evaluable.isna().any() or (df.N_Evaluable.astype(int) < 1).any():
        raise ValueError("Original exact perturbations contain zero evaluable samples")
    return df.copy(), found


def resolve_gwas(root: Path, explicit: Path | None):
    name = "GenoPerfVAE_Rice_PH_LMM_GWAS_AllSNP_GWAS.csv"
    candidates = ([explicit] if explicit else [
        root / "results_Rice_PH_LMM_GWAS_PC10" / name,
        root / name,
    ])
    found = next((p for p in candidates if p is not None and p.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            "Cannot locate PC10 full GWAS file; use --gwas-csv. Checked: " + "; ".join(map(str, candidates)))
    gw = pd.read_csv(found, low_memory=False)
    if not {"SNP", "PValue", "FDR_BH", "GWAS_Eligible"}.issubset(gw):
        raise ValueError("GWAS columns SNP, PValue, FDR_BH, GWAS_Eligible required")
    if gw.SNP.duplicated().any():
        raise ValueError("Duplicate SNP ID in full GWAS")
    return gw, found


def setup(args):
    root = args.project_dir.resolve()
    rt = load_local_module(args.roundtrip_script or root / "GenoPerfVAE_Rice_PH_Target_RoundTrip_Validation.py", "rice_ph_roundtrip")
    perturb = load_local_module(args.perturb_script or root / "GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation.py", "rice_ph_perturb")
    rt.set_seed_fixed(args.seeds[0])
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but GPU CUDA is not available")
    ckpt_path = rt.resolve_checkpoint(root, args.checkpoint)
    genotype, phenotype = rt.resolve_data_files(root, args.data_dir, args.genotype, args.phenotype)
    ckpt = rt.torch_load_checkpoint(ckpt_path, device)
    recon = ckpt.get("training_config", {}).get("recon_weight")
    if recon is None or abs(float(recon) - args.expected_recon_weight) > 1e-8:
        if not args.allow_ablation_checkpoint:
            raise RuntimeError("Checkpoint reconstruction weight mismatch: "
                               f"found {recon!r}, expected {args.expected_recon_weight}. "
                               "Set --allow-ablation-checkpoint only for a LABELED ablation.")
        warnings.warn(f"ABLATED CHECKPOINT! reconstruction weight={recon!r}")
    model = rt.build_model_from_checkpoint(ckpt, device).eval()
    model.requires_grad_(False)
    raw, X, y, entries, names = rt.load_reference_data(genotype, phenotype, ckpt)
    raw = np.asarray(raw, np.float32)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32).reshape(-1)
    entries = np.asarray(entries, dtype=str)
    names = validate_ids(names)
    if raw.shape != X.shape or X.shape[1] != len(names) or len(y) != len(X):
        raise RuntimeError("Reference input shape/SNP checkpoint mismatch")
    if not np.isfinite(X).all():
        raise RuntimeError("Checkpoint input has non-finite values")
    if len(X) != 299 or len(names) != 73147:
        if not args.allow_mismatch:
            raise RuntimeError(f"Expected 299 x 73147, got {X.shape}; use --allow-mismatch ONLY if intended")
    plus, minus, nplus, nminus, ncalled, frac, minor, eligible = perturb.compute_eligibility(
        X, raw, args.min_state_count, args.min_state_freq, args.min_called_fraction)
    if eligible.sum() != SNP_QC_EXPECTED and not args.allow_mismatch:
        raise RuntimeError(f"Expected {SNP_QC_EXPECTED} eligible SNPs; got {eligible.sum()}; "
                           "check QC thresholds and data pipeline")
    output = (args.out_dir or root / "results_Rice_PH_GradientSHAP_Comparison").resolve()
    if args.smoke_test:
        output = output.with_name(output.name + "_SMOKETEST")
    output.mkdir(parents=True, exist_ok=True)
    f = FrozenPHModel(model, float(ckpt["scaler_y_mean"]), float(ckpt["scaler_y_scale"]))
    f.eval()
    f.requires_grad_(False)
    return dict(root=root, rt=rt, perturb=perturb, device=device, ckpt=ckpt,
                ckpt_path=ckpt_path, genotype=genotype, phenotype=phenotype,
                model=model, predictor=f, X=X, raw=raw, y=y, entries=entries,
                snps=names, nplus=nplus, nminus=nminus, ncalled=ncalled,
                callrate=frac, maf=minor, eligible=eligible, out=output,
                recon_weight=recon)


def seed_everything(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def do_shap(args, state):
    try:
        from captum.attr import GradientShap
    except ImportError as exc:
        raise ImportError("Missing Captum. Install with: python -m pip install captum") from exc
    t0_all = time.perf_counter()
    d = len(state["snps"])
    n = len(state["X"])
    count_samples = min(n, args.smoke_samples) if args.smoke_test else n
    out = state["out"]
    device = state["device"]
    probe = np.arange(count_samples, dtype=np.int32)
    valid_called = (np.abs(state["raw"] - 1) <= TOL) | (np.abs(state["raw"] + 1) <= TOL)
    records = []
    base_ids = []
    scores_per_seed = []
    attribution_all_per_seed = []
    seed_rank = {}
    for seed in args.seeds:
        seed_everything(seed, device)
        rng = np.random.default_rng(seed)
        nbase = min(args.baseline_size, n)
        background_idx = np.sort(rng.choice(n, nbase, replace=False))
        baseline = torch.from_numpy(np.ascontiguousarray(state["X"][background_idx])).to(device)
        base_ids.extend({"Seed": seed, "Entry": str(state["entries"][i]), "RowIndex0": int(i)}
                        for i in background_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        gshap = GradientShap(state["predictor"], multiply_by_inputs=True)
        total_abs = np.zeros(d, np.float64)
        called_abs = np.zeros(d, np.float64)
        called_count = np.zeros(d, np.int32)
        start = time.perf_counter()
        print(f"[GradientSHAP seed={seed}] {count_samples} queries, {nbase} baselines, "
              f"n_samples={args.n_samples}, batch_size={args.batch_size}", flush=True)
        for start_i in range(0, count_samples, args.batch_size):
            end_i = min(start_i + args.batch_size, count_samples)
            ids = probe[start_i:end_i]
            xb = torch.from_numpy(np.ascontiguousarray(state["X"][ids])).to(device)
            try:
                with torch.enable_grad():
                    attr = gshap.attribute(xb, baselines=baseline,
                                           n_samples=args.n_samples, stdevs=0.0)
                vals = np.abs(attr.detach().float().cpu().numpy()).astype(np.float64)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("GradientSHAP GPU out of memory. Retry with "
                                   "--batch-size 1 and/or --n-samples 8; this "
                                   "changes the reported experimental configuration.") from exc
            if vals.shape != (len(ids), d) or not np.isfinite(vals).all():
                raise RuntimeError("Bad/non-finite GradientSHAP attribution output")
            mask = valid_called[ids]
            total_abs += vals.sum(axis=0)
            called_abs += np.where(mask, vals, 0.).sum(axis=0)
            called_count += mask.sum(axis=0).astype(np.int32)
            del attr, xb, vals
            if end_i == count_samples or end_i % max(args.batch_size * 10, 1) == 0:
                elapsed = time.perf_counter() - start
                print(f"  {end_i}/{count_samples} | {elapsed:.1f}s", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        else:
            peak_mb = np.nan
        dur = time.perf_counter() - start
        called_score = np.divide(called_abs, called_count,
                                 out=np.full(d, np.nan, np.float64), where=called_count > 0)
        all_score = total_abs / count_samples
        if count_samples == n and np.any(state["eligible"] & (called_count != state["ncalled"])):
            raise RuntimeError("Unexpected called observation count mismatch")
        scores_per_seed.append(called_score)
        attribution_all_per_seed.append(all_score)
        ranking, rank = rank_desc(called_score, state["eligible"])
        seed_rank[seed] = ranking
        per_seed = pd.DataFrame({
            "SNPIndex0": np.arange(d), "SNP": state["snps"], "Eligible": state["eligible"],
            "MeanAbsGradientSHAP_Called": called_score,
            "MeanAbsGradientSHAP_AllEntries": all_score,
            "N_CalledQueries": called_count, "Rank_Eligible": rank})
        per_seed.to_csv(out / f"{PREFIX}_Seed{seed}_AllSNP.csv", index=False)
        records.append(dict(Seed=seed, SamplesEvaluated=count_samples,
                            BaselineSize=nbase, Interpolations=args.n_samples,
                            Seconds=dur, PeakGPUAllocatedMB=peak_mb,
                            Top150SNPs=";".join(state["snps"][ranking[:args.top_k]].tolist())))
        print(f"  Seed {seed}: {dur:.2f}s; peak GPU allocated {peak_mb:.1f} MB", flush=True)
        del baseline, gshap
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stacked = np.vstack(scores_per_seed)
    all_stacked = np.vstack(attribution_all_per_seed)
    # Average attribution magnitudes across independent reference-set seeds.
    count_valid = np.isfinite(stacked).sum(axis=0)
    score = np.divide(np.nansum(stacked, axis=0), count_valid,
                      out=np.full(d, np.nan), where=count_valid > 0)
    all_score = np.nanmean(all_stacked, axis=0)
    rank_idx, ranks = rank_desc(score, state["eligible"])
    if len(rank_idx) < args.top_k:
        raise RuntimeError(f"Not enough eligible finite scores: {len(rank_idx)}")
    chrid, pos = state["perturb"].parse_marker_ids(state["snps"])
    tab = pd.DataFrame(dict(SNP=state["snps"], SNPIndex0=np.arange(d),
                            Chromosome=chrid, Position=pos,
                            GWASEligible=state["eligible"],
                            N_Called=state["ncalled"],
                            CalledStateRate=state["callrate"],
                            CalledMinorStateFrequency=state["maf"],
                            MeanAbsGradientSHAP_Called=score,
                            MeanAbsGradientSHAP_AllEntries=all_score,
                            Eligible_SHAP_Rank=ranks,
                            SelectedTopK=np.isin(np.arange(d), rank_idx[:args.top_k])))
    tab.to_csv(out / f"{PREFIX}_AllSNP_Attributions.csv", index=False)
    top = tab.iloc[rank_idx[:args.top_k]].copy()
    top.insert(0, "SelectionRank", np.arange(1, args.top_k + 1))
    top.to_csv(out / f"{PREFIX}_Top{args.top_k}_Candidates.csv", index=False)
    pd.DataFrame(records).drop(columns=["Top150SNPs"]).to_csv(
        out / f"{PREFIX}_SeedRuntime.csv", index=False)
    pd.DataFrame(base_ids).to_csv(out / f"{PREFIX}_BaselineEntries.csv", index=False)
    stability = []
    for a in range(len(args.seeds)):
        for b in range(a + 1, len(args.seeds)):
            seeda, seedb = args.seeds[a], args.seeds[b]
            ia, ib = seed_rank[seeda], seed_rank[seedb]
            sa, sb = scores_per_seed[a], scores_per_seed[b]
            stability.append({"SeedA": seeda, "SeedB": seedb,
                              "EligibleSpearman": spearman(sa[state["eligible"]], sb[state["eligible"]]),
                              "Top32Overlap": int(len(set(ia[:32]) & set(ib[:32]))),
                              "Top50Overlap": int(len(set(ia[:50]) & set(ib[:50]))),
                              "Top150Overlap": int(len(set(ia[:args.top_k]) & set(ib[:args.top_k])))})
    pd.DataFrame(stability, columns=["SeedA", "SeedB", "EligibleSpearman", "Top32Overlap",
                                     "Top50Overlap", "Top150Overlap"]).to_csv(
        out / f"{PREFIX}_SeedStability.csv", index=False)
    summary = dict(Analysis="Post-hoc in-sample Captum GradientSHAP approximate attribution",
                   CandidateN=args.top_k, QuerySamples=count_samples, BackgroundPerSeed=args.baseline_size,
                   Seeds=args.seeds, NInterpPerSample=args.n_samples,
                   EffectiveInputsPerMinibatchApprox=args.batch_size * args.n_samples,
                   SnpsAll=d, SnpsEligible=int(state["eligible"].sum()),
                   Score="Mean absolute sampled GradientSHAP attribution on +/-1 called observations",
                   Baseline="Random REAL observed genotype rows per seed",
                   Stdevs=0, AttributionMethod="Captum GradientShap; multiply_by_inputs=True",
                   SecondsTotal=time.perf_counter() - t0_all,
                   PeakGPUAllocatedMB=max((r["PeakGPUAllocatedMB"] for r in records), default=np.nan),
                   Warning="Same-cohort, correlated SNPs and baseline-dependent approximations. "
                           "Interpolation paths need not be valid haplotypes. NOT exact SHAP/causal.")
    dump_json(out / f"{PREFIX}_Attribution_Config.json", summary)
    print("GradientSHAP complete. Global Top 10 SNPs:")
    print(top[["SelectionRank", "SNP", "MeanAbsGradientSHAP_Called"]].head(10).to_string(index=False))
    return top, summary


def do_compare(args, state):
    if args.smoke_test:
        raise RuntimeError("Comparison is prohibited on --smoke-test results")
    out = state["out"]
    topfile = out / f"{PREFIX}_Top{args.top_k}_Candidates.csv"
    if not topfile.is_file():
        raise FileNotFoundError(f"Run --stage shap first: {topfile}")
    top = pd.read_csv(topfile)
    config_file = out / f"{PREFIX}_Attribution_Config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing GradientSHAP run metadata: {config_file}")
    shap_config = json.loads(config_file.read_text(encoding="utf-8"))
    if (int(shap_config["QuerySamples"]) != len(state["X"]) or
            int(shap_config["SnpsAll"]) != len(state["snps"]) or
            int(shap_config["SnpsEligible"]) != int(state["eligible"].sum()) or
            int(shap_config["CandidateN"]) != args.top_k):
        raise RuntimeError("GradientSHAP attribution run is incomplete or uses incompatible cohort/QC/top-k")
    if len(top) != args.top_k or top.SNP.nunique() != args.top_k:
        raise RuntimeError("Bad GradientSHAP top candidate CSV")
    snp_map = {s:i for i,s in enumerate(state["snps"])}
    if any(s not in snp_map for s in top.SNP):
        raise RuntimeError("GradientSHAP SNP IDs do not match checkpoint")
    for s, ix in zip(top.SNP, top.SNPIndex0):
        if int(ix) != snp_map[s]:
            raise RuntimeError(f"SHAP column-index mismatch at {s}")
    prior, priorfile = get_previous_exact(state["root"], args.original_exact, args.top_k)
    prior["SNPIndex0"] = prior["SNPIndex0"].astype(int)
    for s, ix in zip(prior.SNP, prior.SNPIndex0):
        if s not in snp_map or snp_map[s] != ix:
            raise RuntimeError(f"Original perturbation misaligned SNP ID/index: {s} index {ix}")
    if int(prior.N_Evaluable.max()) > len(state["X"]):
        raise RuntimeError("Old perturbation outputs from incompatible cohort")
    counts_now = ((np.abs(state["raw"] + 1) <= TOL) | (np.abs(state["raw"] - 1) <= TOL)).sum(axis=0)
    for s, num in zip(prior.SNP, prior.N_Evaluable):
        if int(num) != int(counts_now[snp_map[s]]):
            raise RuntimeError(f"Original perturbation came from different samples/called states: {s}")
    # Re-evaluate ONLY SHAP-specific SNPs; original candidate exact scores are reused.
    oldset = set(prior.SNP)
    newset = set(top.SNP)
    missing = [s for s in top.SNP if s not in oldset]
    known = {str(r.SNP): {"MeanAbsDeltaPH": float(r.MeanAbsDeltaPH),
                          "N_Evaluable": int(r.N_Evaluable), "ExistingExact": True}
             for r in prior.itertuples()}
    t0 = time.perf_counter()
    base = state["perturb"].predict_matrix(
        state["model"], state["X"], state["device"], args.exact_batch_size,
        state["predictor"].ymean, state["predictor"].yscale)
    sample_idx = np.arange(len(state["X"]), dtype=np.int32)
    elite_mask = state["y"] >= np.quantile(state["y"], 0.95)
    # Verify several existing finite-difference scores against the current model.
    # Prevents silent comparison against another checkpoint despite identical IDs.
    for s in prior.sort_values("MeanAbsDeltaPH", ascending=False).head(min(3, len(prior))).SNP:
        _check_rows, check = state["perturb"].exact_single_snp_flip(
            state["model"], state["X"], state["raw"], sample_idx, snp_map[s],
            base, state["device"], args.exact_batch_size,
            state["predictor"].ymean, state["predictor"].yscale,
            state["entries"], state["y"], elite_mask)
        reported = float(prior.loc[prior.SNP.eq(s), "MeanAbsDeltaPH"].iloc[0])
        if not np.isclose(check["MeanAbsDeltaPH"], reported, rtol=0.005, atol=0.00005):
            raise RuntimeError(f"Existing original perturbation score differs for {s}: "
                               f"now={check['MeanAbsDeltaPH']:.8g}, previous={reported:.8g}. "
                               "Check checkpoint and cohort before comparing")
    print("Verified reproducibility of up to 3 original exact SNP scores.", flush=True)
    print(f"Comparative exact scoring: {len(missing)} SHAP-only SNPs; {len(oldset & newset)} overlap; "
          f"{len(state['X'])} observed genotypes.", flush=True)
    for k,s in enumerate(missing, 1):
        index = snp_map[s]
        _rows, stats = state["perturb"].exact_single_snp_flip(
            state["model"], state["X"], state["raw"], sample_idx, index,
            base, state["device"], args.exact_batch_size,
            state["predictor"].ymean, state["predictor"].yscale,
            state["entries"], state["y"], elite_mask)
        if stats["N_Evaluable"] < 1:
            raise RuntimeError(f"No called observations in SHAP candidate: {s}")
        known[s] = {"MeanAbsDeltaPH": float(stats["MeanAbsDeltaPH"]),
                    "N_Evaluable": int(stats["N_Evaluable"]), "ExistingExact": False}
        if k % 10 == 0 or k == len(missing):
            print(f"  Exact SHAP-only perturbations: {k}/{len(missing)}", flush=True)
    dur = time.perf_counter() - t0
    print(f"Exact new candidate evaluation complete: {dur:.2f}s", flush=True)
    gw, gwfile = resolve_gwas(state["root"], args.gwas_csv)
    gw = gw.set_index("SNP", verify_integrity=True)
    union = sorted(oldset | newset)
    rows = []
    shap_map = {str(r.SNP): r for r in top.itertuples()}
    original_map = {str(r.SNP): r for r in prior.itertuples()}
    for snp in union:
        if snp not in gw.index:
            raise RuntimeError(f"Candidate missing from full PC10 GWAS: {snp}")
        r = gw.loc[snp]
        if not bool(r.GWAS_Eligible):
            raise RuntimeError(f"Candidate not GWAS eligible: {snp}")
        record = known[snp]
        rows.append(dict(SNP=snp, SNPIndex0=snp_map[snp],
                         Selected_GradientTop150=snp in oldset,
                         Selected_GradientSHAPTop150=snp in newset,
                         OriginalGradientRank=(int(original_map[snp].GradientScreenRank)
                                               if snp in oldset and "GradientScreenRank" in prior else np.nan),
                         OriginalExactRank=(int(original_map[snp].ExactAbsEffectRank_AmongCandidates)
                                            if snp in oldset and "ExactAbsEffectRank_AmongCandidates" in prior else np.nan),
                         SHAPRank=(int(shap_map[snp].SelectionRank) if snp in newset else np.nan),
                         SHAPScoreCalled=(float(shap_map[snp].MeanAbsGradientSHAP_Called)
                                          if snp in newset else np.nan),
                         ExactMeanAbsDeltaPH=record["MeanAbsDeltaPH"],
                         ExactN_Evaluable=record["N_Evaluable"],
                         ExactReusedFromOriginal=record["ExistingExact"],
                         PC10_GWAS_P=float(r.PValue),
                         PC10_GWAS_MinusLog10P=(-np.log10(max(float(r.PValue),1e-300))),
                         PC10_FDR_BH=float(r.FDR_BH),
                         PC10_FDR05=bool(float(r.FDR_BH) < 0.05)))
    merged = pd.DataFrame(rows)
    merged = merged.sort_values(["ExactMeanAbsDeltaPH", "SNP"], ascending=[False, True])
    merged.to_csv(out / f"{PREFIX}_Top150Union_ExactAndGWAS.csv", index=False)
    methods = []
    for label, key in (("GradientScreen_Top150", "Selected_GradientTop150"),
                       ("GradientSHAP_Top150", "Selected_GradientSHAPTop150")):
        df = merged[merged[key]].copy()
        if len(df) != args.top_k:
            raise RuntimeError(f"{label} has {len(df)} instead of {args.top_k} SNPs")
        methods.append(dict(Method=label, SNPCount=len(df),
                            GWAS_P_LT_005=int((df.PC10_GWAS_P < 0.05).sum()),
                            GWAS_P_LT_001=int((df.PC10_GWAS_P < 0.01).sum()),
                            GWAS_FDR_LT_005=int(df.PC10_FDR05.sum()),
                            MeanExactAbsDeltaPH=float(df.ExactMeanAbsDeltaPH.mean()),
                            MedianExactAbsDeltaPH=float(df.ExactMeanAbsDeltaPH.median()),
                            Spearman_Exact_vs_GWASLogP=spearman(df.ExactMeanAbsDeltaPH,
                                                                 df.PC10_GWAS_MinusLog10P)))
    pd.DataFrame(methods).to_csv(out / f"{PREFIX}_MethodComparison_Summary.csv", index=False)
    old_exact_sorted = sorted(oldset, key=lambda s: -known[s]["MeanAbsDeltaPH"])
    new_shap_sorted = top.SNP.astype(str).tolist()
    overlaps = []
    for k in (32, 50, args.top_k):
        k = min(k, args.top_k)
        overlaps.append(dict(K=k, TopK_Overlap_GradientRank_vs_SHAP=int(len(
            set(prior.sort_values("GradientScreenRank").head(k).SNP) & set(new_shap_sorted[:k]))
        ) if "GradientScreenRank" in prior else np.nan,
                             TopK_Overlap_ExactWithinOriginal150_vs_SHAP=int(len(
                                 set(old_exact_sorted[:k]) & set(new_shap_sorted[:k])))))
    pd.DataFrame(overlaps).drop_duplicates("K").to_csv(
        out / f"{PREFIX}_CandidateOverlap.csv", index=False)
    common = sorted(oldset & newset)
    rho = spearman([known[s]["MeanAbsDeltaPH"] for s in common],
                   [float(shap_map[s].MeanAbsGradientSHAP_Called) for s in common])
    summ = dict(Model="Frozen full-data Rice299-PH GenoPerfVAE, reconstruction weight " + str(state["recon_weight"]),
                OriginalCandidates=args.top_k, SHAPCandidates=args.top_k,
                Intersect=len(common), Union=len(union), NewlyEvaluatedExact=len(missing),
                Spearman_SHAPAttribution_vsExactSensitivity_OverlapOnly=rho,
                ExactAdditionalSeconds=dur, OriginalExactFile=str(priorfile), GWASFile=str(gwfile),
                Interpretation="Comparison of different model explanation estimands; in-sample, "
                               "correlated SNPs and no independent causal validation.",
                Warning="Do NOT interpret exact sensitivity endpoint as an unbiased superiority "
                        "test for a method that explicitly selects by exact/gradient sensitivity; "
                        "nor nominal same-cohort GWAS as external validation.")
    dump_json(out / f"{PREFIX}_Comparison_Summary.json", summ)
    print(pd.DataFrame(methods).to_string(index=False))
    print(f"Final merged union ({len(union)}) and statistics: {out}")
    return summ


def self_test():
    print("Running offline synthetic checks (no model/checkpoint needed)...")
    assert rank_desc([2, 4, np.nan, 3], [True, True, True, False])[0].tolist() == [1, 0]
    assert abs(spearman([1,2,3],[3,2,1]) + 1) < 1e-12
    arr = np.array([[1, -1, .25], [-1, 1, .3], [1, 1, -.5]], np.float32)
    called = (np.abs(arr + 1) <= TOL) | (np.abs(arr - 1) <= TOL)
    counts = called.sum(axis=0)
    assert counts.tolist() == [3, 3, 0]
    class Toy(nn.Module):
        def encode(self, x):
            return x, torch.zeros_like(x)
        def __init__(self):
            super().__init__()
            self.predictor = nn.Linear(3,1, bias=False)
            self.predictor.weight.data[:] = torch.tensor([[1, 2, 3]])
    module = FrozenPHModel(Toy(), 4, 2)
    tensor = torch.tensor([[1., 2., 3.]], requires_grad=True)
    value = module(tensor)
    assert torch.allclose(value, torch.tensor([32.]))
    grads, = torch.autograd.grad(value.sum(), tensor)
    assert torch.allclose(grads, torch.tensor([[2.,4.,6.]]))
    print("SELF-TEST PASSED: ranks, Spearman, called-state filter, frozen PH-scale gradients")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["shap", "compare", "all"], default="all")
    parser.add_argument("--project-dir", type=Path, default=Path(r"E:\VGMP\Extended Comparison\GenoPerfVAE"))
    parser.add_argument("--roundtrip-script", type=Path)
    parser.add_argument("--perturb-script", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--genotype", type=Path)
    parser.add_argument("--phenotype", type=Path)
    parser.add_argument("--original-exact", type=Path)
    parser.add_argument("--gwas-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Captum batch size; internally expanded by --n-samples")
    parser.add_argument("--exact-batch-size", type=int, default=8)
    parser.add_argument("--baseline-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=16,
                        help="GradientShap interpolations per query per batch")
    parser.add_argument("--seeds", type=str, default="42", help="Comma separated, e.g. 42,43,44")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--min-state-count", type=int, default=10)
    parser.add_argument("--min-state-freq", type=float, default=0.05)
    parser.add_argument("--min-called-fraction", type=float, default=0.90)
    parser.add_argument("--expected-recon-weight", type=float, default=0.3)
    parser.add_argument("--allow-ablation-checkpoint", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true", help="Skip exact expected cohort/QC counts")
    parser.add_argument("--smoke-test", action="store_true", help="Analyze first few queries only; NOT reportable")
    parser.add_argument("--smoke-samples", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        args.seeds = [int(t.strip()) for t in args.seeds.split(",") if t.strip()]
    except ValueError:
        parser.error("Invalid --seeds. Use integers such as 42,43,44")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or
            min(args.batch_size, args.exact_batch_size, args.baseline_size,
                args.n_samples, args.top_k, args.smoke_samples) < 1):
        parser.error("Positive sizes and unique seeds required")
    if args.smoke_test and args.stage != "shap":
        parser.error("--smoke-test only supports --stage shap")
    return args


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    state = setup(args)
    print("="*80)
    print("Rice299-PH | Frozen GenoPerfVAE GradientSHAP vs exact perturbation")
    print("Shape:", state["X"].shape, "Eligible:", int(state["eligible"].sum()))
    print("Checkpoint:", state["ckpt_path"], "Recon:", state["recon_weight"])
    print("Device:", state["device"], "Output:", state["out"])
    print("SHAP is approximate EXPECTED GRADIENT attribution; not a genotype counterfactual.")
    print("="*80, flush=True)
    if args.stage in ("shap", "all"):
        do_shap(args, state)
    if args.stage in ("compare", "all"):
        do_compare(args, state)
    print("DONE. Do not claim causal effects, strict LD matching, or independent validation.")


if __name__ == "__main__":
    main()
