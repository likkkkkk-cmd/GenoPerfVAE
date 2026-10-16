#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GenoPerfVAE Rice299-PH | Observed-genotype SNP perturbation and matched controls
============================================================================
Post-hoc, frozen, in-sample model interpretation. NO model training and NO decoder.

Workflow:
  1. Restore the Rice299 genotype in the exact checkpoint sample/SNP order.
  2. Keep original numerical input: -1 is valid, +1 is valid; fractional values
     remain unchanged in the MODEL input, but are NOT treated as allele states.
  3. Exclude candidate SNPs without adequate BOTH-called-state support.
  4. Screen all eligible SNPs with an input-gradient approximation on REAL
     observed genotypes (the gradient is only for SHORTLISTING).
  5. Choose low-cost control SNPs matched to the called minor-state frequency,
     chromosome (when possible) and nearest-neighbor LD proxy (when available).
  6. For candidate/control SNPs, modify ONE SNP from a real genotype's called
     state to its other actually observed +/-1 state and run the FROZEN
     encoder -> predictor. This is a *model counterfactual*, not a real line.
  7. Save ranked candidates, matched controls, per-sample effects and audit.

CRITICAL SCIENTIFIC LIMITATIONS
-------------------------------
- This model was trained on the same 299 samples: ALL results are IN-SAMPLE
  model sensitivity, not held-out validation or biological confirmation.
- Each +/-1 flip is a single-position intervention in input feature space;
  it may not preserve LD/haplotype/recombination constraints.
- Directional effects refer to THE NUMERIC +/-1 CODING only. They are not
  allele-identity, additive QTL, causal, nor experimentally validated effects.
- The nearest-neighbor LD quantity is a very local approximate proxy, NOT
  genome-wide LD matching. Matching diagnostics are exported; no unqualified
  claim of strict MAF/LD matching or statistical significance is made.
- The candidate/control comparison is post-selection on this same model;
  a small empirical difference is not an independent statistical test.
- GWAS is NOT computed here; a separate genuine LMM GWAS is required.

Dependencies: numpy, pandas, torch, performer-pytorch, and the previously
created GenoPerfVAE_Rice_PH_Target_RoundTrip_Validation.py in the same directory
(or give --roundtrip-script). No checkpoint is rewritten.

Defaults are designed as a relatively economical first pass: gradient screening
of all 299 samples, then exact flips of the top 32 SNPs and 32 controls on 40
observed samples including all Q95 elite accessions. Choose --all-samples for
more robust but more expensive exact evaluation.

Example:
 python GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation.py
 python GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation.py --top-k 32 \
   --controls-per-candidate 1 --probe-samples 40
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PREFIX = "GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation"
TOL = 1e-6


def load_roundtrip(path: Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Put the previous round-trip script next "
            "to this file, or provide --roundtrip-script."
        )
    spec = importlib.util.spec_from_file_location("Genoperfvae_reference_roundtrip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "set_seed_fixed", "resolve_checkpoint", "resolve_data_files",
        "torch_load_checkpoint", "load_reference_data", "build_model_from_checkpoint",
    )
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise RuntimeError(f"Source round-trip script missing APIs: {missing}")
    return module


def parse_marker_ids(snp_names):
    chromosome = np.empty(len(snp_names), dtype=object)
    position = np.full(len(snp_names), -1, dtype=np.int64)
    pat = re.compile(r"^[Ss](\d+)[_:-](\d+)$")
    for j, snp in enumerate(snp_names):
        match = pat.match(str(snp))
        chromosome[j] = match.group(1) if match else ""
        if match:
            position[j] = int(match.group(2))
    return chromosome, position


def called_masks(X):
    """Only exactly coded +/-1 entries are called allele proxies."""
    plus = np.isfinite(X) & (np.abs(X - 1.0) <= TOL)
    minus = np.isfinite(X) & (np.abs(X + 1.0) <= TOL)
    return plus, minus


def local_adjacent_ld_proxy(X, called, chr_ids, pos, max_bp=200_000, min_pairs=20):
    """Nearest-SNP r^2, valid only for adjacent SNPs on same chromosome.

    Uses +/-1 called genotypes and pairwise-complete observations. Does NOT
    estimate full LD blocks. Missing values and fractional encodings are ignored.
    """
    n, d = X.shape
    out = np.full(d, np.nan, dtype=np.float32)
    if d < 2:
        return out
    adjacent = ((chr_ids[:-1] != "") & (chr_ids[:-1] == chr_ids[1:]) &
                (pos[:-1] >= 0) & (pos[1:] >= 0) &
                (np.abs(pos[1:] - pos[:-1]) <= max_bp))
    if not adjacent.any():
        return out
    pairmask = called[:, :-1] & called[:, 1:]
    pairmask &= adjacent[None, :]
    a = (X[:, :-1] < 0) & pairmask
    b = (X[:, 1:] < 0) & pairmask
    n_pair = pairmask.sum(axis=0).astype(np.float64)
    na = a.sum(axis=0).astype(np.float64)
    nb = b.sum(axis=0).astype(np.float64)
    nab = (a & b).sum(axis=0).astype(np.float64)
    den = na * (n_pair - na) * nb * (n_pair - nb)
    usable = adjacent & (n_pair >= min_pairs) & (den > 0)
    r2 = np.full(d - 1, np.nan, dtype=np.float64)
    r2[usable] = (n_pair[usable] * nab[usable] - na[usable] * nb[usable]) ** 2 / den[usable]
    r2 = np.clip(r2, 0, 1)
    for j in np.flatnonzero(np.isfinite(r2)):
        for k in (j, j + 1):
            if not np.isfinite(out[k]) or out[k] < r2[j]:
                out[k] = r2[j]
    return out


def compute_eligibility(X, raw, min_state_count, min_state_fraction, min_called_fraction):
    plus, minus = called_masks(raw)
    np1 = plus.sum(axis=0).astype(np.int32)
    nm1 = minus.sum(axis=0).astype(np.int32)
    ncalled = np1 + nm1
    frac_called = ncalled / X.shape[0]
    minor = np.minimum(np1, nm1) / np.maximum(ncalled, 1)
    # Minor-state frequency is a sample-frequency proxy, not diploid allele MAF.
    eligible = ((np1 >= min_state_count) & (nm1 >= min_state_count) &
                (minor >= min_state_fraction) &
                (frac_called >= min_called_fraction))
    return plus, minus, np1, nm1, ncalled, frac_called, minor, eligible


def predict_ph(model, xb: torch.Tensor, ymean: float, yscale: float):
    mu, _ = model.encode(xb)
    return model.predictor(mu).reshape(-1) * yscale + ymean


def predict_matrix(model, X, device, batch_size, ymean, yscale):
    out = np.empty(len(X), np.float32)
    with torch.inference_mode():
        for a in range(0, len(X), batch_size):
            b = min(a + batch_size, len(X))
            xb = torch.from_numpy(np.ascontiguousarray(X[a:b], dtype=np.float32)).to(device)
            pred = predict_ph(model, xb, ymean, yscale)
            out[a:b] = pred.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite model PH prediction")
    return out


def gradient_screen(model, X, raw, device, batch_size, ymean, yscale,
                    elite_mask):
    """Mean |d PH / d SNP| * 2 for +/-1 called positions, full 299 samples.

    *2 corresponds to the numeric distance between -1 and +1. Gradient is
    only a first-order screening score, NOT a finite-difference effect.
    """
    n, d = X.shape
    sums = np.zeros(d, np.float64)
    count = np.zeros(d, np.int32)
    esums = np.zeros(d, np.float64)
    ecount = np.zeros(d, np.int32)
    print("Screening gradients on all observed samples (frozen model)...", flush=True)
    for a in range(0, n, batch_size):
        b = min(a + batch_size, n)
        xb = torch.tensor(np.ascontiguousarray(X[a:b]), device=device,
                          dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            preds = predict_ph(model, xb, ymean, yscale)
            gx = torch.autograd.grad(preds.sum(), xb, retain_graph=False,
                                     create_graph=False, only_inputs=True)[0]
        g = gx.detach().cpu().numpy()
        if not np.isfinite(g).all():
            raise RuntimeError("Non-finite input gradient")
        called = (np.abs(raw[a:b] - 1.) <= TOL) | (np.abs(raw[a:b] + 1.) <= TOL)
        weighted_abs = 2. * np.abs(g)
        sums += np.sum(np.where(called, weighted_abs, 0.), axis=0, dtype=np.float64)
        count += called.sum(axis=0).astype(np.int32)
        if np.any(elite_mask[a:b]):
            ec = called[elite_mask[a:b]]
            ev = weighted_abs[elite_mask[a:b]]
            esums += np.sum(np.where(ec, ev, 0.), axis=0, dtype=np.float64)
            ecount += ec.sum(axis=0).astype(np.int32)
        if b == n or (b % max(batch_size * 5, 1) == 0):
            print(f"  Gradients: {b}/{n}", flush=True)
    out = np.divide(sums, count, out=np.full(d, np.nan, dtype=np.float64), where=count > 0)
    elite = np.divide(esums, ecount, out=np.full(d, np.nan, dtype=np.float64), where=ecount > 0)
    return out, elite


def select_probe_samples(y, elite_mask, desired, seed):
    """Deterministic samples: all Q95 elite plus stratified non-elite across PH."""
    rng = np.random.default_rng(seed)
    elites = np.flatnonzero(elite_mask)
    pool = np.flatnonzero(~elite_mask)
    desired = min(max(desired, len(elites)), len(y))
    if desired == len(y):
        return np.arange(len(y), dtype=np.int32)
    n_other = min(desired - len(elites), len(pool))
    sorted_pool = pool[np.argsort(y[pool], kind="stable")]
    groups = np.array_split(sorted_pool, min(5, max(1, n_other)))
    buckets = [[] for _ in groups]
    for k, group in enumerate(groups):
        buckets[k] = rng.permutation(group).tolist()
    chosen = []
    while len(chosen) < n_other:
        found = False
        for b in buckets:
            if b and len(chosen) < n_other:
                chosen.append(b.pop())
                found = True
        if not found:
            break
    return np.sort(np.r_[elites, np.asarray(chosen, dtype=np.int32)])


def match_controls(candidate_indices, eligible, minor_freq, local_ld, chrom, pos,
                   controls_per_candidate, seed, exclusion_bp=100_000):
    """Greedy no-replacement approximate match; export balance diagnostics.

    Must report *observed* frequency and LD-proxy differences. No strict
    matching is claimed when no sufficiently similar controls exist.
    """
    rng = np.random.default_rng(seed)
    d = len(eligible)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    cand_set = set(map(int, candidate_indices))
    used = set(cand_set)
    all_pool = np.flatnonzero(eligible)
    rows = []
    control_ids = []
    for j in candidate_indices:
        for rank in range(1, controls_per_candidate + 1):
            choices = np.asarray([i for i in all_pool if int(i) not in used], dtype=np.int32)
            if not len(choices):
                rows.append({"CandidateIndex": int(j), "ControlIndex": -1,
                             "MatchRank": rank, "MatchingStatus": "NO_ELIGIBLE_CONTROLS"})
                continue
            # Try not to compare to the candidate itself or its adjacent locus.
            away = ((chrom[choices] != chrom[j]) | (chrom[j] == "") |
                    (pos[choices] < 0) | (pos[j] < 0) |
                    (np.abs(pos[choices] - pos[j]) > exclusion_bp))
            if away.any():
                choices = choices[away]
            delta_freq = np.abs(minor_freq[choices] - minor_freq[j])
            same_chr = (chrom[j] != "") & (chrom[choices] == chrom[j])
            # Adjacent SNP LD r^2 is a weak local proxy. Penalize its mismatch
            # only when both metrics exist.
            have_ld = np.isfinite(local_ld[j]) & np.isfinite(local_ld[choices])
            dld = np.where(have_ld, np.abs(np.nan_to_num(local_ld[choices]) -
                                                np.nan_to_num(local_ld[j])), np.nan)
            ld_cost = np.where(np.isfinite(dld), dld, 0.3)
            same_chr_available = np.any(same_chr & (delta_freq <= 0.05))
            chr_cost = np.where(same_chr, 0., 0.1 if same_chr_available else 0.)
            costs = delta_freq / 0.02 + ld_cost / 0.2 + chr_cost
            costs += rng.uniform(0, 1e-5, len(costs))
            chosen = int(choices[np.argmin(costs)])
            used.add(chosen)
            control_ids.append(chosen)
            absf = abs(float(minor_freq[j] - minor_freq[chosen]))
            ldd = abs(float(local_ld[j] - local_ld[chosen])) if (
                np.isfinite(local_ld[j]) and np.isfinite(local_ld[chosen])) else np.nan
            status = "GOOD_APPROX_MATCH" if (absf <= 0.02 and
                       (not np.isfinite(ldd) or ldd <= 0.15)) else "WEAK_APPROX_MATCH"
            rows.append({"CandidateIndex": int(j), "ControlIndex": chosen,
                         "MatchRank": rank, "MinorStateFrequencyCandidate": float(minor_freq[j]),
                         "MinorStateFrequencyControl": float(minor_freq[chosen]),
                         "AbsMinorStateFrequencyDiff": absf,
                         "AdjacentLDProxyCandidate": float(local_ld[j]),
                         "AdjacentLDProxyControl": float(local_ld[chosen]),
                         "AbsAdjacentLDProxyDiff": ldd,
                         "SameChromosome": bool(chrom[j] != "" and chrom[j] == chrom[chosen]),
                         "MatchingStatus": status})
    return np.asarray(control_ids, dtype=np.int32), pd.DataFrame(rows)


def exact_single_snp_flip(model, X, raw, indices, marker_j, base_ph,
                          device, batch_size, ymean, yscale,
                          entries, y, elite_mask):
    """SNP +/-1 flip in real genotype background for called-only positions."""
    vals = raw[indices, marker_j]
    called = (np.abs(vals - 1.) <= TOL) | (np.abs(vals + 1.) <= TOL)
    sample_idx = indices[called]
    if len(sample_idx) == 0:
        return [], {"N_Evaluable": 0}
    altered_pred = np.empty(len(sample_idx), dtype=np.float32)
    for a in range(0, len(sample_idx), batch_size):
        b = min(a + batch_size, len(sample_idx))
        idx = sample_idx[a:b]
        mat = np.ascontiguousarray(X[idx, :], dtype=np.float32)
        mat[:, marker_j] = -mat[:, marker_j]
        with torch.inference_mode():
            xb = torch.from_numpy(mat).to(device)
            altered_pred[a:b] = predict_ph(model, xb, ymean, yscale).detach().cpu().numpy()
    xold = X[sample_idx, marker_j]
    dph = altered_pred.astype(np.float64) - base_ph[sample_idx].astype(np.float64)
    # A directional +1 minus -1 local contrast, including a sign correction
    # for samples flipped from +1 to -1. Not a causal/additive QTL effect.
    directed = np.where(xold < 0, dph, -dph)
    elite = elite_mask[sample_idx]
    good = np.isfinite(dph)
    if not good.all():
        raise RuntimeError("Non-finite perturbation effects")
    nonzero = np.abs(directed) > 1e-7
    mean_sign = np.sign(directed.mean())
    stability = float(np.mean(np.sign(directed[nonzero]) == mean_sign)) if nonzero.any() and mean_sign else np.nan
    rows = [{"Entry": str(entries[i]), "Observed_PH": float(y[i]),
             "EliteQ95": bool(elite_mask[i]), "SNPIndex": int(marker_j),
             "OriginalNumericState": float(X[i, marker_j]),
             "FlippedNumericState": float(-X[i, marker_j]),
             "BaselineModel_PH": float(base_ph[i]),
             "PerturbedModel_PH": float(altered_pred[k]),
             "DeltaPH_FlipMinusBaseline": float(dph[k]),
             "DirectionalPH_PlusOneMinusMinusOne": float(directed[k])}
            for k, i in enumerate(sample_idx)]
    summary = {"N_Evaluable": int(len(sample_idx)),
               "N_Elite_Evaluable": int(elite.sum()),
               "N_OrigMinusOne": int(np.sum(xold < 0)),
               "N_OrigPlusOne": int(np.sum(xold > 0)),
               "MeanAbsDeltaPH": float(np.mean(np.abs(dph))),
               "MedianAbsDeltaPH": float(np.median(np.abs(dph))),
               "MeanDirectedEffectPH_PlusMinus": float(np.mean(directed)),
               "SDDirectedEffectPH_PlusMinus": float(np.std(directed, ddof=1)) if len(directed) > 1 else np.nan,
               "DirectionalSignConsistency": stability,
               "MeanAbsDeltaPH_Elite": float(np.mean(np.abs(dph[elite]))) if elite.any() else np.nan,
               "MeanAbsDeltaPH_NonElite": float(np.mean(np.abs(dph[~elite]))) if (~elite).any() else np.nan,
               "MeanDirectedPH_Elite": float(np.mean(directed[elite])) if elite.any() else np.nan,
               "MeanDirectedPH_NonElite": float(np.mean(directed[~elite])) if (~elite).any() else np.nan}
    return rows, summary


def run_self_test():
    X = np.asarray([
        [-1, 1, 1, -1, 0.75, 1],
        [1, -1, 1, -1, .76, 1],
        [-1, -1, -1, 1, 0.80, -1],
        [1, 1, -1, 1, 0.78, -1],
        [-1, 1, -1, -1, 0.70, 1],
        [1, -1, 1, 1, 0.79, -1],
    ], dtype=np.float32)
    plus, minus = called_masks(X)
    assert plus.sum() + minus.sum() == 30
    p, m, np1, nm1, ncalled, fcalled, minor, eligible = compute_eligibility(
        X, X, 2, .2, .8)
    assert eligible.tolist() == [True, True, True, True, False, True]
    chrids = np.asarray(["1"] * 4 + ["2"] * 2, dtype=object)
    pos = np.asarray([10, 30, 50, 70, 5, 10])
    ld = local_adjacent_ld_proxy(X, p | m, chrids, pos, max_bp=30, min_pairs=4)
    assert len(ld) == X.shape[1]
    probe = select_probe_samples(np.asarray([1,2,3,4,5,6]), np.asarray([False,False,False,False,True,True]), 4,42)
    assert len(probe) == 4 and {4,5}.issubset(set(probe.tolist()))
    class SimpleModel(torch.nn.Module):
        def encode(self, xb):
            return xb[:, :1] * .3 + xb[:, 1:2] * .1, None
        def predictor(self, z):
            return z
    mock = SimpleModel()
    preds = predict_matrix(mock, X, torch.device("cpu"), 2, 110., 10.)
    assert np.allclose(preds, 110. + 3 * X[:,0] + X[:,1], atol=1e-4)
    e_mask = np.asarray([False,False,False,False,True,True])
    scores, _ = gradient_screen(mock, X, X, torch.device("cpu"), 2, 110.,10.,e_mask)
    assert np.allclose(scores[:2], [6.,2.], atol=1e-5)
    details, result = exact_single_snp_flip(mock, X, X, np.arange(6), 0, preds,
       torch.device("cpu"), 2, 110., 10., np.array(list("abcdef")),np.arange(6),e_mask)
    assert len(details)==6 and abs(result["MeanAbsDeltaPH"]-6) < 1e-4
    assert abs(result["MeanDirectedEffectPH_PlusMinus"]-6) < 1e-4
    controls, match = match_controls([0], eligible, minor, ld, chrids, pos, 1,42,0)
    assert len(controls)==1 and controls[0]!=0 and len(match)==1
    print("SELF-TEST PASSED: called masks, eligibility, LD proxy, samples, "
          "gradient screening, exact SNP flips, controls")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--genotype")
    ap.add_argument("--phenotype")
    ap.add_argument("--data-dir")
    ap.add_argument("--roundtrip-script")
    ap.add_argument("--output-dir")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--controls-per-candidate", type=int, default=1)
    ap.add_argument("--probe-samples", type=int, default=40)
    ap.add_argument("--all-samples", action="store_true")
    ap.add_argument("--min-state-count", type=int, default=10)
    ap.add_argument("--min-state-freq", type=float, default=0.05)
    ap.add_argument("--min-called-fraction", type=float, default=0.90)
    ap.add_argument("--elite-quantile", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expected-recon-weight", type=float, default=0.3,
                    help="Weight used by original manuscript model; default 0.3")
    ap.add_argument("--allow-ablation-checkpoint", action="store_true",
                    help="Explicitly allow recon=1.0 or another ablation checkpoint")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return
    if (args.top_k < 1 or args.batch_size < 1 or args.controls_per_candidate < 0 or
        args.probe_samples < 1 or args.min_state_count < 1 or not 0 < args.min_state_freq < .5 or
        not 0 < args.min_called_fraction <= 1 or not 0 < args.elite_quantile < 1):
        ap.error("Invalid top-k, batch, sampling, allele support or quantile arguments")
    t0 = time.perf_counter()
    here = Path(__file__).resolve().parent
    sourcepath = Path(args.roundtrip_script) if args.roundtrip_script else here / "GenoPerfVAE_Rice_PH_Target_RoundTrip_Validation.py"
    source = load_roundtrip(sourcepath)
    source.set_seed_fixed(args.seed)
    ckpt_path = source.resolve_checkpoint(here, args.checkpoint)
    geno_path, phe_path = source.resolve_data_files(
        here, explicit_data_dir=args.data_dir,
        explicit_genotype=args.genotype, explicit_phenotype=args.phenotype)
    outdir = (Path(args.output_dir) if args.output_dir else
              here / "results_Rice_PH_ObservedGenotype_SNP_Perturbation")
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ckpt = source.torch_load_checkpoint(ckpt_path, device)
    actual_recon_weight = ckpt.get("training_config", {}).get("recon_weight", None)
    original_model_match = (actual_recon_weight is not None and
                            abs(float(actual_recon_weight) - args.expected_recon_weight) <= 1e-8)
    if not original_model_match:
        message = ("Checkpoint reconstruction weight mismatch: checkpoint="
                   f"{actual_recon_weight}, manuscript model expected="
                   f"{args.expected_recon_weight}. This may be your recon=1.0 "
                   "ablation, not the originally reported recon=0.3 model.")
        if not args.allow_ablation_checkpoint:
            raise RuntimeError(message + " Restore the original checkpoint "
                "or explicitly pass --allow-ablation-checkpoint for a labeled ablation run.")
        warnings.warn(message + " Explicit ablation mode enabled.")
    model = source.build_model_from_checkpoint(ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    raw, X, y, entries, snps = source.load_reference_data(geno_path, phe_path, ckpt)
    raw = np.asarray(raw, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    entries = np.asarray(entries, dtype=str)
    if X.ndim != 2 or not np.isfinite(X).all() or len(X) != len(y):
        raise ValueError("Invalid checkpoint-aligned data")
    if (np.abs(raw - X) > TOL).any() and np.isnan(raw).sum() == 0:
        raise ValueError("Unexpected genotype differences without true NaN imputation")
    ymean, yscale = float(ckpt["scaler_y_mean"]), float(ckpt["scaler_y_scale"])
    elite_thr = float(np.quantile(y, args.elite_quantile))
    elite_mask = (y >= elite_thr)
    if len(X) != 299:
        warnings.warn(f"Aligned N={len(X)} instead of 299. Check dataset version")
    print(f"Rice299 PH | {device} | X={X.shape} | Elite={elite_mask.sum()}")
    print("Checkpoint:", ckpt_path)
    print("Source: REAL observed genotypes -> frozen encoder -> predictor")
    print("Decoder NOT used. Input fractional values unchanged; flips only on +/-1 called entries.")
    plus, minus, np1, nm1, ncalled, frac_called, minor, eligible = compute_eligibility(
        X, raw, args.min_state_count, args.min_state_freq, args.min_called_fraction)
    print(f"Called-state eligible SNPs: {eligible.sum()}/{len(snps)}")
    if not eligible.any():
        raise ValueError("No SNP passed called support filters")
    chrid, pos = parse_marker_ids(snps)
    called = plus | minus
    ldproxy = local_adjacent_ld_proxy(X, called, chrid, pos)
    print("Local adjacent-LD proxy available for:", np.isfinite(ldproxy).sum(), "SNPs")
    base_ph = predict_matrix(model, X, device, args.batch_size, ymean, yscale)
    screen, screen_elite = gradient_screen(model, X, raw, device, args.batch_size,
                                         ymean, yscale, elite_mask)
    allowed = np.flatnonzero(eligible & np.isfinite(screen))
    ranked = allowed[np.argsort(-screen[allowed], kind="stable")]
    top = ranked[:args.top_k]
    controls, match = match_controls(top, eligible, minor, ldproxy, chrid, pos,
                                     args.controls_per_candidate, args.seed)
    selected = np.asarray(list(dict.fromkeys(np.r_[top, controls].tolist())), dtype=np.int32)
    probe = np.arange(len(X), dtype=np.int32) if args.all_samples else select_probe_samples(
        y, elite_mask, args.probe_samples, args.seed)
    print(f"Shortlist: {len(top)} candidates, {len(controls)} controls; "
          f"exact evaluation on {len(probe)} observed genotypes")
    screen_rank = np.zeros(len(snps), dtype=np.int32)
    screen_rank[ranked] = np.arange(1, len(ranked)+1, dtype=np.int32)
    per_snp = pd.DataFrame({
        "SNPIndex0": np.arange(len(snps), dtype=np.int32), "SNP": snps,
        "Chromosome": chrid, "Position": pos,
        "CalledPlusOne_N": np1, "CalledMinusOne_N": nm1, "CalledN": ncalled,
        "CalledFraction": frac_called, "CalledMinorStateFrequency": minor,
        "Eligible": eligible, "AdjacentLD_R2_Proxy": ldproxy,
        "GradientProxy_MeanAbs2dPHdx": screen,
        "GradientProxy_Elite_MeanAbs2dPHdx": screen_elite,
        "EligibleGradientScreenRank": screen_rank,
        "IsTopCandidate": np.isin(np.arange(len(snps)), top),
        "IsMatchedControl": np.isin(np.arange(len(snps)), controls),
    })
    prefix = str(outdir / PREFIX)
    per_snp.to_csv(prefix + "_AllSNPGradientScreen.csv", index=False)
    if len(match):
        match["CandidateSNP"] = match["CandidateIndex"].map(lambda j: str(snps[int(j)]))
        match["ControlSNP"] = match["ControlIndex"].map(
            lambda j: str(snps[int(j)]) if int(j)>=0 else "")
    match.to_csv(prefix + "_Matching.csv", index=False)
    detailed = []
    evaluated = []
    control_set = set(map(int, controls))
    for k, marker in enumerate(selected, start=1):
        details, metrics = exact_single_snp_flip(
            model, X, raw, probe, int(marker), base_ph, device, args.batch_size,
            ymean, yscale, entries, y, elite_mask)
        for row in details:
            row["SNP"] = str(snps[marker])
            row["Role"] = "TopGradientCandidate" if marker in top else "MatchedControl"
        detailed.extend(details)
        evaluated.append({"SNP": str(snps[marker]), "SNPIndex0": int(marker),
                          "Role": "TopGradientCandidate" if marker in top else "MatchedControl",
                          "Chromosome": str(chrid[marker]), "Position": int(pos[marker]),
                          "CalledMinorStateFrequency": float(minor[marker]),
                          "CalledMinusOne_N": int(nm1[marker]),
                          "CalledPlusOne_N": int(np1[marker]),
                          "AdjacentLD_R2_Proxy": float(ldproxy[marker]),
                          "GradientProxy_MeanAbs2dPHdx": float(screen[marker]),
                          "GradientScreenRank": int(screen_rank[marker]),
                          **metrics})
        if k % 8 == 0 or k == len(selected):
            print(f"  Exact finite differences: {k}/{len(selected)} SNPs", flush=True)
    exact = pd.DataFrame(evaluated)
    if len(exact):
        exact["ExactAbsEffectRank_AmongCandidates"] = np.nan
        c = exact.Role == "TopGradientCandidate"
        exact.loc[c, "ExactAbsEffectRank_AmongCandidates"] = (
            exact.loc[c, "MeanAbsDeltaPH"].rank(ascending=False, method="min"))
        exact = exact.sort_values(["Role", "MeanAbsDeltaPH"], ascending=[False, False])
    pd.DataFrame(detailed).to_csv(prefix + "_PerSampleEffects.csv", index=False)
    exact.to_csv(prefix + "_ExactPerturbations.csv", index=False)
    candidates = exact[exact.Role == "TopGradientCandidate"].copy()
    candidates.to_csv(prefix + "_CandidatesForGWAS.csv", index=False)
    pd.DataFrame({"Entry": entries, "ObservedPH": y, "ModelPH_InSample": base_ph,
                  "EliteQ95": elite_mask, "InExactProbeSubset": np.isin(np.arange(len(y)), probe)}).to_csv(
                      prefix + "_ReferenceSamples.csv", index=False)
    summary = {
        "Dataset": "Rice299", "Trait": "PH",
        "AnalysisRole": "in-sample frozen-model sensitivity (NOT causal / external validation)",
        "Samples": int(len(X)), "SNPs": int(X.shape[1]),
        "EliteN": int(elite_mask.sum()), "EliteQ95": elite_thr,
        "EligibleSNPs": int(eligible.sum()), "EligibleFrequencyThreshold": args.min_state_freq,
        "MinPerStateCount": args.min_state_count,
        "TopCandidates": int(len(top)), "MatchedControls": int(len(controls)),
        "ExactProbeSamples": int(len(probe)),
        "ModelCheckpointReconWeight": actual_recon_weight,
        "OriginalReconWeightMatches": bool(original_model_match),
        "MeanAbsPredictionDifference_TopCandidates": float(candidates["MeanAbsDeltaPH"].mean()) if len(candidates) else np.nan,
        "MeanAbsPredictionDifference_Controls": float(exact.loc[exact.Role == "MatchedControl", "MeanAbsDeltaPH"].mean()) if len(controls) else np.nan,
        "MatchingGoodApproxCount": int(np.sum(match.get("MatchingStatus", pd.Series([], dtype=str)) == "GOOD_APPROX_MATCH")),
        "Seconds": time.perf_counter() - t0,
        "GWASPerformed": False,
        "Warning": "No decoder; no independence; in-sample, post-selection, allele coding numeric, single-SNP counterfactual may violate haplotypes",
    }
    pd.DataFrame([summary]).to_csv(prefix + "_Summary.csv", index=False)
    with open(prefix + "_Config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "paths": {"checkpoint": str(ckpt_path),
                   "genotype": str(geno_path), "phenotype": str(phe_path),
                   "roundtrip_script": str(sourcepath)}, "summary": summary,
                   "method": {"screen": "mean |2*d(PH)/d(Xj)| on observed called entries; only selection",
                              "exact": "one-SNP flip +/-1 on observed called genotype; encoder -> predictor",
                              "controls": "frequency + adjacent SNP LD proxy, approximate, diagnostics required",
                              "phenotype": "inverse standardized PH via checkpoint scaler",
                              "selection": "in-sample and post-selection; no causal or biological validation"}},
                  f, indent=2, ensure_ascii=False)
    print("\nRESULTS (IN-SAMPLE FROZEN-MODEL PERTURBATION)")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("Saved to:", outdir)
    print("Do NOT report these as independent GWAS, QTL, causal or breedable SNP effects.")


if __name__ == "__main__":
    main()
