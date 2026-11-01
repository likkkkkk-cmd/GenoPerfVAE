#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare LGCG Rice299-PH candidate/background gene lists for agriGO Full GO.

Source is the *LGCG Top-N GO Slim analysis* gene export, not a legacy PerfVAE
Top32/67-gene list. No neural model or GO statistics are executed here.
The candidate and background lists are frozen, matched exactly to the prior
GO Slim run and suitable as identical query/reference inputs for full GO and
Plant GO Slim analyses. Different gene counts at Top32/150/200 are expected.

The source analysis uses MSU6 5-kb SNP-to-gene mapping for both groups and
stable LOC_Os IDs. MSU6/MSU7 nearby-gene discrepancies are recorded, NOT
silently assumed absent. Enrichment is NOT computed by this script.

Offline usage:
  python GenoPerfVAE_LGCG_Rice_PH_FullGO_Prepare_GeneLists.py --top-k 150
  python GenoPerfVAE_LGCG_Rice_PH_FullGO_Prepare_GeneLists.py --top-k 32 \
      --root YOUR_PROJECT_DIRECTORY
  python GenoPerfVAE_LGCG_Rice_PH_FullGO_Prepare_GeneLists.py \
      --go-dir PATH_TO_03_GOSlim_Enrichment --top-k 150
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent
GO_PREFIX = "GenoPerfVAE_LGCG_Rice_PH_TopN_GO_Enrichment"
OUT_PREFIX = "GenoPerfVAE_LGCG_Rice_PH_FullGO"
GENE_ID = re.compile(r"LOC_Os(?:0[1-9]|1[0-2])g\d{5}$")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_genes(path):
    if not path.is_file():
        raise FileNotFoundError(f"LGCG GO gene export missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames or not {"GeneID", "InCandidate"} <= set(rdr.fieldnames):
            raise ValueError(f"Required GeneID and InCandidate columns missing: {path}")
        records = list(rdr)
    if not records:
        raise ValueError(f"Empty gene set: {path}")
    selected = set()
    flags = {}
    for rec in records:
        g = str(rec.get("GeneID", "")).strip()
        if not GENE_ID.fullmatch(g):
            raise ValueError(f"Invalid MSU LOC_Os gene ID {g!r} in {path}")
        if g in flags:
            raise ValueError(f"Duplicated gene ID {g!r} in {path}")
        flag = str(rec.get("InCandidate", "")).strip().lower()
        if flag not in {"true", "false", "1", "0"}:
            raise ValueError(f"Invalid InCandidate value {flag!r} for {g}")
        flags[g] = flag in {"true", "1"}
        selected.add(g)
    return selected, flags


def run(root, top_k=150, go_dir=None, cand_input=None, bg_input=None,
        out_dir=None, expected_eligible_snps=38796):
    root = Path(root).expanduser().resolve()
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")
    source_dir = (Path(go_dir).expanduser().resolve() if go_dir else
        root / f"results_Rice_PH_Top{top_k}_LGCG_FinalExpansion" / "03_GOSlim_Enrichment")
    candidate_file = (Path(cand_input) if cand_input else
        source_dir / f"{GO_PREFIX}_CandidateGenes.csv")
    background_file = (Path(bg_input) if bg_input else
        source_dir / f"{GO_PREFIX}_AllSNP_Mappable_BackgroundGenes.csv")
    summary_file = source_dir / f"{GO_PREFIX}_Summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError(
            "Cannot verify LGCG GO input provenance without a matching GO Slim "
            f"summary: {summary_file}. Run the LGCG Top-N GO stage first."
        )
    with summary_file.open("r", encoding="utf-8-sig") as fh:
        previous = json.load(fh)
    label = str(previous.get("Model", ""))
    if "LGCG" not in label.upper():
        raise RuntimeError(f"Wrong source model in GO summary: {label!r}; requires LGCG")
    if str(previous.get("Phenotype", "")).upper() != "PH":
        raise RuntimeError("GO summary is not for Rice299-PH")
    if int(previous.get("CandidateSNPs", -1)) != top_k:
        raise RuntimeError(
            f"LGCG GO Top-N mismatch: source={previous.get('CandidateSNPs')} "
            f"but --top-k={top_k}"
        )
    count_eligible = int(previous.get("EligibleGWASSNPs", -1))
    if expected_eligible_snps > 0 and count_eligible != expected_eligible_snps:
        raise RuntimeError(
            f"Eligible SNP mismatch: source={count_eligible} "
            f"expected={expected_eligible_snps}"
        )
    gene_method = str(previous.get("GeneMappingSystemForStatisticalTest", ""))
    if "MSU6" not in gene_method:
        raise RuntimeError(
            "GO summary lacks documented common MSU6 SNP-to-gene mapping; "
            "candidate/background gene lists may be incomparable."
        )
    candidates, cflags = read_genes(candidate_file)
    background, bflags = read_genes(background_file)
    if not candidates.issubset(background):
        raise RuntimeError(
            f"Candidate genes absent from background: {sorted(candidates-background)[:12]}"
        )
    if not all(cflags.values()):
        raise RuntimeError("Candidate file contains InCandidate=False genes")
    in_bg = {gene for gene, is_candidate in bflags.items() if is_candidate}
    if in_bg != candidates:
        raise RuntimeError(
            "Source gene-set membership differs between candidate and "
            f"background tables: extra={sorted(in_bg-candidates)[:8]}, "
            f"missing={sorted(candidates-in_bg)[:8]}"
        )
    n_prev_c = int(previous.get("CandidateUniqueGenes", -1))
    n_prev_b = int(previous.get("BackgroundSNP_MappableGenes", -1))
    if len(candidates) != n_prev_c or len(background) != n_prev_b:
        raise RuntimeError(
            f"GO source gene counts differ from exports: candidate "
            f"{len(candidates)}/{n_prev_c}, background {len(background)}/{n_prev_b}"
        )
    output = (Path(out_dir).expanduser().resolve() if out_dir else
        root / f"results_Rice_PH_Top{top_k}_LGCG_FullGO_Prepare")
    output.mkdir(parents=True, exist_ok=True)
    candidate_txt = output / f"{OUT_PREFIX}_Top{top_k}_Candidates{len(candidates)}_LOC.txt"
    background_txt = output / f"{OUT_PREFIX}_Top{top_k}_Background{len(background)}_LOC.txt"
    candidate_txt.write_text("\n".join(sorted(candidates)) + "\n", encoding="utf-8")
    background_txt.write_text("\n".join(sorted(background)) + "\n", encoding="utf-8")
    differences = int(previous.get("CandidateSNPsWithMSU6MSU7GeneDifferences", -1))
    equivalent = previous.get("CandidateSourceAndTargetGeneSetsMatchExactly")
    audit = {
        "status": "PASS", "model": label, "trait": "Rice299 PH",
        "task": "Frozen LGCG GO gene-list preparation; no enrichment test",
        "top_k_snp_candidates": top_k,
        "eligible_snp_count": count_eligible,
        "candidate_gene_count": len(candidates),
        "background_gene_count": len(background),
        "candidate_within_background": True,
        "background_candidate_membership_exact": True,
        "input_gene_mapping": gene_method,
        "source_MSU6_MSU7_gene_sets_match_exactly": equivalent,
        "candidate_snps_with_cross_assembly_gene_differences": differences,
        "cross_assembly_note": (
            "The GO analysis uses a shared MSU6 5-kb gene mapping; MSU7 "
            "discrepancies are reported and do not invalidate the shared "
            "MSU6-based candidate/background universe."
        ),
        "candidate_source_file": str(candidate_file),
        "background_source_file": str(background_file),
        "go_summary_file": str(summary_file),
        "go_summary_sha256": sha256(summary_file),
        "candidate_source_sha256": sha256(candidate_file),
        "background_source_sha256": sha256(background_file),
        "candidate_list_file": str(candidate_txt),
        "background_list_file": str(background_txt),
        "candidate_list_sha256": sha256(candidate_txt),
        "background_list_sha256": sha256(background_txt),
        "usage": (
            "Use identical candidate and custom reference gene lists for "
            "Full GO and Plant GO Slim in agriGO. Do not filter genes by "
            "GO annotation or preselect significant terms."
        ),
    }
    audit_file = output / f"{OUT_PREFIX}_Top{top_k}_GeneSet_Audit.json"
    audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS | LGCG Top{top_k} | candidate genes={len(candidates)} | "
          f"background genes={len(background)}")
    print(f"MSU6/MSU7 nearby-gene discrepancies in {differences} candidate SNP(s)")
    print("Candidate gene list:", candidate_txt)
    print("Background gene list:", background_txt)
    print("Provenance audit:", audit_file)
    return audit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--go-dir", type=Path, default=None)
    p.add_argument("--candidates", type=Path, default=None)
    p.add_argument("--background", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--expected-eligible-snps", type=int, default=38796,
                   help="0 disables fixed eligible-SNP count check")
    args = p.parse_args(argv)
    return run(args.root, args.top_k, args.go_dir, args.candidates,
               args.background, args.out_dir, args.expected_eligible_snps)


if __name__ == "__main__":
    main()
