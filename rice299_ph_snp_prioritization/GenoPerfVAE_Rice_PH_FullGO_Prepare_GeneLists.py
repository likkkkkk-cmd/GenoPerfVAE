#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare STRICTLY frozen 67-candidate/24,005-background gene lists for full GO.

Use these exact same lists in BOTH Full GO and Plant GO Slim analyses in agriGO v2.
No neural model, SNP ranking, or data filtering is performed here.

From E:\\VGMP\\Extended Comparison\\GenoPerfVAE:
  python GenoPerfVAE_Rice_PH_FullGO_Prepare_GeneLists.py

Optionally:
  python GenoPerfVAE_Rice_PH_FullGO_Prepare_GeneLists.py --root "D:\\path\\GenoPerfVAE"
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

DEFAULT_ROOT = Path(r"E:\VGMP\Extended Comparison\GenoPerfVAE")
PREVIOUS = "GenoPerfVAE_Rice_PH_GO_Enrichment"
ID_PATTERN = re.compile(r"LOC_Os(?:0[1-9]|1[0-2])g\d{5}$")


def read_genes(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required gene table not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "GeneID" not in reader.fieldnames:
            raise ValueError(f"Expected a GeneID column in {path}")
        rows = list(reader)
    genes = []
    flags = {}
    for r in rows:
        gid = str(r.get("GeneID", "")).strip()
        if not ID_PATTERN.fullmatch(gid):
            raise ValueError(f"Invalid / non-MSU LOC_Os gene ID: {gid!r} in {path}")
        if gid in flags:
            raise ValueError(f"Duplicate GeneID {gid!r} in {path}")
        flags[gid] = str(r.get("InCandidate", "")).strip().lower()
        genes.append(gid)
    return set(genes), flags


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def run(root, cand_input=None, bg_input=None, out_dir=None):
    old = root / "results_Rice_PH_GO_Enrichment"
    cand = cand_input or old / f"{PREVIOUS}_Corrected67_CandidateGenes.csv"
    bg = bg_input or old / f"{PREVIOUS}_AllSNP_Mappable_BackgroundGenes.csv"
    out = out_dir or root / "results_Rice_PH_FullGO_Prepare"
    candidates, c_flags = read_genes(cand)
    background, b_flags = read_genes(bg)
    if len(candidates) != 67:
        raise RuntimeError(f"Stop: expected 67 candidates, found {len(candidates)}")
    if len(background) != 24005:
        raise RuntimeError(f"Stop: expected 24,005 eligible-SNP-mappable background genes, found {len(background)}")
    if not candidates.issubset(background):
        raise RuntimeError(f"Stop: candidates missing from background: {sorted(candidates-background)}")
    if any(v != 'true' for v in c_flags.values()):
        raise RuntimeError("Stop: candidate CSV has GeneID rows not marked InCandidate=True")
    bg_flagged = {g for g, v in b_flags.items() if v == 'true'}
    if bg_flagged != candidates:
        raise RuntimeError(f"Stop: background candidate membership differs; extra={sorted(bg_flagged-candidates)} missing={sorted(candidates-bg_flagged)}")
    previous_summary = old / f"{PREVIOUS}_Summary.json"
    if previous_summary.exists():
        with previous_summary.open("r", encoding="utf-8-sig") as f:
            prior = json.load(f)
        for field, expected in [('CandidateUniqueGenes', 67), ('BackgroundSNP_MappableGenes', 24005), ('CandidateSNPs', 32), ('EligibleGWASSNPs', 38796)]:
            if prior.get(field) != expected:
                raise RuntimeError(f"Stop: prior GO Slim summary {field}={prior.get(field)}; expected {expected}")
        if prior.get('CandidateSourceAndTargetGeneSetsMatchExactly') is not True:
            raise RuntimeError('Stop: MSU6/MSU7 candidate gene-set equivalence was not documented')
    else:
        raise FileNotFoundError(f"Previous GO Slim summary not found, cannot verify provenance: {previous_summary}")
    out.mkdir(parents=True, exist_ok=True)
    candidate_txt = out / 'GenoPerfVAE_Rice_PH_FullGO_Candidate67_LOC.txt'
    background_txt = out / 'GenoPerfVAE_Rice_PH_FullGO_Background24005_LOC.txt'
    candidate_txt.write_text('\n'.join(sorted(candidates)) + '\n', encoding='utf-8')
    background_txt.write_text('\n'.join(sorted(background)) + '\n', encoding='utf-8')
    audit = {
        'status': 'PASS', 'purpose': 'Generate unchanged GO-enrichment gene sets; NOT GO statistics',
        'candidate_gene_count': len(candidates), 'background_gene_count': len(background),
        'candidate_within_background': True,
        'candidate_snp_count_source': 32, 'eligible_snp_count_source': 38796,
        'reference': 'MSU6 position-matched SNP->gene mapping, stable MSU7 LOC_Os IDs',
        'gene_window_bp': 5000, 'no_GO_annotation_preselection': True,
        'candidate_source_file': str(cand), 'background_source_file': str(bg),
        'candidate_file': str(candidate_txt), 'background_file': str(background_txt),
        'candidate_sha256': sha256(candidate_txt), 'background_sha256': sha256(background_txt),
        'instructions': 'Use both exact gene lists (Full GO and Plant GO slim) as query and custom reference in agriGO v2. Do not select significant subset before BH correction.'
    }
    audit_path = out / 'GenoPerfVAE_Rice_PH_FullGO_GeneSet_Audit.json'
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: Candidate genes:',len(candidates),' Background genes:',len(background))
    print('Candidate:', candidate_txt)
    print('Background:', background_txt)
    print('Audit:', audit_path)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description='Prepare and validate frozen 67/24005 MSU LOC_Os gene sets')
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--candidates',type=Path,default=None)
    ap.add_argument('--background',type=Path,default=None)
    ap.add_argument('--out-dir',type=Path,default=None)
    args=ap.parse_args()
    run(args.root,args.candidates,args.background,args.out_dir)
