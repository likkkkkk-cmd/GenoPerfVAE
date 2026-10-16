#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rice299-PH: Top-N GO Slim overrepresentation analysis.

Design: 38,796 QC-eligible SNPs define an MSU6 5-kb SNP-mappable gene
background. Top-N candidates are selected by frozen-model input gradients,
then evaluated by exact +/-1 flips on the 299 observed genotype backgrounds.
Top-N SNPs must undergo genome-wide unique MSU6 -> MSU7 coordinate audit.
The statistical test maps BOTH candidates and background by the same MSU6
GFF3 and joins GO Slim by stable LOC_Os IDs; MSU7 release-dependent gene-set
differences are reported in a separate file rather than silently ignored.

Test Fisher/hypergeometric and BH-FDR with matched-SNP and chromosome-shift
empirical nulls (neither is a strict LD-matched test). No causal or external
validation claims. This is Plant GO Slim, not comprehensive GO.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(r"E:\VGMP\Extended Comparison\PerVAE")
PREFIX = "GenoPerfVAE_Rice_PH_TopN_GO_Enrichment"
GOSLIM_URL = "https://rice.uga.edu/osa1r7_download/osa1_r7.all_models.GOSlim.txt.gz"
OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
LOCRE = re.compile(r"LOC_Os\d{2}g\d+", re.IGNORECASE)
GORE = re.compile(r"GO:\d{7}", re.IGNORECASE)
SNPRE = re.compile(r"^S(\d{1,2})_(\d+)$", re.IGNORECASE)


def clean(s):
    return str(s or "").strip().lstrip("\ufeff")


def read_text(path):
    path = Path(path)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("r", encoding="utf-8-sig", errors="replace")


def read_rows(path):
    with read_text(path) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        return list(reader)


def csv_save(path, rows, header):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_true(v):
    return clean(v).lower() in {"true", "yes", "1"}


def parse_chrom(v):
    raw = clean(v).lower().split("|")[0].split()[0]
    raw = re.sub(r"^(?:chromosome|chr)[_\-]?", "", raw)
    if raw.isdigit() and 1 <= int(raw) <= 12:
        return int(raw)
    return None


def read_msu6_aliases(fasta):
    """Read header lines from actual compressed MSU6 chromosome FASTA.

    Avoid loading reference sequence into RAM; do not blindly infer 13101 etc.
    """
    aliases = {}
    found = set()
    with read_text(fasta) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            token = line[1:].strip().split()[0]
            chr_ = parse_chrom(token)
            if chr_ is None:
                continue
            found.add(chr_)
            parts = [str(chr_), f"Chr{chr_}", f"Chr{chr_:02d}", f"{chr_:02d}"] + token.split("|")
            for part in parts:
                k = part.strip().lower()
                if k in aliases and aliases[k] != chr_:
                    raise ValueError(f"Conflicting FASTA seqid alias: {part}")
                aliases[k] = chr_
    if found != set(range(1, 13)):
        raise ValueError(f"Expected 12 MSU6 chromosome FASTA entries, got {sorted(found)}")
    return aliases


@dataclass(frozen=True)
class Gene:
    gid: str
    chrom: int
    start: int
    end: int
    strand: str


def load_msu6_genes(path, aliases):
    """Gene entries only, same MSU6 locus logic as coordinate-audit program."""
    genes = {}
    gene_rows = 0
    with read_text(path) as f:
        for line in f:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\r\n").split("\t")
            if len(p) != 9 or p[2].lower() not in {"gene", "pseudogene", "transposable_element_gene", "ncrna_gene", "transcription_unit", "transposable_element"}:
                continue
            ch = parse_chrom(p[0]) or aliases.get(p[0].strip().lower())
            if ch is None:
                continue
            match = LOCRE.search(p[8])
            if not match:
                continue
            try:
                a, b = int(p[3]), int(p[4])
            except ValueError:
                continue
            if a < 1 or b < a:
                continue
            gene_rows += 1
            gid = match.group(0)
            key = (ch, gid)
            if key in genes:
                old = genes[key]
                genes[key] = Gene(gid, ch, min(old.start, a), max(old.end, b), old.strand)
            else:
                genes[key] = Gene(gid, ch, a, b, p[6])
    if len(genes) < 50000:
        raise ValueError(f"MSU6 GFF parsed only {len(genes)} loci, expected about 57,624: {path}")
    by_ch = defaultdict(list)
    for g in genes.values():
        by_ch[g.chrom].append(g)
    indexes = {}
    for ch in range(1, 13):
        arr = sorted(by_ch[ch], key=lambda g: (g.start, g.end, g.gid))
        starts, prefix_max, mm = [], [], 0
        for g in arr:
            starts.append(g.start)
            mm = max(mm, g.end)
            prefix_max.append(mm)
        indexes[ch] = (arr, starts, prefix_max)
    return indexes, len(genes), gene_rows


def near_genes(ch, pos, indexes, window):
    """Return ALL gene bodies or genes within window of SNP coordinate."""
    if ch not in indexes or pos <= 0:
        return set()
    arr, starts, prefix_end = indexes[ch]
    hi = bisect.bisect_right(starts, pos + window)
    lo = bisect.bisect_left(prefix_end, pos - window, 0, hi)
    return {g.gid for g in arr[lo:hi] if g.end >= pos - window}


def load_gwas_eligible(path):
    rows = read_rows(path)
    needed = {"SNP", "GWAS_Eligible", "Chromosome", "Position", "Minor_State_Frequency", "Call_Rate"}
    if not needed.issubset(rows[0]):
        raise ValueError(f"AllSNP GWAS missing columns: {sorted(needed - set(rows[0]))}")
    eligible = {}
    for r in rows:
        if not parse_true(r["GWAS_Eligible"]):
            continue
        name = clean(r["SNP"])
        m = SNPRE.fullmatch(name)
        if not m:
            raise ValueError(f"Eligible SNP has unknown naming scheme: {name}")
        ch, pos = int(m.group(1)), int(m.group(2))
        if ch != int(r["Chromosome"]) or pos != int(r["Position"]):
            raise ValueError(f"GWAS coordinate and SNP identifier disagree: {name}")
        if name in eligible:
            raise ValueError(f"Duplicate GWAS SNP: {name}")
        maf = float(r["Minor_State_Frequency"])
        call = float(r["Call_Rate"])
        if not (0.0 <= maf <= 0.5 and 0.0 <= call <= 1.0):
            raise ValueError(f"Invalid QC stats for {name}")
        eligible[name] = (ch, pos, maf, call)
    if not eligible:
        raise ValueError("AllSNP GWAS contained no GWAS_Eligible SNPs")
    return eligible, len(rows)


def load_audited_candidates(path):
    rows = read_rows(path)
    candidates = {}
    for r in rows:
        if clean(r.get("RoleGroup", "Candidate")) != "Candidate":
            continue
        name = clean(r.get("SNP"))
        if name in candidates:
            raise ValueError(f"Duplicate audited candidate: {name}")
        if not (clean(r.get("MappingStatus")) in {"UNIQUE_EXACT_FLANK", "UNIQUE_TWO_ANCHOR_ALIGNMENT"} and
                parse_true(r.get("GenomewideUniqueMappingAndSourceAlleleConfirmed"))):
            raise ValueError(f"Candidate {name}: MSU7 mapping NOT genome-wide unique and allele-confirmed")
        a = set(g.strip() for g in clean(r.get("MSU6_NearGeneIDs")).split(";") if g.strip())
        b = set(g.strip() for g in clean(r.get("MSU7_Remapped_NearGeneIDs")).split(";") if g.strip())
        # All tested SNPs and the gene-universe must use ONE common coordinate
        # system: original HapMap MSU6. MSU7 near-gene IDs may differ by release;
        # report differences separately rather than silently replacing the mapping.
        candidates[name] = (int(r["MSU6_Chr"]), int(r["MSU6_Position"]), a, r)
    if not (1 <= len(candidates) <= 1000):
        raise ValueError(f"Unexpected audited candidate SNP count: {len(candidates)}")
    return candidates


def load_goslim(path):
    """Official RGAP7 file has transcript_id TAB GO_ID; collapse isoforms to locus.

    A GO term can appear on multiple transcripts. A gene contributes once to
    each term. Never treat transcripts as separate biological genes.
    """
    go_by_gene = defaultdict(set)
    terms_with_names = {}
    row_n = 0
    with read_text(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            match = LOCRE.search(parts[0])
            if not match:
                continue
            ids = set(GORE.findall(parts[1]))
            if not ids:
                continue
            gid = match.group(0)
            row_n += 1
            for go in ids:
                go_by_gene[gid].add(go)
                if len(parts) >= 3 and not GORE.search(parts[2]):
                    s = clean(parts[2])
                    if s and len(s) < 100 and s.lower() not in {"p", "c", "f", "bp", "mf", "cc"}:
                        terms_with_names.setdefault(go, s)
    if len(go_by_gene) < 1000 or len({t for v in go_by_gene.values() for t in v}) < 10:
        raise ValueError(f"GO Slim file parsed poorly: genes={len(go_by_gene)}; check file: {path}")
    return go_by_gene, terms_with_names, row_n


def load_go_obo(path):
    if not path or not Path(path).is_file():
        return {}
    result = {}
    term = None
    def finish(record):
        if record and record.get("id", "").startswith("GO:") and "name" in record and not record.get("is_obsolete", False):
            result[record["id"]] = (record["name"], record.get("namespace", ""))
    with read_text(path) as f:
        for line in f:
            s = line.strip()
            if s == "[Term]":
                finish(term)
                term = {}
            elif s.startswith("[") and s != "[Term]":
                finish(term)
                term = None
            elif term is not None and ": " in s:
                k, v = s.split(": ", 1)
                if k in {"id", "name", "namespace"}:
                    term[k] = v
                elif k == "is_obsolete" and v.lower() == "true":
                    term[k] = True
    finish(term)
    return result


def hypergeom_sf(x, universe, term_size, draws):
    """One-sided P[X >= x] via log-gamma / recurrence; no scipy needed."""
    if x <= 0:
        return 1.0
    hi = min(term_size, draws)
    lo = max(0, draws + term_size - universe)
    if x > hi:
        return 0.0
    x = max(x, lo)
    def logchoose(n, k):
        if k < 0 or k > n:
            return -math.inf
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    logs = [logchoose(term_size, k) + logchoose(universe - term_size, draws - k) - logchoose(universe, draws)
            for k in range(x, hi+1)]
    mx = max(logs)
    return min(1.0, math.exp(mx) * sum(math.exp(v - mx) for v in logs))


def bh_adjust(pvals):
    n = len(pvals)
    out = [1.0] * n
    last = 1.0
    for rank, i in reversed(list(enumerate(sorted(range(n), key=lambda i: pvals[i]), 1))):
        last = min(last, pvals[i] * n / rank)
        out[i] = min(1.0, last)
    return out


def go_counts(genes, gene_go):
    c = Counter()
    for g in genes:
        c.update(gene_go.get(g, ()))
    return c


def prepare_match_pools(candidates, eligible, by_chr, maf_tol=0.05, call_tol=0.05):
    """Precompute matching strata exactly once, not 2000 times."""
    candidate_set = set(candidates)
    pools, diagnostics = {}, []
    for name in candidates:
        ch, _, maf, call = eligible[name]
        for delta in (1, 2, 4):
            pool = [p[0] for p in by_chr[ch] if p[0] not in candidate_set and
                    abs(p[2] - maf) <= maf_tol * delta and
                    abs(p[3] - call) <= call_tol * delta]
            if len(pool) >= 32:
                break
        if not pool:
            raise RuntimeError(f"No matched eligible SNPs for {name}")
        pools[name] = pool
        diagnostics.append({"CandidateSNP":name,"PoolSize":len(pool),"ToleranceExpansion":delta})
    return pools, diagnostics


def sample_matched_set(candidates, pools, rnd):
    """Draw 32 unique markers from previously defined matched pools."""
    picks, used = [], set()
    for name in candidates:
        pool = pools[name]
        for _ in range(60):
            snp = rnd.choice(pool)
            if snp not in used:
                break
        else:
            available = [s for s in pool if s not in used]
            if not available:
                raise RuntimeError(f"Exhausted matched SNP pool for {name}")
            snp = rnd.choice(available)
        picks.append(snp)
        used.add(snp)
    return picks


def positional_shift_set(candidates, eligible, by_chr, chrom_lengths, rnd):
    """Shift ALL candidate positions on a chromosome by one circular offset.

    Choose closest GWAS-eligible marker to each shifted position; constrained
    to be unique within a trial. Preserves genomic spacing approximately, not
    exact LD r^2, not MAF/call-rate. Record relocation discrepancy.
    """
    grouped = defaultdict(list)
    for name in candidates:
        ch, pos, _, _ = eligible[name]
        grouped[ch].append((name, pos))
    output, distance_errors, used = [], [], set(candidates)
    for ch, group in grouped.items():
        arr = by_chr[ch]
        poses = [v[1] for v in arr]
        clen = chrom_lengths[ch]
        shift = rnd.randrange(1, clen)
        for _, pos in group:
            target = (pos - 1 + shift) % clen + 1
            ix = bisect.bisect_left(poses, target)
            options = []
            for step in range(25):
                for k in ((ix + step) % len(arr), (ix - step - 1) % len(arr)):
                    if arr[k][0] not in used:
                        # circular distance
                        d = abs(poses[k] - target)
                        options.append((min(d, clen - d), arr[k][0]))
                if options:
                    break
            if not options:
                raise RuntimeError("Cannot draw distinct markers in positional shift")
            d, snp = min(options)
            output.append(snp)
            distance_errors.append(d)
            used.add(snp)
    return output, distance_errors


def write_png(path, records):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[NOTE] matplotlib missing: skip PNG (all CSV/JSON produced).")
        return
    any_all_significant = any(float(r["Fisher_FDR_BH"]) < .05 and
                              float(r["MatchedSNP_Empirical_FDR_BH"]) < .05 and
                              float(r["Shift_Empirical_FDR_BH"]) < .05 for r in records)
    records = sorted(records, key=lambda r: (float(r["Fisher_FDR_BH"]), -int(r["CandidateGeneCount"])))[:15]
    if not records:
        return
    yy = list(range(len(records)))
    xs = [min(10, float(r["FoldEnrichment"])) for r in records]
    counts = [int(r["CandidateGeneCount"]) for r in records]
    q = [float(r["Fisher_FDR_BH"]) for r in records]
    colors = [-math.log10(max(1e-12, v)) for v in q]
    names = [str(r["GO_Name"])[:44] + ("…" if len(str(r["GO_Name"])) > 44 else "") for r in records]
    fig, ax = plt.subplots(figsize=(9.3, max(4.7, 0.38*len(records)+1.9)))
    scat = ax.scatter(xs, yy, s=[35+36*n for n in counts], c=colors, cmap="viridis", vmin=0, vmax=max(1.301, max(colors)), edgecolor="black", linewidth=0.35)
    ax.set_yticks(yy, [f"{n} ({r['GO_ID']})" for n, r in zip(names, records)])
    ax.invert_yaxis()
    ax.set_xlabel("Fold enrichment versus SNP-mappable GO-annotated background (max displayed 10)")
    ax.set_title("Rice299-PH | exploratory MSU7 GO Slim enrichment")
    bar = fig.colorbar(scat, ax=ax, shrink=.85, pad=.04)
    bar.set_label("-log10(Fisher BH-FDR)")
    if not any_all_significant:
        ax.text(.98, .98, "No term passes ALL three FDR<0.05 checks", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, bbox={"facecolor":"white", "alpha":.9, "edgecolor":"#999"})
    fig.tight_layout()
    fig.savefig(path, dpi=230)
    plt.close(fig)


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (scientific annotation download)"})
    tmp = Path(str(path) + ".partial")
    with urllib.request.urlopen(req, timeout=90) as src, tmp.open("wb") as dst:
        while True:
            block = src.read(1024*1024)
            if not block: break
            dst.write(block)
    if tmp.stat().st_size < 1000:
        raise ValueError("Downloaded file unexpectedly small")
    tmp.replace(path)


def run(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--all-snp", type=Path, default=None, help="GWAS PC10 full genomewide AllSNP_GWAS.csv")
    p.add_argument("--candidate-audit", type=Path, default=None)
    p.add_argument("--msu6-gff", type=Path, default=None)
    p.add_argument("--msu6-fasta", type=Path, default=None)
    p.add_argument("--goslim", type=Path, default=None)
    p.add_argument("--obo", type=Path, default=None)
    p.add_argument("--download-goslim", action="store_true")
    p.add_argument("--download-obo", action="store_true")
    p.add_argument("--window", type=int, default=5000)
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--maf-tol", type=float, default=.05)
    p.add_argument("--call-tol", type=float, default=.05)
    p.add_argument("--min-term-genes", type=int, default=5)
    p.add_argument("--max-term-fraction", type=float, default=.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.window != 5000:
        raise ValueError("This expansion requires the original 5000-bp window; rerun the coordinate audit and revise the protocol before changing")
    if args.permutations < 199:
        raise ValueError("Use at least 199 permutations; recommended 2000 or more")
    root = args.root
    gw_path = args.all_snp or root / "results_Rice_PH_LMM_GWAS_PC10" / "GenoPerfVAE_Rice_PH_LMM_GWAS_AllSNP_GWAS.csv"
    au_path = args.candidate_audit or root / "results_Rice_PH_MSU6_MSU7_Coordinate_Audit" / "GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit_Candidates_CoordinateAudit.csv"
    gff = args.msu6_gff or root / "all.gff3.gz"
    fasta = args.msu6_fasta or root / "all.con.gz"
    goslim = args.goslim or root / "osa1_r7.all_models.GOSlim.txt.gz"
    obo = args.obo or root / "go-basic.obo"
    out = args.out_dir or root / "results_Rice_PH_TopN_GO_Enrichment"
    if args.download_goslim and not goslim.is_file():
        print("Downloading official MSU7 Plant GO Slim:", GOSLIM_URL, flush=True)
        try:
            download(GOSLIM_URL, goslim)
        except Exception as exc:
            raise RuntimeError(f"GO Slim download failed ({exc}). Download manually from {GOSLIM_URL} to {goslim}") from exc
    if args.download_obo and not obo.is_file():
        print("Downloading GO ontology labels:", OBO_URL, flush=True)
        try:
            download(OBO_URL, obo)
        except Exception as exc:
            print(f"[WARNING] Could not download GO term names: {exc}. IDs alone remain valid.", flush=True)
    for path in (gw_path, au_path, gff, fasta, goslim):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
    out.mkdir(parents=True, exist_ok=True)
    print("="*83)
    print("Rice299-PH | TOP-N GO SLIM ENRICHMENT | SEQUENCE-AUDITED SNPs + MATCHED BACKGROUND")
    print("="*83)
    print("All eligible SNPs:", gw_path)
    print("Candidate audit:", au_path)
    print("MSU6 gene coordinates:", gff)
    print("MSU7 GO Slim:", goslim)
    print("Output:", out, "| permutations per mode:", args.permutations, flush=True)

    eligible, all_n = load_gwas_eligible(gw_path)
    candidates = load_audited_candidates(au_path)
    print("GWAS eligible:", len(eligible), "of", all_n, "| audited candidates:", len(candidates), flush=True)
    if len(eligible) != 38796:
        raise ValueError(f"GWAS eligibility differs from finalized Rice299 analysis: {len(eligible)} != 38796")
    if not set(candidates) <= set(eligible):
        raise ValueError(f"Some candidate SNPs are not in GWAS QC eligible markers: {sorted(set(candidates)-set(eligible))}")
    for snp, (ch, pos, _, _) in candidates.items():
        if eligible[snp][:2] != (ch,pos):
            raise ValueError(f"Coordinate disagreement GWAS vs audit: {snp}")

    aliases = read_msu6_aliases(fasta)
    print("MSU6 assembly aliases:", sorted((k,v) for k,v in aliases.items() if k.isnumeric() and len(k)>4), flush=True)
    indexes, n_genes, n_features = load_msu6_genes(gff, aliases)
    print("MSU6 loci:", n_genes, "parsed gene features:", n_features, flush=True)
    if n_genes != 57624:
        raise ValueError(f"MSU6 gene count changed from coordinate audit: {n_genes} != 57624")

    snp_genes = {}
    all_gene_ids = set()
    for snp, (ch,pos,maf,call) in eligible.items():
        hits = near_genes(ch,pos,indexes,args.window)
        snp_genes[snp] = hits
        all_gene_ids.update(hits)
    candidate_genes = set()
    for snp, (_,_,previous,_) in candidates.items():
        computed = snp_genes[snp]
        if computed != previous:
            raise RuntimeError(f"Candidate gene audit discrepancy: {snp}: computed={sorted(computed)}, audited={sorted(previous)}")
        candidate_genes.update(computed)
    if not candidate_genes:
        raise RuntimeError("No candidate genes fall within 5 kb; cannot test GO")
    if not candidate_genes <= all_gene_ids:
        raise RuntimeError("Candidate genes must be in eligible SNP-mappable background")
    mapping_discrepancies = []
    for snp, (_, _, msu6_ids, audit_row) in candidates.items():
        msu7_ids = set(g.strip() for g in clean(audit_row.get("MSU7_Remapped_NearGeneIDs", "")).split(";") if g.strip())
        if msu6_ids != msu7_ids:
            mapping_discrepancies.append({"SNP": snp,
                "MSU6_Genes": ";".join(sorted(msu6_ids)),
                "MSU7_Genes": ";".join(sorted(msu7_ids)),
                "OnlyMSU6": ";".join(sorted(msu6_ids-msu7_ids)),
                "OnlyMSU7": ";".join(sorted(msu7_ids-msu6_ids))})
    csv_save(out / f"{PREFIX}_CrossAssembly_Differences.csv", mapping_discrepancies,
             ["SNP", "MSU6_Genes", "MSU7_Genes", "OnlyMSU6", "OnlyMSU7"])
    print(f"Candidate = {len(candidate_genes)} genes; SNP-mappable background = {len(all_gene_ids)} genes", flush=True)
    print(f"Candidate SNPs with release-dependent 5-kb gene mapping: {len(mapping_discrepancies)}/{len(candidates)}", flush=True)
    if mapping_discrepancies:
        print('[CAUTION] The GO universe and candidate mapping use MSU6 consistently; MSU7 differs for some SNPs. See CrossAssembly_Differences.csv.', flush=True)

    gene_go, embedded_names, n_assoc = load_goslim(goslim)
    obo_terms = load_go_obo(obo)
    annotated_universe = all_gene_ids & set(gene_go)
    annotated_candidates = candidate_genes & annotated_universe
    if len(annotated_candidates) < 10:
        raise ValueError(f"Only {len(annotated_candidates)} of {len(candidate_genes)} candidates have official GO Slim; inspect input")
    if len(annotated_universe) < 1000:
        raise ValueError("Too few GO-annotated background genes; annotation file or MSU6/MSU7 ID join may be wrong")
    print(f"GO-associated models/rows={n_assoc}; candidate annotated {len(annotated_candidates)}/{len(candidate_genes)}; "
          f"background annotated {len(annotated_universe)}/{len(all_gene_ids)}; GO term names={len(obo_terms)}", flush=True)

    # Write gene-level provenance, including unannotated loci.
    def gene_prov(g):
        return {"GeneID": g, "InCandidate": g in candidate_genes,
                "GOSlimAnnotated": g in gene_go,
                "GOSlimIDs": ";".join(sorted(gene_go.get(g, ())))}
    csv_save(out / f"{PREFIX}_AllSNP_Mappable_BackgroundGenes.csv",
             [gene_prov(g) for g in sorted(all_gene_ids)],
             ["GeneID", "InCandidate", "GOSlimAnnotated", "GOSlimIDs"])
    csv_save(out / f"{PREFIX}_CandidateGenes.csv",
             [gene_prov(g) for g in sorted(candidate_genes)],
             ["GeneID", "InCandidate", "GOSlimAnnotated", "GOSlimIDs"])
    gene_links = []
    for snp in candidates:
        for g in sorted(snp_genes[snp]):
            gene_links.append({"SNP":snp,"GeneID":g,"MSU6_Chr":eligible[snp][0],"MSU6_Pos":eligible[snp][1],"Window_bp":args.window})
    csv_save(out / f"{PREFIX}_CandidateSNP_GeneLinks.csv", gene_links,
             ["SNP", "GeneID", "MSU6_Chr", "MSU6_Pos", "Window_bp"])
    (out / f"{PREFIX}_CandidateGenes_LOC.txt").write_text("\n".join(sorted(candidate_genes))+"\n",encoding="utf-8")
    (out / f"{PREFIX}_BackgroundGenes_LOC.txt").write_text("\n".join(sorted(all_gene_ids))+"\n",encoding="utf-8")

    N, n = len(annotated_universe), len(annotated_candidates)
    bg_count = go_counts(annotated_universe, gene_go)
    obs_count = go_counts(annotated_candidates, gene_go)
    terms = sorted(t for t, k in bg_count.items() if args.min_term_genes <= k <= int(N * args.max_term_fraction))
    if not terms:
        raise ValueError("No GO terms passed background-size filter")
    expected = [obs_count.get(t, 0) for t in terms]
    print("GO Slim terms formally tested:", len(terms), "| observed positive:", sum(v>0 for v in expected), flush=True)

    by_chr = defaultdict(list)
    for snp,(ch,pos,maf,call) in eligible.items():
        by_chr[ch].append((snp,pos,maf,call))
    for ch in by_chr:
        by_chr[ch].sort(key=lambda row: row[1])
    # MSU6 chromosome sizes from the actual FASTA header section; if files
    # differ, fail rather than use mismatched physical coordinate lengths.
    # The lengths below are audit-verified for MSU6 v6.0 all.con.gz.
    chrom_lengths = {1:43268879,2:35930381,3:36406689,4:35278225,5:29894789,6:31246789,
                     7:29696629,8:28439308,9:23011239,10:23134759,11:28512666,12:27497214}
    if any(max(row[1] for row in arr) > chrom_lengths[ch] for ch, arr in by_chr.items()):
        raise ValueError("GWAS SNP coordinates exceed audited MSU6 chromosome reference")
    pools, pool_diagnostics = prepare_match_pools(list(candidates), eligible, by_chr,
                                                  args.maf_tol, args.call_tol)
    csv_save(out / f"{PREFIX}_MatchedSNP_PoolDiagnostics.csv", pool_diagnostics,
             ["CandidateSNP", "PoolSize", "ToleranceExpansion"])
    print("Matched SNP candidate pools: min size", min(v["PoolSize"] for v in pool_diagnostics),
          "| tolerance expanded for", sum(v["ToleranceExpansion"] > 1 for v in pool_diagnostics), "candidate(s)", flush=True)
    rng = random.Random(args.seed)
    tallies_matched = [0] * len(terms)
    tallies_shift = [0] * len(terms)
    matched_gene_counts, shift_gene_counts, shift_d = [], [], []
    term_to_ix = {t:i for i,t in enumerate(terms)}
    candidate_names = list(candidates)
    def trial_evaluate(names, tally, counts_out):
        genes = set().union(*(snp_genes[s] for s in names)) & annotated_universe
        counts_out.append(len(genes))
        gc = go_counts(genes, gene_go)
        for go, x in gc.items():
            j = term_to_ix.get(go)
            if j is not None and x >= expected[j]:
                tally[j] += 1
        # Zero-observed GO term counts: empirical p must equal 1.
    for rep in range(args.permutations):
        matched = sample_matched_set(candidate_names, pools, rng)
        trial_evaluate(matched, tallies_matched, matched_gene_counts)
        shifted, discrepancies = positional_shift_set(candidate_names, eligible, by_chr, chrom_lengths, rng)
        trial_evaluate(shifted, tallies_shift, shift_gene_counts)
        shift_d.extend(discrepancies)
        if (rep+1) % 500 == 0 or rep+1 == args.permutations:
            print(f"  SNP permutations: {rep+1}/{args.permutations} complete", flush=True)

    fishp = [hypergeom_sf(expected[i], N, bg_count[t], n) for i,t in enumerate(terms)]
    eq = lambda z: [(v + 1) / (args.permutations + 1) for v in z]
    emp_matched, emp_shift = eq(tallies_matched), eq(tallies_shift)
    fishq, matchq, shiftq = bh_adjust(fishp), bh_adjust(emp_matched), bh_adjust(emp_shift)
    result_rows = []
    for i,t in enumerate(terms):
        k, K = expected[i], bg_count[t]
        name, namespace = obo_terms.get(t, (embedded_names.get(t, t), ""))
        annotated_hits = sorted(g for g in annotated_candidates if t in gene_go[g])
        fold = (k/n) / (K/N) if K and n else 0.0
        result_rows.append({
            "GO_ID":t, "GO_Name":name, "GO_Namespace":namespace,
            "CandidateGeneCount":k, "CandidateGOAnnotatedTotal":n,
            "BackgroundGeneCount":K, "BackgroundGOAnnotatedTotal":N,
            "FoldEnrichment":f"{fold:.8g}",
            "Fisher_P":f"{fishp[i]:.10g}", "Fisher_FDR_BH":f"{fishq[i]:.10g}",
            "MatchedSNP_Empirical_P":f"{emp_matched[i]:.10g}",
            "MatchedSNP_Empirical_FDR_BH":f"{matchq[i]:.10g}",
            "Shift_Empirical_P":f"{emp_shift[i]:.10g}",
            "Shift_Empirical_FDR_BH":f"{shiftq[i]:.10g}",
            "FisherSignificant":fishq[i] < .05,
            "MatchedSignificant":matchq[i] < .05,
            "ShiftSignificant":shiftq[i] < .05,
            "AllThreeFDRPass":fishq[i] < .05 and matchq[i] < .05 and shiftq[i] < .05,
            "CandidateGenesInTerm":";".join(annotated_hits)
        })
    result_rows.sort(key=lambda r:(float(r["Fisher_FDR_BH"]),float(r["Fisher_P"]),r["GO_ID"]))
    result_keys = list(result_rows[0])
    csv_save(out / f"{PREFIX}_Full_GOSlim_Statistics.csv", result_rows, result_keys)
    csv_save(out / f"{PREFIX}_GO_Terms_With_CandidateHits.csv",
             [r for r in result_rows if int(r["CandidateGeneCount"]) > 0], result_keys)

    unannot_candidate = len(candidate_genes)-n
    sig_fish = sum(r["FisherSignificant"] for r in result_rows)
    sig_m = sum(r["MatchedSignificant"] for r in result_rows)
    sig_s = sum(r["ShiftSignificant"] for r in result_rows)
    sig_all = sum(r["AllThreeFDRPass"] for r in result_rows)
    def stats(vals):
        return {"mean":sum(vals)/len(vals), "min":min(vals),"max":max(vals)}
    summary = {
        "Study":"Rice299 PH model-based SNP prioritization",
        "Status":"EXPLORATORY SNP-prioritized GO Slim analysis; NOT causal / external validation",
        "Phenotype":"PH", "SampleSize":299,
        "AllInputSNPs":all_n,"EligibleGWASSNPs":len(eligible),"CandidateSNPs":len(candidates),
        "CoordinateSystem":"HapMap SNPs in MSU6; GFF3 MSU6; MSU7 GO Slim by stable LOC_Os gene identifiers",
        "CandidateSourceAndTargetGeneSetsMatchExactly":len(mapping_discrepancies)==0,
        "CandidateUniqueGenes":len(candidate_genes),
        "CandidateSNPsWithMSU6MSU7GeneDifferences":len(mapping_discrepancies),
        "GeneMappingSystemForStatisticalTest":"MSU6 GFF on both candidate and background SNPs; MSU7 official Plant GO Slim IDs by stable LOC_Os names",

        "BackgroundSNP_MappableGenes":len(all_gene_ids),
        "CandidateGenesWithOfficialGOSlim":n,
        "CandidateGenesWithoutOfficialGOSlim":unannot_candidate,
        "BackgroundGenesWithOfficialGOSlim":N,
        "BackgroundGenesWithoutOfficialGOSlim":len(all_gene_ids)-N,
        "Ontology":"MSU7 official Plant GO Slim; NOT full GO ontology; per-gene isoforms deduplicated",
        "EligibleGOTermsTested":len(terms),
        "MinGenesPerTermInBackground":args.min_term_genes,
        "Fisher_FDR_05_Terms":sig_fish,
        "MatchedSNP_Empirical_FDR_05_Terms":sig_m,
        "ChromosomeShift_Empirical_FDR_05_Terms":sig_s,
        "AllThree_FDR_05_Terms":sig_all,
        "PermutationsPerNull":args.permutations,
        "RandomSeed":args.seed,
        "GenicMappingWindow_bp":args.window,
        "EmpiricalNullMatched":"chromosome + MAF tolerance + called-state call-rate tolerance; without replacement; excludes candidate SNPs; DOES NOT match LD",
        "EmpiricalNullShift":"each chromosome shifted circularly with one random offset; nearest eligible SNP; approximate within-chr spacing, NOT genotype LD or MAF matched",
        "MatchedNull_PoolDiagnostics":pool_diagnostics,
        "MatchedNull_AnnotatedGenesPerSample":stats(matched_gene_counts),
        "ShiftNull_AnnotatedGenesPerSample":stats(shift_gene_counts),
        "ShiftMarkerToTarget_bp":stats(shift_d),
        "Caveat":"same-cohort SNP selection; not experimental/causal validation; GO coverage incomplete; SNP annotations are proximity-only; both SNP nulls approximate",
        "Inputs":{"GWASAllSNP":str(gw_path),"CandidateAudit":str(au_path),"MSU6GFF":str(gff),"MSU6FASTA":str(fasta),"OfficialGOSlim":str(goslim),"GoBasicOBO":str(obo) if obo.is_file() else None}
    }
    (out / f"{PREFIX}_Summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    write_png(out / f"{PREFIX}_GOSlim_Bubble.png", result_rows)
    msg = ["Rice299-PH | exploratory GO Slim enrichment", "", "Important: GO SLIM categories only, NOT the whole GO ontology.",
           f"GWAS QC SNPs {len(eligible)}; candidate SNPs {len(candidates)}; mapped candidate genes {len(candidate_genes)}.",
           f"SNP-mappable background genes {len(all_gene_ids)}; GO-covered background {N}.",
           f"Official GO Slim annotated candidate genes {n}/{len(candidate_genes)}; not annotated {unannot_candidate}/{len(candidate_genes)}.",
           f"Tested GO Slim terms {len(terms)}; Fisher FDR<0.05 {sig_fish}; matched null FDR<0.05 {sig_m}; shift null FDR<0.05 {sig_s}; all three {sig_all}.",
           "", "NO functional SNP causality is established by these results.",
           "All background and candidate genes were mapped using MSU6 GFF and a shared 5-kb window; MSU7 sequence liftover confirms SNP positions; MSU7 GO IDs joined via LOC_Os names.",
           "The two permutation nulls are sensitivity analyses and do not establish linkage-disequilibrium-controlled enrichment.",
           "If no GO term passes all checks, report negative enrichment rather than asserting a pathway.",
           "", "Top 10 by Fisher significance:"]
    for r in result_rows[:10]:
        msg.append(f"{r['GO_ID']} {r['GO_Name']}: hits={r['CandidateGeneCount']} Fisher q={r['Fisher_FDR_BH']} matched q={r['MatchedSNP_Empirical_FDR_BH']} shift q={r['Shift_Empirical_FDR_BH']}")
    (out / f"{PREFIX}_ReadMe_Results.txt").write_text("\n".join(msg)+"\n",encoding="utf-8")
    print("="*83)
    print("GO SLIM ENRICHMENT COMPLETE")
    print(f"Candidate genes: {len(candidate_genes)} | annotated: {n} | background mappable: {len(all_gene_ids)} | annotated: {N}")
    print(f"Tested: {len(terms)} GO terms; Fisher FDR<0.05: {sig_fish}; matched: {sig_m}; shifted: {sig_s}; all checks: {sig_all}")
    print("Saved to:", out)
    print("No causal claims; LD r2-matching NOT performed; GO Slim is not full GO.")
    return summary


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
