#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rice299-PH reference-coordinate audit, MSU6 versus MSU7 (MSU6 GFF3 assembly-ID support).

No model fitting, phenotype use, genotype changes, or SNP recoding.

IMPORTANT: MSU6 ``all.seq.gz`` stores gene/TU sequences, NOT chromosome
pseudomolecules. This script requires MSU6 ``all.con.gz`` (full chromosomes).
See the official MSU6 README:
https://rice.uga.edu/pub/data/Eukaryotic_Projects/o_sativa/annotation_dbs/pseudomolecules/version_6.0/README

Windows (from E:\\VGMP\\Extended Comparison\\GenoPerfVAE):
    python GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit.py

Optional:
    python GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit.py --flank 150 --window 5000

The outputs distinguish same-position reference concordance from UNIQUE
sequence-based remapping. A HapMap allele match alone does NOT validate an
assembly, and a nearby gene is NOT a causal target.

Dependencies: Python 3.9+ standard library only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from urllib.parse import unquote

PROJECT = Path(r"E:\VGMP\Extended Comparison\GenoPerfVAE")
DATA = Path(r"E:\VGMP\Extended Comparison\data")
PREFIX = "GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit"
SNPRX = re.compile(r"^S(\d{1,2})_(\d+)$", re.I)
LOCRX = re.compile(r"LOC_Os\d{2}g\d+", re.I)
BASES = set("ACGT")
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def clean(s):
    return str(s or "").strip().lstrip("\ufeff")


def chnum(s):
    s = clean(s).split()[0].split("|")[0].lower()
    s = re.sub(r"^(?:chromosome[_-]?|chr[_-]?|os[_-]?)", "", s)
    if re.fullmatch(r"0*([1-9]|1[012])", s):
        v = int(s)
        return v if 1 <= v <= 12 else None
    return None


def marker_coords(s):
    m = SNPRX.fullmatch(clean(s))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def open_txt(path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("r", encoding="utf-8-sig", errors="replace")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_alleles(s):
    # SNP metadata can include IUPAC / indels; restrict reference comparisons
    # to explicit single-nucleotide A/C/G/T alleles, not numeric genotypes.
    found = set()
    for piece in re.split(r"[/,;|\\\s]+", clean(s).upper()):
        if piece in BASES:
            found.add(piece)
    return found


def allele_match(ref_base, alleles):
    if ref_base not in BASES:
        return "REFERENCE_N_OR_UNAVAILABLE"
    if not alleles:
        return "ALLELES_NOT_CALLABLE"
    direct = ref_base in alleles
    reverse = ref_base in {x.translate(COMPLEMENT) for x in alleles}
    if direct and reverse:
        return "BOTH_STRANDS_COMPATIBLE"
    if direct:
        return "DIRECT"
    if reverse:
        return "REVERSE_COMPLEMENT"
    return "NOT_IN_HAPMAP_ALLELES"


def register_fasta_gff_aliases(header, alias_map):
    """Collect chromosome aliases from legacy ``>chr|asmbl_id database``.

    MSU6 all.gff3 may use the second field (asmbl_id) rather than Chr1.
    Only trust numeric assembly IDs explicitly linked to a chromosome FASTA
    header; never infer a chromosome from an arbitrary numeric GFF seqid.
    """
    chrom = chnum(header)
    if chrom is None:
        return
    token = header.strip().split()[0]
    # Header example documented by the official MSU6 README: >chr|asmbl_id database
    aliases = {str(chrom), f"{chrom:02d}", f"chr{chrom}", f"chr{chrom:02d}",
               f"chromosome{chrom}", f"chromosome{chrom:02d}"}
    for part in token.split("|"):
        part = part.strip()
        if not part:
            continue
        aliases.add(part)
        # Some mirrors include assembly=11667 in place of a bare 11667.
        match = re.fullmatch(r"(?:asmbl_id|assembly|asmbl)[=:](\d+)", part, re.I)
        if match:
            aliases.add(match.group(1))
    for alias in aliases:
        key = alias.lower()
        previous = alias_map.get(key)
        if previous is not None and previous != chrom:
            raise ValueError(f"Conflicting MSU6 FASTA sequence alias {alias}: chromosomes {previous} and {chrom}")
        alias_map[key] = chrom


def iter_chrom_fasta(path, *, allow_partial=False, aliases=None):
    """Stream full chromosomes; reject gene/TU FASTA and duplicate chroms."""
    seen = set()
    with open_txt(path) as fh:
        label, pieces = None, []
        for line in fh:
            if line.startswith(">"):
                if label is not None:
                    ch = chnum(label)
                    if ch is not None:
                        if ch in seen:
                            raise ValueError(f"Duplicate chromosome {ch} in {path.name}; not a valid chromosome FASTA")
                        seen.add(ch)
                        seq = "".join(pieces).upper().replace("U", "T")
                        if not allow_partial and len(seq) < 1_000_000:
                            raise ValueError(f"Chromosome {ch} has only {len(seq)} bp in {path.name}; likely gene/TU FASTA")
                        yield ch, seq
                label = line[1:].strip()
                if aliases is not None:
                    register_fasta_gff_aliases(label, aliases)
                if re.search(r"LOC_OS\d{2}G\d+", label, re.I):
                    raise ValueError(f"{path.name} appears to contain gene/TU sequences, NOT a chromosome assembly. MSU6 requires all.con.gz")
                pieces = []
            elif label is None:
                if line.strip():
                    raise ValueError(f"FASTA file missing '>' header: {path}")
            else:
                line = line.strip()
                if line:
                    pieces.append(line)
        if label is not None:
            ch = chnum(label)
            if ch is not None:
                if ch in seen:
                    raise ValueError(f"Duplicate chromosome {ch} in {path}")
                seen.add(ch)
                seq = "".join(pieces).upper().replace("U", "T")
                if not allow_partial and len(seq) < 1_000_000:
                    raise ValueError(f"Chromosome {ch} unexpectedly short ({len(seq)} bp)")
                yield ch, seq
    if not seen:
        raise ValueError(f"No chromosomes 1–12 found in {path}; check FASTA header/format")
    if not allow_partial and seen != set(range(1, 13)):
        raise ValueError(f"Reference incomplete: {path.name} contains chromosomes {sorted(seen)}; expected 1–12")


def read_hapmap_metadata(path, *, limit=0):
    items = {}
    duplicates, bad = 0, 0
    with open_txt(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Empty HapMap: {path}")
        reader.fieldnames = [clean(x) for x in reader.fieldnames]
        for f in ("rs#", "chrom", "pos", "alleles"):
            if f not in reader.fieldnames:
                raise ValueError(f"HapMap is missing required metadata column {f!r}; found {reader.fieldnames[:15]}")
        for row in reader:
            name = clean(row["rs#"])
            if not name:
                bad += 1
                continue
            if name in items:
                duplicates += 1
                continue
            try:
                ch = chnum(row["chrom"])
                pos = int(clean(row["pos"]))
            except (ValueError, TypeError):
                bad += 1
                continue
            id_ch, id_pos = marker_coords(name)
            items[name] = {"SNP": name, "Chr": ch, "Position": pos,
                           "SNP_ID_ChrPosMatch": id_ch == ch and id_pos == pos,
                           "HapMapAlleles": clean(row["alleles"]),
                           "HapMapStrand": clean(row.get("strand", "")),
                           "HapMapAssemblyField": clean(row.get("assembly#", "")),
                           "AlleleSet": split_alleles(row["alleles"])}
            if limit and len(items) >= limit:
                break
    if not items:
        raise ValueError("No usable SNP metadata in HapMap")
    return items, {"hapmap_usable_rows": len(items), "duplicate_ID_rows": duplicates, "bad_rows": bad}


def read_candidate_table(path):
    with open_txt(path) as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or "SNP" not in rdr.fieldnames:
            raise ValueError("CandidatesTable.csv must contain 'SNP' column")
        data = {}
        for row in rdr:
            name = clean(row.get("SNP"))
            if not name:
                continue
            role = clean(row.get("RoleGroup", "Candidate"))
            if role not in {"Candidate", "Control"}:
                continue
            if name in data:
                raise ValueError(f"Duplicated candidate/control SNP: {name}")
            data[name] = row
    if not data:
        raise ValueError("No candidate/control SNP rows found")
    print("Loaded target variants:", len(data), "Candidates:", sum(v.get("RoleGroup") == "Candidate" for v in data.values()), flush=True)
    return data


@dataclass
class Gene:
    chrom: int
    start: int
    end: int
    strand: str
    gid: str
    source: str
    ftype: str
    desc: str


def gff_attrs(s):
    out = {}
    for fragment in s.split(";"):
        if "=" in fragment:
            a, b = fragment.split("=", 1)
        elif " " in fragment:
            a, b = fragment.split(" ", 1)
        else:
            continue
        out[a.strip().lower()] = unquote(b.strip().strip('"'))
    return out


def read_gff(path, seqid_aliases=None):
    genes, fallback = {}, {}
    cnt = Counter()
    observed_seqids = Counter()
    observed_features = Counter()
    example_records = []
    with open_txt(path) as f:
        for line in f:
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) != 9:
                cnt["malformed"] += 1
                continue
            # MSU6 may use an assembly ID (numeric) instead of chromosome name.
            # A numeric ID is accepted ONLY when it appeared in the matching
            # chromosome FASTA header, never from guesswork.
            seqid = fields[0].strip()
            observed_seqids[seqid] += 1
            ft = fields[2].lower()
            observed_features[ft] += 1
            if len(example_records) < 4:
                example_records.append((seqid, fields[2], fields[8][:180]))
            chrom = chnum(seqid)
            if chrom is None and seqid_aliases is not None:
                chrom = seqid_aliases.get(seqid.lower())
            if chrom is None:
                cnt["unrecognized_seqid"] += 1
                continue
            cnt["recognized_chromosome_records"] += 1
            is_gene = ft in {"gene", "pseudogene", "transposable_element_gene", "ncrna_gene", "transcription_unit", "transposable_element"}
            is_rna = ft in {"mrna", "transcript", "ncrna", "trna", "rrna", "pseudogenic_transcript"}
            if not is_gene and not is_rna:
                continue
            try:
                a, b = int(fields[3]), int(fields[4])
            except ValueError:
                continue
            if a < 1 or b < a:
                continue
            attrs = gff_attrs(fields[8])
            # Older GFF3 versions can store pub_locus in other attributes.
            # Searching the complete attributes does not invent locus IDs.
            m = LOCRX.search(unquote(fields[8]))
            # Don't merge generic transcript IDs without robust locus ID.
            if not m:
                cnt["missing_loc_os_id"] += 1
                continue
            gid = m.group(0)
            d = attrs.get("note", attrs.get("description", attrs.get("product", "")))
            g = Gene(chrom, a, b, fields[6], gid, fields[1], fields[2], d)
            key = chrom, gid
            target = genes if is_gene else fallback
            if key in target:
                old = target[key]
                old.start, old.end = min(old.start, a), max(old.end, b)
                if not old.desc and d:
                    old.desc = d
            else:
                target[key] = g
            cnt["gene_feature" if is_gene else "transcript_feature"] += 1
    for key, v in fallback.items():
        if key not in genes:
            genes[key] = v
    out = defaultdict(list)
    for g in genes.values():
        out[g.chrom].append(g)
    for ch in out:
        out[ch].sort(key=lambda g: (g.start, g.end, g.gid))
    if len(genes) < 5:
        observed = observed_seqids.most_common(8)
        types = observed_features.most_common(10)
        aliases = list(seqid_aliases.items())[:12] if seqid_aliases is not None else []
        raise ValueError(
            f"Parsed only {len(genes)} gene models from {path}.\n"
            f"  GFF3 sequence IDs: {observed}\n"
            f"  Feature types: {types}\n"
            f"  Records with known chromosome: {cnt['recognized_chromosome_records']}\n"
            f"  Missing LOC_Os gene identifiers: {cnt['missing_loc_os_id']}\n"
            f"  FASTA-derived aliases (examples): {aliases}\n"
            f"  First GFF3 records: {example_records}\n"
            "  Check that the MSU6 all.con.gz and all.gff3.gz are from the same release."
        )
    cnt["unique_genes"] = len(genes)
    cnt["gff_unrecognized_seqids"] = dict(observed_seqids) if len(observed_seqids) <= 30 else dict(observed_seqids.most_common(30))
    cnt["gff_feature_type_top"] = dict(observed_features.most_common(20))
    cnt["chromosome_recognized_records"] = cnt["recognized_chromosome_records"]
    print(f"  {path.name}: {len(genes):,} unique loci; "
          f"chromosome-matched rows={cnt['recognized_chromosome_records']:,}; "
          f"unmapped seqID rows={cnt['unrecognized_seqid']:,}; "
          f"missing LOC IDs={cnt['missing_loc_os_id']:,}", flush=True)
    return out, dict(cnt)


def gene_dist(g, p):
    if p < g.start:
        return g.start - p
    if p > g.end:
        return p - g.end
    return 0


def gene_region(g, p, upstream, downstream):
    if g.start <= p <= g.end:
        return "gene_body"
    if g.strand not in "+-":
        return "flanking_unknown_strand"
    direction = ("upstream" if p < g.start else "downstream") if g.strand == "+" else ("upstream" if p > g.end else "downstream")
    d = gene_dist(g, p)
    return direction if d <= (upstream if direction == "upstream" else downstream) else "intergenic_nearby"


def find_genes(by_ch, chrom, pos, radius, upstream, downstream):
    if chrom not in by_ch or pos is None or pos <= 0:
        return [], None
    arr = by_ch[chrom]
    # 50k genes total so linear chromosome scan is fine for 64 SNPs.
    nearest = min(arr, key=lambda g: (gene_dist(g, pos), g.gid)) if arr else None
    near = [g for g in arr if gene_dist(g, pos) <= radius]
    near.sort(key=lambda g: (gene_dist(g, pos), g.gid))
    return near, nearest


def quick_flank_status(source, target):
    if not source or not target or len(source) != len(target):
        return "OUT_OF_RANGE", ""
    positions = [(x, y) for x, y in zip(source, target) if x in BASES and y in BASES]
    if len(positions) < max(10, int(0.8 * len(source))):
        return "INSUFFICIENT_ACGT", ""
    ident = sum(a == b for a, b in positions) / len(positions)
    if ident == 1.0:
        return "IDENTICAL_WINDOW", ident
    return "DIFFERENT_WINDOW", round(ident, 6)


def genome_stage_msu6(path, hap, targets, flank, fixed_flank, allow_partial, seqid_aliases=None):
    bychr = defaultdict(list)
    for name, row in hap.items():
        if row["Chr"] is not None and row["Position"] > 0:
            bychr[row["Chr"]].append(name)
    source_lengths = {}
    for ch, seq in iter_chrom_fasta(path, allow_partial=allow_partial, aliases=seqid_aliases):
        source_lengths[str(ch)] = len(seq)
        print(f"  MSU6 chr {ch}: {len(seq):,} bp, {len(bychr[ch]):,} HapMap SNPs", flush=True)
        for name in bychr[ch]:
            row = hap[name]
            p = row["Position"]
            row["MSU6_Base"] = seq[p - 1] if p <= len(seq) else ""
            l, r = p - fixed_flank - 1, p + fixed_flank
            row["MSU6_FixedFlank"] = seq[l:r] if l >= 0 and r <= len(seq) else ""
            if name in targets:
                lo, hi = max(0, p - 1 - flank), min(len(seq), p + flank)
                q = seq[lo:hi]
                row["MSU6_QueryFlank"] = q
                row["MSU6_CenterIndex"] = p - 1 - lo
                row["MSU6_CandidateWindowComplete"] = len(q) == (2 * flank + 1)
        del seq
    return source_lengths


def genome_stage_msu7(path, hap, fixed_flank, allow_partial):
    # We keep MSU7 in memory for a truly genome-wide unique remapping search.
    genome = {}
    for ch, seq in iter_chrom_fasta(path, allow_partial=allow_partial):
        genome[ch] = seq
        print(f"  MSU7 chr {ch}: {len(seq):,} bp", flush=True)
    for row in hap.values():
        ch, p = row["Chr"], row["Position"]
        seq = genome.get(ch)
        if not seq or p > len(seq) or p <= 0:
            row["MSU7_SamePos_Base"] = ""
            row["FixedFlankStatus"] = "OUT_OF_RANGE"
            row["FixedFlankIdentity"] = ""
            continue
        row["MSU7_SamePos_Base"] = seq[p - 1]
        lo, hi = p - fixed_flank - 1, p + fixed_flank
        b = seq[lo:hi] if lo >= 0 and hi <= len(seq) else ""
        status, sim = quick_flank_status(row.get("MSU6_FixedFlank", ""), b)
        row["FixedFlankStatus"], row["FixedFlankIdentity"] = status, sim
    return genome


def occurrences(haystack, needle, limit=50):
    if not needle or "N" in needle or any(v not in BASES for v in needle):
        return []
    out, start = [], 0
    while len(out) < limit:
        p = haystack.find(needle, start)
        if p < 0:
            break
        out.append(p)
        start = p + 1
    return out


def semiglobal_align(query, target, target_start):
    """Global alignment of query to a substring of target (free target ends).

    Query positions and target positions are 0-based within strings. Linear-gap
    traceback maps the query central base to an exact 1-based target position.
    """
    n, m = len(query), len(target)
    if not n or not m:
        return None
    match, mismatch, gap = 2, -3, -4
    prior = [0] * (m + 1)
    traceback = [bytearray(m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cur = [gap * i] + [0] * m
        traceback[i][0] = 1
        a = query[i - 1]
        for j in range(1, m + 1):
            b = target[j - 1]
            diag = prior[j - 1] + (match if a == b and a in BASES else mismatch)
            up = prior[j] + gap
            left = cur[j - 1] + gap
            if diag >= up and diag >= left:
                cur[j], traceback[i][j] = diag, 0
            elif up >= left:
                cur[j], traceback[i][j] = up, 1
            else:
                cur[j], traceback[i][j] = left, 2
        prior = cur
    j = max(range(m + 1), key=lambda c: prior[c])
    score = prior[j]
    i = n
    # Count query alignments and map central position in traceback.
    center = n // 2   # caller supplies a full symmetrical flank; enforce upstream
    mapping, same, paired, insdel = {}, 0, 0, 0
    while i > 0:
        choice = traceback[i][j]
        if choice == 0 and j > 0:
            mapping[i - 1] = j - 1
            a, b = query[i - 1], target[j - 1]
            paired += 1
            if a == b and a in BASES:
                same += 1
            i -= 1
            j -= 1
        elif choice == 1 or j == 0:
            insdel += 1
            i -= 1
        else:
            insdel += 1
            j -= 1
    aligned = paired + insdel
    identity = same / max(1, aligned)
    return {"AlignmentScore": score, "AlignedIdentity": round(identity, 6),
            "AlignedQueryFraction": round(paired / n, 6),
            "IndelColumns": insdel, "Map": mapping,
            "MappedPosition": target_start + mapping[center] if center in mapping else None}


def map_locus(query, center_index, source_chr, source_pos, genome, flank, min_identity, max_shift, mode, max_hits):
    """Map one MSU6 flank to MSU7, NEVER infer positions from SNP ID alone.

    Unique exact full-flank hits are strongest evidence. For imperfect mapping,
    two spatially consistent independent anchors and high-identity gapped alignment
    are required. Ambiguous or single-anchor hits are not accepted as verified.
    """
    if not query or len(query) < 101 or query.count("N") > len(query) * 0.2:
        return {"MappingStatus": "MSU6_FLANK_UNUSABLE", "MappingEvidence": "missing/short/N-rich MSU6 sequence"}
    if center_index != len(query) // 2:
        return {"MappingStatus": "MSU6_FLANK_AT_CHR_EDGE", "MappingEvidence": "requires symmetric reference window"}
    chroms = ([source_chr] if mode == "same-chr" else sorted(genome))
    chroms = [c for c in chroms if c in genome]
    rc = query.translate(COMPLEMENT)[::-1]
    full_hits = []
    for c in chroms:
        seq = genome[c]
        for orient, q in (("+", query), ("-", rc)):
            for idx in occurrences(seq, q, max_hits):
                hit_center = idx + (center_index if orient == "+" else len(query) - 1 - center_index) + 1
                full_hits.append((c, hit_center, orient))
                if len(full_hits) > max_hits:
                    return {"MappingStatus": "MULTIPLE_FULL_FLANK_HITS", "MappingEvidence": f">{max_hits} identical mappings"}
    full_hits = list(dict.fromkeys(full_hits))
    if len(full_hits) == 1:
        c, p, strand = full_hits[0]
        return {"MappingStatus": "UNIQUE_EXACT_FLANK", "MappingEvidence": f"unique full {len(query)}-bp sequence; genomewide" if mode == "genomewide" else "unique within tested chromosome only",
                "MSU7_MappedChr": c, "MSU7_MappedPos": p, "MapStrand": strand,
                "AlignmentIdentity": 1.0, "AlignmentQueryFraction": 1.0,
                "AnchorSupport": 2, "MappingSearchScope": mode}
    if len(full_hits) > 1:
        return {"MappingStatus": "MULTIPLE_FULL_FLANK_HITS", "MappingEvidence": f"{len(full_hits)} identical mappings", "MappingSearchScope": mode}

    # Left and right exact anchors, separated by most of the flank. If one
    # anchor is absent, do not silently assign a shifted position.
    width = min(65, max(30, (len(query) - 1) // 3))
    anchors = [("L", 5, width), ("R", len(query) - 5 - width, width)]
    all_hits = defaultdict(list)
    for orient, oq in (("+", query), ("-", rc)):
        oc = center_index if orient == "+" else len(oq) - 1 - center_index
        for tag, offset, k in anchors:
            anchor = oq[offset:offset + k]
            if len(anchor) != k or any(x not in BASES for x in anchor):
                continue
            for c in chroms:
                for start in occurrences(genome[c], anchor, max_hits):
                    predicted = start + oc - offset + 1
                    all_hits[(c, orient)].append((predicted, tag))
    candidates = []
    for (c, orient), hits in all_hits.items():
        for predicted, _ in hits:
            good = [(p, tag) for p, tag in hits if abs(p - predicted) <= max_shift]
            support = len({tag for p, tag in good})
            if support < 2:
                continue
            nominal = int(round(median([p for p, tag in good])))
            if nominal <= 0:
                continue
            # Query and target can differ by small indels; allow extra ends.
            extra = max_shift + 12
            seq = genome[c]
            lo = max(0, nominal - 1 - center_index - extra)
            hi = min(len(seq), nominal - 1 + (len(query) - center_index) + extra)
            target = seq[lo:hi]
            q_oriented = query if orient == "+" else rc
            alignment = semiglobal_align(q_oriented, target, lo + 1)
            if not alignment or alignment["MappedPosition"] is None:
                continue
            pos = alignment["MappedPosition"]
            candidates.append({"chr": c, "pos": pos, "strand": orient,
                               "identity": alignment["AlignedIdentity"],
                               "fraction": alignment["AlignedQueryFraction"],
                               "anchors": support, "score": alignment["AlignmentScore"],
                               "gaps": alignment["IndelColumns"]})
    # Deduplicate multiple seed hypotheses converging on same physical location.
    byloc = {}
    for a in candidates:
        k = (a["chr"], a["pos"], a["strand"])
        if k not in byloc or a["score"] > byloc[k]["score"]:
            byloc[k] = a
    ranked = sorted(byloc.values(), key=lambda v: (v["identity"], v["score"]), reverse=True)
    eligible = [v for v in ranked if v["identity"] >= min_identity and v["fraction"] >= 0.98]
    if not eligible:
        return {"MappingStatus": "NO_HIGH_CONFIDENCE_ANCHORED_MAPPING",
                "MappingEvidence": f"two-anchor high-identity mapping not found (candidates={len(ranked)})",
                "MappingSearchScope": mode}
    best = eligible[0]
    if len(eligible) > 1:
        second = eligible[1]
        if abs(best["identity"] - second["identity"]) <= 0.01 and (best["chr"], best["pos"], best["strand"]) != (second["chr"], second["pos"], second["strand"]):
            return {"MappingStatus": "AMBIGUOUS_ANCHORED_MAPPING",
                    "MappingEvidence": f"at least two comparable alignments ({best['chr']}:{best['pos']}, {second['chr']}:{second['pos']})",
                    "MappingSearchScope": mode}
    return {"MappingStatus": "UNIQUE_TWO_ANCHOR_ALIGNMENT", "MappingEvidence": "two distant anchors, gapped alignment; mapping unique above threshold",
            "MSU7_MappedChr": best["chr"], "MSU7_MappedPos": best["pos"],
            "MapStrand": best["strand"], "AlignmentIdentity": best["identity"],
            "AlignmentQueryFraction": best["fraction"], "AnchorSupport": best["anchors"],
            "AlignmentIndelColumns": best["gaps"], "MappingSearchScope": mode}


def gene_report(by_ch, chrom, pos, radius, upstream, downstream, label, snp):
    near, nearest = find_genes(by_ch, chrom, pos, radius, upstream, downstream)
    output = {
        f"{label}_NearGeneCount": len(near),
        f"{label}_NearestGeneID": nearest.gid if nearest else "",
        f"{label}_NearestGeneDistance_bp": gene_dist(nearest, pos) if nearest else "",
        f"{label}_NearestGeneRegion": gene_region(nearest, pos, upstream, downstream) if nearest else "",
        f"{label}_GeneBodyOverlaps": sum(gene_dist(g, pos) == 0 for g in near),
        f"{label}_NearGeneIDs": ";".join(g.gid for g in near)
    }
    links = [{"SNP": snp, "Assembly": label, "Chromosome": chrom,
              "Position": pos, "GeneID": g.gid, "GeneStart": g.start,
              "GeneEnd": g.end, "GeneStrand": g.strand,
              "Distance_bp": gene_dist(g, pos),
              "Region": gene_region(g, pos, upstream, downstream),
              "GeneDescriptionFromGFF": g.desc, "GeneFeature": g.ftype,
              "IsNearestGene": bool(nearest and g.gid == nearest.gid)} for g in near]
    return output, links


def locate(args, basename, alternatives=()):
    p = Path(basename)
    if p.is_file():
        return p
    for path in alternatives:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Required file missing: {basename}. Tested: {', '.join(map(str, alternatives))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project-dir", type=Path, default=PROJECT)
    ap.add_argument("--hapmap", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--controls", type=Path, default=None, help="Optional separate matched ControlsTable.csv; auto-detect if present")
    ap.add_argument("--msu6-fasta", type=Path, default=None, help="MSU6 chromosomes: all.con.gz (NOT all.seq.gz)")
    ap.add_argument("--msu7-fasta", type=Path, default=None, help="MSU7 chromosomes: osa1_r7.asm.chrs.fa.gz")
    ap.add_argument("--msu6-gff", type=Path, default=None, help="all.gff3.gz")
    ap.add_argument("--msu7-gff", type=Path, default=None, help="osa1_r7.all_models.gff3.gz")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--flank", type=int, default=150, help="MSU6 +/- bp for candidate remapping (default 150)")
    ap.add_argument("--fixed-flank", type=int, default=30, help="+/- bp for same-position whole HapMap concordance")
    ap.add_argument("--window", type=int, default=5000, help="Gene proximity radius bp")
    ap.add_argument("--upstream", type=int, default=2000)
    ap.add_argument("--downstream", type=int, default=2000)
    ap.add_argument("--map-scope", choices=("genomewide", "same-chr"), default="genomewide")
    ap.add_argument("--min-map-identity", type=float, default=0.97)
    ap.add_argument("--max-indel-shift", type=int, default=40)
    ap.add_argument("--max-hits", type=int, default=40, help="maximum reported hits per anchor per chromosome")
    ap.add_argument("--max-hapmap-snps", type=int, default=0, help="0 = all; positive = quick audit")
    ap.add_argument("--allow-partial-reference", action="store_true", help="TEST ONLY: allow fewer/small chromosomes")
    args = ap.parse_args(argv)
    if args.flank < 100 or args.fixed_flank < 10 or args.window < 0 or args.max_indel_shift < 5:
        ap.error("flank>=100, fixed-flank>=10, window>=0, max-indel-shift>=5 required")
    project = args.project_dir
    if args.msu6_fasta and args.msu6_fasta.name.lower().startswith("all.seq"):
        ap.error("all.seq.gz is gene/TU sequences, NOT a genome. Download MSU6 all.con.gz instead.")
    paths = {
        "hapmap": args.hapmap or locate(args, DATA / "MET_crfilt_.75_allchrom.hmp.txt", [project / "MET_crfilt_.75_allchrom.hmp.txt"]),
        "candidates": args.candidates or locate(args, project / "results_Rice_PH_Candidate_Gene_Annotation" / "GenoPerfVAE_Rice_PH_Candidate_Gene_Annotation_CandidatesTable.csv", [project / "GenoPerfVAE_Rice_PH_Candidate_Gene_Annotation_CandidatesTable.csv"]),
        "msu6_fasta": args.msu6_fasta or locate(args, project / "all.con.gz", [project / "all.con", project / "msu6_all.con.gz"]),
        "msu7_fasta": args.msu7_fasta or locate(args, project / "osa1_r7.asm.chrs.fa.gz", [project / "osa1_r7.asm.fa.gz", project / "osa1_r7.asm.chrs.fa"]),
        "msu6_gff": args.msu6_gff or locate(args, project / "all.gff3.gz", [project / "all.gff3"]),
        "msu7_gff": args.msu7_gff or locate(args, project / "osa1_r7.all_models.gff3.gz", [project / "osa1_r7.all_models.gff3"]),
    }
    if paths["msu6_fasta"].name.lower().startswith("all.seq"):
        ap.error("STOP: MSU6 all.seq.gz contains TU sequences, NOT chromosomes. Need all.con.gz.")
    for k, p in paths.items():
        if not p.is_file():
            ap.error(f"{k} not found: {p}")
    out = args.out_dir or project / "results_Rice_PH_MSU6_MSU7_Coordinate_Audit"
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 84, "\nRice299-PH | MSU6 → MSU7 COORDINATE AUDIT (OBSERVED HAPMAP)\n" + "=" * 84, flush=True)
    for k, v in paths.items():
        print(f"{k}: {v}")
    print("Output:", out, "| scope:", args.map_scope, flush=True)
    targets = read_candidate_table(paths["candidates"])
    controls_path = args.controls
    if controls_path is None:
        default_ctrl = project / "results_Rice_PH_Candidate_Gene_Annotation" / "GenoPerfVAE_Rice_PH_Candidate_Gene_Annotation_ControlsTable.csv"
        if default_ctrl.is_file():
            controls_path = default_ctrl
    if controls_path is not None:
        if not controls_path.is_file():
            ap.error(f"Controls file not found: {controls_path}")
        controls = read_candidate_table(controls_path)
        if any(clean(v.get("RoleGroup")) != "Control" for v in controls.values()):
            ap.error("--controls includes non-Control rows; refusing to merge")
        duplicate = set(targets) & set(controls)
        if duplicate:
            ap.error(f"Duplicated candidate/control IDs: {sorted(duplicate)[:5]}")
        targets.update(controls)
        paths["controls"] = controls_path
        print("Combined candidate and control SNPs:", len(targets), flush=True)
    hap, hstat = read_hapmap_metadata(paths["hapmap"], limit=args.max_hapmap_snps)
    print("HapMap metadata SNPs:", len(hap), "| duplicate ID:", hstat["duplicate_ID_rows"], flush=True)
    for name in targets:
        if name not in hap:
            print(f"WARNING: target {name} missing from original HapMap! It cannot be verified.", flush=True)
    print("Loading MSU6 chromosome sequences (discard each after extracting needed bases/windows)...", flush=True)
    msu6_gff_aliases = {}
    lens6 = genome_stage_msu6(paths["msu6_fasta"], hap, targets, args.flank, args.fixed_flank, args.allow_partial_reference, msu6_gff_aliases)
    # Show aliases derived from the authoritative FASTA metadata, not guessed.
    unusual_aliases = [(sid, chrom) for sid, chrom in msu6_gff_aliases.items() if sid.isdigit() and int(sid) > 12]
    print("MSU6 chromosome FASTA assembly-ID aliases:", unusual_aliases[:20], flush=True)
    print("Loading MSU7 chromosomes for genome-wide mapping...", flush=True)
    gen7 = genome_stage_msu7(paths["msu7_fasta"], hap, args.fixed_flank, args.allow_partial_reference)
    print("Parsing MSU6/MSU7 GFF3...", flush=True)
    genes6, gstat6 = read_gff(paths["msu6_gff"], seqid_aliases=msu6_gff_aliases)
    genes7, gstat7 = read_gff(paths["msu7_gff"])
    print("Gene locus counts: MSU6", gstat6["unique_genes"], "MSU7", gstat7["unique_genes"], flush=True)

    all_rows = []
    for name, item in hap.items():
        a6 = allele_match(item.get("MSU6_Base", ""), item["AlleleSet"])
        a7 = allele_match(item.get("MSU7_SamePos_Base", ""), item["AlleleSet"])
        all_rows.append({"SNP": name, "Chr": item["Chr"], "HapMapPosition": item["Position"],
                         "SNP_ID_ChrPosMatch": item["SNP_ID_ChrPosMatch"],
                         "HapMapAlleles": item["HapMapAlleles"], "HapMapStrand": item["HapMapStrand"],
                         "MSU6_ReferenceBase": item.get("MSU6_Base", ""),
                         "MSU6_AlleleCompatibility": a6,
                         "MSU7_SamePosReferenceBase": item.get("MSU7_SamePos_Base", ""),
                         "MSU7_SamePosAlleleCompatibility": a7,
                         "ReferenceBaseSamePositionMatch": item.get("MSU6_Base", "") == item.get("MSU7_SamePos_Base", "") and item.get("MSU6_Base", "") in BASES,
                         "SamePositionFlankStatus": item.get("FixedFlankStatus", ""),
                         "SamePositionFlankIdentity": item.get("FixedFlankIdentity", "")})
    outprefix = out / PREFIX
    allcols = ["SNP", "Chr", "HapMapPosition", "SNP_ID_ChrPosMatch", "HapMapAlleles", "HapMapStrand",
               "MSU6_ReferenceBase", "MSU6_AlleleCompatibility", "MSU7_SamePosReferenceBase", "MSU7_SamePosAlleleCompatibility",
               "ReferenceBaseSamePositionMatch", "SamePositionFlankStatus", "SamePositionFlankIdentity"]
    write_csv(Path(str(outprefix) + "_AllHapMap_ReferenceCheck.csv"), all_rows, allcols)

    # chromosome-summary denominators EXCLUDE invalid/non-ACGT reference bases.
    bychr = defaultdict(list)
    for r in all_rows:
        if r["Chr"]:
            bychr[r["Chr"]].append(r)
    summary_chrom = []
    for c, rows in sorted(bychr.items()):
        direct6 = [r for r in rows if r["MSU6_ReferenceBase"] in BASES and r["HapMapAlleles"]]
        direct7 = [r for r in rows if r["MSU7_SamePosReferenceBase"] in BASES and r["HapMapAlleles"]]
        flank = [r for r in rows if r["SamePositionFlankStatus"] in ("IDENTICAL_WINDOW", "DIFFERENT_WINDOW")]
        def amatch(x):
            return x in ("DIRECT", "REVERSE_COMPLEMENT", "BOTH_STRANDS_COMPATIBLE")
        summary_chrom.append({"Chr": c, "HapMapRows": len(rows),
                              "MSU6ValidReferenceAndAlleles_N": len(direct6),
                              "MSU6AlleleCompatible_N": sum(amatch(r["MSU6_AlleleCompatibility"]) for r in direct6),
                              "MSU6AlleleCompatible_Fraction": round(sum(amatch(r["MSU6_AlleleCompatibility"]) for r in direct6) / len(direct6), 5) if direct6 else "",
                              "MSU7SamePosValidReferenceAndAlleles_N": len(direct7),
                              "MSU7SamePosAlleleCompatible_N": sum(amatch(r["MSU7_SamePosAlleleCompatibility"]) for r in direct7),
                              "MSU7SamePosAlleleCompatible_Fraction": round(sum(amatch(r["MSU7_SamePosAlleleCompatibility"]) for r in direct7) / len(direct7), 5) if direct7 else "",
                              "SamePosFlankComparable_N": len(flank),
                              "SamePosFlankIdentical_N": sum(r["SamePositionFlankStatus"] == "IDENTICAL_WINDOW" for r in flank),
                              "SamePosFlankIdentical_Fraction": round(sum(r["SamePositionFlankStatus"] == "IDENTICAL_WINDOW" for r in flank) / len(flank), 5) if flank else "",
                              "RefLengthsMSU6": lens6.get(str(c), ""), "RefLengthsMSU7": len(gen7[c]) if c in gen7 else ""})
    write_csv(Path(str(outprefix) + "_ByChromosome_Summary.csv"), summary_chrom, list(summary_chrom[0]))

    print("Remapping candidate/control variant flanks, n =", len(targets), flush=True)
    detailed, links = [], []
    for i, (name, target) in enumerate(targets.items(), 1):
        src = hap.get(name)
        idch, idpos = marker_coords(name)
        base = {"SNP": name, "RoleGroup": target.get("RoleGroup", ""),
                "ExactPerturbationRank": target.get("RankByExactPerturbation", ""),
                "MeanAbsDeltaPH": target.get("MeanAbsDeltaPH", ""),
                "GWAS_PC10_PValue": target.get("PValue", ""),
                "GWAS_PC10_FDR": target.get("FDR_BH", ""),
                "OldMSU7NearestGeneID_Unverified": target.get("NearestGeneID", ""),
                "OldMSU7NearestGeneDistance_Unverified": target.get("NearestGeneDistance_bp", ""),
                "MSU6_Chr": src["Chr"] if src else idch,
                "MSU6_Position": src["Position"] if src else idpos,
                "HapMapPresent": bool(src), "SNP_ID_HapMap_ChrPosAgree": src["SNP_ID_ChrPosMatch"] if src else False}
        if src:
            base["HapMapAlleles"] = src["HapMapAlleles"]
            base["MSU6_ReferenceBase"] = src.get("MSU6_Base", "")
            base["MSU6_AlleleCompatibility"] = allele_match(src.get("MSU6_Base", ""), src["AlleleSet"])
            base["MSU7_SamePosReferenceBase"] = src.get("MSU7_SamePos_Base", "")
            base["SamePositionFlankStatus"] = src.get("FixedFlankStatus", "")
            base["SamePositionFlankIdentity"] = src.get("FixedFlankIdentity", "")
            mapper = map_locus(src.get("MSU6_QueryFlank", ""), src.get("MSU6_CenterIndex", -1), src["Chr"], src["Position"], gen7,
                               args.flank, args.min_map_identity, args.max_indel_shift, args.map_scope, args.max_hits)
        else:
            mapper = {"MappingStatus": "NOT_PRESENT_IN_HAPMAP", "MappingEvidence": "No source row; no conversion"}
        base.update(mapper)
        mapped_ch = mapper.get("MSU7_MappedChr")
        mapped_pos = mapper.get("MSU7_MappedPos")
        accepted = mapper.get("MappingStatus") in {"UNIQUE_EXACT_FLANK", "UNIQUE_TWO_ANCHOR_ALIGNMENT"} and args.map_scope == "genomewide"
        if src:
            accepted = accepted and src["SNP_ID_ChrPosMatch"] and base["MSU6_AlleleCompatibility"] != "NOT_IN_HAPMAP_ALLELES"
            # For confident coordinate remapping, explicit callable alleles required.
            accepted = accepted and base["MSU6_AlleleCompatibility"] not in ("REFERENCE_N_OR_UNAVAILABLE", "ALLELES_NOT_CALLABLE")
        base["GenomewideUniqueMappingAndSourceAlleleConfirmed"] = bool(accepted)
        if mapped_ch and mapped_pos:
            base["MSU7_MappedReferenceBase"] = gen7[mapped_ch][mapped_pos - 1]
            base["PositionShift_bp_ifSameChr"] = mapped_pos - src["Position"] if src and mapped_ch == src["Chr"] else ""
            base["MSU7_MappedAlleleCompatibility"] = allele_match(base["MSU7_MappedReferenceBase"], src["AlleleSet"]) if src else ""
        else:
            base["MSU7_MappedReferenceBase"] = ""
            base["PositionShift_bp_ifSameChr"] = ""
            base["MSU7_MappedAlleleCompatibility"] = ""
        # MSU6 annotation valid provisionally even if hapmap allele uncertain;
        # MSU7 remapped annotation ONLY when genomewide unique.
        if src and src["Chr"] and src["Position"]:
            report6, lk6 = gene_report(genes6, src["Chr"], src["Position"], args.window, args.upstream, args.downstream, "MSU6", name)
            base.update(report6)
            links += [dict(x, MappingStatus="MSU6_SOURCE_POSITION", CoordinateEvidence="HapMap source coordinates, not independent phenotype validation") for x in lk6]
            if accepted:
                report7, lk7 = gene_report(genes7, mapped_ch, mapped_pos, args.window, args.upstream, args.downstream, "MSU7_Remapped", name)
                base.update(report7)
                links += [dict(x, MappingStatus=mapper["MappingStatus"], CoordinateEvidence=mapper["MappingEvidence"]) for x in lk7]
                base["NearestGeneIDChanged6vs7"] = (base["MSU6_NearestGeneID"] != base["MSU7_Remapped_NearestGeneID"])
                base["PreviousRawMSU7AnnotationMatchesRemapped"] = (target.get("NearestGeneID", "") == base["MSU7_Remapped_NearestGeneID"])
        detailed.append(base)
        print(f"  [{i:02d}/{len(targets)}] {name}: {mapper.get('MappingStatus')} | msu7={mapped_ch}:{mapped_pos} | verified_for_GFF={accepted}", flush=True)

    combined_path = Path(str(outprefix) + "_CandidateAndControl_CoordinateAudit.csv")
    prefer = ["SNP", "RoleGroup", "ExactPerturbationRank", "MeanAbsDeltaPH", "GWAS_PC10_PValue", "GWAS_PC10_FDR",
              "MSU6_Chr", "MSU6_Position", "HapMapPresent", "SNP_ID_HapMap_ChrPosAgree", "HapMapAlleles",
              "MSU6_ReferenceBase", "MSU6_AlleleCompatibility", "MSU7_SamePosReferenceBase", "SamePositionFlankStatus",
              "SamePositionFlankIdentity", "MappingStatus", "MappingEvidence", "MSU7_MappedChr", "MSU7_MappedPos",
              "MapStrand", "AlignmentIdentity", "AnchorSupport", "MappingSearchScope", "PositionShift_bp_ifSameChr",
              "MSU7_MappedReferenceBase", "MSU7_MappedAlleleCompatibility", "GenomewideUniqueMappingAndSourceAlleleConfirmed",
              "MSU6_NearestGeneID", "MSU6_NearestGeneDistance_bp", "MSU6_NearestGeneRegion",
              "MSU7_Remapped_NearestGeneID", "MSU7_Remapped_NearestGeneDistance_bp", "MSU7_Remapped_NearestGeneRegion",
              "OldMSU7NearestGeneID_Unverified", "OldMSU7NearestGeneDistance_Unverified", "PreviousRawMSU7AnnotationMatchesRemapped",
              "NearestGeneIDChanged6vs7", "MSU6_NearGeneIDs", "MSU7_Remapped_NearGeneIDs"]
    all_keys = list(dict.fromkeys(prefer + [k for r in detailed for k in r]))
    write_csv(combined_path, detailed, all_keys)
    candidates_only = [r for r in detailed if r["RoleGroup"] == "Candidate"]
    write_csv(Path(str(outprefix) + "_Candidates_CoordinateAudit.csv"), candidates_only, all_keys)
    if links:
        write_csv(Path(str(outprefix) + "_SNP_GeneLinks_CrossAssembly.csv"), links, list(dict.fromkeys(k for r in links for k in r)))
    else:
        write_csv(Path(str(outprefix) + "_SNP_GeneLinks_CrossAssembly.csv"), [], ["SNP", "Assembly", "GeneID", "Distance_bp", "Region"])

    statuses = Counter(r.get("MappingStatus", "") for r in detailed)
    summary = {"tool": PREFIX,
               "reference_note": "MSU6 full chromosomes must be all.con.gz; all.seq.gz is NOT chromosome FASTA",
               "reference_origin": "MSU6 Release 6.0; MSU7 UGA/IRGSP-1.0; versions must NOT be conflated",
               "inputs": {k: str(v) for k, v in paths.items()},
               "dataset": "Rice299, PH", "no_neural_network_training": True,
               "hapmap": hstat, "target_snp_count": len(detailed),
               "candidates": len(candidates_only), "controls": len(detailed) - len(candidates_only),
               "accepted_genomewide_coordinate_mappings": sum(r.get("GenomewideUniqueMappingAndSourceAlleleConfirmed", False) for r in detailed),
               "mapping_status": dict(statuses),
               "msu6_gene_count": gstat6["unique_genes"], "msu7_gene_count": gstat7["unique_genes"],
               "msu6_gff_parser_stats": gstat6, "msu7_gff_parser_stats": gstat7,
               "msu6_chromosome_lengths": lens6, "msu7_chromosome_lengths": {str(ch): len(seq) for ch, seq in gen7.items()},
               "same_coordinate_only_is_not_liftover": True,
               "gene_proximity_does_not_prove_causality": True,
               "genomewide_comparison": summary_chrom,
               "parameters": {"flank": args.flank, "fixed_flank": args.fixed_flank,
                              "min_map_identity": args.min_map_identity, "max_indel_shift": args.max_indel_shift,
                              "map_scope": args.map_scope, "gene_window": args.window, "upstream": args.upstream, "downstream": args.downstream}}
    with Path(str(outprefix) + "_Summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\n" + "=" * 84)
    print("COORDINATE AUDIT FINISHED")
    print("HapMap SNPs:", len(hap), "| candidates:", len(candidates_only), "| controls:", len(detailed) - len(candidates_only))
    print("Genome-wide uniquely mapped AND source allele compatible:", summary["accepted_genomewide_coordinate_mappings"])
    print("Mapping statuses:", dict(statuses))
    print("Files:")
    for fname in ("_AllHapMap_ReferenceCheck.csv", "_ByChromosome_Summary.csv", "_Candidates_CoordinateAudit.csv",
                  "_CandidateAndControl_CoordinateAudit.csv", "_SNP_GeneLinks_CrossAssembly.csv", "_Summary.json"):
        print(" ", str(outprefix) + fname)
    print("WARNING: Alignments show reference-coordinate concordance, not genetic causality or independent biological validation.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, OSError, csv.Error) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
