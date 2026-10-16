#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rice299-PH final Top-N extension: 150 default, 200 optional.

This script does not train GenoPerfVAE. It reuses the FINAL recon=0.3 checkpoint,
retains the full 299 observed genotype backgrounds, and retains all previously
specified called-state QC thresholds. TOP-N = gradient preselection, not
all-38,796-SNP exact finite-difference ranking.

Stages:
  perturb: run frozen-model full-cohort finite-difference on TopN (long GPU step).
  audit: prepare TopN candidate file, liftover each from MSU6 to MSU7 (CPU).
  go: run the same GO Slim/background method with matched SNP permutations (CPU).
  all: run all 3 stages.ython GenoPerfVAE_Rice_PH_TopN_Expansion.py --top-k 150 --stage go --permutations 2000

Repeat the study with fixed N=150 as a candidate-set-size sensitivity analysis.
Do NOT choose N after inspecting enrichment significance.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"E:\VGMP\Extended Comparison\GenoPerfVAE")
PERTURB = 'GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation.py'
AUDIT = 'GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit.py'
GO = 'GenoPerfVAE_Rice_PH_TopN_GO_Enrichment.py'
PFX_P = 'GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation'
PFX_A = 'GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit'
PFX_G = 'GenoPerfVAE_Rice_PH_TopN_GO_Enrichment'


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def run_command(argv):
    print('\n' + '='*85, flush=True)
    print('RUNNING:', subprocess.list2cmdline([str(x) for x in argv]), flush=True)
    print('='*85, flush=True)
    subprocess.run([str(x) for x in argv], check=True)


def read_rows(path):
    with require(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def safe_float(v):
    try: return float(v)
    except (ValueError, TypeError): return float('nan')


def generate_audit_table(perturb_exact, output_csv, top_k):
    rows = read_rows(perturb_exact)
    cand = [r for r in rows if r.get('Role') == 'TopGradientCandidate']
    if len(cand) != top_k or len(set(x.get('SNP','') for x in cand)) != top_k:
        raise ValueError(f'Expected {top_k} unique TopGradientCandidate rows, found {len(cand)}')
    if any(int(float(r['N_Evaluable'])) <= 0 for r in cand):
        raise ValueError('At least one TopN SNP has zero callable perturbation samples')
    cand.sort(key=lambda r: (-safe_float(r.get('MeanAbsDeltaPH')), r['SNP']))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, ['SNP','RoleGroup','MeanAbsDeltaPH','RankByExactPerturbation',
                                   'GradientScreenRank','N_Evaluable','MSU6_Chr','MSU6_Position'])
        writer.writeheader()
        for i, r in enumerate(cand,1):
            writer.writerow({'SNP':r['SNP'],'RoleGroup':'Candidate','MeanAbsDeltaPH':r['MeanAbsDeltaPH'],
                             'RankByExactPerturbation':i, 'GradientScreenRank':r['GradientScreenRank'],
                             'N_Evaluable':r['N_Evaluable'], 'MSU6_Chr':r['Chromosome'],
                             'MSU6_Position':r['Position']})
    return cand


def compare_to_original_top32(root, outdir, cand):
    """Check reproducibility of Top32 selection when expanding the shortlist."""
    options = [
        root / 'results_Rice_PH_ObservedGenotype_SNP_Perturbation_All299' / (PFX_P + '_ExactPerturbations.csv'),
        root / 'results_Rice_PH_ObservedGenotype_SNP_Perturbation' / (PFX_P + '_ExactPerturbations.csv'),
    ]
    found = next((p for p in options if p.is_file()), None)
    if found is None:
        print('[NOTICE] Original Top32 exact perturbation table not found; skip overlap check.')
        return
    original = [r for r in read_rows(found) if r.get('Role') == 'TopGradientCandidate']
    names_old = {r['SNP'] for r in original}
    names_new = {r['SNP'] for r in cand}
    summary = {'OriginalResults':str(found), 'OriginalCandidateCount':len(original),
               'ExpansionCandidates':len(cand), 'OldCandidateIDsIncludedInExpansion':len(names_old & names_new),
               'MissingOriginalIDs':sorted(names_old-names_new),
               'SelectionFullyNested':len(original)==32 and names_old <= names_new}
    (outdir / 'Top32_vs_TopN_SelectionQC.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    print('Top32 inclusion verification:',summary['OldCandidateIDsIncludedInExpansion'],'/',len(original))
    if len(original)==32 and not summary['SelectionFullyNested']:
        print('[WARNING] Old and extended gradient selection differ: do not treat this as a strictly nested Top32->TopN comparison.')


def gwas_crosscheck(root, outdir, cand):
    path = root / 'results_Rice_PH_LMM_GWAS_PC10' / 'GenoPerfVAE_Rice_PH_LMM_GWAS_AllSNP_GWAS.csv'
    if not path.is_file():
        print('[NOTICE] PC10 full GWAS table not found. Skipping GWAS crosscheck:', path)
        return
    wanted = {x['SNP']:x for x in cand}
    result = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            name = r.get('SNP','')
            if name in wanted:
                if name in result: raise ValueError('Repeated SNP in full GWAS: ' + name)
                result[name] = r
    if set(result) != set(wanted):
        raise ValueError(f'GWAS file missing {len(set(wanted)-set(result))} candidate SNP(s)')
    dest = outdir / 'TopN150or200_Candidate_PC10_GWAS_Comparison.csv'
    with dest.open('w', encoding='utf-8-sig', newline='') as f:
        cols = ['SNP', 'GradientScreenRank','RankByExactPerturbation','MeanAbsDeltaPH',
                'PC10_PValue','PC10_FDR_BH','PC10_GWAS_PRank','PC10_MinusLog10P']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for i,r in enumerate(cand,1):
            d=result[r['SNP']]
            w.writerow({'SNP':r['SNP'],'GradientScreenRank':r['GradientScreenRank'],
                        'RankByExactPerturbation':i,'MeanAbsDeltaPH':r['MeanAbsDeltaPH'],
                        'PC10_PValue':d.get('PValue',''), 'PC10_FDR_BH':d.get('FDR_BH',''),
                        'PC10_GWAS_PRank':d.get('GWAS_PRank',''),
                        'PC10_MinusLog10P':d.get('MinusLog10P','')})
    print('Full-GWAS candidate comparison saved:', dest)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--top-k', type=int, choices=[150,200], default=150)
    p.add_argument('--stage', choices=['perturb','audit','go','all','prepare'], default='all')
    p.add_argument('--project-dir', type=Path, default=ROOT)
    p.add_argument('--permutations',type=int,default=2000)
    p.add_argument('--controls-per-candidate',type=int,choices=[0,1],default=0,
                   help='0 is the efficient GO-size extension; 1 also computes matched-control model responses (much slower)')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--overwrite-perturb',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'p.csv';b=Path(tmp)/'a.csv'
            with a.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,['SNP','Role','N_Evaluable','MeanAbsDeltaPH','GradientScreenRank','Chromosome','Position'])
                w.writeheader()
                for i in range(args.top_k):
                    w.writerow({'SNP':f'S1_{1000+i}','Role':'TopGradientCandidate',
                        'N_Evaluable':'299','MeanAbsDeltaPH':str(i*0.001),
                        'GradientScreenRank':str(i+1),'Chromosome':1,'Position':1000+i})
                w.writerow({'SNP':'S2_1200','Role':'MatchedControl','N_Evaluable':299,
                    'MeanAbsDeltaPH':0.01,'GradientScreenRank':190,'Chromosome':2,'Position':1200})
            rows=generate_audit_table(a,b,args.top_k)
            assert len(rows)==args.top_k and len(read_rows(b))==args.top_k
            assert read_rows(b)[0]['SNP'] == f'S1_{1000+args.top_k-1}'
            print('SELF-TEST PASSED: exact role filtering, full count, ranking and CSV export')
        return
    root=args.project_dir
    tag=f'Top{args.top_k}'
    home=root / f'results_Rice_PH_{tag}_FinalExpansion'
    pout=home / '01_ObservedGenotype_ExactPerturbation'
    aout=home / '02_MSU6_MSU7_CoordinateAudit'
    gout=home / '03_GOSlim_Enrichment'
    cand_csv=home / f'Rice_PH_{tag}_AuditedCandidates_Input.csv'
    p_exact=pout / (PFX_P + '_ExactPerturbations.csv')
    a_exact=aout / (PFX_A + '_Candidates_CoordinateAudit.csv')
    steps=['perturb','audit','go'] if args.stage == 'all' else [args.stage]
    for step in steps:
        if step == 'perturb':
            require(root / PERTURB)
            if p_exact.is_file() and not args.overwrite_perturb:
                print('Existing perturbation result detected; skipping expensive GPU stage:',p_exact)
            else:
                run_command([sys.executable,root / PERTURB,'--top-k',args.top_k,
                    '--controls-per-candidate',args.controls_per_candidate, '--all-samples',
                    '--batch-size',args.batch_size,'--device',args.device,
                    '--seed',42,'--output-dir',pout])
            generate_audit_table(p_exact,cand_csv,args.top_k)
            gwas_crosscheck(root,home,read_rows(cand_csv))
            compare_to_original_top32(root,home,read_rows(cand_csv))
        elif step == 'prepare':
            generate_audit_table(p_exact,cand_csv,args.top_k)
            gwas_crosscheck(root,home,read_rows(cand_csv))
            compare_to_original_top32(root,home,read_rows(cand_csv))
        elif step == 'audit':
            generate_audit_table(p_exact,cand_csv,args.top_k)
            # A separate dummy project-dir prevents the old audit program from
            # silently auto-loading OLD Top32 controls from the original project.
            empty_root=home / '_NoLegacyControlAutoload'; empty_root.mkdir(parents=True,exist_ok=True)
            hapmap=root.parent / 'data' / 'MET_crfilt_.75_allchrom.hmp.txt'
            fa7=root / 'osa1_r7.asm.chrs.fa.gz'
            if not fa7.is_file(): fa7=root / 'osa1_r7.asm.fa.gz'
            needed=[root/AUDIT,hapmap,root/'all.con.gz',fa7,
                    root/'all.gff3.gz',root/'osa1_r7.all_models.gff3.gz']
            for path in needed:require(path)
            run_command([sys.executable,root / AUDIT,'--project-dir',empty_root,
                '--candidates',cand_csv,'--hapmap',hapmap,
                '--msu6-fasta',root/'all.con.gz','--msu7-fasta',fa7,
                '--msu6-gff',root/'all.gff3.gz',
                '--msu7-gff',root/'osa1_r7.all_models.gff3.gz',
                '--map-scope','genomewide','--window',5000,'--out-dir',aout])
            audited=read_rows(a_exact)
            passcount=sum(str(r.get('GenomewideUniqueMappingAndSourceAlleleConfirmed','')).lower()=='true' for r in audited)
            print(f'TopN target SNPs with supported genomewide liftover: {passcount}/{args.top_k}')
            if passcount!=args.top_k:
                raise RuntimeError('Some candidate coordinates are not uniquely validated: inspect the audit table; DO NOT silently use them in GO')
        elif step == 'go':
            require(a_exact)
            require(root/GO)
            require(root/'osa1_r7.all_models.GOSlim.txt.gz')
            require(root/'results_Rice_PH_LMM_GWAS_PC10' / 'GenoPerfVAE_Rice_PH_LMM_GWAS_AllSNP_GWAS.csv')
            run_command([sys.executable,root/GO,'--root',root,'--candidate-audit',a_exact,
                '--out-dir',gout,'--permutations',args.permutations,'--seed',42])
            summary=gout / (PFX_G+'_Summary.json')
            if summary.is_file():
                d=json.loads(summary.read_text(encoding='utf-8'))
                print('EXPANSION SUMMARY:', json.dumps({k:d.get(k) for k in
                    ['CandidateSNPs','CandidateUniqueGenes','BackgroundSNP_MappableGenes',
                    'CandidateGenesWithOfficialGOSlim','EligibleGOTermsTested',
                    'Fisher_FDR_05_Terms','MatchedSNP_Empirical_FDR_05_Terms',
                    'ChromosomeShift_Empirical_FDR_05_Terms']},indent=2,ensure_ascii=False))
    print('COMPLETED stage=',args.stage,'TopK=',args.top_k,'Output=',home)
    print('No independent biological validation / causality is implied. GO scores must be BH adjusted.')


if __name__=='__main__':
    try:main()
    except (FileNotFoundError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:
        print('[ERROR]',str(e), file=sys.stderr)
        sys.exit(2)
