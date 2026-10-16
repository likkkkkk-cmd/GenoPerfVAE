# Rice299-PH SNP Prioritization and Exploratory Functional Interpretation

Scripts supporting **Section 3.7** of the revised GenoPerfVAE manuscript: *Model-Based SNP Prioritization and Exploratory Functional Interpretation in Rice*.

> **Data policy:** The repository does **not** redistribute the Rice299 genotype/phenotype files, source HapMap, MSU6/MSU7 assemblies or annotations, nor any other third-party data. Obtain the original datasets from the publications/resources cited in **manuscript Section 2.1** and the genome references from their original providers. This README describes required local inputs; it does not imply they are bundled or that redistribution permission has been granted.

## Scope and interpretation

This workflow uses a **full-cohort GenoPerfVAE model trained on 299 Rice299 accessions** for post-hoc analysis of plant height (PH). Of 73,147 input SNPs, 38,796 satisfy the called-state eligibility rules; the **top 150 SNPs are selected by input-gradient screening** and then assessed using **exact single-marker changes between the observed numeric genotype states −1 and +1**, with model parameters held fixed. The analysis is complemented by observed-PH LOCO-LMM GWAS, MSU6-to-MSU7 coordinate auditing, gene annotation, Plant GO Slim analysis, and a GradientSHAP comparison.

**Limitations:** These are *in-sample model-sensitivity analyses*, not an independent cohort validation. Single-SNP perturbations may not preserve haplotypes/linkage disequilibrium; GWAS is performed on the same rice cohort; gene proximity and nominal associations do **not** establish causal variants or experimental functionality. No prioritized SNP passed genome-wide FDR correction in the reported analysis.

## Included Python scripts

| Script | Purpose |
| --- | --- |
| `GenoPerfVAE_Rice_PH_Final_PostHoc_Train.py` | Fit final PH model to all 299 matched accessions and save checkpoint. Not a cross-validation accuracy estimate. |
| `GenoPerfVAE_Rice_PH_Target_RoundTrip_Validation.py` | **Import dependency** providing checkpoint/model/data loading utilities for the perturbation workflow. Its historical latent-optimization entry point is **not** part of the revised Section 3.7 workflow. |
| `GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation.py` | Screen eligible SNPs with gradients and evaluate exact frozen-model genotype-state perturbations on observed accessions. |
| `GenoPerfVAE_Rice_PH_TopN_Expansion.py` | Orchestrate `perturb`, `audit`, and `go` stages with `--top-k 150`. |
| `GenoPerfVAE_Rice_PH_LMM_GWAS.py` | LOCO-LMM association analysis of **observed** PH values; run separately, with `--pcs 10` for the reported main analysis. |
| `GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit.py` | Validate SNP coordinates by reference-sequence mapping and audit nearby gene assignments. |
| `GenoPerfVAE_Rice_PH_TopN_GO_Enrichment.py` | GO Slim overrepresentation with the eligible-SNP-derived gene background and empirical SNP-level nulls. |
| `GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py` | Run GradientSHAP and compare its selected top-150 markers against the exact-perturbation workflow. |

**Naming note:** Figure plotting script names retain their historical `Figure10*` prefix; the revised manuscript refers to the panels as **Figure 9A–D**. Keep filenames unchanged unless the scripts and documentation are updated together.

## Environment

A GPU and a PyTorch environment are recommended for model training, exhaustive input-gradient screening, exact perturbations, and GradientSHAP. Representative Python dependencies are:

```powershell
conda activate pytorch
python -m pip install numpy pandas scipy scikit-learn matplotlib torch performer-pytorch captum
```

The command above lists library families; it is **not a tested, version-pinned environment specification**. Record the exact Python/PyTorch/CUDA/dependency versions used for the reported runs before claiming environment-level reproducibility.

### Required local inputs

- `MET_crfilt_.90_outliers_removed_for_RRBlup_line_corrected.csv`: Rice299 numeric genotype matrix, including `Entry` and the SNP columns in checkpoint order.
- `rice299_PH.csv`: observed rice PH phenotypes, matched using `Entry`.
- `MET_crfilt_.75_allchrom.hmp.txt`: source HapMap used in the MSU6→MSU7 coordinate audit.
- `all.con.gz` and `all.gff3.gz`: MSU6 chromosome sequence and GFF3 annotation.
- `osa1_r7.asm.chrs.fa.gz` (or `osa1_r7.asm.fa.gz`), `osa1_r7.all_models.gff3.gz`, and `osa1_r7.all_models.GOSlim.txt.gz`: MSU7 sequence and annotation inputs.
- For the Figure 9C plot, the existing `GenoPerfVAE_Rice_PH_Top150_GWAS_MatchedNull_Distributions.csv` containing 2,000 rows of nominal GWAS hit-count randomizations. **Its stand-alone generating script is not identified in the available script inventory; see Reproducibility checklist.**
- For Figures 9B–D, the derived `GenoPerfVAE_Rice_PH_Top150_Perturbation_GWAS_MSU7_Merged.csv` (or a verified equivalently structured merged table). Its construction needs to be documented/automated to enable clean-clone reproduction.

### Original local path layout

The existing scripts were authored against a Windows working directory of:

```text
E:\VGMP\Extended Comparison\
├── data\
│   ├── MET_crfilt_.90_outliers_removed_for_RRBlup_line_corrected.csv
│   ├── rice299_PH.csv
│   └── MET_crfilt_.75_allchrom.hmp.txt
└── GenoPerfVAE\
    ├── [the 12 Python scripts listed above]
    ├── all.con.gz
    ├── all.gff3.gz
    ├── osa1_r7.asm.chrs.fa.gz
    ├── osa1_r7.all_models.gff3.gz
    └── osa1_r7.all_models.GOSlim.txt.gz
```

**Important portability caveat:** Several scripts use this absolute Windows path as a default; the top-N orchestration script expects **its dependency scripts and reference files inside its `--project-dir` directory**, and it looks for the HapMap in the **parent directory's `data` folder**. Merely copying the scripts into a new GitHub subfolder does **not** make the commands below portable to a new machine. Before releasing the repository as fully runnable, replace fixed local paths with portable command-line arguments or repository-relative paths and test a clean clone. The commands below document the **original runtime layout**, not an independently verified clean-clone installation.

## Execution flow

```text
[1] Final PH checkpoint (299 accessions)
                 |
                 v
[2] Gradient screening (38,796 eligible SNPs)
       -> Top 150 -> exact -1/+1 perturbation (frozen model)
                 |
          +------+------------------+
          |                         |
          v                         v
[3] PC10 LOCO-LMM GWAS       [4] MSU6-to-MSU7 audit
 (observed phenotypes)                  |
          |                             v
          |                   [5] Plant GO Slim analysis
          |                     (2,000 null trials)
          v
[6] GradientSHAP comparison (using the same checkpoint and PC10 GWAS)
          |
          v
[7] Prepare/verify Figure 9 input tables -> plotting scripts A–D
```

### Step 1 — Fit the final PH model

Run from the original `GenoPerfVAE` directory:

```bat
conda activate pytorch
cd /d "E:\VGMP\Extended Comparison\GenoPerfVAE"
python GenoPerfVAE_Rice_PH_Final_PostHoc_Train.py --data-dir "E:\VGMP\Extended Comparison\data" --output-dir "results_Rice_PH_Final_PostHoc"
```

Expected checkpoint: `results_Rice_PH_Final_PostHoc/GenoPerfVAE_Rice_PH_Final_PostHoc.pth`.

This is a **single full-data fitted model** for post-hoc interpretation. Use the separate 10-fold evaluation to report prediction generalization.

### Step 2 — Gradient screening and exact perturbation (Top 150)

```bat
python GenoPerfVAE_Rice_PH_TopN_Expansion.py --project-dir "E:\VGMP\Extended Comparison\GenoPerfVAE" --top-k 150 --stage perturb --device cuda --batch-size 8
```

The orchestrator invokes `GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation.py`. It evaluates the selected **150 markers** on all aligned samples; it does **not** perform exact perturbation for all 38,796 eligible SNPs.

Primary output:

```text
results_Rice_PH_Top150_FinalExpansion/01_ObservedGenotype_ExactPerturbation/
    GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation_ExactPerturbations.csv
```

By default `--controls-per-candidate 0`: the Top150 expansion does **not** re-run 150 matched-control exact-perturbation experiments. To force a re-run of a previously generated perturbation CSV, explicitly use `--overwrite-perturb` after checking existing outputs.

### Step 3 — Observed-PH LOCO-LMM GWAS (PC10)

Use the Top150 exact-perturbation candidate file as input; run 10-PC GWAS into the fixed directory required by the orchestrator. For example:

```bat
python GenoPerfVAE_Rice_PH_LMM_GWAS.py --pcs 10 --checkpoint "results_Rice_PH_Final_PostHoc\GenoPerfVAE_Rice_PH_Final_PostHoc.pth" --genotype "E:\VGMP\Extended Comparison\data\MET_crfilt_.90_outliers_removed_for_RRBlup_line_corrected.csv" --phenotype "E:\VGMP\Extended Comparison\data\rice299_PH.csv" --candidate-file "results_Rice_PH_Top150_FinalExpansion\01_ObservedGenotype_ExactPerturbation\GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation_ExactPerturbations.csv" --matching-file "results_Rice_PH_Top150_FinalExpansion\optional_top150_controls_not_used.csv" --out-dir "results_Rice_PH_LMM_GWAS_PC10"
```

The intentionally nonexistent optional `--matching-file` path prevents inadvertently reusing historical **Top32** matched-control pairs for the Top150 run. This does not disable the genome-wide association analysis. A main required output is:

```text
results_Rice_PH_LMM_GWAS_PC10/GenoPerfVAE_Rice_PH_LMM_GWAS_AllSNP_GWAS.csv
```

Optional sensitivity checks use `--pcs 3` and `--pcs 5`, each in their own output directory. Do not overwrite the PC10 results.

### Step 4 — MSU6→MSU7 sequence-based coordinate audit

```bat
python GenoPerfVAE_Rice_PH_TopN_Expansion.py --project-dir "E:\VGMP\Extended Comparison\GenoPerfVAE" --top-k 150 --stage audit
```

The orchestrator invokes `GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit.py` and checks the uniqueness/allele compatibility of the candidate mappings. Required outputs are under:

```text
results_Rice_PH_Top150_FinalExpansion/02_MSU6_MSU7_CoordinateAudit/
```

A nearby gene is a genomic-context hypothesis, not a validated regulatory target.

### Step 5 — Plant GO Slim overrepresentation and randomizations

```bat
python GenoPerfVAE_Rice_PH_TopN_Expansion.py --project-dir "E:\VGMP\Extended Comparison\GenoPerfVAE" --top-k 150 --stage go --permutations 2000
```

The orchestrator invokes `GenoPerfVAE_Rice_PH_TopN_GO_Enrichment.py`. GO candidates/background are mapped by the same MSU6 gene models with a ±5-kb rule, with official MSU7 Plant GO Slim annotations and matched-SNP and chromosome-shift empirical checks. Outputs include:

```text
results_Rice_PH_Top150_FinalExpansion/03_GOSlim_Enrichment/
    GenoPerfVAE_Rice_PH_TopN_GO_Enrichment_Summary.json
    GenoPerfVAE_Rice_PH_TopN_GO_Enrichment_Full_GOSlim_Statistics.csv
```

**The GO empirical null distributions are not the same file as the GWAS hit-count null-distribution input needed for Figure 9C.** Do not confuse these analyses.

After all prerequisites are available, the orchestrator also supports:

```bat
python GenoPerfVAE_Rice_PH_TopN_Expansion.py --project-dir "E:\VGMP\Extended Comparison\GenoPerfVAE" --top-k 150 --stage all --device cuda --batch-size 8 --permutations 2000
```

This covers **perturb → coordinate audit → GO**; it does **not** run model fitting, GWAS, GradientSHAP, or Figure 9 plotting.

### Step 6 — GradientSHAP comparison

```bat
python GenoPerfVAE_Rice_PH_GradientSHAP_Comparison.py --project-dir "E:\VGMP\Extended Comparison\GenoPerfVAE" --stage all --device cuda --top-k 150 --baseline-size 32 --n-samples 16 --seeds 42,43,44
```

This script requires the final checkpoint, prior Top150 exact perturbations, and the existing PC10 GWAS output. It compares alternative SNP rankings under the **same fitted predictor**, not independent biological validation.

### Step 7 — Figure 9A–D plotting

The plotting scripts have historical `Figure10*` filenames but correspond to the **revised Figure 9**. Use the actual fully verified input-file paths; the examples below assume that `MERGED.csv` and `MATCHED_NULL.csv` have been prepared in the current directory.

```bat
REM Figure 9A: 150-marker exact perturbation ranking
python GenoPerfVAE_Rice_PH_Figure10A_Plot.py --csv "results_Rice_PH_Top150_FinalExpansion\01_ObservedGenotype_ExactPerturbation\GenoPerfVAE_Rice_PH_ObservedGenotype_SNP_Perturbation_ExactPerturbations.csv" --out-dir "figure9"

REM Figure 9B: sensitivity versus GWAS
python GenoPerfVAE_Rice_PH_Figure10B_Plot.py --merged "MERGED.csv" --out-dir "figure9"

REM Figure 9C: GWAS nominal-hit matched/shift null distributions
python GenoPerfVAE_Rice_PH_Figure10C_Plot.py --null-csv "MATCHED_NULL.csv" --merged "MERGED.csv" --out-dir "figure9"

REM Figure 9D: full reference-supported annotation (NOT preview mode)
python GenoPerfVAE_Rice_PH_Figure10D_Plot.py --merged "MERGED.csv" --audit "results_Rice_PH_Top150_FinalExpansion\02_MSU6_MSU7_CoordinateAudit\GenoPerfVAE_Rice_PH_MSU6_MSU7_Coordinate_Audit_Candidates_CoordinateAudit.csv" --msu7-gff "osa1_r7.all_models.gff3.gz" --out-dir "figure9"
```

`MERGED.csv` refers to a verified file such as `GenoPerfVAE_Rice_PH_Top150_Perturbation_GWAS_MSU7_Merged.csv`, and `MATCHED_NULL.csv` refers to `GenoPerfVAE_Rice_PH_Top150_GWAS_MatchedNull_Distributions.csv`. **They are placeholders, not filenames automatically produced by the above commands.** The Figure 9D script requires `--msu7-gff` to generate a publication-ready reference figure; `--preview-without-gff` is provisional only.

## Reported analysis cross-checks

Values from the revised Section 3.7 manuscript should be checked against the actual outputs when the workflow is rerun:

- Rice accessions: **299**; original SNP columns: **73,147**.
- QC-eligible markers: **38,796**; gradient-screened candidate list: **150**.
- Main GWAS comparison: **PC10 LOCO-LMM**, performed on observed PH.
- Among these 150 candidates: **27** nominal GWAS hits at *P* < 0.05, **12** at *P* < 0.01, **0** at genome-wide FDR < 0.05.
- GO analysis: **2,000** matched-marker and **2,000** chromosome-shift randomizations; no GO Slim category survived FDR correction in the reported analysis.
- GradientSHAP stability: **seeds 42, 43, and 44**, with 32 baseline genotypes and 16 interpolation samples per accession.

Do not present these checks as newly computed results from this README; they are expected values from the revised manuscript.

## Reproducibility checklist before public release

- [ ] Verify the corresponding full-cohort checkpoint and the code/training configuration that produced it; do not accidentally use a cross-validation or ablation checkpoint.
- [ ] Convert fixed Windows file paths to configurable/relative paths and test from a fresh clone.
- [ ] Record exact Python, Torch, CUDA and dependency versions and, ideally, a version-pinned environment file.
- [ ] Record licenses and provenance for genotype, phenotype, rice genome FASTA, HapMap, GFF3 and GO annotations. Do not upload third-party raw data or large reference archives without checking redistribution permissions.
- [ ] Locate/document the program that generated `GenoPerfVAE_Rice_PH_Top150_GWAS_MatchedNull_Distributions.csv`. The available `Figure10C_Plot.py` **only consumes** this file; it does not generate the GWAS nulls.
- [ ] Document/code the join that generated `GenoPerfVAE_Rice_PH_Top150_Perturbation_GWAS_MSU7_Merged.csv`, or provide it with a clear provenance statement.
- [ ] Keep Top32 legacy exploratory runs separate from final Top150 runs. Historical target-genotype design/decoder diagnostics are not part of Section 3.7.
- [ ] Inspect CSV metadata, script comments and saved figures for stale figure numbering or inconsistent dataset/loss labels.
- [ ] Verify results from the real input files and include only the checked, shareable summary outputs.
- [ ] Keep patient/private information, secrets, access tokens, huge checkpoints and proprietary input files out of Git history.

## Citation

If this repository is used in a publication, cite the final GenoPerfVAE paper when its bibliographic metadata is available. The current analysis provides exploratory marker prioritization, not genetic causality or functional validation.

## Publishing this folder to GitHub

Suggested repository location: `GenoPerfVAE/analysis/rice299_ph_snp_prioritization/` (or `GenoPerfVAE/rice299_ph_snp_prioritization/`). Keep the original script basenames unless you update their imports/launch commands. Put this README and `.gitignore` into the same folder as the scripts.

If you already have a local clone of your GenoPerfVAE repository, from its root:

```bash
git status
git add analysis/rice299_ph_snp_prioritization/
git diff --cached --stat
git commit -m "Add Rice299 PH Top150 SNP prioritization workflow"
git push
```

The `git add` path is an **example**: change it if you chose a different directory. Before `git commit`, inspect the staged files (`git diff --cached --name-only`) and verify that no raw data, reference archives, model checkpoints, access tokens or unauthorized materials are included. A `.gitignore` does not automatically untrack files that were already committed.
