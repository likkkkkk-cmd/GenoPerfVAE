# Reproducibility and experiment scope

## Repository scope

This repository provides genomic prediction and SNP-prioritization **code, not datasets**. No raw or processed third-party genotype or phenotype files are bundled. Download the original datasets through the publications and associated resources identified in **Section 2.1 (Datasets)** of the GenoPerfVAE manuscript, subject to provider conditions. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

All ten root-level Python scripts passed static syntax parsing in the reviewed upload; this does not establish end-to-end execution or identical replication of the reported experiments.

## Reproducing a local Wheat2000 analysis

1. Obtain the Wheat2000 data from its original source identified in manuscript Section 2.1; prepare the genotype/phenotype inputs to match the code's expected format and **sample order**.
2. For a local grain-length analysis, put authorized copies of `2000gene.pkl` and `2000_3_phe_length.txt` in a local `example_data/` folder. They are **not included in the public repository and must not be committed**.
3. Install dependencies. Optionally run `python validate_example_data.py --data-dir example_data` for a basic two-file dimensionality/row-count check. Load trusted pickle inputs only.
4. Adjust the selected model's dataset path and trait selection. Most historic entry points expect `data/`, six phenotypes or machine-specific Windows locations.
5. Run the target program; verify the code's active hyperparameters, fold settings, local environment, model outputs and saved metrics.
6. For DeepGS/DNNGP, the two parity shards (`FOLD_PARITY=0` and `FOLD_PARITY=1`) are needed to cover all ten folds, if using their provided sharded execution mode.

For all six Wheat2000 traits, obtain all six phenotype files. Rice299, Wheat599 and Maize1404 require independent data acquisition and processing.

## Evaluation and experiment records

For fair evaluation, document: sample IDs and SNP order, fold membership, which preprocessing parameters are fitted on train only, genotype missing-state semantics, phenotype transformations, validation and early stopping, seeds, training epochs, batch size, optimizer/learning rate/weight decay, loss weights, model dimensions, search budgets and reported evaluation metrics. Baseline scripts have different original/adapted training procedures and should not be described as uniformly optimized without supporting logs.

## Rice299 PH SNP prioritization

Read [the dedicated Section 3.7 workflow](rice299_ph_snp_prioritization/README.md). Its 299-accession full-cohort model is for post-hoc model sensitivity rather than held-out predictive generalization. The top-150 workflow uses the 38,796 eligible SNPs for gradient screening and re-evaluates the 150 chosen markers by fixed-model exact genotype-state perturbation, with additional same-cohort GWAS, coordinate auditing, Plant GO Slim and GradientSHAP comparisons. None of the input data or reference assemblies is included in this repository.

A stand-alone generator for the 2,000-trial GWAS nominal-hit null-distribution CSV used in Fig. 9C has not yet been identified in the reviewed scripts, and some tools still contain machine-specific local paths. Therefore an unqualified claim of full one-command reproducibility is not currently supported.

## Capturing local environment versions

Record versions on the machine used for the reported runs:

```bash
python --version
python -m pip freeze > environment-original-pip-freeze.txt
python -m pip check
```

Inspect the exported file for internal URLs or private paths before sharing.
