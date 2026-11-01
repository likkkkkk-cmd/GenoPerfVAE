# GenoPerfVAE

**Genomic prediction and exploratory model-based SNP prioritization in crops**

This repository contains research code for GenoPerfVAE, implementations/adaptations of genomic-prediction baselines, and a dedicated analysis workflow for exploratory SNP prioritization in rice plant height. Refer to the revised GenoPerfVAE manuscript for methodological details, including **Section 2.1 (Datasets)** for the original publications and data sources.

> **Data availability:** **No genotype, phenotype, or other third-party dataset is distributed in this repository**, including processed data. We have not obtained permission to redistribute these datasets. Users should obtain the original data from the providers and papers cited in **Section 2.1 (Datasets)** of our manuscript, subject to each source's access conditions, and prepare local inputs following the experiment-specific code and manuscript methods. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
>
> **Reproducibility status:** The ten root-level Python scripts were checked for Python syntax, but a clean-clone end-to-end run has not been verified. Some scripts use historical Windows paths and expect multiple local phenotype files. Do not assume the code runs without configuration. Known issues are recorded in [CODE_AUDIT_REPORT.md](CODE_AUDIT_REPORT.md).

## Repository structure

```text
GenoPerfVAE/
├── README.md
├── requirements.txt
├── requirements-baselines.txt
├── .gitignore
├── DATA_AVAILABILITY.md
├── REPRODUCIBILITY.md
├── CODE_AUDIT_REPORT.md
├── validate_example_data.py          # Optional validator for user-obtained local Wheat2000 data
├── example_data/
│   └── README.md                     # Instructions only; NO dataset files committed
├── rice299_ph_snp_prioritization/
│   ├── README.md
│   └── [analysis and plotting scripts]
├── GenoPerfVAE.py
├── Performer.py
├── CLCNet.py
├── DeepGS.py
├── DNNGP.py
├── GBLUP.py
├── MeNet.py
├── SVR.py
├── VMGP.py
└── XGBoost.py
```

The optional `example_data/` directory contains **documentation only** in the public repository. Users may place their own legitimately obtained copies of `2000gene.pkl` and `2000_3_phe_length.txt` there **locally**. `.gitignore` excludes these local data files from Git tracking. The directory itself does **not** provide downloads.

## Models and entry points

| Python script | Research component | Important note |
| --- | --- | --- |
| `GenoPerfVAE.py` | GenoPerfVAE prediction benchmark | Wheat2000 configuration; commonly expects six phenotype files. Review active loss weights against manuscript. |
| `Performer.py` | Performer-only baseline | Uses a data directory and multiple phenotype files. |
| `CLCNet.py` | CLCNet adaptation | LightGBM feature selection; uses historical absolute input paths. |
| `DeepGS.py` | DeepGS adaptation | By default processes one parity of the ten folds. |
| `DNNGP.py` | DNNGP adaptation | By default processes one parity of the ten folds. |
| `GBLUP.py` | Genomic BLUP baseline | Requires locally obtained genotype and phenotype files. |
| `MeNet.py` | MeNet adaptation | Current entry point selects hardness; source-data paths require adjustment. |
| `SVR.py` | Support vector regression | Requires local trait inputs. |
| `VMGP.py` | VMGP-S comparison | Imports `vmgp_model.py`, which must be supplied under an appropriate upstream license; see code audit. |
| `XGBoost.py` | XGBoost benchmark | GPU-oriented defaults and a stale introductory comment; see code audit. |

These files are experiment-specific research programs, **not a unified command-line application**. Baseline implementations may use different native or adapted training settings; identical tuning budgets must not be assumed.

## Installation

Use a Python version compatible with the PyTorch build selected for your operating system and GPU. Python 3.10 is a suggested starting point, not a validated version lock.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-baselines.txt  # for baseline-specific dependencies
```

Dependencies are an audited **package inventory**, not a pinned reproduction environment. Record the exact versions from the original execution platform to reproduce reported training results.

## Obtaining research datasets

**Data are not hosted by this GitHub repository.** The GenoPerfVAE manuscript's **Section 2.1 (Datasets)** cites the source studies used for the four genomic datasets:

| Dataset | Species | Where to find the original data source |
| --- | --- | --- |
| Rice299 | Rice | Original rice breeding population and the subsequent Rice299 cohort described/cited in manuscript Section 2.1 |
| Wheat2000 | Wheat | Source study and associated data-access references cited in Section 2.1 |
| Wheat599 | Wheat | Source study and associated data-access references cited in Section 2.1 |
| Maize1404 | Maize | Source study and associated data-access references cited in Section 2.1 |

Download data from the **original providers**, not from this repository. Some script input files are processed versions of those data: a source download may require phenotype matching, filtering, imputation or other transformations to reproduce the experiment-specific inputs. Consult [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

### Local Wheat2000 format check (optional)

After obtaining and preparing a trusted copy of the Wheat2000 genotype and length phenotype locally, a basic structural check can be run:

```bash
python validate_example_data.py --data-dir example_data
```

This command **does not download data**, train a model, or independently verify accession identities. Never load an untrusted `.pkl` file because Python pickle files may execute code.

The Wheat2000 genotype (`2000gene.pkl`) and length phenotype (`2000_3_phe_length.txt`) shown in this documentation are **expected local filenames only**, not files included in the repository. To run all six Wheat2000 traits, obtain the other phenotype files as well. The scripts may require updating local data paths and trait selections.

## Rice SNP prioritization (Section 3.7)

See [rice299_ph_snp_prioritization/README.md](rice299_ph_snp_prioritization/README.md) for the frozen-model SNP-sensitivity pipeline, LOCO-LMM GWAS, MSU6→MSU7 reference mapping, GO analysis, and GradientSHAP comparison. Its required Rice299 datasets, reference genomes and derived intermediate tables are **not distributed here**. SNP prioritization is exploratory and does not establish causal variants or validated biological effects.

## Further documentation

- [Data availability and original-source access](DATA_AVAILABILITY.md)
- [Reproducibility and setup notes](REPRODUCIBILITY.md)
- [Code audit and current limitations](CODE_AUDIT_REPORT.md)
- [Release checklist](GITHUB_RELEASE_CHECKLIST.md)

## Citation and code licensing

Please cite the source publications listed in manuscript Section 2.1 when using their data. Cite the original methods for reused baseline implementations, in addition to the GenoPerfVAE manuscript once its final citation is available. The exact paper metadata and repository-wide software license should be confirmed before finalizing the provided citation/license templates. **Data redistribution rights and software code licenses are separate matters.**
