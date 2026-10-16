# GenoPerfVAE

**Genomic prediction and exploratory model-based SNP prioritization in crops**

GenoPerfVAE is a variational neural model for predicting quantitative crop phenotypes from genome-wide SNP inputs. This repository contains the main prediction benchmark, implementations/adaptations of comparison models, and a Rice299 plant-height workflow for exploratory SNP prioritization.

The main model combines a chunked Performer branch with a dense shortcut branch. Each branch estimates posterior parameters for its allocated latent dimensions; these parameters are concatenated into a shared variational representation used for phenotype prediction and genotype reconstruction. Training combines phenotype-prediction loss, reconstruction loss, and KL regularization.

The repository contains experiment-specific research scripts. Input data must be obtained separately, and some baseline and analysis programs require local path configuration. A clean-clone end-to-end reproduction has not yet been verified.

## Repository structure

```text
GenoPerfVAE/
├── README.md
├── GenoPerfVAE.py                       # Main prediction benchmark
├── Performer.py                        # Performer-only comparison
├── CLCNet.py
├── DeepGS.py
├── DNNGP.py
├── GBLUP.py
├── MeNet.py
├── SVR.py
├── VMGP.py
├── XGBoost.py
├── requirements.txt
├── requirements-baselines.txt
├── validate_example_data.py             # Local Wheat2000 format check
├── DATA_AVAILABILITY.md
├── REPRODUCIBILITY.md
├── CODE_AUDIT_REPORT.md
├── CITATION.cff.template
├── LICENSE-MIT.template
├── example_data/
│   └── README.md                        # Documentation only
└── rice299_ph_snp_prioritization/
    ├── README.md
    └── *.py                            # Rice299 analysis scripts
```

This README uses **GenoPerfVAE** as the model name and `GenoPerfVAE.py` as the main filename. Save the main script under that filename before using the training command below. Analysis script filenames must remain consistent with their imports and orchestration commands.

## Installation

Use a Python environment compatible with your chosen PyTorch build and hardware. Python 3.10 is a suggested starting point; it is not a validated version requirement. A CUDA-capable GPU is recommended for neural-model training.

```bash
python -m pip install -r requirements.txt
```

For comparison models requiring additional packages:

```bash
python -m pip install -r requirements-baselines.txt
```

The requirements files list dependencies without version pins. For reproduction of reported experiments, use the versions recorded on the original execution platform. Installing these dependencies does not supply the separate `vmgp_model.py` source required by `VMGP.py`.

## Datasets

The study uses four crop datasets:

| Dataset | Crop |
| --- | --- |
| Rice299 | Rice |
| Wheat2000 | Wheat |
| Wheat599 | Wheat |
| Maize1404 | Maize |

**No third-party genotype, phenotype, or reference-genome data are included.** Obtain the original data through the source publications and resources identified in Section 2.1 (Datasets) of the accompanying manuscript, subject to their access conditions. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

Original downloads may require sample matching, filtering, imputation, or other processing before they match the script inputs. The default main entry point is configured for **Wheat2000**; running it does not automatically reproduce the other three datasets.

## Run the Wheat2000 benchmark

Run the following commands from the repository root.

### 1. Prepare local inputs

Create a local `data/` directory beside `GenoPerfVAE.py`. The default six-trait configuration expects:

```text
data/
├── 2000gene.pkl
├── 2000_1_phe_tkw.txt
├── 2000_2_phe_testw.txt
├── 2000_3_phe_length.txt
├── 2000_4_phe_width.txt
├── 2000_5_phe_hardness.txt
└── 2000_6_phe_Prot.txt
```

These are expected local filenames, not downloadable files bundled with this repository. Keep local research data out of version control.

| Input | Expected format |
| --- | --- |
| `2000gene.pkl` | A pickled two-dimensional numeric matrix with samples in rows and SNPs in columns; do not include an extra sample-ID column in the numeric matrix. |
| Phenotype `.txt` files | One phenotype value per sample, in exactly the same sample order as the genotype matrix. |

The phenotype parser takes the **first numeric token** from a line. Numeric sample IDs must not precede the phenotype value. Matching row counts alone do not verify sample identity or order.

In the supplied Wheat2000 configuration, `-1` is retained as a valid genotype value. True NaNs are imputed using parameters fitted on each outer training fold. Phenotypes are standardized within each training fold and predictions are transformed back to the original scale for evaluation.

After obtaining trusted local inputs, optionally check the genotype and grain-length phenotype:

```bash
python validate_example_data.py --data-dir data
```

This checks basic dimensions and row counts; it does not train a model or verify accession identities. Load pickle files only from trusted sources.

### 2. Check the experiment configuration

The supplied main script uses the following active Wheat2000 settings:

| Setting | Value |
| --- | --- |
| Evaluation | Shuffled 10-fold cross-validation |
| Random seed | `42` |
| Training per fold | Fixed `200` epochs; no internal validation split or early stopping |
| Batch size | `48` |
| Optimizer | Adam, learning rate `1e-4`, weight decay `0` |
| Gradient clipping | Global norm `1.0` |
| SNP chunk size | `32` |
| Latent allocation | Performer `32` + dense shortcut `16` = `48` dimensions |
| Predictor | `48 → 64 → 1`, with BatchNorm and GELU; no predictor dropout |
| Loss weights | Prediction `1.50`, reconstruction `0.30`, KL `0.80`; Pearson term `0.0` |

These describe the supplied configuration, not a verified parameter specification for every manuscript experiment. Use the active constants in the script when checking the loss weights; some historical comments contain different values.

By default, `RUN_ALL_6_TRAITS = True`. To run **grain length only**, set these two constants in `GenoPerfVAE.py` before execution:

```python
RUN_ALL_6_TRAITS = False
SCREEN_TRAITS = ["length"]
```

For that configuration, only `2000gene.pkl` and `2000_3_phe_length.txt` are needed. This entry point is configured through source constants rather than a unified command-line interface.

### 3. Start training

```bash
python GenoPerfVAE.py
```

The script selects CUDA when available and otherwise uses the CPU. Review the final console output and any failed-trait report to confirm that all intended traits completed.

### 4. Inspect the outputs

Results are written to an automatically created directory beside the main script matching `results_*wheat2000*`, with subdirectories for each trait/configuration. Exact filename prefixes are determined by the script.

| Output | Contents |
| --- | --- |
| `*_FoldResults.csv` | Metrics and execution statistics for each fold |
| `*_Predictions.csv` | Observed and out-of-fold predicted phenotype values |
| `*_10FoldMembership.csv` | Sample membership in the cross-validation folds |
| `*_Summary.csv` | Per-configuration or combined trait summaries |
| `Wheat2000_SNP_Order.csv` | Positional SNP-index order used by the benchmark |
| `*_FailedTraits.csv` | Traits that raised errors, when applicable |

Reported prediction metrics include Pearson correlation, coefficient of determination (`R2`), MSE, RMSE, normalized RMSE, and MAE. In the main script, normalized RMSE divides RMSE by the sample standard deviation of the observed values in the evaluated set. `R2` is not the square of Pearson correlation.

## Comparison models

| Script | Component | Configuration note |
| --- | --- | --- |
| `Performer.py` | Performer-only comparison | Uses multiple Wheat2000 phenotype files by default. |
| `CLCNet.py` | CLCNet adaptation | Uses LightGBM feature selection and historical absolute input paths. |
| `DeepGS.py` | DeepGS adaptation | Default fold-parity setting runs half of the ten folds. |
| `DNNGP.py` | DNNGP adaptation | Default fold-parity setting runs half of the ten folds. |
| `GBLUP.py` | Genomic BLUP | Requires local genotype and phenotype inputs. |
| `MeNet.py` | MeNet adaptation | Current entry point selects hardness and uses absolute paths. |
| `SVR.py` | Support vector regression | Requires local trait inputs and path configuration. |
| `VMGP.py` | VMGP-S comparison | Requires the separately supplied `vmgp_model.py` module. |
| `XGBoost.py` | XGBoost benchmark | Uses GPU-oriented defaults; check its active dataset configuration. |

For the provided DeepGS/DNNGP sharded execution mode, both `FOLD_PARITY=0` and `FOLD_PARITY=1` are required to cover all ten folds. Review each script's paths, phenotype selection, preprocessing, and training settings before running comparisons. A shared fold seed alone does not establish identical tuning budgets or evaluation procedures.

## Rice299 plant-height SNP prioritization

See [the dedicated analysis README](rice299_ph_snp_prioritization/README.md) for the detailed workflow associated with manuscript Section 3.7.

The reported workflow fits a full-cohort model to 299 accessions, screens eligible SNPs using input gradients, and evaluates the selected top 150 markers with exact single-marker changes between observed numeric genotype states `-1` and `+1`, holding model parameters fixed. Additional analyses include observed-phenotype LOCO-LMM GWAS, MSU6-to-MSU7 coordinate auditing, GO Slim analysis, and a GradientSHAP comparison.

These are **exploratory, in-sample model-sensitivity analyses**. The full-cohort fit is separate from cross-validation estimates of prediction performance. Single-marker perturbations may violate linkage or haplotype constraints, and the supplementary analyses do not establish causal variants or experimentally validated biological effects.

## Reproducibility and current limitations

- Several comparison and analysis scripts retain absolute Windows paths. Configure them for the local machine before execution.
- `VMGP.py` has an external local-module dependency that is absent from this archive.
- The Rice299 analysis README describes four figure-plotting scripts that are absent from this archive. A generator for the Figure 9C GWAS null-distribution table and the construction of the merged plotting table also remain to be documented or supplied.
- Dependency versions are not locked, and full training/result reproduction has not been verified from a fresh clone. Syntax checks alone do not establish reproducibility.

Record input provenance, sample/SNP ordering, fold assignments, preprocessing, random seeds, active hyperparameters, dependency versions, and hardware when reproducing an experiment. More details are available in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [CODE_AUDIT_REPORT.md](CODE_AUDIT_REPORT.md).

## Citation and licensing

When using this work, cite the GenoPerfVAE manuscript once its bibliographic metadata is available, the original data-source publications, and the original methods used for comparison models.

`CITATION.cff.template` and `LICENSE-MIT.template` are templates. They are not finalized citation metadata or a finalized repository-wide license. Confirm the project license and the provenance/licenses of reused implementations before completing the release. Dataset access conditions are independent of software licensing.
