# Data availability and access

## No datasets are redistributed

**This GitHub repository contains source code and documentation only; it does not provide or redistribute any genotype or phenotype data, whether raw or processed.** We have not obtained authorization from the original data providers to redistribute their datasets or derivative data files. For that reason, all dataset files are omitted from the public repository.

This also applies to the previously discussed Wheat2000 local files:

- `2000gene.pkl` — a processed Wheat2000 genotype matrix;
- `2000_3_phe_length.txt` — the corresponding grain-length phenotype input.

These filenames are mentioned only to document the expected **local input format**. **Neither file is uploaded to this repository.** The optional `example_data/README.md` explains how users can configure a local directory without committing dataset files.

## How to obtain the original research datasets

The datasets used for GenoPerfVAE are described in **Section 2.1 (Datasets) of the manuscript**, with citations to their original source publications and the associated data resources. Researchers should consult those papers and obtain the datasets **directly from the original authors, repositories, or officially identified data providers** as permitted by their access and use conditions.

The study includes:

| Dataset | Crop | Access guidance |
| --- | --- | --- |
| Rice299 | Rice | Refer to the original rice-population publication and the subsequent 299-accession benchmark source identified in manuscript §2.1. The Spindel et al. source population and the Rice299 subset are not identical data files. |
| Wheat2000 | Wheat | Refer to the original Wheat2000 source study and data source identified in manuscript §2.1. |
| Wheat599 | Wheat | Refer to the original Wheat599 source study and data source identified in manuscript §2.1. |
| Maize1404 | Maize | Refer to the original Maize1404 source study and data source identified in manuscript §2.1. |

Availability remains subject to the original providers' terms. **Public accessibility does not, by itself, establish permission to republish data in a new GitHub repository.** No new download URL or redistribution license is asserted here.

## Preparing the local inputs

After obtaining original datasets, readers should follow the dataset-specific processing and alignment procedures described in the manuscript and implemented in the relevant scripts. Downloaded source tables may not be identical to the processed model inputs. In particular:

- Preserve genotype–phenotype sample correspondence and the required sample/marker order.
- Follow the genotype missing-value coding of each dataset. Do not globally treat `-1` as missing if the script considers it a valid genotype state.
- Check which imputation, feature filtering, or standardization steps must be fit **within** each training fold to avoid data leakage.
- For the rice Section 3.7 analysis, obtain the correct Rice299 cohort and the original reference sequences, HapMap and MSU6/MSU7 annotation sources separately.
- When using pickled inputs, load `.pkl` files only from sources you trust.

## Scope of reproducibility

The repository documents the computational methods but **does not contain a complete ready-to-run data bundle**. Users need to download and prepare source data separately and may need to adjust local file paths in the research scripts. Additional derived intermediates and script portability issues are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [rice299_ph_snp_prioritization/README.md](rice299_ph_snp_prioritization/README.md).
