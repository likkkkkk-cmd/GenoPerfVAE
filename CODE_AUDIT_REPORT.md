# Static code and data compatibility audit (16 October 2026)

This report is based on direct inspection of the **ten root-level `.py` files supplied for review**. All ten parsed successfully under Python AST; **model training and result replication were not executed**. The two Wheat2000 local file names were shown in a screenshot; **no dataset bytes were supplied or checked**. Neither file is redistributed through this repository.

## Critical issues before claiming a runnable example

| Model script | Observed issue | Action required |
| --- | --- | --- |
| `GenoPerfVAE.py` | `RUN_ALL_6_TRAITS = True`; `SCREEN_TRAITS` is not limited to length; input-directory resolver does not inspect `example_data/`. Its introductory comment says reconstruction/KL weights **0.5/0.8**, but the active `RECON_WEIGHT` / `KL_WEIGHT` constants are **0.30/0.80**. | For a locally prepared Wheat2000 length-only experiment, add `example_data` to the resolver and set `RUN_ALL_6_TRAITS=False; SCREEN_TRAITS=["length"]`. Correct stale comment, but **do not silently change the actual training weights**. |
| `Performer.py` | Data resolver searches `data/` beside or above the script and the default loop covers six traits. | Provide data-path override and restrict the trait loop to length. |
| `CLCNet.py` | `GENO_FILE` and `PHENOTYPE_DIR` are hard-coded to `E:\VGMP\wheat2000\SNP_origin`. | Change both to repository-relative `example_data/`; note chromosome-aware feature selection may require an additional chromosome-label file. |
| `DeepGS.py` | Assumes `data/` and all six phenotype files, resolving them before training. `FOLD_PARITY=0` by default trains only folds 1,3,5,7,9. | Restrict `PHENOTYPE_SPECS` to length; adapt directory and run both fold parities to reproduce all 10 folds. |
| `DNNGP.py` | Same six-file `data/` assumption and split fold-parity behavior as DeepGS. | Restrict to length; adapt directory and run both fold parities. |
| `GBLUP.py` | Searches `data/`/historical directories, not `example_data/` by default; loops over six traits. | Set data directory and restrict to length (missing traits are handled inside its loop). |
| `MeNet.py` | The `__main__` block is currently configured only for **hardness**, using an absolute `E:\...` path; **length is commented out**. | Switch the active phenotype to length and update both paths. |
| `SVR.py` | Searches `data/`/historical directories and loops over six traits. | Set data directory and restrict to length if demonstrating only the public files. |
| `VMGP.py` | `from vmgp_model import VMGP` requires a local module **not included among the 10 uploaded files**. `PHENO_FILE` points to Prot while `TRAIT_NAME` is `tkw`; both use absolute paths. | Restore the compatible upstream `vmgp_model.py` under its correct license, or document VMGP as a non-runnable reference script. Set the matching phenotype and trait label. |
| `XGBoost.py` | Introductory docstring incorrectly refers to **Wheat599/Env1–4**; the actual main code uses `2000gene.pkl` and six Wheat2000 phenotypes; default `device='cuda'`. | Correct the stale description, choose compatible device and set the example path/trait. |

## What user-obtained local Wheat2000 inputs support

`example_data/2000gene.pkl` and `example_data/2000_3_phe_length.txt` are expected **local-only inputs** for a Wheat2000 length experiment after per-script configuration. They are NOT uploaded to GitHub and cannot produce six-trait summaries, the other three crop datasets, or the rice SNP-prioritization analysis.

The scripts generally pair Wheat2000 genotype and phenotype **by row order**. A matching row count is necessary, but is **not enough** to prove that the two files represent the same accession order. Check against the original provenance records and use `validate_example_data.py` for basic format checks.

## Dependencies

The ten scripts import `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `performer_pytorch`, `lightgbm`, `xgboost`, and `tqdm` (specific needs vary). `vmgp_model` is a *local source file*, not a pip package. The SNP-analysis subdirectory additionally uses `captum`, `matplotlib` and possibly other packages documented there. The `requirements*.txt` files supplied here are **unlocked inventories**, not verified exact versions.

## Additional items to verify before resubmission

- Run an actual length-only experiment from a freshly cloned repo after making paths and trait selection portable.
- Check any model-specific missing data and preprocessing rules; identical fold seeds do not imply identical training/validation processes.
- Verify Rice299-PH analysis scripts, their dependencies and any derived GWAS-null tables independently of the two Wheat2000 example inputs.
- Add baseline source citations and inspect redistribution licenses, especially the separate VMGP implementation.
- Keep all raw and processed datasets, including `.pkl` and phenotype `.txt` files, out of GitHub. Download them independently from the source papers in manuscript Section 2.1.
