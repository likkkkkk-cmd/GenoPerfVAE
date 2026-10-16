# Local Wheat2000 input directory (no data files distributed)

**This folder contains documentation only in the public GitHub repository.** Neither genotype nor phenotype datasets are included.

The original datasets are identified through the papers and associated resources cited in **Section 2.1 (Datasets)** of the GenoPerfVAE manuscript. Please obtain them directly from the original providers, subject to the providers' conditions. The repository authors have not obtained permission to redistribute the raw or processed datasets.

For an optional **local** Wheat2000 grain-length check, once you have legitimately obtained/prepared the correct version and verified the sample order, place these files in this folder *on your own computer*:

```text
example_data/
├── README.md                 # committed documentation
├── 2000gene.pkl              # LOCAL ONLY; ignored by Git
└── 2000_3_phe_length.txt     # LOCAL ONLY; ignored by Git
```

From the repository root, run `python validate_example_data.py --data-dir example_data` to check the numeric shape and row count. This does not validate sample identity or run training. Most model entry points still expect their historic data paths and/or the full six Wheat2000 traits, so adjust the paths/trait selection before running them. Load `.pkl` files only from trusted providers.
