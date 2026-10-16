# -*- coding: utf-8 -*-
r"""
SVR_Wheat2000_6Traits_10Fold_FairComparison.py

Support Vector Regression (SVR) benchmark on Wheat2000.

TRAITS
------
tkw
testw
length
width
hardness
Prot

DATA
----
Genotype:
    2000gene.pkl

Phenotypes:
    2000_1_phe_tkw.txt
    2000_2_phe_testw.txt
    2000_3_phe_length.txt
    2000_4_phe_width.txt
    2000_5_phe_hardness.txt
    2000_6_phe_Prot.txt

Wheat2000 samples are aligned by row order, following the current benchmark
loader convention used in the other Wheat2000 scripts.

CROSS-VALIDATION
----------------
KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

Within every fold:
    X:
        SimpleImputer(strategy="most_frequent") fitted on TRAIN only
        StandardScaler fitted on TRAIN only

    y:
        StandardScaler fitted on TRAIN only

    Test:
        transform only with TRAIN-fitted preprocessing

Predictions are inverse-transformed to original phenotype units before metrics.

SVR
---
Fixed hyperparameters:
    kernel="rbf"
    C=1.0
    epsilon=0.1
    gamma="scale"
    cache_size=4000 MB

No hyperparameter tuning is performed.

UNIFIED METRICS
---------------
Per fold:
    Pearson
    R2
    MSE
    RMSE
    nRMSE = RMSE / SD(y_test, ddof=1)
    MAE

Computational:
    Fit_Time_s
    Inference_Time_s
    Inference_ms_per_sample
    Support_Vectors

Summary:
    mean ± sample SD across 10 folds
    pooled OOF metrics

OUTPUT
------
Per trait:
    AlignedSamples.csv
    FoldResults.csv
    Predictions.csv
    10FoldMembership.csv
    Summary.csv

Combined:
    SVR_Wheat2000_6Traits_10Fold_Summary.csv
"""

from __future__ import annotations

import gc
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================

SEED = 42
N_SPLITS = 10

TRAIT_FILES = {
    "tkw": "2000_1_phe_tkw.txt",
    "testw": "2000_2_phe_testw.txt",
    "length": "2000_3_phe_length.txt",
    "width": "2000_4_phe_width.txt",
    "hardness": "2000_5_phe_hardness.txt",
    "Prot": "2000_6_phe_Prot.txt",
}

SVR_KERNEL = "rbf"
SVR_C = 1.0
SVR_EPSILON = 0.1
SVR_GAMMA = "scale"
SVR_CACHE_SIZE_MB = 4000


# =============================================================================
# 2. Path resolution
# =============================================================================

def resolve_wheat2000_data_dir(
    base_dir,
):
    base_dir = Path(
        base_dir
    )

    candidates = [
        base_dir / "data",
        base_dir.parent / "data",
        base_dir.parent / "data" / "wheat2000",
        Path(r"E:\VGMP\wheat2000"),
    ]

    for data_dir in candidates:
        genotype_file = (
            data_dir
            / "2000gene.pkl"
        )

        if genotype_file.exists():
            print(
                "Resolved Wheat2000 data directory:",
                data_dir
            )
            return data_dir

    raise FileNotFoundError(
        "Could not locate Wheat2000 genotype file 2000gene.pkl."
    )


def resolve_trait_file(
    data_dir,
    trait,
    preferred_name,
):
    data_dir = Path(
        data_dir
    )

    preferred = (
        data_dir
        / preferred_name
    )

    if preferred.exists():
        return preferred

    trait_index = {
        "tkw": 1,
        "testw": 2,
        "length": 3,
        "width": 4,
        "hardness": 5,
        "Prot": 6,
    }[
        trait
    ]

    pattern = (
        f"2000_{trait_index}_phe_*.txt"
    )

    matches = sorted(
        data_dir.glob(
            pattern
        )
    )

    if len(matches) == 1:
        print(
            f"{trait}: using phenotype file "
            f"{matches[0].name}"
        )
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find phenotype file for {trait}. "
            f"Expected {preferred_name!r} or a unique {pattern!r}."
        )

    raise RuntimeError(
        f"Multiple phenotype files match {pattern!r}: "
        f"{[p.name for p in matches]}"
    )


# =============================================================================
# 3. Data loading
# =============================================================================

def load_wheat2000_genotype(
    genotype_file,
):
    obj = pd.read_pickle(
        genotype_file
    )

    if isinstance(
        obj,
        pd.DataFrame,
    ):
        X = obj.values

    elif isinstance(
        obj,
        pd.Series,
    ):
        X = obj.to_numpy()

    else:
        X = np.asarray(
            obj
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D genotype matrix, got shape {X.shape}."
        )

    sample_rows = np.arange(
        X.shape[
            0
        ],
        dtype=int,
    )

    print("=" * 90)
    print("Wheat2000 genotype")
    print("=" * 90)
    print(
        f"Samples       : "
        f"{X.shape[0]}"
    )
    print(
        f"SNPs          : "
        f"{X.shape[1]}"
    )
    print(
        f"True NaN      : "
        f"{int(np.isnan(X).sum())}"
    )
    print(
        f"Genotype range: "
        f"{float(np.nanmin(X)):.6f} "
        f"to "
        f"{float(np.nanmax(X)):.6f}"
    )

    return (
        np.ascontiguousarray(
            X,
            dtype=np.float32,
        ),
        sample_rows,
    )


def first_numeric_from_line(
    line,
):
    line = line.strip()

    if not line:
        return None

    tokens = re.split(
        r"[\s,]+",
        line,
    )

    for token in tokens:
        if not token:
            continue

        try:
            return float(
                token
            )

        except ValueError:
            continue

    return None


def load_wheat2000_phenotype(
    phenotype_file,
):
    values = []

    with open(
        phenotype_file,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:
        for line in handle:
            value = first_numeric_from_line(
                line
            )

            if value is not None:
                values.append(
                    value
                )

    if not values:
        raise ValueError(
            f"No numeric phenotype values found in {phenotype_file}."
        )

    y = np.asarray(
        values,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(
            y
        )
    ):
        raise ValueError(
            f"Phenotype contains NaN/Inf: {phenotype_file}"
        )

    return y


# =============================================================================
# 4. Unified metrics
# =============================================================================

def safe_pearson(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if (
        len(y_true) < 2
        or np.std(y_true) < 1e-12
        or np.std(y_pred) < 1e-12
    ):
        return np.nan

    return float(
        stats.pearsonr(
            y_true,
            y_pred,
        )[0]
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    mse = float(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    y_sd = (
        float(
            np.std(
                y_true,
                ddof=1,
            )
        )
        if len(y_true) > 1
        else np.nan
    )

    return {
        "Pearson":
            safe_pearson(
                y_true,
                y_pred,
            ),
        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),
        "MSE":
            mse,
        "RMSE":
            rmse,
        "nRMSE":
            (
                float(
                    rmse
                    / y_sd
                )
                if (
                    np.isfinite(
                        y_sd
                    )
                    and y_sd
                    > 1e-12
                )
                else np.nan
            ),
        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),
    }


# =============================================================================
# 5. One trait
# =============================================================================

def run_trait(
    X,
    y,
    sample_rows,
    trait,
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame({
        "Sample_Row_0Based":
            sample_rows,
        "Observed":
            y,
    }).to_csv(
        output_dir
        / (
            f"Wheat2000_"
            f"{trait}_"
            "AlignedSamples.csv"
        ),
        index=False,
    )

    kf = KFold(
        n_splits=
            N_SPLITS,
        shuffle=True,
        random_state=
            SEED,
    )

    fold_rows = []
    prediction_rows = []
    membership_rows = []

    print(
        "\n"
        + "#" * 100
    )
    print(
        f"SVR | Wheat2000 | {trait}"
    )
    print(
        f"10-fold CV | "
        f"RBF | C={SVR_C} | "
        f"epsilon={SVR_EPSILON} | "
        f"gamma={SVR_GAMMA}"
    )
    print(
        "#" * 100
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        kf.split(
            X
        ),
        start=1,
    ):
        print(
            "\n"
            + "-" * 100
        )
        print(
            f"{trait} | "
            f"Fold {fold}/{N_SPLITS}"
        )
        print(
            "-" * 100
        )

        # ---------------------------------------------------------
        # Train-only genotype imputation.
        # ---------------------------------------------------------
        imputer = SimpleImputer(
            strategy=
                "most_frequent",
            keep_empty_features=
                True,
        )

        X_train_imp = (
            imputer
            .fit_transform(
                X[
                    train_idx
                ]
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        X_test_imp = (
            imputer
            .transform(
                X[
                    test_idx
                ]
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        if (
            X_train_imp.shape[
                1
            ]
            != X.shape[
                1
            ]
        ):
            raise RuntimeError(
                "SimpleImputer changed SNP dimensionality."
            )

        # ---------------------------------------------------------
        # Train-only feature standardization.
        # ---------------------------------------------------------
        scaler_x = StandardScaler()

        X_train = (
            scaler_x
            .fit_transform(
                X_train_imp
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        X_test = (
            scaler_x
            .transform(
                X_test_imp
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        # ---------------------------------------------------------
        # Train-only phenotype standardization.
        # ---------------------------------------------------------
        scaler_y = StandardScaler()

        y_train_scaled = (
            scaler_y
            .fit_transform(
                y[
                    train_idx
                ]
                .reshape(
                    -1,
                    1,
                )
            )
            .reshape(-1)
        )

        y_test = (
            y[
                test_idx
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        model = SVR(
            kernel=
                SVR_KERNEL,
            C=
                SVR_C,
            epsilon=
                SVR_EPSILON,
            gamma=
                SVR_GAMMA,
            cache_size=
                SVR_CACHE_SIZE_MB,
        )

        fit_start = (
            time.perf_counter()
        )

        model.fit(
            X_train,
            y_train_scaled,
        )

        fit_time = (
            time.perf_counter()
            - fit_start
        )

        infer_start = (
            time.perf_counter()
        )

        y_pred_scaled = model.predict(
            X_test
        )

        inference_time = (
            time.perf_counter()
            - infer_start
        )

        y_pred = (
            scaler_y
            .inverse_transform(
                y_pred_scaled
                .reshape(
                    -1,
                    1,
                )
            )
            .reshape(-1)
        )

        metrics = calculate_metrics(
            y_test,
            y_pred,
        )

        support_vectors = int(
            np.sum(
                model.n_support_
            )
        )

        fold_row = {
            "Model":
                "SVR",
            "Dataset":
                "Wheat2000",
            "Trait":
                trait,
            "Fold":
                fold,
            "Train_N":
                len(
                    train_idx
                ),
            "Test_N":
                len(
                    test_idx
                ),
            "SNP_Dimension_D":
                X.shape[
                    1
                ],
            **metrics,
            "Observed_Mean":
                float(
                    np.mean(
                        y_test
                    )
                ),
            "Prediction_Mean":
                float(
                    np.mean(
                        y_pred
                    )
                ),
            "Mean_Bias_PredMinusObs":
                float(
                    np.mean(
                        y_pred
                        - y_test
                    )
                ),
            "Observed_SD":
                float(
                    np.std(
                        y_test,
                        ddof=1,
                    )
                ),
            "Prediction_SD":
                float(
                    np.std(
                        y_pred,
                        ddof=1,
                    )
                ),
            "Kernel":
                SVR_KERNEL,
            "C":
                SVR_C,
            "Epsilon":
                SVR_EPSILON,
            "Gamma":
                SVR_GAMMA,
            "Support_Vectors":
                support_vectors,
            "Fit_Time_s":
                float(
                    fit_time
                ),
            "Inference_Time_s":
                float(
                    inference_time
                ),
            "Inference_ms_per_sample":
                float(
                    inference_time
                    * 1000.0
                    / len(
                        test_idx
                    )
                ),
            "Seed":
                SEED,
            "X_Processing":
                (
                    "TrainOnlyMostFrequentImputer"
                    "+TrainOnlyStandardScaler"
                ),
            "Y_Processing":
                (
                    "TrainOnlyStandardScaler"
                    "+InverseTransformForMetrics"
                ),
        }

        fold_rows.append(
            fold_row
        )

        for (
            idx,
            obs,
            pred,
        ) in zip(
            test_idx,
            y_test,
            y_pred,
        ):
            prediction_rows.append({
                "Trait":
                    trait,
                "Fold":
                    fold,
                "Aligned_Row_0Based":
                    int(
                        idx
                    ),
                "Original_Row_0Based":
                    int(
                        sample_rows[
                            idx
                        ]
                    ),
                "Observed":
                    float(
                        obs
                    ),
                "Predicted":
                    float(
                        pred
                    ),
            })

        for idx in train_idx:
            membership_rows.append({
                "Trait":
                    trait,
                "Fold":
                    fold,
                "Aligned_Row_0Based":
                    int(
                        idx
                    ),
                "Original_Row_0Based":
                    int(
                        sample_rows[
                            idx
                        ]
                    ),
                "Set":
                    "Train",
            })

        for idx in test_idx:
            membership_rows.append({
                "Trait":
                    trait,
                "Fold":
                    fold,
                "Aligned_Row_0Based":
                    int(
                        idx
                    ),
                "Original_Row_0Based":
                    int(
                        sample_rows[
                            idx
                        ]
                    ),
                "Set":
                    "Test",
            })

        print(
            f"Pearson="
            f"{metrics['Pearson']:.4f} | "
            f"R2="
            f"{metrics['R2']:.4f} | "
            f"RMSE="
            f"{metrics['RMSE']:.4f} | "
            f"nRMSE="
            f"{metrics['nRMSE']:.4f} | "
            f"MAE="
            f"{metrics['MAE']:.4f} | "
            f"SV="
            f"{support_vectors} | "
            f"Fit="
            f"{fit_time:.2f}s"
        )

        pd.DataFrame(
            fold_rows
        ).to_csv(
            output_dir
            / (
                f"SVR_Wheat2000_"
                f"{trait}_"
                "FoldResults_CHECKPOINT.csv"
            ),
            index=False,
        )

        del (
            model,
            imputer,
            scaler_x,
            scaler_y,
            X_train_imp,
            X_test_imp,
            X_train,
            X_test,
        )

        gc.collect()

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = (
        pd.DataFrame(
            prediction_rows
        )
        .sort_values(
            "Aligned_Row_0Based"
        )
    )

    membership_df = pd.DataFrame(
        membership_rows
    )

    fold_df.to_csv(
        output_dir
        / (
            f"SVR_Wheat2000_"
            f"{trait}_"
            "FoldResults.csv"
        ),
        index=False,
    )

    pred_df.to_csv(
        output_dir
        / (
            f"SVR_Wheat2000_"
            f"{trait}_"
            "Predictions.csv"
        ),
        index=False,
    )

    membership_df.to_csv(
        output_dir
        / (
            f"SVR_Wheat2000_"
            f"{trait}_"
            "10FoldMembership.csv"
        ),
        index=False,
    )

    summary = {
        "Model":
            "SVR",
        "Dataset":
            "Wheat2000",
        "Trait":
            trait,
        "N":
            len(
                X
            ),
        "SNP_Dimension_D":
            X.shape[
                1
            ],
        "Kernel":
            SVR_KERNEL,
        "C":
            SVR_C,
        "Epsilon":
            SVR_EPSILON,
        "Gamma":
            SVR_GAMMA,
    }

    for metric in (
        "Pearson",
        "R2",
        "MSE",
        "RMSE",
        "nRMSE",
        "MAE",
    ):
        summary[
            f"{metric}_Mean"
        ] = fold_df[
            metric
        ].mean()

        summary[
            f"{metric}_SD"
        ] = fold_df[
            metric
        ].std(
            ddof=1
        )

    oof = calculate_metrics(
        pred_df[
            "Observed"
        ].to_numpy(),
        pred_df[
            "Predicted"
        ].to_numpy(),
    )

    for (
        metric,
        value,
    ) in oof.items():
        summary[
            f"OOF_{metric}"
        ] = value

    summary.update({
        "Fit_Time_Mean_s":
            fold_df[
                "Fit_Time_s"
            ].mean(),
        "Fit_Time_SD_s":
            fold_df[
                "Fit_Time_s"
            ].std(
                ddof=1
            ),
        "Inference_Mean_ms_sample":
            fold_df[
                "Inference_ms_per_sample"
            ].mean(),
        "Inference_SD_ms_sample":
            fold_df[
                "Inference_ms_per_sample"
            ].std(
                ddof=1
            ),
        "Support_Vectors_Mean":
            fold_df[
                "Support_Vectors"
            ].mean(),
        "Support_Vectors_SD":
            fold_df[
                "Support_Vectors"
            ].std(
                ddof=1
            ),
    })

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        output_dir
        / (
            f"SVR_Wheat2000_"
            f"{trait}_"
            "Summary.csv"
        ),
        index=False,
    )

    print(
        "\n"
        + "=" * 96
    )
    print(
        f"{trait} | "
        "10-fold summary"
    )
    print(
        "=" * 96
    )
    print(
        f"Pearson : "
        f"{summary['Pearson_Mean']:.4f} ± "
        f"{summary['Pearson_SD']:.4f}"
    )
    print(
        f"R2      : "
        f"{summary['R2_Mean']:.4f} ± "
        f"{summary['R2_SD']:.4f}"
    )
    print(
        f"RMSE    : "
        f"{summary['RMSE_Mean']:.4f} ± "
        f"{summary['RMSE_SD']:.4f}"
    )
    print(
        f"nRMSE   : "
        f"{summary['nRMSE_Mean']:.4f} ± "
        f"{summary['nRMSE_SD']:.4f}"
    )
    print(
        f"MAE     : "
        f"{summary['MAE_Mean']:.4f} ± "
        f"{summary['MAE_SD']:.4f}"
    )

    return summary_df


# =============================================================================
# 6. Main
# =============================================================================

def main():
    base_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    data_dir = resolve_wheat2000_data_dir(
        base_dir
    )

    (
        X_all,
        sample_rows_all,
    ) = load_wheat2000_genotype(
        data_dir
        / "2000gene.pkl"
    )

    output_root = (
        base_dir
        / (
            "results_svr_"
            "wheat2000_6traits_10fold"
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    failed = []

    for (
        trait,
        preferred_name,
    ) in TRAIT_FILES.items():
        try:
            phenotype_file = resolve_trait_file(
                data_dir=
                    data_dir,
                trait=
                    trait,
                preferred_name=
                    preferred_name,
            )

            y_all = load_wheat2000_phenotype(
                phenotype_file
            )

            if (
                len(
                    y_all
                )
                != len(
                    X_all
                )
            ):
                n_common = min(
                    len(
                        y_all
                    ),
                    len(
                        X_all
                    ),
                )

                print(
                    f"WARNING [{trait}]: "
                    f"genotype N={len(X_all)}, "
                    f"phenotype N={len(y_all)}; "
                    f"using first {n_common} row-aligned samples."
                )

                X_trait = np.ascontiguousarray(
                    X_all[
                        :n_common
                    ],
                    dtype=np.float32,
                )

                y_trait = np.ascontiguousarray(
                    y_all[
                        :n_common
                    ],
                    dtype=np.float32,
                )

                sample_rows = sample_rows_all[
                    :n_common
                ]

            else:
                X_trait = X_all
                y_trait = y_all
                sample_rows = sample_rows_all

            summary_df = run_trait(
                X=
                    X_trait,
                y=
                    y_trait,
                sample_rows=
                    sample_rows,
                trait=
                    trait,
                output_dir=
                    output_root
                    / trait,
            )

            summary_df[
                "Phenotype_File"
            ] = str(
                phenotype_file
            )

            summaries.append(
                summary_df
            )

            pd.concat(
                summaries,
                ignore_index=True,
            ).to_csv(
                output_root
                / (
                    "SVR_Wheat2000_"
                    "6Traits_10Fold_"
                    "Summary_CHECKPOINT.csv"
                ),
                index=False,
            )

            if (
                X_trait
                is not X_all
            ):
                del (
                    X_trait,
                    y_trait,
                    sample_rows,
                )

            gc.collect()

        except Exception as exc:
            print(
                f"[ERROR] "
                f"{trait}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed.append({
                "Trait":
                    trait,
                "Reason":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            })

            gc.collect()

    if summaries:
        combined = pd.concat(
            summaries,
            ignore_index=True,
        )

        combined.to_csv(
            output_root
            / (
                "SVR_Wheat2000_"
                "6Traits_10Fold_"
                "Summary.csv"
            ),
            index=False,
        )

        display_cols = [
            "Trait",
            "N",
            "SNP_Dimension_D",
            "Pearson_Mean",
            "Pearson_SD",
            "R2_Mean",
            "R2_SD",
            "MSE_Mean",
            "MSE_SD",
            "RMSE_Mean",
            "RMSE_SD",
            "nRMSE_Mean",
            "nRMSE_SD",
            "MAE_Mean",
            "MAE_SD",
            "Fit_Time_Mean_s",
            "Inference_Mean_ms_sample",
            "Support_Vectors_Mean",
        ]

        print(
            "\n"
            + "=" * 150
        )
        print(
            "SVR Wheat2000 "
            "6-trait 10-fold summary"
        )
        print(
            "=" * 150
        )

        print(
            combined[
                display_cols
            ].to_string(
                index=False
            )
        )

    if failed:
        pd.DataFrame(
            failed
        ).to_csv(
            output_root
            / (
                "SVR_Wheat2000_"
                "FailedTraits.csv"
            ),
            index=False,
        )


if __name__ == "__main__":
    main()
