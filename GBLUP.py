# -*- coding: utf-8 -*-
r"""
GBLUP_Wheat2000_6Traits_10Fold_FairComparison.py

Leakage-safe Python implementation of GBLUP for Wheat2000.

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

Alignment:
    row order

Phenotype parser:
    first numeric value from each non-empty line.

CROSS-VALIDATION
----------------
KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

For every outer fold:
    1. Fit SNP imputation on TRAIN only.
    2. Estimate marker centering / allele-frequency parameters on TRAIN only.
    3. Build K_train,train and K_test,train using TRAIN parameters.
    4. Estimate lambda = sigma_e^2 / sigma_g^2 by profile REML
       using TRAIN phenotypes only.
    5. Predict held-out TEST phenotypes by GBLUP.
    6. Evaluate in the ORIGINAL phenotype scale.

No phenotype StandardScaler is used.

GRM CODING
----------
The script automatically detects the approximate marker coding from TRAIN only:

A) Approximately [-1, 1]:
       dosage = genotype + 1
       VanRaden scaling

B) Approximately [0, 2]:
       dosage = genotype
       VanRaden scaling

C) Other numeric coding:
       train-centered marker covariance GRM
       denominator = sum of train marker variances

The selected GRM mode is recorded for every fold.

GBLUP MODEL
-----------
    y = 1*mu + g + e

    g ~ N(0, sigma_g^2 K)
    e ~ N(0, sigma_e^2 I)

    lambda = sigma_e^2 / sigma_g^2

For computational efficiency on Wheat2000, K_train is eigendecomposed once
per fold. REML optimization over lambda then uses the eigensystem rather than
performing a new Cholesky factorization for every objective evaluation.

UNIFIED METRICS
---------------
Per fold:
    Pearson
    R2
    MSE
    RMSE
    nRMSE = RMSE / SD(y_test, ddof=1)
    MAE

Additional GBLUP diagnostics:
    Lambda
    Sigma_g2
    Sigma_e2
    H2 = Sigma_g2 / (Sigma_g2 + Sigma_e2)
    Fit_Time_s
    Inference_Time_s
    Inference_ms_per_sample

OUTPUT
------
Per trait:
    AlignedSamples.csv
    FoldResults.csv
    Predictions.csv
    10FoldMembership.csv
    Summary.csv

Combined:
    GBLUP_Wheat2000_6Traits_10Fold_Summary.csv
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
from scipy.optimize import minimize_scalar

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold

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

TRAIT_INDEX = {
    "tkw": 1,
    "testw": 2,
    "length": 3,
    "width": 4,
    "hardness": 5,
    "Prot": 6,
}

LOG_LAMBDA_BOUNDS = (-10.0, 10.0)
EIGEN_EPS = 1e-10


# =============================================================================
# 2. Paths
# =============================================================================

def resolve_wheat2000_data_dir(base_dir):
    base_dir = Path(base_dir)

    candidates = [
        base_dir / "data",
        base_dir.parent / "data",
        base_dir.parent / "data" / "wheat2000",
        Path(r"E:\VGMP\wheat2000"),
    ]

    for data_dir in candidates:
        if (data_dir / "2000gene.pkl").exists():
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
    data_dir = Path(data_dir)

    preferred = (
        data_dir
        / preferred_name
    )

    if preferred.exists():
        return preferred

    pattern = (
        f"2000_{TRAIT_INDEX[trait]}_phe_*.txt"
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
            f"Expected {preferred_name!r} or unique {pattern!r}."
        )

    raise RuntimeError(
        f"Multiple phenotype files match {pattern!r}: "
        f"{[p.name for p in matches]}"
    )


# =============================================================================
# 3. Data loading
# =============================================================================

def load_wheat2000_genotype(genotype_file):
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
            f"Expected a 2D genotype matrix, got {X.shape}."
        )

    sample_rows = np.arange(
        X.shape[0],
        dtype=int,
    )

    print("=" * 92)
    print("Wheat2000 genotype")
    print("=" * 92)
    print(
        "Samples       :",
        X.shape[0],
    )
    print(
        "SNPs          :",
        X.shape[1],
    )
    print(
        "True NaN      :",
        int(
            np.isnan(X).sum()
        ),
    )
    print(
        "Genotype range:",
        float(
            np.nanmin(X)
        ),
        "to",
        float(
            np.nanmax(X)
        ),
    )

    return (
        np.ascontiguousarray(
            X,
            dtype=np.float32,
        ),
        sample_rows,
    )


def first_numeric_from_line(line):
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


def load_wheat2000_phenotype(phenotype_file):
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
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(y)
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
# 5. Train-only preprocessing and GRM
# =============================================================================

def make_imputer():
    try:
        return SimpleImputer(
            strategy=
                "most_frequent",
            keep_empty_features=
                True,
        )
    except TypeError:
        return SimpleImputer(
            strategy=
                "most_frequent",
        )


def detect_grm_mode(
    X_train,
):
    finite = X_train[
        np.isfinite(
            X_train
        )
    ]

    if finite.size == 0:
        raise ValueError(
            "No finite genotype values are available."
        )

    x_min = float(
        np.min(finite)
    )

    x_max = float(
        np.max(finite)
    )

    # Approximately -1/0/1 coding, possibly with fractional imputation.
    if (
        x_min >= -1.05
        and x_max <= 1.05
    ):
        return (
            "VanRaden_Minus1_0_1",
            x_min,
            x_max,
        )

    # Approximately 0/1/2 dosage coding.
    if (
        x_min >= -0.05
        and x_max <= 2.05
    ):
        return (
            "VanRaden_0_1_2",
            x_min,
            x_max,
        )

    return (
        "TrainCenteredMarkerCovariance",
        x_min,
        x_max,
    )


def build_grm_train_test(
    X_train_raw,
    X_test_raw,
):
    """
    Fit imputation and GRM parameters on TRAIN only.

    Returns
    -------
    K_train       : n_train x n_train
    K_test_train  : n_test  x n_train
    diagnostics   : dict
    """
    imputer = make_imputer()

    X_train = (
        imputer
        .fit_transform(
            X_train_raw
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    X_test = (
        imputer
        .transform(
            X_test_raw
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    if (
        X_train.shape[1]
        != X_test.shape[1]
    ):
        raise RuntimeError(
            "Train/test SNP dimensions differ after imputation."
        )

    (
        mode,
        train_min,
        train_max,
    ) = detect_grm_mode(
        X_train
    )

    if mode == "VanRaden_Minus1_0_1":
        M_train = (
            X_train
            + 1.0
        )

        M_test = (
            X_test
            + 1.0
        )

        p = np.mean(
            M_train,
            axis=0,
            dtype=np.float64,
        ) / 2.0

        p = np.clip(
            p,
            0.0,
            1.0,
        )

        marker_mean = (
            2.0
            * p
        ).astype(
            np.float32
        )

        Z_train = (
            M_train
            - marker_mean
        )

        Z_test = (
            M_test
            - marker_mean
        )

        denominator = float(
            2.0
            * np.sum(
                p
                * (1.0 - p)
            )
        )

        mean_p = float(
            np.mean(p)
        )

    elif mode == "VanRaden_0_1_2":
        M_train = X_train
        M_test = X_test

        p = np.mean(
            M_train,
            axis=0,
            dtype=np.float64,
        ) / 2.0

        p = np.clip(
            p,
            0.0,
            1.0,
        )

        marker_mean = (
            2.0
            * p
        ).astype(
            np.float32
        )

        Z_train = (
            M_train
            - marker_mean
        )

        Z_test = (
            M_test
            - marker_mean
        )

        denominator = float(
            2.0
            * np.sum(
                p
                * (1.0 - p)
            )
        )

        mean_p = float(
            np.mean(p)
        )

    else:
        marker_mean = np.mean(
            X_train,
            axis=0,
            dtype=np.float64,
        ).astype(
            np.float32
        )

        Z_train = (
            X_train
            - marker_mean
        )

        Z_test = (
            X_test
            - marker_mean
        )

        denominator = float(
            np.sum(
                np.var(
                    X_train,
                    axis=0,
                    ddof=1,
                    dtype=np.float64,
                )
            )
        )

        mean_p = np.nan

    if (
        not np.isfinite(
            denominator
        )
        or denominator
        <= 1e-12
    ):
        # Defensive fallback using train marker variance.
        denominator = float(
            np.sum(
                np.var(
                    Z_train,
                    axis=0,
                    ddof=1,
                    dtype=np.float64,
                )
            )
        )

        mode = (
            mode
            + "_VarianceFallback"
        )

    if (
        not np.isfinite(
            denominator
        )
        or denominator
        <= 1e-12
    ):
        raise ValueError(
            "Unable to construct a valid GRM denominator."
        )

    # Compute with float32 marker matrices for memory efficiency.
    # Final K matrices are promoted to float64 for REML.
    K_train = (
        Z_train
        @ Z_train.T
    ) / denominator

    K_test_train = (
        Z_test
        @ Z_train.T
    ) / denominator

    K_train = np.asarray(
        (
            K_train
            + K_train.T
        )
        / 2.0,
        dtype=np.float64,
    )

    K_test_train = np.asarray(
        K_test_train,
        dtype=np.float64,
    )

    diagnostics = {
        "GRM_Scaling":
            mode,
        "GRM_Denominator":
            denominator,
        "Train_Genotype_Min":
            train_min,
        "Train_Genotype_Max":
            train_max,
        "Mean_Train_Allele_Frequency":
            mean_p,
        "Train_NaN_Before_Imputation":
            int(
                np.isnan(
                    X_train_raw
                ).sum()
            ),
        "Test_NaN_Before_Imputation":
            int(
                np.isnan(
                    X_test_raw
                ).sum()
            ),
    }

    return (
        K_train,
        K_test_train,
        diagnostics,
    )


# =============================================================================
# 6. Fast profile REML using one eigendecomposition per fold
# =============================================================================

def prepare_reml_eigensystem(
    K_train,
    y_train,
):
    """
    K = U diag(d) U'

    All quantities needed for the one-dimensional lambda optimization are
    projected into the eigenbasis once.
    """
    K_train = np.asarray(
        K_train,
        dtype=np.float64,
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(-1)

    eigvals, U = np.linalg.eigh(
        K_train
    )

    # Numerical roundoff can create tiny negative eigenvalues for a PSD GRM.
    eigvals = np.maximum(
        eigvals,
        0.0,
    )

    one = np.ones(
        len(y_train),
        dtype=np.float64,
    )

    Ut_y = (
        U.T
        @ y_train
    )

    Ut_one = (
        U.T
        @ one
    )

    return {
        "eigvals":
            eigvals,
        "U":
            U,
        "Ut_y":
            Ut_y,
        "Ut_one":
            Ut_one,
    }


def _profile_reml_loglambda_eigen(
    log_lambda,
    eigvals,
    Ut_y,
    Ut_one,
):
    n = len(
        Ut_y
    )

    if n <= 2:
        return np.inf

    lam = float(
        np.exp(
            log_lambda
        )
    )

    denom = (
        eigvals
        + lam
        + EIGEN_EPS
    )

    inv_diag = (
        1.0
        / denom
    )

    xt_vinv_x = float(
        np.sum(
            Ut_one
            * Ut_one
            * inv_diag
        )
    )

    if (
        not np.isfinite(
            xt_vinv_x
        )
        or xt_vinv_x
        <= 1e-14
    ):
        return np.inf

    xt_vinv_y = float(
        np.sum(
            Ut_one
            * Ut_y
            * inv_diag
        )
    )

    mu_hat = (
        xt_vinv_y
        / xt_vinv_x
    )

    Ut_resid = (
        Ut_y
        - mu_hat
        * Ut_one
    )

    q = float(
        np.sum(
            Ut_resid
            * Ut_resid
            * inv_diag
        )
    )

    df = (
        n - 1
    )

    sigma_g2 = (
        q / df
    )

    if (
        not np.isfinite(
            sigma_g2
        )
        or sigma_g2
        <= 1e-14
    ):
        return np.inf

    logdet_A = float(
        np.sum(
            np.log(
                denom
            )
        )
    )

    objective = (
        df
        * np.log(
            sigma_g2
        )
        + logdet_A
        + np.log(
            xt_vinv_x
        )
    )

    if not np.isfinite(
        objective
    ):
        return np.inf

    return float(
        objective
    )


def estimate_reml_components(
    K_train,
    y_train,
):
    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(-1)

    eig = prepare_reml_eigensystem(
        K_train=
            K_train,
        y_train=
            y_train,
    )

    result = minimize_scalar(
        _profile_reml_loglambda_eigen,
        bounds=
            LOG_LAMBDA_BOUNDS,
        args=(
            eig["eigvals"],
            eig["Ut_y"],
            eig["Ut_one"],
        ),
        method="bounded",
        options={
            "xatol": 1e-5,
            "maxiter": 300,
        },
    )

    if not result.success:
        print(
            "WARNING: REML optimizer did not report success:",
            result.message
        )

    lam = float(
        np.exp(
            float(
                result.x
            )
        )
    )

    denom = (
        eig["eigvals"]
        + lam
        + EIGEN_EPS
    )

    inv_diag = (
        1.0
        / denom
    )

    xt_vinv_x = float(
        np.sum(
            eig["Ut_one"]
            * eig["Ut_one"]
            * inv_diag
        )
    )

    xt_vinv_y = float(
        np.sum(
            eig["Ut_one"]
            * eig["Ut_y"]
            * inv_diag
        )
    )

    mu_hat = (
        xt_vinv_y
        / xt_vinv_x
    )

    Ut_resid = (
        eig["Ut_y"]
        - mu_hat
        * eig["Ut_one"]
    )

    alpha_eigen = (
        Ut_resid
        * inv_diag
    )

    alpha = (
        eig["U"]
        @ alpha_eigen
    )

    q = float(
        np.sum(
            Ut_resid
            * Ut_resid
            * inv_diag
        )
    )

    sigma_g2 = float(
        q
        / (
            len(
                y_train
            )
            - 1
        )
    )

    sigma_e2 = float(
        lam
        * sigma_g2
    )

    total_var = (
        sigma_g2
        + sigma_e2
    )

    h2 = (
        float(
            sigma_g2
            / total_var
        )
        if (
            np.isfinite(
                total_var
            )
            and total_var
            > 1e-14
        )
        else np.nan
    )

    return {
        "Mu":
            float(
                mu_hat
            ),
        "Lambda":
            lam,
        "Sigma_g2":
            sigma_g2,
        "Sigma_e2":
            sigma_e2,
        "H2":
            h2,
        "REML_Objective":
            float(
                result.fun
            ),
        "REML_Success":
            bool(
                result.success
            ),
        "Alpha":
            np.asarray(
                alpha,
                dtype=np.float64,
            ),
    }


# =============================================================================
# 7. One trait
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
        "Aligned_Row_0Based":
            np.arange(
                len(X),
                dtype=int,
            ),
        "Original_Row_0Based":
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
        + "#" * 108
    )
    print(
        f"GBLUP | Wheat2000 | {trait}"
    )
    print(
        "10-fold CV | "
        "train-only imputation + train-only GRM parameters + REML"
    )
    print(
        "#" * 108
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
            + "-" * 108
        )
        print(
            f"{trait} | "
            f"Fold {fold}/{N_SPLITS}"
        )
        print(
            "-" * 108
        )

        X_train_raw = X[
            train_idx
        ]

        X_test_raw = X[
            test_idx
        ]

        y_train = y[
            train_idx
        ].astype(
            np.float64,
            copy=False,
        )

        y_test = y[
            test_idx
        ].astype(
            np.float64,
            copy=False,
        )

        fit_start = (
            time.perf_counter()
        )

        (
            K_train,
            K_test_train,
            grm_diag,
        ) = build_grm_train_test(
            X_train_raw=
                X_train_raw,
            X_test_raw=
                X_test_raw,
        )

        reml = estimate_reml_components(
            K_train=
                K_train,
            y_train=
                y_train,
        )

        fit_time = (
            time.perf_counter()
            - fit_start
        )

        infer_start = (
            time.perf_counter()
        )

        y_pred = (
            reml["Mu"]
            + K_test_train
            @ reml["Alpha"]
        )

        inference_time = (
            time.perf_counter()
            - infer_start
        )

        metrics = calculate_metrics(
            y_test,
            y_pred,
        )

        fold_row = {
            "Model":
                "GBLUP",
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
            "Mu":
                reml[
                    "Mu"
                ],
            "Lambda":
                reml[
                    "Lambda"
                ],
            "Sigma_g2":
                reml[
                    "Sigma_g2"
                ],
            "Sigma_e2":
                reml[
                    "Sigma_e2"
                ],
            "H2":
                reml[
                    "H2"
                ],
            "REML_Objective":
                reml[
                    "REML_Objective"
                ],
            "REML_Success":
                reml[
                    "REML_Success"
                ],
            "GRM_Scaling":
                grm_diag[
                    "GRM_Scaling"
                ],
            "GRM_Denominator":
                grm_diag[
                    "GRM_Denominator"
                ],
            "Train_Genotype_Min":
                grm_diag[
                    "Train_Genotype_Min"
                ],
            "Train_Genotype_Max":
                grm_diag[
                    "Train_Genotype_Max"
                ],
            "Mean_Train_Allele_Frequency":
                grm_diag[
                    "Mean_Train_Allele_Frequency"
                ],
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
                    "+TrainOnlyGRM"
                ),
            "Y_Processing":
                "OriginalScale_NoStandardization",
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
            f"h2="
            f"{reml['H2']:.4f} | "
            f"lambda="
            f"{reml['Lambda']:.4g} | "
            f"GRM="
            f"{grm_diag['GRM_Scaling']} | "
            f"Fit="
            f"{fit_time:.2f}s"
        )

        pd.DataFrame(
            fold_rows
        ).to_csv(
            output_dir
            / (
                f"GBLUP_Wheat2000_"
                f"{trait}_"
                "FoldResults_CHECKPOINT.csv"
            ),
            index=False,
        )

        del (
            X_train_raw,
            X_test_raw,
            K_train,
            K_test_train,
            reml,
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
            f"GBLUP_Wheat2000_"
            f"{trait}_"
            "FoldResults.csv"
        ),
        index=False,
    )

    pred_df.to_csv(
        output_dir
        / (
            f"GBLUP_Wheat2000_"
            f"{trait}_"
            "Predictions.csv"
        ),
        index=False,
    )

    membership_df.to_csv(
        output_dir
        / (
            f"GBLUP_Wheat2000_"
            f"{trait}_"
            "10FoldMembership.csv"
        ),
        index=False,
    )

    summary = {
        "Model":
            "GBLUP",
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

    for metric in (
        "Lambda",
        "Sigma_g2",
        "Sigma_e2",
        "H2",
        "Fit_Time_s",
        "Inference_ms_per_sample",
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

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        output_dir
        / (
            f"GBLUP_Wheat2000_"
            f"{trait}_"
            "Summary.csv"
        ),
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        f"{trait} | "
        "10-fold summary"
    )
    print(
        "=" * 100
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
    print(
        f"H2      : "
        f"{summary['H2_Mean']:.4f} ± "
        f"{summary['H2_SD']:.4f}"
    )

    return summary_df


# =============================================================================
# 8. Main
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
            "results_gblup_"
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
                    dtype=np.float64,
                )

                sample_rows = (
                    sample_rows_all[
                        :n_common
                    ]
                )

            else:
                X_trait = X_all
                y_trait = y_all
                sample_rows = (
                    sample_rows_all
                )

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
                    "GBLUP_Wheat2000_"
                    "6Traits_10Fold_"
                    "Summary_CHECKPOINT.csv"
                ),
                index=False,
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
                "GBLUP_Wheat2000_"
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
            "H2_Mean",
            "Lambda_Mean",
            "Fit_Time_s_Mean",
        ]

        print(
            "\n"
            + "=" * 165
        )
        print(
            "GBLUP Wheat2000 "
            "6-trait 10-fold summary"
        )
        print(
            "=" * 165
        )

        print(
            combined[
                display_cols
            ].to_string(
                index=False,
            )
        )

    if failed:
        pd.DataFrame(
            failed
        ).to_csv(
            output_root
            / (
                "GBLUP_Wheat2000_"
                "FailedTraits.csv"
            ),
            index=False,
        )


if __name__ == "__main__":
    main()
