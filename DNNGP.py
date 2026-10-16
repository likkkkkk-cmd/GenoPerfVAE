# -*- coding: utf-8 -*-
"""
DNNGP_Wheat2000_6Traits_10Fold_GenoPerfVAE_YProcessing.py

DNNGP on Wheat2000: six-trait 10-fold cross-validation benchmark.

Directory layout
----------------
Put this script beside a folder named "data":

    DNNGP_Wheat2000_6Traits_10Fold_GenoPerfVAE_YProcessing.py
    data/
        2000gene.pkl
        2000_1_phe_tkw.txt
        2000_2_phe_*.txt
        2000_3_phe_*.txt
        2000_4_phe_*.txt
        2000_5_phe_*.txt
        2000_6_phe_*.txt

The script first looks for the conventional exact filenames, then falls back
to the corresponding prefix pattern (for example 2000_5_phe_*.txt). This makes
the script tolerant to capitalization/name differences such as hardness/Hard.

Traits
------
    tkw
    testw
    length
    width
    hardness
    Prot

Fair-CV data handling
---------------------
1. Load the Wheat2000 genotype matrix from data/2000gene.pkl.
2. Load each phenotype by row order, using the same loader convention as the
   existing Wheat2000 GenoPerfVAE/VMGP-S benchmark.
3. Use:
       KFold(n_splits=10, shuffle=True, random_state=42)
4. Within EACH OUTER FOLD:
       - fit most-frequent SNP imputation on training samples only
       - fit MinMaxScaler((-1, 1)) on training genotype only
       - EXACT GenoPerfVAE y handling: fit StandardScaler on outer-training phenotype only
         and transform both outer-train and outer-test phenotypes
       - transform the held-out test fold with training-fitted preprocessors
5. Train a fresh DNNGP model for every fold.
6. Metrics are calculated after inverse-transforming predictions back to the
   original phenotype scale.

Reported metrics
----------------
Per fold and 10-fold mean ± SD:
    Pearson r
    R2
    MSE
    RMSE
    nRMSE = RMSE / SD(observed phenotype in the test fold)
    MAE

Also reports global out-of-fold (OOF) metrics by concatenating all held-out
test predictions for each trait.

DNNGP settings
--------------
    batch_size   = 48
    epochs       = 150
    optimizer    = Adam
    learning_rate= 1e-5
    weight_decay = 1e-2
    loss          = MSE
    output head   = linear (continuous regression)
"""

import os
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 1. Global configuration
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 10

# Two independent GPU processes; no DDP/NCCL.
# GPU0: FOLD_PARITY=0 -> folds 1,3,5,7,9
# GPU1: FOLD_PARITY=1 -> folds 2,4,6,8,10
FOLD_PARITY = int(os.environ.get("FOLD_PARITY", "0"))
if FOLD_PARITY not in (0, 1):
    raise ValueError("FOLD_PARITY must be 0 or 1")

BATCH_SIZE = 48
EPOCHS = 150
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DNNGP model
#    Kept consistent with the supplied implementation
# ============================================================

class DNNGP(nn.Module):
    def __init__(self, snp_len):
        super(DNNGP, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=24,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.dropout1 = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.dropout2 = nn.Dropout(0.1)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.flatten = nn.Flatten()

        self.num = int(
            np.ceil(
                np.sqrt(
                    np.ceil(snp_len / 24)
                )
            )
        )

        self.fc1 = nn.Linear(
            64 * self.num * self.num,
            64,
        )
        self.fc2 = nn.Linear(
            64,
            1,
        )

        self.relu = nn.ReLU()

    def forward(self, snps):
        x = self.relu(
            self.conv1(snps)
        )
        x = self.dropout1(x)
        x = self.batch_norm1(x)

        x = self.relu(
            self.conv2(x)
        )
        x = self.dropout2(x)

        # Kept exactly as in the supplied implementation:
        # batch_norm2 is defined but is not called here.
        x = self.relu(
            self.conv3(x)
        )

        x = self.flatten(x)

        x = self.relu(
            self.fc1(x)
        )

        # Continuous regression head:
        # do NOT apply ReLU after the final linear layer.
        # A final ReLU can produce an all-zero dead output when fc2
        # initializes on the negative side, which makes every upstream
        # gradient zero and causes Pearson r to be undefined.
        x = self.fc2(x)

        return x


# ============================================================
# 4. Wheat2000 data loading
# ============================================================

class Wheat2000DataProcessor:
    """
    Loader matching the existing Wheat2000 benchmark convention.

    Genotype:
        - .pkl -> pd.read_pickle(...).values
        - otherwise read table and drop first column

    Phenotype:
        - read first numeric value from each non-empty line
        - align with genotype by row order
        - truncate to the common sample count

    Missing genotype processing is NOT done here.
    It is performed inside each CV fold.
    """

    def __init__(
        self,
        genotype_file,
        phenotype_file,
    ):
        self.genotype_file = Path(
            genotype_file
        )
        self.phenotype_file = Path(
            phenotype_file
        )

    def load_raw_data(self):
        if not self.genotype_file.exists():
            raise FileNotFoundError(
                f"Genotype file not found: {self.genotype_file}"
            )

        if not self.phenotype_file.exists():
            raise FileNotFoundError(
                f"Phenotype file not found: {self.phenotype_file}"
            )

        print(
            "Loading Wheat2000 data..."
        )

        if self.genotype_file.suffix.lower() == ".pkl":
            geno_obj = pd.read_pickle(
                self.genotype_file
            )

            X = (
                geno_obj.values
                if hasattr(
                    geno_obj,
                    "values"
                )
                else np.asarray(
                    geno_obj
                )
            )

        else:
            geno_df = pd.read_csv(
                self.genotype_file,
                sep=None,
                engine="python",
            )

            X = geno_df.iloc[
                :,
                1:
            ].values

        y_list = []

        with open(
            self.phenotype_file,
            "r",
            encoding="utf-8",
            errors="strict",
        ) as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()

                if not line:
                    raise ValueError(
                        f"Blank phenotype row at line {line_no} in "
                        f"{self.phenotype_file}. Row-order alignment would break."
                    )

                parts = (
                    line
                    .replace(",", " ")
                    .replace("\t", " ")
                    .split()
                )

                if len(parts) != 1:
                    raise ValueError(
                        f"Expected exactly one phenotype value at line {line_no} "
                        f"in {self.phenotype_file}, got {len(parts)} token(s)."
                    )

                try:
                    value = float(parts[0])
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric phenotype at line {line_no}: {parts[0]!r}"
                    ) from exc

                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite phenotype at line {line_no}: {value!r}"
                    )

                y_list.append(value)

        y = np.asarray(
            y_list,
            dtype=np.float32,
        ).reshape(
            -1,
            1,
        )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D genotype matrix, got {X.shape}"
            )

        if y.shape[0] == 0:
            raise ValueError(
                f"No numeric phenotype values found in {self.phenotype_file}"
            )

        original_x_n = X.shape[0]
        original_y_n = y.shape[0]

        if original_x_n != original_y_n:
            raise ValueError(
                "Genotype/phenotype sample-count mismatch under strict "
                "row-order alignment: "
                f"genotype={original_x_n}, phenotype={original_y_n}."
            )


        print(
            f"Aligned samples: {X.shape[0]}"
        )
        print(
            f"SNPs           : {X.shape[1]}"
        )
        print(
            f"Missing SNPs   : {int(np.isnan(X).sum())}"
        )
        print(
            "Genotype range : "
            f"{float(np.nanmin(X)):.6f} "
            f"to {float(np.nanmax(X)):.6f}"
        )
        print(
            "Phenotype mean/std: "
            f"{float(np.mean(y)):.6f} / "
            f"{float(np.std(y)):.6f}"
        )

        return X, y


# ============================================================
# 5. Locate files in ./data
# ============================================================

def resolve_trait_file(
    data_dir,
    exact_name,
    prefix_pattern,
):
    """
    First try the expected exact filename.
    If it does not exist, find exactly one file matching prefix_pattern.
    """
    data_dir = Path(
        data_dir
    )

    exact = (
        data_dir
        / exact_name
    )

    if exact.exists():
        return exact

    matches = sorted(
        data_dir.glob(
            prefix_pattern
        )
    )

    if len(matches) == 1:
        print(
            f"Using discovered phenotype file: {matches[0].name}"
        )
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find phenotype file '{exact_name}' "
            f"or any file matching '{prefix_pattern}' in {data_dir}"
        )

    raise RuntimeError(
        f"Multiple files match '{prefix_pattern}' in {data_dir}: "
        f"{[p.name for p in matches]}. "
        "Keep only the intended file or edit PHENOTYPE_SPECS."
    )


# ============================================================
# 6. DNNGP reshape
# ============================================================

def reshape_for_dnngp(
    X,
    num,
):
    """
    [N, D] -> [N, 24, num, num]
    with right-side zero padding when needed.
    """
    batch_size = X.shape[0]

    total_needed = (
        24 * num * num
    )

    current_len = X.shape[1]

    if current_len < total_needed:
        X = np.pad(
            X,
            pad_width=(
                (0, 0),
                (
                    0,
                    total_needed
                    - current_len,
                ),
            ),
            mode="constant",
            constant_values=0.0,
        )

    elif current_len > total_needed:
        X = X[
            :,
            :total_needed,
        ]

    return X.reshape(
        batch_size,
        24,
        num,
        num,
    ).astype(
        np.float32
    )


# ============================================================
# 7. Metrics
# ============================================================

def safe_pearson(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)

    if (
        len(y_true) < 2
        or np.std(y_true) < 1e-12
        or np.std(y_pred) < 1e-12
    ):
        return np.nan

    r, _ = stats.pearsonr(
        y_true,
        y_pred,
    )

    return float(r)


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    nRMSE = RMSE / sample SD(y_true), ddof=1.
    """
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    pearson = safe_pearson(
        y_true,
        y_pred,
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    mse = float(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
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

    nrmse = (
        float(
            rmse / y_sd
        )
        if (
            np.isfinite(y_sd)
            and y_sd > 1e-12
        )
        else np.nan
    )

    return {
        "Pearson":
            pearson,
        "R2":
            r2,
        "MSE":
            mse,
        "RMSE":
            rmse,
        "nRMSE":
            nrmse,
        "MAE":
            mae,
    }


# ============================================================
# 8. One trait: 10-fold CV
# ============================================================

def run_trait_10fold(
    genotype_file,
    phenotype_file,
    trait_name,
    output_dir,
    random_state=42,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processor = Wheat2000DataProcessor(
        genotype_file=genotype_file,
        phenotype_file=phenotype_file,
    )

    X_all, y_all = (
        processor.load_raw_data()
    )

    n_samples = X_all.shape[0]
    snp_dim = X_all.shape[1]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        + "=" * 96
    )
    print(
        f"DNNGP | Wheat2000 | {trait_name}"
    )
    print(
        f"Device   : {device}"
    )
    print(
        f"Samples  : {n_samples}"
    )
    print(
        f"SNPs     : {snp_dim}"
    )
    print(
        f"CV       : {N_SPLITS}-fold, seed={random_state}"
    )
    print(
        f"Shard    : FOLD_PARITY={FOLD_PARITY}, "
        f"assigned folds={[f for f in range(1, N_SPLITS + 1) if ((f - 1) % 2) == FOLD_PARITY]}"
    )
    print(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}"
    )
    print(
        "=" * 96
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )

    fold_results = []
    predictions = []
    membership = []

    for fold_idx, (
        train_idx,
        test_idx,
    ) in enumerate(
        kfold.split(X_all),
        start=1,
    ):
        if ((fold_idx - 1) % 2) != FOLD_PARITY:
            continue

        set_seed(
            random_state
        )

        print(
            "\n"
            + "-" * 96
        )
        print(
            f"Trait {trait_name} | "
            f"Fold {fold_idx}/{N_SPLITS}"
        )
        print(
            "-" * 96
        )

        X_train_raw = X_all[
            train_idx
        ]

        X_test_raw = X_all[
            test_idx
        ]

        y_train_raw = y_all[
            train_idx
        ]

        y_test_raw = y_all[
            test_idx
        ]

        train_set = set(np.asarray(train_idx, dtype=int).tolist())
        test_set = set(np.asarray(test_idx, dtype=int).tolist())
        if train_set & test_set:
            raise RuntimeError(
                f"Fold {fold_idx}: train/test overlap detected."
            )
        if len(train_set) + len(test_set) != n_samples:
            raise RuntimeError(
                f"Fold {fold_idx}: train/test do not cover all samples exactly once."
            )

        # ------------------------------------------------------
        # Fold-safe genotype imputation
        # ------------------------------------------------------
        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        X_train_imp = imputer.fit_transform(
            X_train_raw
        ).astype(
            np.float32
        )

        X_test_imp = imputer.transform(
            X_test_raw
        ).astype(
            np.float32
        )

        if (
            X_train_imp.shape[1]
            != snp_dim
        ):
            raise RuntimeError(
                "SimpleImputer changed SNP dimensionality. "
                "Check for SNP columns that are entirely missing in the "
                "training fold."
            )

        # ------------------------------------------------------
        # Fold-safe genotype scaling
        # Same Wheat2000 fair-comparison representation:
        # MinMaxScaler(-1, 1), fit on outer train only.
        # ------------------------------------------------------
        scaler_x = MinMaxScaler(
            feature_range=(-1, 1)
        )

        X_train_scaled = scaler_x.fit_transform(
            X_train_imp
        ).astype(
            np.float32
        )

        X_test_scaled = scaler_x.transform(
            X_test_imp
        ).astype(
            np.float32
        )

        # ------------------------------------------------------
        # Fold-safe phenotype scaling -- copied from GenoPerfVAE Wheat2000
        # EXACT protocol:
        #   scaler_y = StandardScaler()
        #   y_train_scaled = scaler_y.fit_transform(y_train_raw)
        #   y_test_scaled  = scaler_y.transform(y_test_raw)
        # The scaler is fitted ONLY on the outer-training fold.
        # ------------------------------------------------------
        scaler_y = StandardScaler()

        y_train_scaled = scaler_y.fit_transform(
            y_train_raw
        ).astype(
            np.float32
        )

        y_test_scaled = scaler_y.transform(
            y_test_raw
        ).astype(
            np.float32
        )

        # ------------------------------------------------------
        # Fresh model for each fold
        # ------------------------------------------------------
        model = DNNGP(
            snp_len=snp_dim
        ).to(
            device
        )

        num = model.num

        X_train = reshape_for_dnngp(
            X_train_scaled,
            num,
        )

        X_test = reshape_for_dnngp(
            X_test_scaled,
            num,
        )

        # ------------------------------------------------------
        # Initial-output diagnostic.
        # This immediately detects a collapsed/constant regression head.
        # ------------------------------------------------------
        model.eval()
        with torch.no_grad():
            probe_n = min(48, len(X_train))
            probe_x = torch.from_numpy(
                X_train[:probe_n]
            ).to(
                device
            )
            probe_pred = model(
                probe_x
            ).detach().cpu().numpy().reshape(-1)

        print(
            "  Initial output: "
            f"mean={np.mean(probe_pred):.6f}, "
            f"std={np.std(probe_pred):.6f}, "
            f"min={np.min(probe_pred):.6f}, "
            f"max={np.max(probe_pred):.6f}"
        )

        model.train()

        train_dataset = TensorDataset(
            torch.from_numpy(
                X_train
            ),
            torch.from_numpy(
                y_train_scaled
            ),
        )

        loader_generator = (
            torch.Generator()
        )

        loader_generator.manual_seed(
            random_state
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=loader_generator,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        criterion = nn.MSELoss()

        # ------------------------------------------------------
        # Training
        # ------------------------------------------------------
        model.train()

        for epoch in range(
            EPOCHS
        ):
            total_loss = 0.0
            n_batches = 0

            for bx, by in train_loader:
                bx = bx.to(
                    device,
                    non_blocking=True,
                )

                by = by.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                pred = model(
                    bx
                )

                loss = criterion(
                    pred,
                    by
                )

                loss.backward()
                optimizer.step()

                total_loss += float(
                    loss.detach().cpu()
                )

                n_batches += 1

            if (
                epoch == 0
                or (epoch + 1) % 50 == 0
                or epoch + 1 == EPOCHS
            ):
                print(
                    f"  Epoch "
                    f"[{epoch + 1:03d}/{EPOCHS}] "
                    f"Loss="
                    f"{total_loss / max(n_batches, 1):.6f}"
                )

        # ------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------
        model.eval()

        test_dataset = TensorDataset(
            torch.from_numpy(
                X_test
            ),
            torch.from_numpy(
                y_test_scaled
            ),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        pred_scaled_list = []

        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(
                    device,
                    non_blocking=True,
                )

                pred_scaled = model(
                    bx
                )

                pred_scaled_list.append(
                    pred_scaled
                    .cpu()
                    .numpy()
                )

        y_pred_scaled = np.concatenate(
            pred_scaled_list,
            axis=0,
        )

        # ------------------------------------------------------
        # EXACT GenoPerfVAE evaluation handling:
        # inverse-transform BOTH prediction and test phenotype
        # from the fold-specific StandardScaler back to the
        # original phenotype scale before calculating metrics.
        # ------------------------------------------------------
        y_pred = scaler_y.inverse_transform(
            y_pred_scaled
        ).reshape(-1)

        y_true = scaler_y.inverse_transform(
            y_test_scaled
        ).reshape(-1)

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        fold_results.append({
            "Model":
                "DNNGP",
            "Dataset":
                "Wheat2000",
            "Trait":
                trait_name,
            "Fold":
                fold_idx,
            "Train_N":
                len(train_idx),
            "Test_N":
                len(test_idx),
            "SNPs":
                snp_dim,
            "DNNGP_Num":
                num,
            "Padded_Length":
                24 * num * num,
            **metrics,
            "Epochs":
                EPOCHS,
            "Batch_Size":
                BATCH_SIZE,
            "Learning_Rate":
                LEARNING_RATE,
            "Weight_Decay":
                WEIGHT_DECAY,
            "Seed":
                random_state,
            "Y_Processing":
                "GenoPerfVAE_foldwise_StandardScaler",
        })

        for sample_idx, obs, pred in zip(
            test_idx,
            y_true,
            y_pred,
        ):
            predictions.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(sample_idx),
                "Observed":
                    float(obs),
                "Predicted":
                    float(pred),
            })

        for sample_idx in train_idx:
            membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(sample_idx),
                "Set":
                    "Train",
            })

        for sample_idx in test_idx:
            membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(sample_idx),
                "Set":
                    "Test",
            })

        print(
            f"  Pearson r = {metrics['Pearson']:.4f}"
        )
        print(
            f"  R2        = {metrics['R2']:.4f}"
        )
        print(
            f"  MSE       = {metrics['MSE']:.4f}"
        )
        print(
            f"  RMSE      = {metrics['RMSE']:.4f}"
        )
        print(
            f"  nRMSE     = {metrics['nRMSE']:.4f}"
        )
        print(
            f"  MAE       = {metrics['MAE']:.4f}"
        )

        del (
            model,
            optimizer,
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
            X_train,
            X_test,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # ----------------------------------------------------------
    # Save fold/prediction/membership outputs
    # ----------------------------------------------------------
    fold_df = pd.DataFrame(
        fold_results
    )

    pred_df = pd.DataFrame(
        predictions
    ).sort_values(
        by="Sample_Row_0Based"
    )

    membership_df = pd.DataFrame(
        membership
    )

    fold_df.to_csv(
        output_dir
        / f"DNNGP_Wheat2000_{trait_name}_FoldResults.csv",
        index=False,
    )

    pred_df.to_csv(
        output_dir
        / f"DNNGP_Wheat2000_{trait_name}_Predictions.csv",
        index=False,
    )

    membership_df.to_csv(
        output_dir
        / f"DNNGP_Wheat2000_{trait_name}_10FoldMembership.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 10-fold mean ± SD
    # ----------------------------------------------------------
    summary = {
        "Model":
            "DNNGP",
        "Dataset":
            "Wheat2000",
        "Trait":
            trait_name,
        "N":
            n_samples,
        "SNPs":
            snp_dim,
        "Completed_Folds":
            len(fold_df),
        "Expected_Total_Folds":
            N_SPLITS,
        "Fold_Parity":
            FOLD_PARITY,
        "CUDA_Visible_Devices":
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),

        "Pearson_Mean":
            fold_df["Pearson"].mean(),
        "Pearson_SD":
            fold_df["Pearson"].std(
                ddof=1
            ),

        "R2_Mean":
            fold_df["R2"].mean(),
        "R2_SD":
            fold_df["R2"].std(
                ddof=1
            ),

        "MSE_Mean":
            fold_df["MSE"].mean(),
        "MSE_SD":
            fold_df["MSE"].std(
                ddof=1
            ),

        "RMSE_Mean":
            fold_df["RMSE"].mean(),
        "RMSE_SD":
            fold_df["RMSE"].std(
                ddof=1
            ),

        "nRMSE_Mean":
            fold_df["nRMSE"].mean(),
        "nRMSE_SD":
            fold_df["nRMSE"].std(
                ddof=1
            ),

        "MAE_Mean":
            fold_df["MAE"].mean(),
        "MAE_SD":
            fold_df["MAE"].std(
                ddof=1
            ),

        "Epochs":
            EPOCHS,
        "Batch_Size":
            BATCH_SIZE,
        "Learning_Rate":
            LEARNING_RATE,
        "Weight_Decay":
            WEIGHT_DECAY,
        "Seed":
            random_state,
        "Y_Processing":
            "GenoPerfVAE_foldwise_StandardScaler",
    }

    # ----------------------------------------------------------
    # Global out-of-fold metrics
    # ----------------------------------------------------------
    oof_metrics = calculate_metrics(
        pred_df[
            "Observed"
        ].to_numpy(),
        pred_df[
            "Predicted"
        ].to_numpy(),
    )

    summary.update({
        "OOF_Pearson":
            oof_metrics[
                "Pearson"
            ],
        "OOF_R2":
            oof_metrics[
                "R2"
            ],
        "OOF_MSE":
            oof_metrics[
                "MSE"
            ],
        "OOF_RMSE":
            oof_metrics[
                "RMSE"
            ],
        "OOF_nRMSE":
            oof_metrics[
                "nRMSE"
            ],
        "OOF_MAE":
            oof_metrics[
                "MAE"
            ],
    })

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        output_dir
        / f"DNNGP_Wheat2000_{trait_name}_Summary.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 88
    )
    print(
        f"{trait_name} | shard summary ({len(fold_df)} folds completed)"
    )
    print(
        "=" * 88
    )
    print(
        f"Pearson r : "
        f"{summary['Pearson_Mean']:.4f} ± "
        f"{summary['Pearson_SD']:.4f}"
    )
    print(
        f"R2        : "
        f"{summary['R2_Mean']:.4f} ± "
        f"{summary['R2_SD']:.4f}"
    )
    print(
        f"RMSE      : "
        f"{summary['RMSE_Mean']:.4f} ± "
        f"{summary['RMSE_SD']:.4f}"
    )
    print(
        f"nRMSE     : "
        f"{summary['nRMSE_Mean']:.4f} ± "
        f"{summary['nRMSE_SD']:.4f}"
    )
    print(
        f"MAE       : "
        f"{summary['MAE_Mean']:.4f} ± "
        f"{summary['MAE_SD']:.4f}"
    )
    print(
        f"OOF RMSE  : "
        f"{summary['OOF_RMSE']:.4f}"
    )
    print(
        f"OOF nRMSE : "
        f"{summary['OOF_nRMSE']:.4f}"
    )

    return (
        fold_df,
        pred_df,
        summary_df,
    )


# ============================================================
# 9. Run all six traits
# ============================================================

def run_all_traits(
    genotype_file,
    phenotype_files,
    output_root,
    random_state=42,
):
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []
    failed = []

    for trait_idx, (
        trait_name,
        phenotype_file,
    ) in enumerate(
        phenotype_files.items(),
        start=1,
    ):
        print(
            "\n\n"
            + "#" * 100
        )
        print(
            f"Trait "
            f"{trait_idx}/"
            f"{len(phenotype_files)}: "
            f"{trait_name}"
        )
        print(
            f"Phenotype file: "
            f"{phenotype_file}"
        )
        print(
            "#" * 100
        )

        try:
            trait_dir = (
                output_root
                / trait_name
            )

            _, _, summary_df = (
                run_trait_10fold(
                    genotype_file=
                        genotype_file,
                    phenotype_file=
                        phenotype_file,
                    trait_name=
                        trait_name,
                    output_dir=
                        trait_dir,
                    random_state=
                        random_state,
                )
            )

            all_summaries.append(
                summary_df
            )

            # Update checkpoint after every completed trait.
            pd.concat(
                all_summaries,
                ignore_index=True,
            ).to_csv(
                output_root
                / (
                    "DNNGP_Wheat2000_"
                    "6Traits_Summary_CHECKPOINT.csv"
                ),
                index=False,
            )

        except Exception as exc:
            print(
                f"[ERROR] {trait_name}: "
                f"{type(exc).__name__}: {exc}"
            )

            failed.append({
                "Trait":
                    trait_name,
                "Phenotype_File":
                    str(
                        phenotype_file
                    ),
                "Reason":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

    if all_summaries:
        combined = pd.concat(
            all_summaries,
            ignore_index=True,
        )

        combined.to_csv(
            output_root
            / "DNNGP_Wheat2000_6Traits_Summary.csv",
            index=False,
        )

        print(
            "\n"
            + "=" * 118
        )
        print(
            "DNNGP Wheat2000 six-trait "
            "10-fold summary"
        )
        print(
            "=" * 118
        )

        display_cols = [
            "Trait",
            "N",
            "SNPs",
            "Pearson_Mean",
            "Pearson_SD",
            "R2_Mean",
            "R2_SD",
            "RMSE_Mean",
            "RMSE_SD",
            "nRMSE_Mean",
            "nRMSE_SD",
            "MAE_Mean",
            "MAE_SD",
        ]

        print(
            combined[
                display_cols
            ].to_string(
                index=False
            )
        )

    else:
        combined = pd.DataFrame()

    if failed:
        pd.DataFrame(
            failed
        ).to_csv(
            output_root
            / "DNNGP_Wheat2000_FailedTraits.csv",
            index=False,
        )

    return combined


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # The user requested all Wheat2000 data to live in:
    #
    #     <script folder>/data/
    # --------------------------------------------------------
    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parent
    )

    DATA_DIR = (
        BASE_DIR
        / "data"
    )

    GENOTYPE_FILE = (
        DATA_DIR
        / "2000gene.pkl"
    )

    # Exact names are tried first.
    # Prefix fallback handles small filename differences.
    PHENOTYPE_SPECS = {
        "tkw": {
            "exact":
                "2000_1_phe_tkw.txt",
            "pattern":
                "2000_1_phe_*.txt",
        },
        "testw": {
            "exact":
                "2000_2_phe_testw.txt",
            "pattern":
                "2000_2_phe_*.txt",
        },
        "length": {
            "exact":
                "2000_3_phe_length.txt",
            "pattern":
                "2000_3_phe_*.txt",
        },
        "width": {
            "exact":
                "2000_4_phe_width.txt",
            "pattern":
                "2000_4_phe_*.txt",
        },
        "hardness": {
            "exact":
                "2000_5_phe_hardness.txt",
            "pattern":
                "2000_5_phe_*.txt",
        },
        "Prot": {
            "exact":
                "2000_6_phe_Prot.txt",
            "pattern":
                "2000_6_phe_*.txt",
        },
    }

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}\n"
            "Create a 'data' folder beside this script and place "
            "the Wheat2000 files inside it."
        )

    if not GENOTYPE_FILE.exists():
        raise FileNotFoundError(
            f"Wheat2000 genotype file not found: {GENOTYPE_FILE}"
        )

    PHENOTYPE_FILES = {}

    for trait, spec in (
        PHENOTYPE_SPECS.items()
    ):
        PHENOTYPE_FILES[
            trait
        ] = resolve_trait_file(
            data_dir=DATA_DIR,
            exact_name=spec[
                "exact"
            ],
            prefix_pattern=spec[
                "pattern"
            ],
        )

    print(
        "\nResolved Wheat2000 files:"
    )
    print(
        f"  Genotype: {GENOTYPE_FILE}"
    )

    for trait, path in (
        PHENOTYPE_FILES.items()
    ):
        print(
            f"  {trait:<10}: {path.name}"
        )

    OUTPUT_ROOT_BASE = (
        BASE_DIR
        / "results_dnngp_wheat2000_6traits"
    )

    OUTPUT_ROOT = (
        OUTPUT_ROOT_BASE
        / f"shard_{FOLD_PARITY}"
    )

    run_all_traits(
        genotype_file=
            GENOTYPE_FILE,
        phenotype_files=
            PHENOTYPE_FILES,
        output_root=
            OUTPUT_ROOT,
        random_state=
            RANDOM_STATE,
    )
