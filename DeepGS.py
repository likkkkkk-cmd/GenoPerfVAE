# -*- coding: utf-8 -*-
"""
DeepGS_Wheat2000_6Traits_DNNGP_Same10Fold_NativeProtocol_2GPU_FINAL.py

Goal
----
Use EXACTLY the same OUTER 10-fold sample partition as the DNNGP Wheat2000
benchmark, while training DeepGS with a native-style optimization protocol.

Outer CV (must match DNNGP):
    KFold(n_splits=10, shuffle=True, random_state=42)

Two-GPU execution (NO DDP / NO NCCL):
    GPU0 / FOLD_PARITY=0 -> folds 1,3,5,7,9
    GPU1 / FOLD_PARITY=1 -> folds 2,4,6,8,10

Wheat2000 data convention:
    genotype: data/2000gene.pkl
    phenotype: six txt files, NO header, one numeric value per physical row
    genotype row i <-> phenotype row i

DeepGS native-style training protocol used here:
    - genotype input kept on its original numeric coding (NO MinMax scaling)
    - phenotype kept on its original scale (NO StandardScaler)
    - missing SNP imputation fitted on OUTER-TRAIN only
    - inner validation = 10% of OUTER-TRAIN, used for monitoring/model selection
    - optimizer = SGD
    - learning_rate = 0.01
    - momentum = 0.5
    - weight_decay = 1e-5
    - loss = MAE / L1Loss
    - batch_size = 30
    - max num_rounds = 6000
    - early stopping patience = 200 validation checks
    - fresh DeepGS model for each outer fold

Important:
    The OUTER test membership is exactly determined by the same DNNGP KFold.
    No outer-test information is used for imputation, training, validation,
    checkpoint selection, or preprocessing.
"""

from __future__ import annotations

import os
import gc
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from scipy import stats


# ============================================================
# 1. Global configuration
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 10

# DeepGS native-style optimization settings
BATCH_SIZE = 30
NUM_ROUNDS = 6000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
WEIGHT_DECAY = 1e-5
INNER_VAL_RATIO = 0.10
EARLY_STOP_PATIENCE = 200
MIN_DELTA = 1e-6

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# Log every N rounds to avoid enormous logs.
LOG_EVERY = 100

# Independent dual-GPU fold sharding.
FOLD_PARITY = int(os.environ.get("FOLD_PARITY", "0"))
if FOLD_PARITY not in (0, 1):
    raise ValueError("FOLD_PARITY must be 0 or 1")


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DeepGS model
#    Restores the supplied/original DeepGS structure
# ============================================================

class DeepGS(nn.Module):
    def __init__(self, snp_len: int):
        super().__init__()

        self.CNN = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=18,
            stride=1,
        )

        self.pool = nn.MaxPool1d(
            kernel_size=4,
            stride=4,
        )

        # Exact PyTorch output length after Conv1d + MaxPool1d.
        conv_len = snp_len - 18 + 1
        pooled_len = ((conv_len - 4) // 4) + 1

        if pooled_len <= 0:
            raise ValueError(
                f"SNP dimension {snp_len} is too short for DeepGS."
            )

        self.linear1 = nn.Linear(
            pooled_len * 8,
            32,
        )

        self.linear2 = nn.Linear(
            32,
            1,
        )

        self.linear3 = nn.Linear(
            1,
            1,
        )

        self.dropout2 = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout05 = nn.Dropout(0.05)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Keep deterministic Xavier initialization from the supplied PyTorch port.
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = torch.unsqueeze(
            x,
            dim=1,
        )

        x = self.CNN(
            x
        )

        x = self.relu(
            x
        )

        x = self.pool(
            x
        )

        x = self.dropout2(
            x
        )

        x = self.flatten(
            x
        )

        x = self.linear1(
            x
        )

        x = self.relu(
            self.dropout1(
                x
            )
        )

        x = self.linear2(
            x
        )

        x = self.linear3(
            self.dropout05(
                x
            )
        )

        return x


# ============================================================
# 4. Wheat2000 data loading
# ============================================================

def resolve_trait_file(
    data_dir: Path,
    exact_name: str,
    prefix_pattern: str,
) -> Path:
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
            f"Could not find '{exact_name}' or '{prefix_pattern}' "
            f"in {data_dir}"
        )

    raise RuntimeError(
        f"Multiple phenotype files match '{prefix_pattern}': "
        f"{[p.name for p in matches]}"
    )


def load_wheat2000(
    genotype_file: Path,
    phenotype_file: Path,
):
    genotype_file = Path(
        genotype_file
    )

    phenotype_file = Path(
        phenotype_file
    )

    if not genotype_file.exists():
        raise FileNotFoundError(
            f"Genotype file not found: {genotype_file}"
        )

    if not phenotype_file.exists():
        raise FileNotFoundError(
            f"Phenotype file not found: {phenotype_file}"
        )

    print(
        f"Loading genotype : {genotype_file.name}"
    )

    geno_obj = pd.read_pickle(
        genotype_file
    )

    X = (
        geno_obj.values
        if hasattr(
            geno_obj,
            "values",
        )
        else np.asarray(
            geno_obj
        )
    )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    print(
        f"Loading phenotype: {phenotype_file.name}"
    )

    y_values = []

    with open(
        phenotype_file,
        "r",
        encoding="utf-8",
        errors="strict",
    ) as f:
        for line_no, raw_line in enumerate(
            f,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                raise ValueError(
                    f"Blank phenotype row at line {line_no}. "
                    "Strict row alignment would break."
                )

            parts = (
                line
                .replace(",", " ")
                .replace("\t", " ")
                .split()
            )

            if len(parts) != 1:
                raise ValueError(
                    f"Expected exactly one phenotype value at line "
                    f"{line_no}, found {len(parts)} token(s): {parts!r}"
                )

            try:
                value = float(
                    parts[0]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric phenotype at line {line_no}: "
                    f"{parts[0]!r}"
                ) from exc

            if not np.isfinite(
                value
            ):
                raise ValueError(
                    f"Non-finite phenotype at line {line_no}: "
                    f"{value!r}"
                )

            y_values.append(
                value
            )

    y = np.asarray(
        y_values,
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2-D genotype matrix, got {X.shape}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Genotype/phenotype row-count mismatch under strict "
            "row-order alignment: "
            f"genotype={X.shape[0]}, phenotype={y.shape[0]}"
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
        "Raw genotype range: "
        f"{float(np.nanmin(X)):.6f} to "
        f"{float(np.nanmax(X)):.6f}"
    )
    print(
        "Raw phenotype mean/std: "
        f"{float(np.mean(y)):.6f} / "
        f"{float(np.std(y)):.6f}"
    )

    return (
        X,
        y,
    )


# ============================================================
# 5. Metrics
# ============================================================

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

    r, _ = stats.pearsonr(
        y_true,
        y_pred,
    )

    return float(
        r
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
        if len(
            y_true
        ) > 1
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
                    rmse / y_sd
                )
                if (
                    np.isfinite(
                        y_sd
                    )
                    and y_sd > 1e-12
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


# ============================================================
# 6. Validation helper
# ============================================================

@torch.no_grad()
def evaluate_mae(
    model,
    loader,
    device,
):
    model.eval()

    total_abs = 0.0
    total_n = 0

    for bx, by in loader:
        bx = bx.to(
            device,
            non_blocking=True,
        )

        by = by.to(
            device,
            non_blocking=True,
        )

        pred = model(
            bx
        )

        total_abs += float(
            torch.sum(
                torch.abs(
                    pred - by
                )
            ).cpu()
        )

        total_n += int(
            by.numel()
        )

    return (
        total_abs
        / max(
            total_n,
            1,
        )
    )


# ============================================================
# 7. One trait: same DNNGP outer 10-fold + DeepGS protocol
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

    X_all, y_all = load_wheat2000(
        genotype_file,
        phenotype_file,
    )

    n_samples = X_all.shape[0]
    snp_dim = X_all.shape[1]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    assigned_folds = [
        f
        for f in range(
            1,
            N_SPLITS + 1,
        )
        if (
            (f - 1) % 2
        ) == FOLD_PARITY
    ]

    print(
        "\n"
        + "=" * 104
    )
    print(
        f"DeepGS native protocol | Wheat2000 | {trait_name}"
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
        f"Outer CV : KFold(10, shuffle=True, random_state={random_state})"
    )
    print(
        f"Shard    : FOLD_PARITY={FOLD_PARITY}, "
        f"assigned folds={assigned_folds}"
    )
    print(
        f"CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}"
    )
    print(
        "DeepGS protocol: raw X, raw y, SGD(lr=0.01,momentum=0.5,"
        "wd=1e-5), MAE, batch=30, max_rounds=6000, "
        f"early_stop_patience={EARLY_STOP_PATIENCE}"
    )
    print(
        "=" * 104
    )

    # EXACT same outer splitter as DNNGP Wheat2000.
    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )

    fold_results = []
    predictions = []
    membership = []

    for fold_idx, (
        outer_train_idx,
        test_idx,
    ) in enumerate(
        kfold.split(
            X_all
        ),
        start=1,
    ):
        if (
            (fold_idx - 1) % 2
        ) != FOLD_PARITY:
            continue

        set_seed(
            random_state
        )

        print(
            "\n"
            + "-" * 104
        )
        print(
            f"Trait {trait_name} | Fold {fold_idx}/{N_SPLITS}"
        )
        print(
            "-" * 104
        )

        # ------------------------------------------------------
        # Verify outer-fold integrity
        # ------------------------------------------------------
        outer_train_set = set(
            np.asarray(
                outer_train_idx,
                dtype=int,
            ).tolist()
        )

        test_set = set(
            np.asarray(
                test_idx,
                dtype=int,
            ).tolist()
        )

        if outer_train_set & test_set:
            raise RuntimeError(
                f"Fold {fold_idx}: outer train/test overlap detected."
            )

        if (
            len(
                outer_train_set
            )
            + len(
                test_set
            )
            != n_samples
        ):
            raise RuntimeError(
                f"Fold {fold_idx}: outer train/test do not cover "
                "all samples exactly once."
            )

        # ------------------------------------------------------
        # Native-style inner 10% validation from outer-train only
        # ------------------------------------------------------
        inner_train_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=INNER_VAL_RATIO,
            random_state=random_state,
            shuffle=True,
        )

        inner_train_set = set(
            np.asarray(
                inner_train_idx,
                dtype=int,
            ).tolist()
        )

        val_set = set(
            np.asarray(
                val_idx,
                dtype=int,
            ).tolist()
        )

        if inner_train_set & val_set:
            raise RuntimeError(
                f"Fold {fold_idx}: inner train/val overlap detected."
            )

        if inner_train_set & test_set:
            raise RuntimeError(
                f"Fold {fold_idx}: inner train/test overlap detected."
            )

        if val_set & test_set:
            raise RuntimeError(
                f"Fold {fold_idx}: validation/test overlap detected."
            )

        if (
            inner_train_set
            | val_set
        ) != outer_train_set:
            raise RuntimeError(
                f"Fold {fold_idx}: inner train + val does not exactly "
                "reconstruct the DNNGP outer-training fold."
            )

        print(
            f"Outer train/test : "
            f"{len(outer_train_idx)} / {len(test_idx)}"
        )
        print(
            f"Inner train/val  : "
            f"{len(inner_train_idx)} / {len(val_idx)}"
        )

        X_train_raw = X_all[
            inner_train_idx
        ].astype(
            np.float32,
            copy=True,
        )

        X_val_raw = X_all[
            val_idx
        ].astype(
            np.float32,
            copy=True,
        )

        X_test_raw = X_all[
            test_idx
        ].astype(
            np.float32,
            copy=True,
        )

        y_train = y_all[
            inner_train_idx
        ].astype(
            np.float32,
            copy=True,
        )

        y_val = y_all[
            val_idx
        ].astype(
            np.float32,
            copy=True,
        )

        y_test = y_all[
            test_idx
        ].astype(
            np.float32,
            copy=True,
        )

        # ------------------------------------------------------
        # Train-only missing SNP imputation.
        # NO genotype MinMaxScaler: keep DeepGS raw numeric coding.
        # ------------------------------------------------------
        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        X_train = (
            imputer
            .fit_transform(
                X_train_raw
            )
            .astype(
                np.float32
            )
        )

        X_val = (
            imputer
            .transform(
                X_val_raw
            )
            .astype(
                np.float32
            )
        )

        X_test = (
            imputer
            .transform(
                X_test_raw
            )
            .astype(
                np.float32
            )
        )

        if (
            X_train.shape[1]
            != snp_dim
        ):
            raise RuntimeError(
                "SimpleImputer changed SNP dimensionality. "
                "A SNP column may be entirely missing in the inner train set."
            )

        print(
            "Input range after train-fitted imputation: "
            f"train=[{float(np.min(X_train)):.6f}, "
            f"{float(np.max(X_train)):.6f}] "
            f"val=[{float(np.min(X_val)):.6f}, "
            f"{float(np.max(X_val)):.6f}] "
            f"test=[{float(np.min(X_test)):.6f}, "
            f"{float(np.max(X_test)):.6f}]"
        )

        print(
            "Phenotype kept on original scale: "
            f"train mean/std={float(np.mean(y_train)):.6f}/"
            f"{float(np.std(y_train)):.6f}"
        )

        # ------------------------------------------------------
        # Fresh DeepGS
        # ------------------------------------------------------
        model = DeepGS(
            snp_len=snp_dim
        ).to(
            device
        )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            f"Parameters: "
            f"{total_params / 1e6:.3f} M "
            f"(trainable {trainable_params / 1e6:.3f} M)"
        )

        # Initial output diagnostic.
        model.eval()

        with torch.no_grad():
            probe_n = min(
                64,
                len(
                    X_train
                ),
            )

            probe_pred = (
                model(
                    torch.from_numpy(
                        X_train[
                            :probe_n
                        ]
                    ).to(
                        device
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        print(
            "  Initial output: "
            f"mean={np.mean(probe_pred):.6f}, "
            f"std={np.std(probe_pred):.6f}, "
            f"min={np.min(probe_pred):.6f}, "
            f"max={np.max(probe_pred):.6f}"
        )

        # ------------------------------------------------------
        # DataLoaders
        # ------------------------------------------------------
        train_ds = TensorDataset(
            torch.from_numpy(
                X_train
            ),
            torch.from_numpy(
                y_train
            ),
        )

        val_ds = TensorDataset(
            torch.from_numpy(
                X_val
            ),
            torch.from_numpy(
                y_val
            ),
        )

        test_ds = TensorDataset(
            torch.from_numpy(
                X_test
            ),
            torch.from_numpy(
                y_test
            ),
        )

        train_generator = torch.Generator()
        train_generator.manual_seed(
            random_state
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=train_generator,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        # ------------------------------------------------------
        # Native-style optimizer / objective
        # ------------------------------------------------------
        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

        criterion = nn.L1Loss(
            reduction="mean"
        )

        best_val_mae = np.inf
        best_round = 0
        best_state = None
        no_improve_rounds = 0

        # ------------------------------------------------------
        # Training: fixed 6000 rounds.
        # Validation is used to retain the best model.
        # ------------------------------------------------------
        for round_idx in range(
            1,
            NUM_ROUNDS + 1,
        ):
            model.train()

            total_abs = 0.0
            total_n = 0

            for batch_idx, (
                bx,
                by,
            ) in enumerate(
                train_loader
            ):
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

                if (
                    round_idx == 1
                    and batch_idx == 0
                ):
                    grad_sq_sum = 0.0
                    grad_params = 0

                    for p in model.parameters():
                        if p.grad is not None:
                            g = p.grad.detach().float()

                            grad_sq_sum += float(
                                torch.sum(
                                    g * g
                                ).cpu()
                            )

                            grad_params += 1

                    print(
                        "  First-batch gradient: "
                        f"L2={grad_sq_sum ** 0.5:.6e}, "
                        f"params_with_grad={grad_params}"
                    )

                optimizer.step()

                total_abs += float(
                    torch.sum(
                        torch.abs(
                            pred.detach()
                            - by
                        )
                    ).cpu()
                )

                total_n += int(
                    by.numel()
                )

            train_mae = (
                total_abs
                / max(
                    total_n,
                    1,
                )
            )

            val_mae = evaluate_mae(
                model,
                val_loader,
                device,
            )

            if (
                np.isfinite(
                    val_mae
                )
                and val_mae < (
                    best_val_mae
                    - MIN_DELTA
                )
            ):
                best_val_mae = float(
                    val_mae
                )

                best_round = int(
                    round_idx
                )

                best_state = deepcopy(
                    {
                        k:
                            v.detach().cpu().clone()
                        for k, v in (
                            model.state_dict().items()
                        )
                    }
                )

                no_improve_rounds = 0

            else:
                no_improve_rounds += 1

            if (
                round_idx == 1
                or round_idx % LOG_EVERY == 0
                or round_idx == NUM_ROUNDS
            ):
                print(
                    f"  Round [{round_idx:04d}/{NUM_ROUNDS}] "
                    f"Train_MAE={train_mae:.6f} "
                    f"Val_MAE={val_mae:.6f} "
                    f"Best={best_val_mae:.6f} "
                    f"BestRound={best_round} "
                    f"NoImprove={no_improve_rounds}/{EARLY_STOP_PATIENCE} "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )

            if no_improve_rounds >= EARLY_STOP_PATIENCE:
                print(
                    f"  Early stopping at round {round_idx}. "
                    f"Best round={best_round}, "
                    f"Best Val_MAE={best_val_mae:.6f}"
                )
                break

        if best_state is None:
            raise RuntimeError(
                f"Fold {fold_idx}: no finite validation MAE "
                "was observed during training."
            )

        model.load_state_dict(
            best_state
        )

        model.to(
            device
        )

        model.eval()

        # ------------------------------------------------------
        # Best-model diagnostics
        # ------------------------------------------------------
        with torch.no_grad():
            train_diag_n = min(
                256,
                len(
                    X_train
                ),
            )

            train_diag_pred = (
                model(
                    torch.from_numpy(
                        X_train[
                            :train_diag_n
                        ]
                    ).to(
                        device
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        print(
            "  Best-model training prediction diagnostic: "
            f"mean={np.mean(train_diag_pred):.6f}, "
            f"std={np.std(train_diag_pred):.6e}, "
            f"min={np.min(train_diag_pred):.6f}, "
            f"max={np.max(train_diag_pred):.6f}"
        )

        with torch.no_grad():
            print(
                "  Weight diagnostic: "
                f"conv_mean_abs="
                f"{model.CNN.weight.abs().mean().item():.6e}, "
                f"fc1_mean_abs="
                f"{model.linear1.weight.abs().mean().item():.6e}, "
                f"fc2_mean_abs="
                f"{model.linear2.weight.abs().mean().item():.6e}, "
                f"fc3_mean_abs="
                f"{model.linear3.weight.abs().mean().item():.6e}"
            )

        # ------------------------------------------------------
        # Outer held-out test evaluation
        # ------------------------------------------------------
        pred_list = []

        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(
                    device,
                    non_blocking=True,
                )

                pred = model(
                    bx
                )

                pred_list.append(
                    pred
                    .detach()
                    .cpu()
                    .numpy()
                )

        y_pred = np.concatenate(
            pred_list,
            axis=0,
        ).reshape(-1)

        y_true = y_test.reshape(
            -1
        )

        pred_std = float(
            np.std(
                y_pred
            )
        )

        pred_range = float(
            np.max(
                y_pred
            )
            - np.min(
                y_pred
            )
        )

        print(
            "  Test prediction diagnostic: "
            f"mean={np.mean(y_pred):.6f}, "
            f"std={pred_std:.6e}, "
            f"min={np.min(y_pred):.6f}, "
            f"max={np.max(y_pred):.6f}, "
            f"range={pred_range:.6e}"
        )

        if (
            pred_std < 1e-8
            or pred_range < 1e-8
        ):
            print(
                "  WARNING: near-constant test prediction detected; "
                "Pearson may be undefined."
            )

        fold_metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        fold_results.append({
            "Model":
                "DeepGS",
            "Dataset":
                "Wheat2000",
            "Trait":
                trait_name,
            "Fold":
                fold_idx,
            "Outer_Train_N":
                len(
                    outer_train_idx
                ),
            "Inner_Train_N":
                len(
                    inner_train_idx
                ),
            "Val_N":
                len(
                    val_idx
                ),
            "Test_N":
                len(
                    test_idx
                ),
            "SNPs":
                snp_dim,
            **fold_metrics,
            "Best_Val_MAE":
                best_val_mae,
            "Best_Round":
                best_round,
            "Stopped_Round":
                round_idx,
            "Max_Num_Rounds":
                NUM_ROUNDS,
            "Early_Stop_Patience":
                EARLY_STOP_PATIENCE,
            "Min_Delta":
                MIN_DELTA,
            "Batch_Size":
                BATCH_SIZE,
            "Learning_Rate":
                LEARNING_RATE,
            "Momentum":
                MOMENTUM,
            "Weight_Decay":
                WEIGHT_DECAY,
            "Seed":
                random_state,
            "Outer_Split":
                "DNNGP_exact_KFold_10_shuffle_seed42",
            "X_Processing":
                "OuterInnerTrain_ImputerOnly_RawCoding",
            "Y_Processing":
                "RawPhenotype_NoScaling",
            "Optimizer":
                "SGD",
            "Loss":
                "MAE_L1",
            "Prediction_Std":
                pred_std,
            "Prediction_Range":
                pred_range,
            "Fold_Parity":
                FOLD_PARITY,
            "CUDA_Visible_Devices":
                os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "",
                ),
        })

        # Held-out prediction table
        for (
            sample_idx,
            obs,
            pred,
        ) in zip(
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
                    int(
                        sample_idx
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

        # Membership includes inner train / val / test.
        for sample_idx in inner_train_idx:
            membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(
                        sample_idx
                    ),
                "Set":
                    "Train",
            })

        for sample_idx in val_idx:
            membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(
                        sample_idx
                    ),
                "Set":
                    "Validation",
            })

        for sample_idx in test_idx:
            membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(
                        sample_idx
                    ),
                "Set":
                    "Test",
            })

        print(
            f"  Pearson r = "
            f"{fold_metrics['Pearson']:.4f}"
        )
        print(
            f"  R2        = "
            f"{fold_metrics['R2']:.4f}"
        )
        print(
            f"  MSE       = "
            f"{fold_metrics['MSE']:.4f}"
        )
        print(
            f"  RMSE      = "
            f"{fold_metrics['RMSE']:.4f}"
        )
        print(
            f"  nRMSE     = "
            f"{fold_metrics['nRMSE']:.4f}"
        )
        print(
            f"  MAE       = "
            f"{fold_metrics['MAE']:.4f}"
        )

        del (
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            train_ds,
            val_ds,
            test_ds,
            X_train,
            X_val,
            X_test,
            best_state,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # ========================================================
    # Save this GPU shard
    # ========================================================
    fold_df = pd.DataFrame(
        fold_results
    )

    pred_df = (
        pd.DataFrame(
            predictions
        )
        .sort_values(
            [
                "Fold",
                "Sample_Row_0Based",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    membership_df = (
        pd.DataFrame(
            membership
        )
        .sort_values(
            [
                "Fold",
                "Set",
                "Sample_Row_0Based",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    fold_name = (
        f"DeepGS_Wheat2000_"
        f"{trait_name}_FoldResults.csv"
    )

    pred_name = (
        f"DeepGS_Wheat2000_"
        f"{trait_name}_Predictions.csv"
    )

    mem_name = (
        f"DeepGS_Wheat2000_"
        f"{trait_name}_10FoldMembership.csv"
    )

    fold_df.to_csv(
        output_dir
        / fold_name,
        index=False,
    )

    pred_df.to_csv(
        output_dir
        / pred_name,
        index=False,
    )

    membership_df.to_csv(
        output_dir
        / mem_name,
        index=False,
    )

    summary = {
        "Model":
            "DeepGS",
        "Dataset":
            "Wheat2000",
        "Trait":
            trait_name,
        "N":
            n_samples,
        "SNPs":
            snp_dim,
        "Completed_Folds":
            len(
                fold_df
            ),
        "Expected_Total_Folds":
            N_SPLITS,
        "Fold_Parity":
            FOLD_PARITY,
        "Max_Num_Rounds":
            NUM_ROUNDS,
        "Early_Stop_Patience":
            EARLY_STOP_PATIENCE,
        "Min_Delta":
            MIN_DELTA,
        "Batch_Size":
            BATCH_SIZE,
        "Learning_Rate":
            LEARNING_RATE,
        "Momentum":
            MOMENTUM,
        "Weight_Decay":
            WEIGHT_DECAY,
        "Seed":
            random_state,
        "Outer_Split":
            "DNNGP_exact_KFold_10_shuffle_seed42",
        "X_Processing":
            "OuterInnerTrain_ImputerOnly_RawCoding",
        "Y_Processing":
            "RawPhenotype_NoScaling",
        "Optimizer":
            "SGD",
        "Loss":
            "MAE_L1",
    }

    for col in [
        "Pearson",
        "R2",
        "MSE",
        "RMSE",
        "nRMSE",
        "MAE",
    ]:
        summary[
            f"{col}_Mean"
        ] = float(
            fold_df[
                col
            ].mean()
        )

        summary[
            f"{col}_SD"
        ] = float(
            fold_df[
                col
            ].std(
                ddof=1
            )
        )

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        output_dir
        / (
            f"DeepGS_Wheat2000_"
            f"{trait_name}_Summary.csv"
        ),
        index=False,
    )

    print(
        "\n"
        + "=" * 92
    )
    print(
        f"{trait_name} | shard summary "
        f"({len(fold_df)} folds completed)"
    )
    print(
        "=" * 92
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

    return (
        fold_df,
        pred_df,
        summary_df,
    )


# ============================================================
# 8. Run all six traits
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

    summaries = []
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
            + "#" * 104
        )
        print(
            f"Trait {trait_idx}/"
            f"{len(phenotype_files)}: "
            f"{trait_name}"
        )
        print(
            f"Phenotype file: "
            f"{phenotype_file}"
        )
        print(
            "#" * 104
        )

        try:
            _, _, summary_df = run_trait_10fold(
                genotype_file=
                    genotype_file,
                phenotype_file=
                    phenotype_file,
                trait_name=
                    trait_name,
                output_dir=
                    output_root
                    / trait_name,
                random_state=
                    random_state,
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
                    "DeepGS_Wheat2000_"
                    "6Traits_Summary_CHECKPOINT.csv"
                ),
                index=False,
            )

        except Exception as exc:
            print(
                f"[ERROR] {trait_name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
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

    if summaries:
        combined = pd.concat(
            summaries,
            ignore_index=True,
        )

        combined.to_csv(
            output_root
            / "DeepGS_Wheat2000_6Traits_Summary.csv",
            index=False,
        )

    else:
        combined = pd.DataFrame()

    if failed:
        pd.DataFrame(
            failed
        ).to_csv(
            output_root
            / "DeepGS_Wheat2000_FailedTraits.csv",
            index=False,
        )

    return combined


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    set_seed(
        RANDOM_STATE
    )

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

    PHENOTYPE_SPECS = {
        "tkw": (
            "2000_1_phe_tkw.txt",
            "2000_1_phe_*.txt",
        ),
        "testw": (
            "2000_2_phe_testw.txt",
            "2000_2_phe_*.txt",
        ),
        "length": (
            "2000_3_phe_length.txt",
            "2000_3_phe_*.txt",
        ),
        "width": (
            "2000_4_phe_width.txt",
            "2000_4_phe_*.txt",
        ),
        "hardness": (
            "2000_5_phe_hardness.txt",
            "2000_5_phe_*.txt",
        ),
        "Prot": (
            "2000_6_phe_Prot.txt",
            "2000_6_phe_*.txt",
        ),
    }

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}"
        )

    if not GENOTYPE_FILE.exists():
        raise FileNotFoundError(
            f"Genotype file not found: {GENOTYPE_FILE}"
        )

    PHENOTYPE_FILES = {
        trait:
            resolve_trait_file(
                DATA_DIR,
                exact_name,
                pattern,
            )
        for trait, (
            exact_name,
            pattern,
        ) in PHENOTYPE_SPECS.items()
    }

    print(
        "\nResolved Wheat2000 files:"
    )

    print(
        f"  Genotype: "
        f"{GENOTYPE_FILE}"
    )

    for trait, path in (
        PHENOTYPE_FILES.items()
    ):
        print(
            f"  {trait:<10}: "
            f"{path.name}"
        )

    OUTPUT_ROOT = (
        BASE_DIR
        / (
            "results_deepgs_wheat2000_"
            "DNNGPsame10fold_native"
        )
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
