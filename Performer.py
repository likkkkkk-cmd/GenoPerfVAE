# -*- coding: utf-8 -*-
r"""
Performer_Wheat2000_6Traits_10Fold_FairComparison.py

Pure / Standard Performer on Wheat2000: six traits, controlled 10-fold CV.

This script keeps the user's supplied Pure Performer architecture and
model-specific training settings, while adapting the data loader to the
Wheat2000 benchmark.

Expected Wheat2000 files
------------------------
Genotype:
    2000gene.pkl

Phenotypes:
    2000_1_phe_tkw.txt
    2000_2_phe_testw.txt
    2000_3_phe_length.txt
    2000_4_phe_width.txt
    2000_5_phe_hardness.txt
    2000_6_phe_Prot.txt

For phenotype files 2-6, exact filenames are attempted first. If an exact
filename is unavailable, the script accepts a unique corresponding pattern:
    2000_2_phe_*.txt
    ...
    2000_6_phe_*.txt

Traits
------
    tkw
    testw
    length
    width
    hardness
    Prot

Data convention
---------------
- Genotype .pkl is loaded with pd.read_pickle(...).values.
- Phenotypes are read as the first numeric value from each non-empty line.
- Genotype and phenotype are aligned by row order.
- If their row counts differ, both are truncated to the common count and a
  warning is printed.
- Missing genotype processing is done INSIDE CV only.

Cross-validation
----------------
Outer:
    KFold(n_splits=10, shuffle=True, random_state=42)

Inside each outer-training fold:
    85% inner-train / 15% validation

Preprocessing is fitted ONLY on inner-train:
    SimpleImputer(strategy="most_frequent")
    StandardScaler(X)
    StandardScaler(y)

Validation is used only for early stopping / scheduler decisions.
The held-out outer test fold never contributes to imputation, scaling,
training, or model selection.

Pure Performer architecture
---------------------------
Adaptive SNP chunking:
    target_chunk_size = 128
    performer_dim      = 256
    heads              = 8
    depth              = 4
    dropout            = 0.3

Prediction head:
    256 -> 128 -> 64 -> 1
    BatchNorm1d(128)
    GELU / dropout

Training
--------
    optimizer       = AdamW
    lr              = 1e-4
    weight_decay    = 1e-2
    batch_size      = 48
    max_epochs      = 200
    loss            = MSE
    gradient clip   = 1.0
    LR scheduler    = ReduceLROnPlateau
    LR patience     = 10
    LR factor       = 0.5
    early stop      = 30 epochs

Metrics
-------
Per fold and 10-fold mean ± sample SD:
    Pearson
    R2
    MSE
    RMSE
    nRMSE = RMSE / SD(observed phenotype in held-out fold)
    MAE

Also saves:
    - outer/inner split memberships
    - held-out predictions
    - fold-level metrics
    - OOF metrics
    - six-trait combined summary

Data directory search order
---------------------------
If this script is placed in:
    E:\VGMP\Extended Comparison\Performer\

it searches:
    1) <script_dir>\data\
    2) <script_dir>\..\data\

Thus both layouts are supported:
    Performer\script.py + Performer\data\...
or:
    Extended Comparison\Performer\script.py
    Extended Comparison\data\...
"""

from __future__ import annotations

import copy
import gc
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from performer_pytorch import Performer
except ImportError as exc:
    raise ImportError(
        "performer-pytorch is required. "
        "Install it with: pip install performer-pytorch"
    ) from exc

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================

RANDOM_STATE = 42
N_SPLITS = 10
INNER_VAL_RATIO = 0.15

TRAIT_SPECS = {
    "tkw": {
        "exact": "2000_1_phe_tkw.txt",
        "pattern": "2000_1_phe_*.txt",
    },
    "testw": {
        "exact": "2000_2_phe_testw.txt",
        "pattern": "2000_2_phe_*.txt",
    },
    "length": {
        "exact": "2000_3_phe_length.txt",
        "pattern": "2000_3_phe_*.txt",
    },
    "width": {
        "exact": "2000_4_phe_width.txt",
        "pattern": "2000_4_phe_*.txt",
    },
    "hardness": {
        "exact": "2000_5_phe_hardness.txt",
        "pattern": "2000_5_phe_*.txt",
    },
    "Prot": {
        "exact": "2000_6_phe_Prot.txt",
        "pattern": "2000_6_phe_*.txt",
    },
}

TARGET_CHUNK_SIZE = 128
PERFORMER_DIM = 256
NUM_HEADS = 8
PERFORMER_DEPTH = 4
DROPOUT = 0.3

BATCH_SIZE = 48
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

LR_PATIENCE = 10
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 30

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


# =============================================================================
# 2. Reproducibility
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 3. Pure Performer architecture
# =============================================================================

class AdaptiveChunkedSNP_Performer(nn.Module):
    def __init__(
        self,
        snp_dim,
        target_chunk_size=128,
        performer_dim=256,
        num_heads=8,
        dropout=0.2,
        depth=4,
    ):
        super().__init__()

        self.snp_dim = int(snp_dim)

        self.chunk_size = int(target_chunk_size)
        while self.chunk_size % num_heads != 0:
            self.chunk_size += 1

        self.num_chunks = int(
            np.ceil(
                self.snp_dim / self.chunk_size
            )
        )

        self.padding_needed = (
            self.num_chunks * self.chunk_size
            - self.snp_dim
        )

        self.chunk_norm = nn.LayerNorm(
            self.chunk_size
        )

        self.pos_emb = nn.Parameter(
            torch.randn(
                1,
                self.num_chunks,
                self.chunk_size,
            )
        )

        self.performer = Performer(
            dim=self.chunk_size,
            depth=depth,
            heads=num_heads,
            dim_head=(
                self.chunk_size
                // num_heads
            ),
            ff_dropout=dropout,
            attn_dropout=dropout,
            causal=False,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(
                self.chunk_size * 2,
                performer_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        self.max_pool = (
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        if self.padding_needed > 0:
            x = F.pad(
                x,
                (
                    0,
                    self.padding_needed,
                ),
            )

        x = x.view(
            batch_size,
            self.num_chunks,
            self.chunk_size,
        )

        x = (
            self.chunk_norm(x)
            + self.pos_emb
        )

        x = self.performer(
            x
        )

        x_t = x.transpose(
            1,
            2,
        )

        avg_f = (
            self.avg_pool(x_t)
            .squeeze(-1)
        )

        max_f = (
            self.max_pool(x_t)
            .squeeze(-1)
        )

        return self.output_proj(
            torch.cat(
                [
                    avg_f,
                    max_f,
                ],
                dim=1,
            )
        )


class PurePerformerPredictor(nn.Module):
    def __init__(
        self,
        snp_dim,
        performer_dim=256,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = (
            AdaptiveChunkedSNP_Performer(
                snp_dim=snp_dim,
                target_chunk_size=
                    TARGET_CHUNK_SIZE,
                performer_dim=
                    performer_dim,
                num_heads=
                    NUM_HEADS,
                dropout=
                    dropout,
                depth=
                    PERFORMER_DEPTH,
            )
        )

        self.head = nn.Sequential(
            nn.Linear(
                performer_dim,
                128,
            ),
            nn.BatchNorm1d(
                128
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                128,
                64,
            ),
            nn.GELU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):
        return self.head(
            self.encoder(x)
        )


# =============================================================================
# 4. Wheat2000 file/path helpers
# =============================================================================

def resolve_data_dir(
    base_dir,
):
    base_dir = Path(
        base_dir
    )

    candidates = [
        base_dir / "data",
        base_dir.parent / "data",
    ]

    for data_dir in candidates:
        if (
            data_dir.exists()
            and (
                data_dir
                / "2000gene.pkl"
            ).exists()
        ):
            print(
                "Resolved Wheat2000 "
                f"data directory: {data_dir}"
            )
            return data_dir

    raise FileNotFoundError(
        "Could not locate Wheat2000 genotype file 2000gene.pkl.\n"
        "Checked:\n"
        + "\n".join(
            f"  {d}"
            for d in candidates
        )
    )


def resolve_trait_file(
    data_dir,
    exact_name,
    pattern,
):
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
            pattern
        )
    )

    if len(matches) == 1:
        print(
            "Using discovered phenotype file: "
            f"{matches[0].name}"
        )
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find '{exact_name}' "
            f"or unique '{pattern}' "
            f"in {data_dir}"
        )

    raise RuntimeError(
        f"Multiple files match '{pattern}': "
        f"{[p.name for p in matches]}"
    )


# =============================================================================
# 5. Wheat2000 data loader
# =============================================================================

def load_genotype(
    genotype_file,
):
    genotype_file = Path(
        genotype_file
    )

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
            f"Expected 2D genotype matrix, "
            f"got shape {X.shape}"
        )

    print(
        f"Genotype shape : "
        f"{X.shape}"
    )
    print(
        f"Missing values : "
        f"{int(np.isnan(X).sum())}"
    )
    print(
        "Genotype range : "
        f"{float(np.nanmin(X)):.6f} "
        f"to "
        f"{float(np.nanmax(X)):.6f}"
    )

    return X


def first_numeric_from_line(
    line,
):
    """
    Return the first token in a text line that can be parsed as float.
    Supports whitespace/comma separated phenotype text files.
    """
    stripped = (
        line.strip()
    )

    if not stripped:
        return None

    tokens = re.split(
        r"[\s,]+",
        stripped,
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


def load_phenotype_vector(
    phenotype_file,
):
    phenotype_file = Path(
        phenotype_file
    )

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

    if len(values) == 0:
        raise ValueError(
            f"No numeric phenotype values found "
            f"in {phenotype_file}"
        )

    y = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )

    return y


def load_wheat2000_trait(
    genotype_file,
    phenotype_file,
    trait_name,
):
    print(
        "\n"
        + "=" * 96
    )
    print(
        f"Loading Wheat2000 | "
        f"{trait_name}"
    )
    print(
        f"Phenotype file: "
        f"{Path(phenotype_file).name}"
    )
    print(
        "=" * 96
    )

    X = load_genotype(
        genotype_file
    )

    y = load_phenotype_vector(
        phenotype_file
    )

    if X.shape[0] != y.shape[0]:
        n_common = min(
            X.shape[0],
            y.shape[0],
        )

        print(
            "WARNING: genotype/phenotype "
            "row counts differ; using common "
            f"row count {n_common}. "
            f"Genotype={X.shape[0]}, "
            f"phenotype={y.shape[0]}"
        )

        X = X[
            :n_common
        ]

        y = y[
            :n_common
        ]

    if not np.all(
        np.isfinite(y)
    ):
        raise ValueError(
            f"{trait_name} phenotype "
            "contains non-finite values."
        )

    sample_rows = np.arange(
        X.shape[0],
        dtype=int,
    )

    print(
        f"Aligned samples     : "
        f"{X.shape[0]}"
    )
    print(
        f"SNPs                : "
        f"{X.shape[1]}"
    )
    print(
        "Phenotype mean/std  : "
        f"{float(np.mean(y)):.6f} / "
        f"{float(np.std(y)):.6f}"
    )

    return (
        X,
        y,
        sample_rows,
    )


# =============================================================================
# 6. Metrics
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
        or np.std(
            y_true
        ) < 1e-12
        or np.std(
            y_pred
        ) < 1e-12
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


# =============================================================================
# 7. Helpers
# =============================================================================

def safe_train_batch_size(
    n_train,
    preferred=48,
):
    """
    Avoid BatchNorm1d receiving a final training batch of one sample.
    """
    bs = min(
        int(preferred),
        int(n_train),
    )

    while (
        bs > 2
        and n_train % bs == 1
    ):
        bs -= 1

    return bs


@torch.no_grad()
def evaluate_mse(
    model,
    loader,
    device,
):
    model.eval()

    sse = 0.0
    n = 0

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

        sse += float(
            torch.sum(
                (
                    pred
                    - by
                ) ** 2
            ).cpu()
        )

        n += int(
            by.numel()
        )

    return (
        sse
        / max(
            n,
            1,
        )
    )


# =============================================================================
# 8. One trait: 10-fold CV
# =============================================================================

def run_trait_10fold(
    X_all,
    y_all,
    sample_rows,
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

    n_samples = X_all.shape[
        0
    ]
    snp_dim = X_all.shape[
        1
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        + "#" * 104
    )
    print(
        "Pure Performer | "
        f"Wheat2000 | "
        f"{trait_name}"
    )
    print(
        f"Device       : "
        f"{device}"
    )
    print(
        f"Samples      : "
        f"{n_samples}"
    )
    print(
        f"SNPs         : "
        f"{snp_dim}"
    )
    print(
        "Outer CV     : "
        f"10-fold, seed="
        f"{random_state}"
    )
    print(
        f"Inner val    : "
        f"{INNER_VAL_RATIO:.0%}"
    )
    print(
        "X processing : "
        "inner-train most_frequent "
        "+ StandardScaler"
    )
    print(
        "y processing : "
        "inner-train StandardScaler"
    )
    print(
        "#" * 104
    )

    kfold = KFold(
        n_splits=
            N_SPLITS,
        shuffle=True,
        random_state=
            random_state,
    )

    fold_rows = []
    prediction_rows = []
    membership_rows = []

    for fold_idx, (
        outer_train_idx,
        test_idx,
    ) in enumerate(
        kfold.split(
            X_all
        ),
        start=1,
    ):
        set_seed(
            random_state
        )

        print(
            "\n"
            + "-" * 104
        )
        print(
            f"{trait_name} | "
            f"Fold "
            f"{fold_idx}/"
            f"{N_SPLITS}"
        )
        print(
            "-" * 104
        )

        (
            inner_train_idx,
            val_idx,
        ) = train_test_split(
            outer_train_idx,
            test_size=
                INNER_VAL_RATIO,
            random_state=
                random_state,
            shuffle=True,
        )

        X_train_raw = (
            X_all[
                inner_train_idx
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        X_val_raw = (
            X_all[
                val_idx
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        X_test_raw = (
            X_all[
                test_idx
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        y_train_raw = (
            y_all[
                inner_train_idx
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        y_val_raw = (
            y_all[
                val_idx
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        y_test_raw = (
            y_all[
                test_idx
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        # -----------------------------------------------------
        # Fold-safe imputation.
        # -----------------------------------------------------
        imputer = SimpleImputer(
            strategy=
                "most_frequent"
        )

        X_train_imp = (
            imputer
            .fit_transform(
                X_train_raw
            )
            .astype(
                np.float32
            )
        )

        X_val_imp = (
            imputer
            .transform(
                X_val_raw
            )
            .astype(
                np.float32
            )
        )

        X_test_imp = (
            imputer
            .transform(
                X_test_raw
            )
            .astype(
                np.float32
            )
        )

        if (
            X_train_imp.shape[
                1
            ]
            != snp_dim
        ):
            raise RuntimeError(
                "SimpleImputer changed "
                "SNP dimensionality. "
                "A marker may be entirely "
                "missing in inner training."
            )

        # -----------------------------------------------------
        # Preserve supplied Pure Performer X processing:
        # StandardScaler, but fit only on inner train.
        # -----------------------------------------------------
        scaler_x = StandardScaler()

        X_train = (
            scaler_x
            .fit_transform(
                X_train_imp
            )
            .astype(
                np.float32
            )
        )

        X_val = (
            scaler_x
            .transform(
                X_val_imp
            )
            .astype(
                np.float32
            )
        )

        X_test = (
            scaler_x
            .transform(
                X_test_imp
            )
            .astype(
                np.float32
            )
        )

        scaler_y = StandardScaler()

        y_train = (
            scaler_y
            .fit_transform(
                y_train_raw
            )
            .astype(
                np.float32
            )
        )

        y_val = (
            scaler_y
            .transform(
                y_val_raw
            )
            .astype(
                np.float32
            )
        )

        y_test_scaled = (
            scaler_y
            .transform(
                y_test_raw
            )
            .astype(
                np.float32
            )
        )

        actual_batch = (
            safe_train_batch_size(
                len(
                    inner_train_idx
                ),
                BATCH_SIZE,
            )
        )

        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            random_state
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(
                    X_train
                ),
                torch.from_numpy(
                    y_train
                ),
            ),
            batch_size=
                actual_batch,
            shuffle=True,
            generator=
                generator,
            num_workers=
                NUM_WORKERS,
            pin_memory=
                PIN_MEMORY,
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(
                    X_val
                ),
                torch.from_numpy(
                    y_val
                ),
            ),
            batch_size=
                BATCH_SIZE,
            shuffle=False,
            num_workers=
                NUM_WORKERS,
            pin_memory=
                PIN_MEMORY,
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(
                    X_test
                ),
                torch.from_numpy(
                    y_test_scaled
                ),
            ),
            batch_size=
                BATCH_SIZE,
            shuffle=False,
            num_workers=
                NUM_WORKERS,
            pin_memory=
                PIN_MEMORY,
        )

        model = (
            PurePerformerPredictor(
                snp_dim=
                    snp_dim,
                performer_dim=
                    PERFORMER_DIM,
                dropout=
                    DROPOUT,
            )
            .to(
                device
            )
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
            f"Outer train/test: "
            f"{len(outer_train_idx)} / "
            f"{len(test_idx)}"
        )
        print(
            f"Inner train/val : "
            f"{len(inner_train_idx)} / "
            f"{len(val_idx)}"
        )
        print(
            "Performer shape : "
            f"chunk="
            f"{model.encoder.chunk_size}, "
            f"chunks="
            f"{model.encoder.num_chunks}, "
            f"padding="
            f"{model.encoder.padding_needed}"
        )
        print(
            f"Parameters      : "
            f"{total_params / 1e6:.3f} M"
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=
                LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY,
        )

        scheduler = (
            optim.lr_scheduler
            .ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=
                    LR_PATIENCE,
                factor=
                    LR_FACTOR,
            )
        )

        criterion = nn.MSELoss()

        best_val_loss = np.inf
        best_state = None
        best_epoch = 0
        patience_counter = 0
        stopped_epoch = 0

        pbar = tqdm(
            range(
                1,
                MAX_EPOCHS
                + 1,
            ),
            desc=(
                f"Performer "
                f"{trait_name} "
                f"Fold {fold_idx}"
            ),
            leave=False,
        )

        for epoch in pbar:
            model.train()

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
                    by,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

            val_loss = evaluate_mse(
                model,
                val_loader,
                device,
            )

            scheduler.step(
                val_loss
            )

            if (
                np.isfinite(
                    val_loss
                )
                and val_loss
                < best_val_loss
            ):
                best_val_loss = float(
                    val_loss
                )

                best_epoch = int(
                    epoch
                )

                best_state = copy.deepcopy(
                    {
                        key:
                            value
                            .detach()
                            .cpu()
                            .clone()
                        for (
                            key,
                            value,
                        ) in (
                            model
                            .state_dict()
                            .items()
                        )
                    }
                )

                patience_counter = 0

            else:
                patience_counter += 1

            pbar.set_postfix({
                "ValMSE":
                    f"{val_loss:.4f}",
                "Best":
                    f"{best_val_loss:.4f}",
                "Wait":
                    (
                        f"{patience_counter}/"
                        f"{EARLY_STOP_PATIENCE}"
                    ),
            })

            stopped_epoch = (
                epoch
            )

            if (
                patience_counter
                >= EARLY_STOP_PATIENCE
            ):
                break

        if best_state is None:
            raise RuntimeError(
                f"No finite validation "
                f"loss for {trait_name}, "
                f"fold {fold_idx}."
            )

        model.load_state_dict(
            best_state
        )

        model.to(
            device
        )

        model.eval()

        pred_scaled_list = []

        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(
                    device,
                    non_blocking=True,
                )

                pred_scaled_list.append(
                    model(
                        bx
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        y_pred_scaled = (
            np.concatenate(
                pred_scaled_list,
                axis=0,
            )
        )

        y_pred = (
            scaler_y
            .inverse_transform(
                y_pred_scaled
            )
            .reshape(-1)
        )

        y_true = (
            y_test_raw
            .reshape(-1)
        )

        fold_metrics = (
            calculate_metrics(
                y_true,
                y_pred,
            )
        )

        prediction_std = float(
            np.std(
                y_pred
            )
        )

        fold_rows.append({
            "Model":
                "PurePerformer",
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
            "Prediction_Std":
                prediction_std,
            "Best_Val_MSE":
                best_val_loss,
            "Best_Epoch":
                best_epoch,
            "Stopped_Epoch":
                stopped_epoch,
            "Chunk_Size":
                model.encoder.chunk_size,
            "Num_Chunks":
                model.encoder.num_chunks,
            "Padding":
                model.encoder.padding_needed,
            "Performer_Dim":
                PERFORMER_DIM,
            "Heads":
                NUM_HEADS,
            "Depth":
                PERFORMER_DEPTH,
            "Dropout":
                DROPOUT,
            "Batch_Size":
                actual_batch,
            "Learning_Rate":
                LEARNING_RATE,
            "Weight_Decay":
                WEIGHT_DECAY,
            "Total_Params_M":
                total_params
                / 1e6,
            "Trainable_Params_M":
                trainable_params
                / 1e6,
            "Seed":
                random_state,
            "X_Processing":
                (
                    "InnerTrain_Imputer"
                    "+StandardScaler"
                ),
            "Y_Processing":
                "InnerTrain_StandardScaler",
            "Loss":
                "MSE",
        })

        for (
            sample_idx,
            obs,
            pred,
        ) in zip(
            test_idx,
            y_true,
            y_pred,
        ):
            prediction_rows.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(
                        sample_rows[
                            sample_idx
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

        for sample_idx in (
            inner_train_idx
        ):
            membership_rows.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(
                        sample_rows[
                            sample_idx
                        ]
                    ),
                "Set":
                    "InnerTrain",
            })

        for sample_idx in val_idx:
            membership_rows.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(
                        sample_rows[
                            sample_idx
                        ]
                    ),
                "Set":
                    "Validation",
            })

        for sample_idx in test_idx:
            membership_rows.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Row_0Based":
                    int(
                        sample_rows[
                            sample_idx
                        ]
                    ),
                "Set":
                    "Test",
            })

        print(
            f"Best epoch = "
            f"{best_epoch}"
        )
        print(
            f"Pearson    = "
            f"{fold_metrics['Pearson']:.4f}"
        )
        print(
            f"R2         = "
            f"{fold_metrics['R2']:.4f}"
        )
        print(
            f"MSE        = "
            f"{fold_metrics['MSE']:.4f}"
        )
        print(
            f"RMSE       = "
            f"{fold_metrics['RMSE']:.4f}"
        )
        print(
            f"nRMSE      = "
            f"{fold_metrics['nRMSE']:.4f}"
        )
        print(
            f"MAE        = "
            f"{fold_metrics['MAE']:.4f}"
        )

        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            test_loader,
            X_train_imp,
            X_val_imp,
            X_test_imp,
            X_train,
            X_val,
            X_test,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # =========================================================================
    # Save detailed outputs
    # =========================================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = (
        pd.DataFrame(
            prediction_rows
        )
        .sort_values(
            by=
                "Sample_Row_0Based"
        )
    )

    membership_df = pd.DataFrame(
        membership_rows
    )

    fold_df.to_csv(
        output_dir
        / (
            f"Performer_Wheat2000_"
            f"{trait_name}_"
            "FoldResults.csv"
        ),
        index=False,
    )

    pred_df.to_csv(
        output_dir
        / (
            f"Performer_Wheat2000_"
            f"{trait_name}_"
            "Predictions.csv"
        ),
        index=False,
    )

    membership_df.to_csv(
        output_dir
        / (
            f"Performer_Wheat2000_"
            f"{trait_name}_"
            "10FoldMembership.csv"
        ),
        index=False,
    )

    # =========================================================================
    # Summary
    # =========================================================================

    summary = {
        "Model":
            "PurePerformer",
        "Dataset":
            "Wheat2000",
        "Trait":
            trait_name,
        "N":
            n_samples,
        "SNPs":
            snp_dim,
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

    oof_metrics = (
        calculate_metrics(
            pred_df[
                "Observed"
            ].to_numpy(),
            pred_df[
                "Predicted"
            ].to_numpy(),
        )
    )

    for (
        metric,
        value,
    ) in (
        oof_metrics.items()
    ):
        summary[
            f"OOF_{metric}"
        ] = value

    summary[
        "Best_Epoch_Mean"
    ] = fold_df[
        "Best_Epoch"
    ].mean()

    summary[
        "Best_Epoch_SD"
    ] = fold_df[
        "Best_Epoch"
    ].std(
        ddof=1
    )

    summary[
        "Chunk_Size"
    ] = int(
        fold_df[
            "Chunk_Size"
        ].iloc[0]
    )

    summary[
        "Num_Chunks"
    ] = int(
        fold_df[
            "Num_Chunks"
        ].iloc[0]
    )

    summary[
        "Padding"
    ] = int(
        fold_df[
            "Padding"
        ].iloc[0]
    )

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        output_dir
        / (
            f"Performer_Wheat2000_"
            f"{trait_name}_"
            "Summary.csv"
        ),
        index=False,
    )

    print(
        "\n"
        + "=" * 96
    )
    print(
        f"{trait_name} | "
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

    return (
        fold_df,
        pred_df,
        summary_df,
    )


# =============================================================================
# 9. Six-trait runner
# =============================================================================

def run_all_traits(
    data_dir,
    output_root,
    random_state=42,
):
    data_dir = Path(
        data_dir
    )

    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    genotype_file = (
        data_dir
        / "2000gene.pkl"
    )

    if not genotype_file.exists():
        raise FileNotFoundError(
            f"Genotype file not found: "
            f"{genotype_file}"
        )

    all_summaries = []
    failed = []

    for (
        trait_idx,
        (
            trait_name,
            spec,
        ),
    ) in enumerate(
        TRAIT_SPECS.items(),
        start=1,
    ):
        print(
            "\n\n"
            + "#" * 108
        )
        print(
            f"Trait "
            f"{trait_idx}/"
            f"{len(TRAIT_SPECS)}: "
            f"{trait_name}"
        )
        print(
            "#" * 108
        )

        try:
            phenotype_file = (
                resolve_trait_file(
                    data_dir=
                        data_dir,
                    exact_name=
                        spec[
                            "exact"
                        ],
                    pattern=
                        spec[
                            "pattern"
                        ],
                )
            )

            (
                X,
                y,
                sample_rows,
            ) = load_wheat2000_trait(
                genotype_file=
                    genotype_file,
                phenotype_file=
                    phenotype_file,
                trait_name=
                    trait_name,
            )

            trait_dir = (
                output_root
                / trait_name
            )

            trait_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame({
                "Sample_Row_0Based":
                    sample_rows,
                trait_name:
                    y.reshape(
                        -1
                    ),
            }).to_csv(
                trait_dir
                / (
                    f"Wheat2000_"
                    f"{trait_name}_"
                    "AlignedSamples.csv"
                ),
                index=False,
            )

            (
                _,
                _,
                summary_df,
            ) = run_trait_10fold(
                X_all=
                    X,
                y_all=
                    y,
                sample_rows=
                    sample_rows,
                trait_name=
                    trait_name,
                output_dir=
                    trait_dir,
                random_state=
                    random_state,
            )

            summary_df[
                "Phenotype_File"
            ] = str(
                phenotype_file
            )

            all_summaries.append(
                summary_df
            )

            pd.concat(
                all_summaries,
                ignore_index=True,
            ).to_csv(
                output_root
                / (
                    "Performer_Wheat2000_"
                    "6Traits_"
                    "Summary_CHECKPOINT.csv"
                ),
                index=False,
            )

            del X, y
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"[ERROR] "
                f"{trait_name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed.append({
                "Trait":
                    trait_name,
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
            / (
                "Performer_Wheat2000_"
                "6Traits_Summary.csv"
            ),
            index=False,
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
            "Best_Epoch_Mean",
        ]

        print(
            "\n"
            + "=" * 126
        )
        print(
            "Pure Performer Wheat2000 "
            "six-trait 10-fold summary"
        )
        print(
            "=" * 126
        )
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
            / (
                "Performer_Wheat2000_"
                "FailedTraits.csv"
            ),
            index=False,
        )

    return combined


# =============================================================================
# 10. Main
# =============================================================================

if __name__ == "__main__":

    set_seed(
        RANDOM_STATE
    )

    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parent
    )

    DATA_DIR = resolve_data_dir(
        BASE_DIR
    )

    OUTPUT_ROOT = (
        BASE_DIR
        / (
            "results_performer_"
            "wheat2000_6traits"
        )
    )

    run_all_traits(
        data_dir=
            DATA_DIR,
        output_root=
            OUTPUT_ROOT,
        random_state=
            RANDOM_STATE,
    )
