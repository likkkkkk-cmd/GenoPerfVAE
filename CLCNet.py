#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLCNet_Wheat2000_10Fold_AllTraits.py

Wheat2000 adapter for the official CLCNet repository.

What is retained from the uploaded CLCNet code
-----------------------------------------------
1. CLCNet architecture:
   input -> [4096, 2048, 1024] shared representation
   + residual projection
   -> phenotype prediction head
   -> auxiliary contrastive head

2. Training objective:
   MSE(pred1, y1) + MSE(pred2, y2)
   + mean[(distance(aux1, aux2) - |y1-y2|^2)^2]

3. Optimizer / scheduler defaults:
   SGD(lr=1e-4, momentum=0.9)
   ReduceLROnPlateau(factor=0.8, patience=3, min_lr=1e-6)

4. LightGBM feature selection:
   global feature importance + optional chromosome-wise importance,
   selecting SNPs with gain importance > 0.

Changes made for a fair Wheat2000 benchmark
-------------------------------------------
- Directly reads the Wheat2000 .pkl genotype used by GenoPerfVAE.
- Automatically discovers phenotype files named 2000_*_phe_*.txt.
- Uses 10-fold CV with random_state=42.
- LightGBM feature selection is performed INSIDE each outer training fold
  to prevent test-fold leakage.
- Phenotype standardization is fitted on the outer training fold only.
- Test samples are never used for feature selection or fitting.
- Each fold starts from a fresh CLCNet model.
- Pearson r, R2 and MAE are reported on the original phenotype scale.
- Results are checkpointed after every fold and every trait.

Chromosome-aware feature selection
----------------------------------
If chromosome labels are available, set CHROMOSOME_FILE to a text file
containing one chromosome label per SNP in the same order as genotype
columns. Alternatively, if the genotype DataFrame column names contain
patterns such as chr1, chr2, ..., chromosome labels are detected
automatically.

If chromosome labels cannot be obtained, the script still runs using
the official GLOBAL LightGBM feature-selection component only. In that
case it prints a warning because this is not the full chromosome-aware
preprocessing of CLCNet.

Required packages
-----------------
torch, numpy, pandas, scikit-learn, lightgbm, tqdm
"""

import os
import re
import gc
import glob
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightgbm as lgb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error


# =============================================================================
# Global configuration
# =============================================================================

RANDOM_STATE = 42
N_SPLITS = 10

EPOCHS = 100
BATCH_SIZE = 32

LEARNING_RATE = 1e-4
MOMENTUM = 0.9

SHARED_DIM = [4096, 2048, 1024]

LGB_NUM_BOOST_ROUND = 100
LGB_IMPORTANCE_THRESHOLD = 0.0

# Official selector keeps all features with importance > threshold.
# Set to an integer (e.g. 10000) only if GPU memory becomes limiting.
MAX_SELECTED_FEATURES = None

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility over maximum speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# CLCNet model -- retained from uploaded model.py
# =============================================================================

class CLCNet(nn.Module):
    def __init__(
        self,
        shuffle=False,
        seed=42,
        input_dim=67500,
        shared_dim=(4096, 2048, 1024),
    ):
        super().__init__()

        self.shuffle = shuffle
        self.seed = seed

        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim[0]),
            nn.ReLU(),
            nn.Linear(shared_dim[0], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[2]),
        )

        self.shared_layer_1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim[2]),
        )

        self.main_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], 1),
        )

        self.aux_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[0]),
        )

    def forward(self, x):
        # The official code has a shuffle option, but training uses shuffle=False.
        shared_features = self.shared_layer(x)
        shared_features = shared_features + self.shared_layer_1(x)

        main_output = self.main_task_layer(shared_features)
        aux_output = self.aux_task_layer(shared_features)

        aux_output = F.normalize(aux_output, p=2, dim=1)

        return main_output, aux_output


# =============================================================================
# Official CLCNet contrastive objective
# =============================================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, phenotype_diff):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return torch.mean(
            (euclidean_distance - phenotype_diff) ** 2
        )


class PairDataset(Dataset):
    """
    Matches the official MyDatasetTrain behavior:
    each sample is paired with one randomly selected sample.
    """

    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(
            np.asarray(y, dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        idx1 = index
        idx2 = torch.randint(
            0,
            len(self.X),
            (1,),
        ).item()

        feature1 = self.X[idx1]
        feature2 = self.X[idx2]

        label1 = self.y[idx1]
        label2 = self.y[idx2]

        # Same definition as uploaded train.py
        label_diff = torch.abs(
            label1 - label2
        ) ** 2

        return (
            feature1,
            feature2,
            label1.unsqueeze(0),
            label2.unsqueeze(0),
            label_diff,
        )


class PredictionDataset(Dataset):
    def __init__(self, X):
        self.X = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]


# =============================================================================
# Wheat2000 loading
# =============================================================================

def load_wheat2000_genotype(geno_path):
    """
    Compatible with the user's Wheat2000 2000gene.pkl.

    Returns
    -------
    X : np.ndarray [n_samples, n_snps]
    marker_names : list[str]
    dataframe_columns_available : bool
    """

    geno_path = Path(geno_path)

    if not geno_path.exists():
        raise FileNotFoundError(
            f"Genotype file not found: {geno_path}"
        )

    if geno_path.suffix.lower() in {
        ".pkl",
        ".pickle",
    }:
        obj = pd.read_pickle(geno_path)

        if isinstance(obj, pd.DataFrame):
            X = obj.values
            marker_names = [
                str(c) for c in obj.columns
            ]
            has_column_names = True
        else:
            X = np.asarray(obj)
            marker_names = [
                f"SNP_{i}"
                for i in range(X.shape[1])
            ]
            has_column_names = False

    elif geno_path.suffix.lower() == ".npy":
        X = np.load(geno_path)
        marker_names = [
            f"SNP_{i}"
            for i in range(X.shape[1])
        ]
        has_column_names = False

    else:
        df = pd.read_csv(
            geno_path,
            sep=None,
            engine="python",
        )

        # Match the style of the user's earlier Wheat2000 loader:
        # if a non-genotype identifier column exists, remove it.
        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        if numeric_df.shape[1] == 0:
            raise ValueError(
                "No numeric genotype columns were found."
            )

        X = numeric_df.values
        marker_names = [
            str(c) for c in numeric_df.columns
        ]
        has_column_names = True

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Genotype must be 2-D, got shape {X.shape}"
        )

    print(
        f"Loaded genotype: samples={X.shape[0]}, SNPs={X.shape[1]}"
    )

    return X, marker_names, has_column_names


def load_wheat2000_phenotype(pheno_path):
    """
    Match the user's existing Wheat2000 loader:
    take the first numeric token from each non-empty line.
    """

    values = []

    with open(
        pheno_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            tokens = re.split(
                r"[\s,;\t]+",
                line,
            )

            found = None

            for token in tokens:
                try:
                    found = float(token)
                    break
                except ValueError:
                    continue

            if found is not None:
                values.append(found)

    if not values:
        raise ValueError(
            f"No numeric phenotype values found in {pheno_path}"
        )

    y = np.asarray(
        values,
        dtype=np.float32,
    )

    return y


def infer_trait_name(path):
    """
    Example:
    2000_1_phe_tkw.txt -> tkw
    2000_6_phe_Prot.txt -> Prot
    """

    stem = Path(path).stem

    match = re.search(
        r"_phe_(.+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return stem


def discover_phenotype_files(
    phenotype_dir,
    pattern="2000_*_phe_*.txt",
):
    files = sorted(
        glob.glob(
            str(
                Path(phenotype_dir)
                / pattern
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No phenotype files matching "
            f"{pattern!r} were found in "
            f"{phenotype_dir}"
        )

    trait_files = {}

    for file_path in files:
        trait = infer_trait_name(
            file_path
        )

        trait_files[trait] = file_path

    return trait_files


# =============================================================================
# Chromosome labels
# =============================================================================

def load_chromosome_labels(
    chromosome_file,
    n_snps,
):
    if chromosome_file is None:
        return None

    chromosome_file = Path(
        chromosome_file
    )

    if not chromosome_file.exists():
        raise FileNotFoundError(
            f"Chromosome file not found: "
            f"{chromosome_file}"
        )

    labels = []

    with open(
        chromosome_file,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                labels.append(
                    line.split()[0]
                )

    labels = np.asarray(
        labels,
        dtype=object,
    )

    if len(labels) != n_snps:
        raise ValueError(
            "Chromosome label count does not match "
            f"number of SNPs: {len(labels)} vs {n_snps}"
        )

    return labels


def infer_chromosomes_from_marker_names(
    marker_names,
):
    """
    Attempts to detect chromosome identifiers from marker names.

    Supported examples:
      chr1_xxx
      Chr2:12345
      chromosome3_...
      1_12345   (only when a clear prefix pattern exists)
    """

    labels = []

    for name in marker_names:
        s = str(name)

        match = re.search(
            r"(?:chr(?:omosome)?)[_\-:\s]*"
            r"([0-9]+|[A-Za-z]+)",
            s,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        labels.append(
            match.group(1)
        )

    labels = np.asarray(
        labels,
        dtype=object,
    )

    if len(np.unique(labels)) <= 1:
        return None

    return labels


# =============================================================================
# LightGBM feature selection
# =============================================================================

def _fit_lgb_importance(
    X_train,
    y_train,
    seed,
):
    train_data = lgb.Dataset(
        X_train.astype(np.float32),
        label=y_train.astype(np.float32),
        free_raw_data=True,
    )

    params = {
        "objective": "regression",
        "metric": "mse",
        "force_col_wise": True,
        "device": "cpu",
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": max(
            1,
            (os.cpu_count() or 2) - 1,
        ),
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=LGB_NUM_BOOST_ROUND,
    )

    importance = booster.feature_importance(
        importance_type="gain",
    )

    return importance


def select_clcnet_features(
    X_train,
    y_train,
    chromosome_labels=None,
    seed=42,
    max_selected_features=None,
):
    """
    Fold-wise implementation of the uploaded:
      ChromosomeAwareProcessor.py
      + ChromosomeAwareSelected.py

    It combines:
      1) globally important SNPs
      2) chromosome-specific important SNPs

    Only OUTER TRAINING samples are used.
    """

    n_snps = X_train.shape[1]

    print(
        "  LightGBM global feature selection..."
    )

    global_importance = _fit_lgb_importance(
        X_train,
        y_train,
        seed,
    )

    selected = set(
        np.where(
            global_importance
            > LGB_IMPORTANCE_THRESHOLD
        )[0].tolist()
    )

    global_selected_count = len(
        selected
    )

    local_selected_count = 0

    if chromosome_labels is not None:
        unique_chromosomes = pd.unique(
            chromosome_labels
        )

        print(
            "  Chromosome-aware selection "
            f"across {len(unique_chromosomes)} chromosomes..."
        )

        for chrom in unique_chromosomes:
            chrom_idx = np.where(
                chromosome_labels == chrom
            )[0]

            if len(chrom_idx) == 0:
                continue

            local_importance = _fit_lgb_importance(
                X_train[:, chrom_idx],
                y_train,
                seed,
            )

            local_idx = np.where(
                local_importance
                > LGB_IMPORTANCE_THRESHOLD
            )[0]

            mapped_idx = chrom_idx[
                local_idx
            ]

            before = len(selected)

            selected.update(
                mapped_idx.tolist()
            )

            local_selected_count += (
                len(selected) - before
            )

    selected = np.asarray(
        sorted(selected),
        dtype=np.int64,
    )

    # Safety fallback.
    if len(selected) == 0:
        print(
            "  WARNING: no SNP had positive gain. "
            "Using the single most important SNP."
        )

        selected = np.asarray(
            [
                int(
                    np.argmax(
                        global_importance
                    )
                )
            ],
            dtype=np.int64,
        )

    if (
        max_selected_features
        is not None
        and len(selected)
        > max_selected_features
    ):
        # Rank the union by GLOBAL gain for memory control.
        ranking = selected[
            np.argsort(
                global_importance[
                    selected
                ]
            )[::-1]
        ]

        selected = np.sort(
            ranking[
                :max_selected_features
            ]
        )

    print(
        f"  Selected SNPs: {len(selected)} "
        f"(global positive-gain={global_selected_count}, "
        f"additional local={local_selected_count})"
    )

    return selected, global_importance


# =============================================================================
# Training / prediction
# =============================================================================

def train_one_fold(
    model,
    X_train,
    y_train,
    device,
    seed,
):
    set_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        PairDataset(
            X_train,
            y_train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=3,
        min_lr=1e-6,
    )

    mse = nn.MSELoss()
    contrastive_loss = ContrastiveLoss()

    model.train()

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.time()

        model.train()

        total_loss = 0.0
        total_batches = 0

        for (
            inputs1,
            inputs2,
            labels1,
            labels2,
            diffs,
        ) in train_loader:

            input1 = inputs1.to(
                device,
                non_blocking=True,
            )
            input2 = inputs2.to(
                device,
                non_blocking=True,
            )

            label1 = labels1.to(
                device,
                non_blocking=True,
            )
            label2 = labels2.to(
                device,
                non_blocking=True,
            )
            diff = diffs.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            out1, aux1 = model(
                input1
            )
            out2, aux2 = model(
                input2
            )

            prediction_loss = (
                mse(out1, label1)
                + mse(out2, label2)
            )

            auxiliary_loss = (
                contrastive_loss(
                    aux1,
                    aux2,
                    diff,
                )
            )

            loss = (
                prediction_loss
                + auxiliary_loss
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        mean_loss = (
            total_loss
            / max(
                total_batches,
                1,
            )
        )

        scheduler.step(
            mean_loss
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):
            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"    Epoch {epoch:3d}/{EPOCHS} "
                f"| loss={mean_loss:.6f} "
                f"| lr={current_lr:.2e} "
                f"| time={time.time()-epoch_start:.2f}s"
            )

    training_time = (
        time.time() - start_time
    )

    return training_time


@torch.no_grad()
def predict(
    model,
    X,
    device,
):
    model.eval()

    loader = DataLoader(
        PredictionDataset(X),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    preds = []

    start = time.time()

    for x in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        pred, _ = model(x)

        preds.append(
            pred.view(-1)
            .cpu()
            .numpy()
        )

    elapsed = time.time() - start

    preds = np.concatenate(
        preds
    )

    inference_ms_per_sample = (
        1000.0
        * elapsed
        / max(
            len(preds),
            1,
        )
    )

    return (
        preds,
        inference_ms_per_sample,
    )


# =============================================================================
# Metrics
# =============================================================================

def safe_pearson(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )


# =============================================================================
# One trait: 10-fold CV
# =============================================================================

def run_trait(
    X_full,
    y_full,
    trait_name,
    marker_names,
    chromosome_labels,
    output_root,
):
    print(
        "\n"
        + "=" * 92
    )
    print(
        f"CLCNet | Wheat2000 | Trait: {trait_name}"
    )
    print(
        "=" * 92
    )

    n = min(
        len(y_full),
        X_full.shape[0],
    )

    X = X_full[:n]
    y = np.asarray(
        y_full[:n],
        dtype=np.float32,
    )

    # Remove missing phenotype values.
    valid_mask = np.isfinite(y)

    X = X[
        valid_mask
    ]
    y = y[
        valid_mask
    ]

    original_sample_indices = np.arange(
        n
    )[
        valid_mask
    ]

    print(
        f"Samples retained: {len(y)}"
    )

    if len(y) < N_SPLITS:
        raise ValueError(
            f"Not enough samples for {N_SPLITS}-fold CV."
        )

    trait_dir = (
        Path(output_root)
        / trait_name
    )
    trait_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    prediction_frames = []
    split_rows = []

    for fold_idx, (
        train_idx,
        test_idx,
    ) in enumerate(
        kfold.split(X),
        start=1,
    ):
        print(
            "\n"
            + "-" * 92
        )
        print(
            f"[{trait_name}] Fold "
            f"{fold_idx}/{N_SPLITS}"
        )
        print(
            f"  Train={len(train_idx)}, "
            f"Test={len(test_idx)}"
        )

        # Same seed for each fold, matching the user's other fair-comparison scripts.
        fold_seed = RANDOM_STATE
        set_seed(
            fold_seed
        )

        X_train_raw = X[
            train_idx
        ]
        X_test_raw = X[
            test_idx
        ]

        y_train_raw = y[
            train_idx
        ]
        y_test_raw = y[
            test_idx
        ]

        # -------------------------------------------------------------
        # Fold-wise genotype imputation
        # -------------------------------------------------------------
        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        X_train = imputer.fit_transform(
            X_train_raw
        ).astype(np.float32)

        X_test = imputer.transform(
            X_test_raw
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Fold-wise phenotype standardization
        # -------------------------------------------------------------
        y_scaler = StandardScaler()

        y_train_scaled = (
            y_scaler.fit_transform(
                y_train_raw.reshape(
                    -1,
                    1,
                )
            )
            .reshape(-1)
            .astype(np.float32)
        )

        # -------------------------------------------------------------
        # Fold-wise feature selection -- no test leakage
        # -------------------------------------------------------------
        feature_start = time.time()

        selected_idx, global_importance = (
            select_clcnet_features(
                X_train=X_train,
                y_train=y_train_scaled,
                chromosome_labels=chromosome_labels,
                seed=fold_seed,
                max_selected_features=MAX_SELECTED_FEATURES,
            )
        )

        feature_selection_time = (
            time.time()
            - feature_start
        )

        X_train_selected = X_train[
            :,
            selected_idx,
        ]

        X_test_selected = X_test[
            :,
            selected_idx,
        ]

        # Save selected indices for reproducibility.
        np.savetxt(
            trait_dir
            / (
                f"{trait_name}_fold"
                f"{fold_idx:02d}_selected_snp_indices.txt"
            ),
            selected_idx,
            fmt="%d",
        )

        # -------------------------------------------------------------
        # Fresh CLCNet for every fold
        # -------------------------------------------------------------
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = CLCNet(
            shuffle=False,
            seed=fold_seed,
            input_dim=X_train_selected.shape[1],
            shared_dim=SHARED_DIM,
        ).to(device)

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # -------------------------------------------------------------
        # Train
        # -------------------------------------------------------------
        training_time = train_one_fold(
            model=model,
            X_train=X_train_selected,
            y_train=y_train_scaled,
            device=device,
            seed=fold_seed,
        )

        # -------------------------------------------------------------
        # Predict
        # -------------------------------------------------------------
        pred_scaled, inference_ms = predict(
            model=model,
            X=X_test_selected,
            device=device,
        )

        pred_original = (
            y_scaler.inverse_transform(
                pred_scaled.reshape(
                    -1,
                    1,
                )
            )
            .reshape(-1)
        )

        # -------------------------------------------------------------
        # Metrics on ORIGINAL phenotype scale
        # -------------------------------------------------------------
        pearson = safe_pearson(
            y_test_raw,
            pred_original,
        )

        r2 = float(
            r2_score(
                y_test_raw,
                pred_original,
            )
        )

        mae = float(
            mean_absolute_error(
                y_test_raw,
                pred_original,
            )
        )

        if torch.cuda.is_available():
            peak_gpu_mb = (
                torch.cuda.max_memory_allocated()
                / 1024 ** 2
            )
        else:
            peak_gpu_mb = np.nan

        result = {
            "Trait":
                trait_name,
            "Fold":
                fold_idx,
            "Train_N":
                len(train_idx),
            "Test_N":
                len(test_idx),
            "Original_SNPs":
                X.shape[1],
            "Selected_SNPs":
                len(selected_idx),
            "Pearson":
                pearson,
            "R2":
                r2,
            "MAE":
                mae,
            "Feature_Selection_Time_s":
                feature_selection_time,
            "Training_Time_s":
                training_time,
            "Inference_ms_per_sample":
                inference_ms,
            "Peak_GPU_Memory_MB":
                peak_gpu_mb,
            "Total_Params":
                total_params,
            "Trainable_Params":
                trainable_params,
            "Epochs":
                EPOCHS,
            "Batch_Size":
                BATCH_SIZE,
            "Learning_Rate":
                LEARNING_RATE,
            "Momentum":
                MOMENTUM,
            "Seed":
                fold_seed,
            "Chromosome_Aware":
                chromosome_labels
                is not None,
        }

        fold_rows.append(
            result
        )

        prediction_frames.append(
            pd.DataFrame({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Index_0Based":
                    original_sample_indices[
                        test_idx
                    ],
                "Observed":
                    y_test_raw,
                "Predicted":
                    pred_original,
            })
        )

        for idx in train_idx:
            split_rows.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Role":
                    "train",
                "Sample_Index_0Based":
                    int(
                        original_sample_indices[
                            idx
                        ]
                    ),
            })

        for idx in test_idx:
            split_rows.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Role":
                    "test",
                "Sample_Index_0Based":
                    int(
                        original_sample_indices[
                            idx
                        ]
                    ),
            })

        print(
            f"  Result: "
            f"Pearson={pearson:.6f}, "
            f"R2={r2:.6f}, "
            f"MAE={mae:.6f}"
        )
        print(
            f"  Selected SNPs: "
            f"{len(selected_idx)} / "
            f"{X.shape[1]}"
        )
        print(
            f"  Feature selection: "
            f"{feature_selection_time:.2f}s"
        )
        print(
            f"  Training time: "
            f"{training_time:.2f}s"
        )

        # -------------------------------------------------------------
        # Checkpoint after EVERY fold
        # -------------------------------------------------------------
        pd.DataFrame(
            fold_rows
        ).to_csv(
            trait_dir
            / (
                f"CLCNet_Wheat2000_"
                f"{trait_name}_FoldResults_CHECKPOINT.csv"
            ),
            index=False,
        )

        pd.concat(
            prediction_frames,
            ignore_index=True,
        ).to_csv(
            trait_dir
            / (
                f"CLCNet_Wheat2000_"
                f"{trait_name}_Predictions_CHECKPOINT.csv"
            ),
            index=False,
        )

        pd.DataFrame(
            split_rows
        ).to_csv(
            trait_dir
            / (
                f"CLCNet_Wheat2000_"
                f"{trait_name}_10FoldIndices.csv"
            ),
            index=False,
        )

        # Free fold objects.
        del model
        del X_train_selected
        del X_test_selected

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # Final trait outputs
    # =========================================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    split_df = pd.DataFrame(
        split_rows
    )

    summary = {
        "Trait":
            trait_name,
        "N":
            len(y),
        "Original_SNPs":
            X.shape[1],
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
        "MAE_Mean":
            fold_df["MAE"].mean(),
        "MAE_SD":
            fold_df["MAE"].std(
                ddof=1
            ),
        "Selected_SNPs_Mean":
            fold_df[
                "Selected_SNPs"
            ].mean(),
        "Selected_SNPs_SD":
            fold_df[
                "Selected_SNPs"
            ].std(
                ddof=1
            ),
        "Feature_Selection_Time_Mean_s":
            fold_df[
                "Feature_Selection_Time_s"
            ].mean(),
        "Training_Time_Mean_s":
            fold_df[
                "Training_Time_s"
            ].mean(),
        "Training_Time_Total_s":
            fold_df[
                "Training_Time_s"
            ].sum(),
        "Inference_Mean_ms_per_sample":
            fold_df[
                "Inference_ms_per_sample"
            ].mean(),
        "Peak_GPU_Memory_Max_MB":
            fold_df[
                "Peak_GPU_Memory_MB"
            ].max(),
        "Chromosome_Aware":
            chromosome_labels
            is not None,
        "CV":
            f"{N_SPLITS}-fold",
        "Random_State":
            RANDOM_STATE,
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    fold_file = (
        trait_dir
        / (
            f"CLCNet_Wheat2000_"
            f"{trait_name}_FoldResults.csv"
        )
    )

    pred_file = (
        trait_dir
        / (
            f"CLCNet_Wheat2000_"
            f"{trait_name}_Predictions.csv"
        )
    )

    summary_file = (
        trait_dir
        / (
            f"CLCNet_Wheat2000_"
            f"{trait_name}_Summary.csv"
        )
    )

    split_file = (
        trait_dir
        / (
            f"CLCNet_Wheat2000_"
            f"{trait_name}_10FoldIndices.csv"
        )
    )

    fold_df.to_csv(
        fold_file,
        index=False,
    )

    pred_df.to_csv(
        pred_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    split_df.to_csv(
        split_file,
        index=False,
    )

    print(
        "\nTrait summary:"
    )
    print(
        summary_df.to_string(
            index=False
        )
    )

    return summary_df


# =============================================================================
# Run all Wheat2000 traits
# =============================================================================

def run_all_traits(
    geno_file,
    phenotype_dir,
    output_root,
    chromosome_file=None,
    phenotype_pattern="2000_*_phe_*.txt",
):
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_full, marker_names, has_columns = (
        load_wheat2000_genotype(
            geno_file
        )
    )

    # -------------------------------------------------------------------------
    # Chromosome labels:
    # 1) explicit file
    # 2) auto-detect from marker names
    # 3) otherwise global feature selection only
    # -------------------------------------------------------------------------

    chromosome_labels = None

    if chromosome_file:
        chromosome_labels = (
            load_chromosome_labels(
                chromosome_file,
                X_full.shape[1],
            )
        )

        print(
            "Chromosome labels loaded "
            "from external file."
        )

    elif has_columns:
        chromosome_labels = (
            infer_chromosomes_from_marker_names(
                marker_names
            )
        )

        if chromosome_labels is not None:
            print(
                "Chromosome labels inferred "
                "from genotype marker names."
            )

    if chromosome_labels is None:
        print(
            "\nWARNING:"
            "\n  Chromosome labels are unavailable."
            "\n  This run will use the GLOBAL LightGBM "
            "feature-selection component only."
            "\n  For the full CLCNet chromosome-aware "
            "preprocessing, provide CHROMOSOME_FILE."
            "\n"
        )
    else:
        print(
            f"Chromosomes detected: "
            f"{len(pd.unique(chromosome_labels))}"
        )

    trait_files = discover_phenotype_files(
        phenotype_dir,
        phenotype_pattern,
    )

    print(
        "\nPhenotype files discovered:"
    )

    for trait, path in trait_files.items():
        print(
            f"  {trait}: {path}"
        )

    all_summaries = []
    failed_traits = []

    for trait_idx, (
        trait,
        pheno_file,
    ) in enumerate(
        trait_files.items(),
        start=1,
    ):
        print(
            "\n"
            + "#" * 92
        )
        print(
            f"Trait {trait_idx}/"
            f"{len(trait_files)}: "
            f"{trait}"
        )
        print(
            "#" * 92
        )

        try:
            y = load_wheat2000_phenotype(
                pheno_file
            )

            summary_df = run_trait(
                X_full=X_full,
                y_full=y,
                trait_name=trait,
                marker_names=marker_names,
                chromosome_labels=chromosome_labels,
                output_root=output_root,
            )

            summary_df[
                "Phenotype_File"
            ] = str(
                pheno_file
            )

            all_summaries.append(
                summary_df
            )

            # Combined checkpoint after every completed trait.
            combined = pd.concat(
                all_summaries,
                ignore_index=True,
            )

            combined.to_csv(
                output_root
                / (
                    "CLCNet_Wheat2000_"
                    "AllTraits_Summary_CHECKPOINT.csv"
                ),
                index=False,
            )

        except Exception as exc:
            print(
                f"\nERROR: trait {trait} failed"
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )

            failed_traits.append({
                "Trait":
                    trait,
                "Phenotype_File":
                    str(pheno_file),
                "Error":
                    f"{type(exc).__name__}: {exc}",
            })

            pd.DataFrame(
                failed_traits
            ).to_csv(
                output_root
                / (
                    "CLCNet_Wheat2000_"
                    "FailedTraits.csv"
                ),
                index=False,
            )

            # Continue with the next trait.
            continue

    if all_summaries:
        combined_summary = pd.concat(
            all_summaries,
            ignore_index=True,
        )

        combined_summary.to_csv(
            output_root
            / (
                "CLCNet_Wheat2000_"
                "AllTraits_Summary.csv"
            ),
            index=False,
        )

        print(
            "\n"
            + "=" * 92
        )
        print(
            "ALL-TRAIT SUMMARY"
        )
        print(
            "=" * 92
        )

        columns_to_show = [
            "Trait",
            "Pearson_Mean",
            "Pearson_SD",
            "R2_Mean",
            "R2_SD",
            "MAE_Mean",
            "MAE_SD",
            "Selected_SNPs_Mean",
            "Chromosome_Aware",
        ]

        print(
            combined_summary[
                [
                    c
                    for c in columns_to_show
                    if c in combined_summary.columns
                ]
            ].to_string(
                index=False
            )
        )

        return combined_summary

    raise RuntimeError(
        "No trait completed successfully."
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Wheat2000 paths
    # -------------------------------------------------------------------------

    GENO_FILE = (
        r"E:\VGMP\wheat2000\SNP_origin\2000gene.pkl"
    )

    PHENOTYPE_DIR = (
        r"E:\VGMP\wheat2000\SNP_origin"
    )

    # Automatically searches:
    # 2000_*_phe_*.txt
    PHENOTYPE_PATTERN = (
        "2000_*_phe_*.txt"
    )

    # -------------------------------------------------------------------------
    # Optional chromosome file
    #
    # One chromosome label per SNP, in the SAME SNP order as 2000gene.pkl.
    #
    # Example:
    # CHROMOSOME_FILE = r"E:\VGMP\wheat2000\SNP_origin\chromosomes.txt"
    #
    # If None, the script attempts to infer chromosomes from marker names.
    # If that also fails, GLOBAL feature selection is used.
    # -------------------------------------------------------------------------

    CHROMOSOME_FILE = None

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    OUTPUT_ROOT = (
        Path(__file__).resolve().parent
        / "results_clcnet_wheat2000_10fold"
    )

    set_seed(
        RANDOM_STATE
    )

    run_all_traits(
        geno_file=GENO_FILE,
        phenotype_dir=PHENOTYPE_DIR,
        output_root=OUTPUT_ROOT,
        chromosome_file=CHROMOSOME_FILE,
        phenotype_pattern=PHENOTYPE_PATTERN,
    )
