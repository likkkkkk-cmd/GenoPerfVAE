# -*- coding: utf-8 -*-
"""
VMGP-S on Wheat2000: controlled 10-fold comparison with GenoPerfVAE
================================================================

This script uses the EXACT SAME Wheat2000 data processing and outer
cross-validation protocol as the paired GenoPerfVAE script:

    1. Same Wheat2000 genotype/phenotype loader.
    2. Same row-order alignment.
    3. Same 10-fold KFold(shuffle=True, random_state=42).
    4. Same fold-wise most-frequent SNP imputation.
    5. Same fold-wise MinMaxScaler((-1, 1)) for genotype.
    6. Same fold-wise StandardScaler for phenotype.
    7. Same outer training/test samples.
    8. All outer-training samples are used for model fitting.
    9. No internal validation split.

VMGP-S retains its original model-specific settings:
    epochs      = 200
    batch size  = 32
    learning rate = 1e-4
    optimizer   = Adam
    vae_ratio   = 0.5
    hidden_dims = [2048, 512]
    latent_dim  = 32
    pre_dim     = 128
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import DataLoader, TensorDataset

from vmgp_model import VMGP


# =============================================================================
# Configuration
# =============================================================================
NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VAE_RATIO = 0.5

LATENT_DIM = 32
HIDDEN_DIMS = [2048, 512]
PRE_DIM = 128

RANDOM_STATE = 42
N_SPLITS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# Wheat2000 loader
# Copied from the paired GenoPerfVAE script to keep data loading identical.
# =============================================================================
class Wheat2000DataProcessor:
    """
    Wheat2000 data loader.

    Genotype:
        - .pkl: pd.read_pickle(...).values
        - otherwise: read table and drop the first column

    Phenotype:
        - read the first numeric value from each non-empty line
        - align genotype and phenotype by row order
        - truncate both to the same number of samples

    Missing-value handling is performed inside each outer training fold.
    """

    def __init__(self, genotype_file, phenotype_file):
        self.genotype_file = genotype_file
        self.phenotype_file = phenotype_file

    def load_raw_data(self):
        print("正在载入 Wheat2000 数据...")

        if self.genotype_file.lower().endswith(".pkl"):
            geno_obj = pd.read_pickle(self.genotype_file)
            X = (
                geno_obj.values
                if hasattr(geno_obj, "values")
                else np.asarray(geno_obj)
            )
        else:
            geno_df = pd.read_csv(
                self.genotype_file,
                sep=None,
                engine="python"
            )
            X = geno_df.iloc[:, 1:].values

        y_list = []

        with open(
            self.phenotype_file,
            "r",
            encoding="utf-8"
        ) as f:
            for line in f:
                parts = (
                    line.strip()
                    .replace(",", " ")
                    .replace("\t", " ")
                    .split()
                )

                if parts:
                    try:
                        y_list.append(float(parts[0]))
                    except ValueError:
                        continue

        y = np.asarray(
            y_list,
            dtype=np.float32
        ).reshape(-1, 1)

        X = np.asarray(
            X,
            dtype=np.float32
        )

        min_samples = min(
            X.shape[0],
            y.shape[0]
        )

        X = X[:min_samples]
        y = y[:min_samples]

        print(
            f"数据对齐完成。样本数: {X.shape[0]}, "
            f"SNP维度: {X.shape[1]}"
        )

        return X, y


# =============================================================================
# Metrics / model utilities
# =============================================================================
def pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if (
        y_true.size < 2
        or np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            y_true,
            y_pred
        )[0, 1]
    )


def count_parameters(
    model: nn.Module
) -> Tuple[int, int]:

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# =============================================================================
# Official VMGP-S objective
# =============================================================================
def vmgp_joint_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    vae_ratio: float = 0.5
):
    mse = nn.functional.mse_loss

    pred_loss = mse(
        y_hat,
        y_true
    )

    kl_loss = -0.5 * torch.mean(
        logvar
        + 1.0
        - mu.pow(2)
        - torch.exp(logvar)
    )

    recon_loss = mse(
        x_hat,
        x_true
    )

    vae_loss = (
        recon_loss
        + kl_loss
    )

    total_loss = (
        (1.0 - vae_ratio) * pred_loss
        + vae_ratio * vae_loss
    )

    return (
        total_loss,
        pred_loss,
        recon_loss,
        kl_loss
    )


def save_fixed_fold_assignments(
    n_samples,
    output_dir,
    random_state=42
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state
    )

    splits = list(
        kf.split(
            np.arange(n_samples)
        )
    )

    fold_membership = np.empty(
        n_samples,
        dtype=int
    )

    for fold_idx, (_, test_idx) in enumerate(
        splits,
        start=1
    ):
        fold_membership[
            test_idx
        ] = fold_idx

    split_df = pd.DataFrame({
        "Sample_Index_0Based":
            np.arange(n_samples),
        "Fold":
            fold_membership
    })

    split_path = os.path.join(
        output_dir,
        "wheat2000_10fold_indices.csv"
    )

    split_df.to_csv(
        split_path,
        index=False
    )

    return splits, split_path


def train_one_fold(
    X_train,
    y_train,
    X_test,
    y_test,
    fold_seed,
    trait_name,
    fold_idx
):

    set_seed(fold_seed)

    train_ds = TensorDataset(
        torch.from_numpy(
            X_train
        ).float(),
        torch.from_numpy(
            y_train
        ).float()
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        fold_seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=loader_generator
    )

    test_x = torch.from_numpy(
        X_test
    ).float()

    # Fresh model every fold.
    model = VMGP(
        input_dim=X_train.shape[1],
        out_dim=1,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        pre_dim=PRE_DIM
    ).to(DEVICE)

    # Match the official VMGP training code.
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(
                m.weight
            )

    model.apply(
        init_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            DEVICE
        )
        cuda_sync()

    epoch_times: List[float] = []
    last_losses = None

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        NUM_EPOCHS
    ):

        model.train()

        cuda_sync()
        epoch_start = (
            time.perf_counter()
        )

        sum_total = 0.0
        sum_pred = 0.0
        sum_recon = 0.0
        sum_kl = 0.0
        n_batches = 0

        for xb, yb in train_loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                y_hat,
                x_hat,
                mu,
                logvar
            ) = model(xb)

            (
                total_loss,
                pred_loss,
                recon_loss,
                kl_loss
            ) = vmgp_joint_loss(
                y_hat,
                yb,
                x_hat,
                xb,
                mu,
                logvar,
                VAE_RATIO
            )

            total_loss.backward()
            optimizer.step()

            sum_total += float(
                total_loss.detach().cpu()
            )
            sum_pred += float(
                pred_loss.detach().cpu()
            )
            sum_recon += float(
                recon_loss.detach().cpu()
            )
            sum_kl += float(
                kl_loss.detach().cpu()
            )
            n_batches += 1

        cuda_sync()
        epoch_times.append(
            time.perf_counter()
            - epoch_start
        )

        last_losses = (
            sum_total / n_batches,
            sum_pred / n_batches,
            sum_recon / n_batches,
            sum_kl / n_batches
        )

        if (
            epoch == 0
            or (epoch + 1) % 20 == 0
            or epoch + 1 == NUM_EPOCHS
        ):
            print(
                f"  [{trait_name}] "
                f"Fold {fold_idx:02d} | "
                f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
                f"total={last_losses[0]:.6f} "
                f"pred={last_losses[1]:.6f} "
                f"recon={last_losses[2]:.6f} "
                f"KL={last_losses[3]:.6f}"
            )

    cuda_sync()

    training_time = (
        time.perf_counter()
        - training_start
    )

    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(
            DEVICE
        ) / (1024 ** 2)
        if torch.cuda.is_available()
        else np.nan
    )

    model.eval()

    preds_scaled = []

    cuda_sync()
    infer_start = (
        time.perf_counter()
    )

    with torch.no_grad():

        for start in range(
            0,
            len(test_x),
            128
        ):

            xb = test_x[
                start:start + 128
            ].to(
                DEVICE,
                non_blocking=True
            )

            pred = model(xb)

            preds_scaled.append(
                pred.detach()
                .cpu()
                .numpy()
            )

    cuda_sync()

    infer_time = (
        time.perf_counter()
        - infer_start
    )

    y_pred_scaled = np.vstack(
        preds_scaled
    ).reshape(-1, 1)

    result = {
        "Trait": trait_name,
        "Fold": fold_idx,
        "N_train": len(X_train),
        "N_test": len(X_test),
        "Pearson_scaled": pearson_r(
            y_test,
            y_pred_scaled.reshape(-1)
        ),
        "Epoch_Time_Mean_s":
            float(np.mean(epoch_times)),
        "Epoch_Time_SD_s":
            float(np.std(epoch_times, ddof=1)),
        "Training_Time_s":
            float(training_time),
        "Inference_ms_per_sample":
            float(
                infer_time
                * 1000.0
                / len(y_test)
            ),
        "Peak_GPU_Memory_MB":
            float(peak_gpu_mb),
        "Total_Params_M":
            total_params / 1e6,
        "Trainable_Params_M":
            trainable_params / 1e6,
        "Final_Total_Loss":
            float(last_losses[0]),
        "Final_Pred_Loss":
            float(last_losses[1]),
        "Final_Recon_Loss":
            float(last_losses[2]),
        "Final_KL_Loss":
            float(last_losses[3]),
        "Seed":
            fold_seed,
        "VAE_Ratio":
            VAE_RATIO,
        "Epochs":
            NUM_EPOCHS,
        "Batch_Size":
            BATCH_SIZE,
        "Learning_Rate":
            LEARNING_RATE
    }

    del model, optimizer, train_loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        result,
        y_pred_scaled.reshape(-1)
    )


def run_experiment(
    geno_path,
    pheno_path,
    trait_name="Trait",
    random_state=42,
    output_dir="results_vmgp_s_wheat2000_10fold"
):

    set_seed(
        random_state
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    processor = Wheat2000DataProcessor(
        geno_path,
        pheno_path
    )

    X_all, y_all = (
        processor.load_raw_data()
    )

    n_samples = X_all.shape[0]
    snp_dim = X_all.shape[1]

    print("=" * 80)
    print(
        "VMGP-S Wheat2000 "
        "fair comparison"
    )
    print(
        f"Trait       : {trait_name}"
    )
    print(
        f"Device      : {DEVICE}"
    )
    print(
        f"Samples     : {n_samples}"
    )
    print(
        f"SNPs        : {snp_dim}"
    )
    print(
        f"Seed        : {random_state}"
    )
    print("=" * 80)

    splits, split_path = (
        save_fixed_fold_assignments(
            n_samples=n_samples,
            output_dir=output_dir,
            random_state=random_state
        )
    )

    all_fold_results = []
    all_predictions = []

    for fold_idx, (
        train_idx,
        test_idx
    ) in enumerate(
        splits,
        start=1
    ):

        # Same fixed seed in every fold as GenoPerfVAE.
        fold_seed = random_state

        X_train_raw = (
            X_all[
                train_idx
            ]
        )

        X_test_raw = (
            X_all[
                test_idx
            ]
        )

        y_train_raw = (
            y_all[
                train_idx
            ]
        )

        y_test_raw = (
            y_all[
                test_idx
            ]
        )

        # -------------------------------------------------------------
        # IDENTICAL fold-wise preprocessing as GenoPerfVAE
        # -------------------------------------------------------------
        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        X_train_imp = (
            imputer
            .fit_transform(
                X_train_raw
            )
            .astype(np.float32)
        )

        X_test_imp = (
            imputer
            .transform(
                X_test_raw
            )
            .astype(np.float32)
        )

        scaler_X = MinMaxScaler(
            feature_range=(-1, 1)
        )

        X_train = (
            scaler_X
            .fit_transform(
                X_train_imp
            )
            .astype(np.float32)
        )

        X_test = (
            scaler_X
            .transform(
                X_test_imp
            )
            .astype(np.float32)
        )

        scaler_y = StandardScaler()

        y_train = (
            scaler_y
            .fit_transform(
                y_train_raw
            )
            .astype(np.float32)
        )

        y_test = (
            scaler_y
            .transform(
                y_test_raw
            )
            .astype(np.float32)
        )

        print(
            f"\n[{trait_name}] "
            f"Fold {fold_idx}/10 | "
            f"train={len(train_idx)} "
            f"test={len(test_idx)} | "
            f"seed={fold_seed}"
        )

        result, y_pred_scaled = (
            train_one_fold(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                fold_seed=fold_seed,
                trait_name=trait_name,
                fold_idx=fold_idx
            )
        )

        # Convert predictions and observations back to original phenotype scale.
        y_pred_original = (
            scaler_y
            .inverse_transform(
                y_pred_scaled
                .reshape(-1, 1)
            )
            .reshape(-1)
        )

        y_true_original = (
            y_test_raw
            .reshape(-1)
        )

        result[
            "Pearson"
        ] = pearson_r(
            y_true_original,
            y_pred_original
        )

        result[
            "R2"
        ] = float(
            r2_score(
                y_true_original,
                y_pred_original
            )
        )

        result[
            "MAE"
        ] = float(
            mean_absolute_error(
                y_true_original,
                y_pred_original
            )
        )

        result[
            "SNP_Dimension_D"
        ] = snp_dim

        result[
            "X_Imputation"
        ] = (
            "most_frequent_train_fold"
        )

        result[
            "X_Scaling"
        ] = (
            "MinMax_-1_1_train_fold"
        )

        result[
            "Y_Scaling"
        ] = (
            "StandardScaler_train_fold"
        )

        result[
            "Internal_Validation"
        ] = False

        all_fold_results.append(
            result
        )

        all_predictions.append(
            pd.DataFrame({
                "Trait":
                    trait_name,
                "Fold":
                    fold_idx,
                "Sample_Index_0Based":
                    test_idx,
                "Observed":
                    y_true_original,
                "Predicted":
                    y_pred_original
            })
        )

        print(
            f"  Result: "
            f"Pearson={result['Pearson']:.6f}, "
            f"R2={result['R2']:.6f}, "
            f"MAE={result['MAE']:.6f}, "
            f"TrainTime={result['Training_Time_s']:.2f}s"
        )

    fold_df = pd.DataFrame(
        all_fold_results
    )

    pred_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    summary_df = pd.DataFrame([{
        "Model":
            "VMGP-S-Wheat2000",
        "Trait":
            trait_name,
        "Pearson_Mean":
            fold_df["Pearson"].mean(),
        "Pearson_SD":
            fold_df["Pearson"].std(ddof=1),
        "R2_Mean":
            fold_df["R2"].mean(),
        "R2_SD":
            fold_df["R2"].std(ddof=1),
        "MAE_Mean":
            fold_df["MAE"].mean(),
        "MAE_SD":
            fold_df["MAE"].std(ddof=1),
        "Epoch_Time_Mean_s":
            fold_df["Epoch_Time_Mean_s"].mean(),
        "Training_Time_Mean_s":
            fold_df["Training_Time_s"].mean(),
        "Peak_GPU_Memory_Mean_MB":
            fold_df["Peak_GPU_Memory_MB"].mean(),
        "Peak_GPU_Memory_Max_MB":
            fold_df["Peak_GPU_Memory_MB"].max(),
        "Inference_Mean_ms_per_sample":
            fold_df["Inference_ms_per_sample"].mean(),
        "Total_Params_M":
            fold_df["Total_Params_M"].iloc[0],
        "Trainable_Params_M":
            fold_df["Trainable_Params_M"].iloc[0]
    }])

    safe_trait = (
        str(trait_name)
        .replace(" ", "_")
    )

    fold_file = os.path.join(
        output_dir,
        f"VMGP_S_Wheat2000_{safe_trait}_FoldResults.csv"
    )

    summary_file = os.path.join(
        output_dir,
        f"VMGP_S_Wheat2000_{safe_trait}_Summary.csv"
    )

    pred_file = os.path.join(
        output_dir,
        f"VMGP_S_Wheat2000_{safe_trait}_Predictions.csv"
    )

    fold_df.to_csv(
        fold_file,
        index=False
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    pred_df.to_csv(
        pred_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("FINISHED")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        f"  {split_path}"
    )
    print(
        f"  {fold_file}"
    )
    print(
        f"  {summary_file}"
    )
    print(
        f"  {pred_file}"
    )

    return (
        fold_df,
        summary_df
    )


if __name__ == "__main__":

    # -----------------------------------------------------------------
    # EDIT THESE PATHS ONLY.
    # Use the SAME files in the paired GenoPerfVAE script.
    # -----------------------------------------------------------------
    GENO_FILE = (
        r"E:\VGMP\wheat2000\SNP_origin\2000gene.pkl"
    )

    PHENO_FILE = (
        r"E:\VGMP\wheat2000\SNP_origin\2000_6_phe_Prot.txt"
    )

    TRAIT_NAME = "tkw"

    OUTPUT_DIR = (
        r"E:\VGMP\wheat2000\fair_compare_Genoperfvae_vmgps\VMGP_S"
    )

    if (
        os.path.exists(
            GENO_FILE
        )
        and os.path.exists(
            PHENO_FILE
        )
    ):
        run_experiment(
            geno_path=GENO_FILE,
            pheno_path=PHENO_FILE,
            trait_name=TRAIT_NAME,
            random_state=RANDOM_STATE,
            output_dir=OUTPUT_DIR
        )

    else:
        print(
            "未找到 Wheat2000 数据文件，"
            "请检查 GENO_FILE 和 PHENO_FILE 路径。"
        )
