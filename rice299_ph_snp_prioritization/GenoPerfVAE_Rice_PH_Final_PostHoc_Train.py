#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GenoPerfVAE_Rice_PH_Final_PostHoc_Train.py

Purpose
-------
Train ONE final GenoPerfVAE model on all aligned Rice299-PH samples for post-hoc
computational genotype exploration only.

IMPORTANT
---------
1) This script is NOT a cross-validation benchmark and its in-sample metrics
   must NOT be reported as generalization performance.
2) It preserves the current Rice299 genotype representation:
      -1 is treated as a VALID genotype value.
      Only true NaN values are imputed by per-SNP most-frequent values.
      No additional SNP scaling is applied.
3) Phenotype PH is standardized with StandardScaler during training.
4) Architecture follows the currently supplied final GenoPerfVAE code:
      chunk size = 32
      Performer readout = 512
      decoder Performer width = 512
      latent = 48 = Performer32 + Linear16
      predictor = 48 -> 64 -> 1, BN + GELU, no predictor dropout
5) Training follows the currently executed configuration in the supplied code:
      Adam(lr=1e-5), weight_decay=0
      fixed 200 epochs
      no validation / early stopping / scheduler
      loss = 1.5*Pred + 0*Pearson + 0.3*Recon + 1.0*KL
      global grad clip = 1.0

Outputs are designed to support the next-step target-genotype round-trip analysis.
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

try:
    from performer_pytorch import Performer
except ImportError as exc:
    raise ImportError(
        "performer-pytorch is required. Install it with:\n"
        "    pip install performer-pytorch"
    ) from exc


# =============================================================================
# 0. Fixed configuration: match the currently supplied final GenoPerfVAE code
# =============================================================================

SEED = 42
TRAIT = "PH"

GENOTYPE_FILENAME = (
    "MET_crfilt_.90_outliers_removed_for_RRBlup_line_corrected.csv"
)
PHENOTYPE_FILENAME = "rice299_PH.csv"

CHUNK_SIZE = 32
PERFORMER_READOUT_DIM = 512
DECODER_PERFORMER_DIM = 512

TOTAL_LATENT_DIM = 48
PERFORMER_LATENT_DIM = 32
LINEAR_LATENT_DIM = 16

DROPOUT = 0.3
BATCH_SIZE = 48
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

PRED_WEIGHT = 1.50
PEARSON_WEIGHT = 0.0
RECON_WEIGHT = 1.0
KL_WEIGHT = 1.00

OUTPUT_BASENAME = "GenoPerfVAE_Rice_PH_Final_PostHoc"


# =============================================================================
# 1. Reproducibility helpers
# =============================================================================

def set_seed_fixed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def safe_batch_size(n_samples: int, preferred: int = BATCH_SIZE) -> int:
    """
    Avoid a final training batch of size 1 because the model contains BatchNorm.
    """
    if n_samples < 2:
        raise ValueError("At least two aligned samples are required.")

    batch = min(preferred, n_samples)

    while batch > 2 and n_samples % batch == 1:
        batch -= 1

    return batch


# =============================================================================
# 2. Loss
# =============================================================================

class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        mu_p = torch.mean(y_pred)
        mu_t = torch.mean(y_true)
        diff_p = y_pred - mu_p
        diff_t = y_true - mu_t

        num = torch.sum(diff_p * diff_t)
        den = torch.sqrt(
            torch.sum(diff_p ** 2) * torch.sum(diff_t ** 2) + 1e-8
        )
        return 1.0 - num / den


class GeneticAwareLoss(nn.Module):
    def __init__(
        self,
        pred_weight=PRED_WEIGHT,
        pearson_weight=PEARSON_WEIGHT,
        recon_weight=RECON_WEIGHT,
        kl_weight=KL_WEIGHT,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pearson = PearsonLoss()

        self.m_weight = float(pred_weight)
        self.p_weight = float(pearson_weight)
        self.r_weight = float(recon_weight)
        self.k_weight = float(kl_weight)

    def forward(self, x_recon, x, y_pred, y, mu, logvar):
        pred_l = self.mse(y_pred, y)
        pearson_l = self.pearson(y_pred, y)
        recon_l = self.mse(x_recon, x)

        # Mean over both batch and latent dimensions, matching current code.
        kl_l = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

        total = (
            self.m_weight * pred_l
            + self.p_weight * pearson_l
            + self.r_weight * recon_l
            + self.k_weight * kl_l
        )

        return total, recon_l, pearson_l, pred_l, kl_l


# =============================================================================
# 3. Current GenoPerfVAE architecture
# =============================================================================

class AdaptiveChunkedSNP_Performer(nn.Module):
    """
    Contiguous SNP chunking + sample-specific learned chunk reweighting
    + positional embedding + Performer + full-token flatten readout.
    """

    def __init__(
        self,
        snp_dim,
        target_chunk_size=CHUNK_SIZE,
        performer_dim=PERFORMER_READOUT_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.snp_dim = int(snp_dim)
        self.chunk_size = int(target_chunk_size)

        remainder = self.snp_dim % self.chunk_size
        self.pad_n = self.chunk_size - remainder if remainder > 0 else 0
        self.num_chunks = (
            self.snp_dim + self.pad_n
        ) // self.chunk_size
        self.sequence_length = self.num_chunks
        self.flatten_readout_dim = self.num_chunks * self.chunk_size

        self.padding_layer = (
            nn.ZeroPad1d((0, self.pad_n))
            if self.pad_n > 0
            else None
        )

        self.chunk_norm = nn.LayerNorm(self.chunk_size)

        hidden_dim = max(self.chunk_size // 4, 32)
        self.chunk_weight_net = nn.Sequential(
            nn.Linear(self.chunk_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.chunk_scale = nn.Parameter(torch.tensor(0.1))

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_chunks, self.chunk_size)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        num_heads = 4
        if self.chunk_size % num_heads != 0:
            raise ValueError(
                f"chunk_size={self.chunk_size} must be divisible by "
                f"num_heads={num_heads}."
            )

        head_dim = self.chunk_size // num_heads

        self.performer = Performer(
            dim=self.chunk_size,
            depth=2,
            heads=num_heads,
            dim_head=head_dim,
            ff_dropout=dropout,
            attn_dropout=dropout,
            causal=False,
        )

        # Preserve the RNG-local construction used in the supplied final code.
        with torch.random.fork_rng(devices=[]):
            self.output_projection = nn.Sequential(
                nn.Linear(self.flatten_readout_dim, performer_dim),
                nn.LayerNorm(performer_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Preserve the corresponding RNG advancement in the supplied code.
        _original_rng_advance = nn.Linear(
            self.chunk_size * 2,
            performer_dim,
        )
        del _original_rng_advance

        self.last_chunk_weights = None

    def forward(self, x):
        if self.padding_layer is not None:
            x = self.padding_layer(
                x.unsqueeze(1)
            ).squeeze(1)

        x = x.reshape(
            x.size(0),
            self.num_chunks,
            self.chunk_size,
        )

        x = self.chunk_norm(x)

        chunk_scores = self.chunk_weight_net(x)
        chunk_weights = torch.softmax(chunk_scores, dim=1)
        scaled_weights = chunk_weights * self.num_chunks
        self.last_chunk_weights = chunk_weights.detach()

        centered_weights = scaled_weights - 1.0
        x = x * (
            1.0
            + self.chunk_scale * centered_weights
        )

        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        x = self.performer(x)

        # P2: preserve all token outputs.
        x_flat = x.flatten(start_dim=1)
        return self.output_projection(x_flat)

    def get_chunk_weights(self):
        if self.last_chunk_weights is None:
            return None
        return self.last_chunk_weights.squeeze(-1)


class AdaptiveChunkedSNP_Performer_Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        snp_dim,
        target_chunk_size=CHUNK_SIZE,
        performer_dim=DECODER_PERFORMER_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.snp_dim = int(snp_dim)
        self.chunk_size = int(target_chunk_size)

        remainder = self.snp_dim % self.chunk_size
        self.pad_n = self.chunk_size - remainder if remainder > 0 else 0
        self.num_chunks = (
            self.snp_dim + self.pad_n
        ) // self.chunk_size

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, performer_dim),
            nn.BatchNorm1d(performer_dim),
            nn.GELU(),
        )

        self.to_seq = nn.Sequential(
            nn.Linear(
                performer_dim,
                self.num_chunks * self.chunk_size
            ),
            nn.BatchNorm1d(
                self.num_chunks * self.chunk_size
            ),
            nn.GELU(),
        )

        num_heads = 4
        if self.chunk_size % num_heads != 0:
            raise ValueError(
                f"chunk_size={self.chunk_size} must be divisible by "
                f"num_heads={num_heads}."
            )

        head_dim = self.chunk_size // num_heads

        self.performer = Performer(
            dim=self.chunk_size,
            depth=2,
            heads=num_heads,
            dim_head=head_dim,
            ff_dropout=dropout,
            attn_dropout=dropout,
            causal=False,
        )

        self.final_head = nn.Linear(
            self.chunk_size,
            self.chunk_size
        )

    def forward(self, z):
        batch_size = z.size(0)

        h = self.latent_proj(z)
        h_seq = self.to_seq(h).view(
            batch_size,
            self.num_chunks,
            self.chunk_size,
        )

        h_decoded = self.performer(h_seq)
        x_flat = self.final_head(
            h_decoded
        ).view(batch_size, -1)

        return x_flat[:, :self.snp_dim]


class EnhancedGeneticModel(nn.Module):
    """
    Final post-hoc PH model:
      Performer branch -> 32 latent dimensions
      Linear shortcut  -> 16 latent dimensions
      Total latent     -> 48 dimensions
    """

    def __init__(
        self,
        snp_dim,
        latent_dim=TOTAL_LATENT_DIM,
        performer_readout_dim=PERFORMER_READOUT_DIM,
        decoder_performer_dim=DECODER_PERFORMER_DIM,
        performer_latent_dim=PERFORMER_LATENT_DIM,
        linear_latent_dim=LINEAR_LATENT_DIM,
        dropout=DROPOUT,
        chunk_size=CHUNK_SIZE,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.performer_readout_dim = int(performer_readout_dim)
        self.decoder_performer_dim = int(decoder_performer_dim)
        self.performer_latent_dim = int(performer_latent_dim)
        self.linear_latent_dim = int(linear_latent_dim)

        if (
            self.performer_latent_dim
            + self.linear_latent_dim
            != self.latent_dim
        ):
            raise ValueError(
                "Branch latent dimensions must sum to total latent_dim: "
                f"{self.performer_latent_dim}+{self.linear_latent_dim}"
                f"!={self.latent_dim}"
            )

        self.snp_performer = AdaptiveChunkedSNP_Performer(
            snp_dim=snp_dim,
            target_chunk_size=chunk_size,
            performer_dim=self.performer_readout_dim,
            dropout=dropout,
        )

        self.linear_shortcut = nn.Sequential(
            nn.Linear(snp_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
        )

        with torch.random.fork_rng(devices=[]):
            self.fc_mu_performer = nn.Linear(
                self.performer_readout_dim,
                self.performer_latent_dim,
            )
            self.fc_logvar_performer = nn.Linear(
                self.performer_readout_dim,
                self.performer_latent_dim,
            )
            self.fc_mu_linear = nn.Linear(
                64,
                self.linear_latent_dim,
            )
            self.fc_logvar_linear = nn.Linear(
                64,
                self.linear_latent_dim,
            )

        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.decoder = AdaptiveChunkedSNP_Performer_Decoder(
            latent_dim=self.latent_dim,
            snp_dim=snp_dim,
            target_chunk_size=chunk_size,
            performer_dim=self.decoder_performer_dim,
            dropout=dropout,
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            return (
                mu
                + torch.randn_like(mu)
                * torch.exp(0.5 * logvar)
            )
        return mu

    def encode(self, x):
        """
        Deterministic encoder access for downstream post-hoc analyses.
        Returns posterior mu/logvar without reparameterization noise.
        """
        h_performer = self.snp_performer(x)
        h_linear = self.linear_shortcut(x)

        mu_performer = self.fc_mu_performer(h_performer)
        logvar_performer = self.fc_logvar_performer(h_performer)

        mu_linear = self.fc_mu_linear(h_linear)
        logvar_linear = self.fc_logvar_linear(h_linear)

        mu = torch.cat(
            [mu_performer, mu_linear],
            dim=1,
        )
        logvar = torch.cat(
            [logvar_performer, logvar_linear],
            dim=1,
        )
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        y_p = self.predictor(z)
        x_r = self.decoder(z)

        return x_r, y_p, mu, logvar


# =============================================================================
# 4. Data loading and alignment
# =============================================================================

def resolve_data_dir(
    base_dir: Path,
    explicit_data_dir=None,
) -> Path:
    """
    Locate only the files needed for this PH post-hoc model.
    """
    if explicit_data_dir is not None:
        candidates = [Path(explicit_data_dir)]
    else:
        candidates = [
            base_dir / "data",
            base_dir.parent / "data",
            base_dir.parent / "data" / "rice299",
            Path(r"E:\VGMP\rice299"),
            base_dir,
        ]

    required = [
        GENOTYPE_FILENAME,
        PHENOTYPE_FILENAME,
    ]

    for data_dir in candidates:
        if all(
            (data_dir / name).exists()
            for name in required
        ):
            return data_dir

    checked = "\n".join(
        f"  - {str(p)}"
        for p in candidates
    )
    raise FileNotFoundError(
        "Could not find the Rice299 PH data files.\n"
        f"Required:\n  - {GENOTYPE_FILENAME}\n"
        f"  - {PHENOTYPE_FILENAME}\n"
        f"Checked:\n{checked}\n"
        "Use --data-dir to specify the folder explicitly."
    )


def load_genotype(genotype_file: Path):
    geno_df = pd.read_csv(genotype_file)

    if "Entry" not in geno_df.columns:
        raise KeyError(
            "Genotype file must contain an 'Entry' column."
        )

    geno_df["Entry"] = (
        geno_df["Entry"]
        .astype(str)
        .str.strip()
    )

    if geno_df["Entry"].duplicated().any():
        dup = (
            geno_df.loc[
                geno_df["Entry"].duplicated(),
                "Entry"
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Duplicate genotype Entry IDs found, e.g. {dup[:5]}"
        )

    snp_cols = [
        c
        for c in geno_df.columns
        if c != "Entry"
    ]

    X = (
        geno_df[snp_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    entries = (
        geno_df["Entry"]
        .astype(str)
        .to_numpy()
    )

    print("=" * 90)
    print("Rice299 genotype")
    print("=" * 90)
    print("File:", genotype_file)
    print("Samples:", X.shape[0])
    print("SNPs:", X.shape[1])
    print("True NaN:", int(np.isnan(X).sum()))
    print(
        "Genotype range (ignoring NaN):",
        float(np.nanmin(X)),
        "to",
        float(np.nanmax(X)),
    )
    print(
        "IMPORTANT: -1 is retained as a VALID genotype value."
    )

    return (
        np.ascontiguousarray(X, dtype=np.float32),
        entries,
        snp_cols,
    )


def align_ph(
    X_genotype,
    genotype_entries,
    phenotype_file: Path,
):
    phe_df = pd.read_csv(phenotype_file)

    if "Entry" not in phe_df.columns or TRAIT not in phe_df.columns:
        raise KeyError(
            f"{phenotype_file} must contain 'Entry' and '{TRAIT}'. "
            f"Found: {list(phe_df.columns)}"
        )

    phe_df["Entry"] = (
        phe_df["Entry"]
        .astype(str)
        .str.strip()
    )

    if phe_df["Entry"].duplicated().any():
        raise ValueError(
            "Duplicate phenotype Entry IDs found."
        )

    phe_df[TRAIT] = pd.to_numeric(
        phe_df[TRAIT],
        errors="coerce",
    )

    geno_index = pd.DataFrame({
        "Entry": genotype_entries,
        "Genotype_Row_0Based": np.arange(
            len(genotype_entries),
            dtype=int,
        ),
    })

    aligned = geno_index.merge(
        phe_df[["Entry", TRAIT]],
        on="Entry",
        how="inner",
        sort=False,
    )

    aligned = (
        aligned
        .dropna(subset=[TRAIT])
        .reset_index(drop=True)
    )

    if aligned.empty:
        raise ValueError(
            "No aligned samples were found for PH."
        )

    rows = aligned[
        "Genotype_Row_0Based"
    ].to_numpy(dtype=int)

    X = np.ascontiguousarray(
        X_genotype[rows],
        dtype=np.float32,
    )

    y = (
        aligned[TRAIT]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )

    entries = (
        aligned["Entry"]
        .astype(str)
        .to_numpy()
    )

    print("\n" + "=" * 90)
    print("Aligned Rice299-PH dataset")
    print("=" * 90)
    print("Aligned samples:", len(X))
    print("SNPs:", X.shape[1])
    print("PH mean:", float(np.mean(y)))
    print("PH SD:", float(np.std(y, ddof=1)))
    print("PH min:", float(np.min(y)))
    print("PH max:", float(np.max(y)))
    print("PH Q90:", float(np.quantile(y, 0.90)))
    print("PH Q95:", float(np.quantile(y, 0.95)))
    print("PH Q97.5:", float(np.quantile(y, 0.975)))

    return X, y, entries


def fit_full_data_preprocessing(X_all, y_all):
    """
    Full-data preprocessing is intentional here because this final model is used
    ONLY for post-hoc computational analysis, not for generalization estimates.
    """

    # True NaN only. -1 is NOT treated as missing.
    imputer = SimpleImputer(
        strategy="most_frequent",
        keep_empty_features=True,
    )

    X_imputed = (
        imputer
        .fit_transform(X_all)
        .astype(np.float32)
    )

    if X_imputed.shape[1] != X_all.shape[1]:
        raise RuntimeError(
            "Imputation changed the SNP dimension unexpectedly."
        )

    if np.isnan(X_imputed).any():
        raise RuntimeError(
            "NaN values remain after imputation."
        )

    # No SNP scaling.
    scaler_y = StandardScaler()
    y_scaled = (
        scaler_y
        .fit_transform(y_all)
        .astype(np.float32)
    )

    return X_imputed, y_scaled, imputer, scaler_y


# =============================================================================
# 5. Training / evaluation
# =============================================================================

def make_loader(X, y, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(X, dtype=np.float32)
        ),
        torch.from_numpy(
            np.ascontiguousarray(y, dtype=np.float32)
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_full_data_model(
    model,
    X_imputed,
    y_scaled,
    device,
):
    loss_fn = GeneticAwareLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_batch = safe_batch_size(
        len(X_imputed),
        BATCH_SIZE,
    )

    loader = make_loader(
        X_imputed,
        y_scaled,
        train_batch,
        shuffle=True,
        seed=SEED,
    )

    history = []

    print("\n" + "=" * 90)
    print("Training final PH post-hoc model on ALL aligned samples")
    print("=" * 90)
    print("Samples:", len(X_imputed))
    print("Batch size:", train_batch)
    print("Epochs:", MAX_EPOCHS)
    print("Validation: NONE")
    print("Early stopping: NONE")
    print("Scheduler: NONE")
    print(
        "This model is for post-hoc computational analysis only; "
        "it is not used for CV performance estimates."
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cuda_sync()
    training_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        sum_total = 0.0
        sum_pred = 0.0
        sum_recon = 0.0
        sum_pearson = 0.0
        sum_kl = 0.0
        n_batches = 0

        cuda_sync()
        epoch_start = time.perf_counter()

        for bx, by in loader:
            bx = bx.to(
                device,
                non_blocking=True,
            )
            by = by.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            xr, yp, mu, lv = model(bx)

            (
                loss,
                recon_l,
                pearson_l,
                pred_l,
                kl_l,
            ) = loss_fn(
                xr,
                bx,
                yp,
                by,
                mu,
                lv,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )

            optimizer.step()

            sum_total += float(loss.detach().item())
            sum_pred += float(pred_l.detach().item())
            sum_recon += float(recon_l.detach().item())
            sum_pearson += float(pearson_l.detach().item())
            sum_kl += float(kl_l.detach().item())
            n_batches += 1

        cuda_sync()
        epoch_time = time.perf_counter() - epoch_start

        n_batches = max(n_batches, 1)

        row = {
            "Epoch": epoch,
            "Total_Loss": sum_total / n_batches,
            "Prediction_MSE": sum_pred / n_batches,
            "Reconstruction_MSE": sum_recon / n_batches,
            "Pearson_Loss": sum_pearson / n_batches,
            "KL_Loss": sum_kl / n_batches,
            "Weighted_Pred": (
                PRED_WEIGHT
                * sum_pred
                / n_batches
            ),
            "Weighted_Recon": (
                RECON_WEIGHT
                * sum_recon
                / n_batches
            ),
            "Weighted_KL": (
                KL_WEIGHT
                * sum_kl
                / n_batches
            ),
            "Epoch_Time_s": epoch_time,
        }

        history.append(row)

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == MAX_EPOCHS
        ):
            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Total={row['Total_Loss']:.6f} | "
                f"Pred={row['Prediction_MSE']:.6f} | "
                f"Recon={row['Reconstruction_MSE']:.6f} | "
                f"KL={row['KL_Loss']:.6f} | "
                f"{epoch_time:.3f}s"
            )

    cuda_sync()
    training_time = time.perf_counter() - training_start

    peak_gpu_mb = (
        float(
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )
        if torch.cuda.is_available()
        else np.nan
    )

    return (
        pd.DataFrame(history),
        float(training_time),
        peak_gpu_mb,
        train_batch,
    )


@torch.no_grad()
def collect_full_data_outputs(
    model,
    X_imputed,
    y_scaled,
    scaler_y,
    device,
):
    model.eval()

    eval_batch = min(
        BATCH_SIZE,
        len(X_imputed),
    )

    loader = make_loader(
        X_imputed,
        y_scaled,
        eval_batch,
        shuffle=False,
        seed=SEED,
    )

    pred_scaled_all = []
    mu_all = []
    logvar_all = []
    recon_mse_per_sample = []

    for bx, by in loader:
        bx = bx.to(device)

        xr, yp, mu, logvar = model(bx)

        pred_scaled_all.append(
            yp.detach().cpu().numpy()
        )
        mu_all.append(
            mu.detach().cpu().numpy()
        )
        logvar_all.append(
            logvar.detach().cpu().numpy()
        )

        sample_recon = torch.mean(
            (xr - bx) ** 2,
            dim=1,
        )
        recon_mse_per_sample.extend(
            sample_recon.detach().cpu().numpy().tolist()
        )

    pred_scaled = np.concatenate(
        pred_scaled_all,
        axis=0,
    )

    mu = np.concatenate(
        mu_all,
        axis=0,
    )

    logvar = np.concatenate(
        logvar_all,
        axis=0,
    )

    pred_real = (
        scaler_y
        .inverse_transform(pred_scaled)
        .reshape(-1)
    )

    return (
        pred_scaled.reshape(-1),
        pred_real,
        mu,
        logvar,
        np.asarray(
            recon_mse_per_sample,
            dtype=np.float64,
        ),
    )


def regression_metrics(y_true, y_pred):
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

    rmse = float(np.sqrt(mse))
    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    if (
        len(y_true) >= 2
        and np.std(y_true) > 1e-12
        and np.std(y_pred) > 1e-12
    ):
        pearson_r = float(
            stats.pearsonr(
                y_true,
                y_pred,
            )[0]
        )
    else:
        pearson_r = np.nan

    return {
        "Pearson_r_InSample": pearson_r,
        "R2_InSample": r2,
        "RMSE_InSample": rmse,
        "MAE_InSample": mae,
    }


# =============================================================================
# 6. Save all assets needed for subsequent round-trip validation
# =============================================================================

def save_outputs(
    output_dir,
    model,
    X_all_raw,
    X_imputed,
    y_all,
    y_scaled,
    entries,
    snp_cols,
    imputer,
    scaler_y,
    history_df,
    training_time,
    peak_gpu_mb,
    train_batch,
    pred_scaled,
    pred_real,
    mu,
    logvar,
    recon_mse_per_sample,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = output_dir / OUTPUT_BASENAME

    # -------------------------------------------------------------------------
    # A. Sample alignment
    # -------------------------------------------------------------------------
    aligned_df = pd.DataFrame({
        "Aligned_Row_0Based": np.arange(
            len(entries),
            dtype=int,
        ),
        "Entry": entries,
        "Observed_PH": y_all.reshape(-1),
        "Scaled_PH": y_scaled.reshape(-1),
    })

    aligned_df.to_csv(
        f"{prefix}_AlignedSamples.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # B. SNP order + imputer statistics + observed state information
    # -------------------------------------------------------------------------
    snp_rows = []

    for j, snp in enumerate(snp_cols):
        raw_col = X_all_raw[:, j]
        imputed_col = X_imputed[:, j]

        observed_raw = raw_col[
            np.isfinite(raw_col)
        ]

        observed_states = (
            sorted(
                np.unique(
                    observed_raw
                ).tolist()
            )
            if len(observed_raw) > 0
            else []
        )

        imputed_states = sorted(
            np.unique(
                imputed_col
            ).tolist()
        )

        snp_rows.append({
            "SNP_Index_0Based": j,
            "SNP": snp,
            "Imputer_MostFrequent": float(
                imputer.statistics_[j]
            ),
            "Raw_Missing_Count": int(
                np.isnan(raw_col).sum()
            ),
            "Observed_Raw_States": ",".join(
                str(float(x))
                for x in observed_states
            ),
            "Observed_Imputed_States": ",".join(
                str(float(x))
                for x in imputed_states
            ),
        })

    pd.DataFrame(
        snp_rows
    ).to_csv(
        f"{prefix}_SNP_Order_Imputer.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # C. Training history
    # -------------------------------------------------------------------------
    history_df.to_csv(
        f"{prefix}_TrainingHistory.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # D. In-sample predictions: sanity check ONLY
    # -------------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "Aligned_Row_0Based": np.arange(
            len(entries),
            dtype=int,
        ),
        "Entry": entries,
        "Observed_PH": y_all.reshape(-1),
        "Predicted_PH_InSample": pred_real,
        "Predicted_PH_Scaled_InSample": pred_scaled,
        "Reconstruction_MSE_PerSample": recon_mse_per_sample,
    })

    pred_df.to_csv(
        f"{prefix}_InSamplePredictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # E. Latent posterior mean/logvar
    # -------------------------------------------------------------------------
    latent_df = pd.DataFrame({
        "Aligned_Row_0Based": np.arange(
            len(entries),
            dtype=int,
        ),
        "Entry": entries,
        "Observed_PH": y_all.reshape(-1),
    })

    for k in range(mu.shape[1]):
        latent_df[f"Mu_{k:02d}"] = mu[:, k]

    for k in range(logvar.shape[1]):
        latent_df[f"LogVar_{k:02d}"] = logvar[:, k]

    latent_df.to_csv(
        f"{prefix}_LatentPosterior.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # F. PH reference distribution for next-step optimization
    # -------------------------------------------------------------------------
    y_flat = y_all.reshape(-1)

    q90 = float(
        np.quantile(
            y_flat,
            0.90,
        )
    )
    q95 = float(
        np.quantile(
            y_flat,
            0.95,
        )
    )
    q975 = float(
        np.quantile(
            y_flat,
            0.975,
        )
    )

    elite_mask = y_flat >= q95
    elite_indices = np.where(
        elite_mask
    )[0]

    phenotype_reference = pd.DataFrame([
        {
            "N_Samples": len(y_flat),
            "PH_Mean": float(
                np.mean(y_flat)
            ),
            "PH_SD_ddof1": float(
                np.std(
                    y_flat,
                    ddof=1,
                )
            ),
            "PH_Min": float(
                np.min(y_flat)
            ),
            "PH_Max": float(
                np.max(y_flat)
            ),
            "PH_Q90": q90,
            "PH_Q95": q95,
            "PH_Q97_5": q975,
            "Elite_Definition": "Observed_PH >= empirical Q95",
            "Elite_N": int(
                elite_mask.sum()
            ),
            "Elite_PH_Mean": float(
                np.mean(
                    y_flat[
                        elite_mask
                    ]
                )
            ),
            "Elite_PH_Min": float(
                np.min(
                    y_flat[
                        elite_mask
                    ]
                )
            ),
            "Elite_PH_Max": float(
                np.max(
                    y_flat[
                        elite_mask
                    ]
                )
            ),
        }
    ])

    phenotype_reference.to_csv(
        f"{prefix}_PhenotypeReference.csv",
        index=False,
    )

    pd.DataFrame({
        "Aligned_Row_0Based": elite_indices,
        "Entry": entries[
            elite_indices
        ],
        "Observed_PH": y_flat[
            elite_indices
        ],
    }).to_csv(
        f"{prefix}_Elite_Q95_Samples.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # G. Summary
    # -------------------------------------------------------------------------
    total_params, trainable_params = count_parameters(
        model
    )

    metrics = regression_metrics(
        y_flat,
        pred_real,
    )

    summary = {
        "Dataset": "Rice299",
        "Trait": TRAIT,
        "Analysis_Role": (
            "Final full-data post-hoc model; "
            "NOT cross-validation performance evaluation"
        ),
        "N_Aligned_Samples": int(
            len(entries)
        ),
        "N_SNPs": int(
            len(snp_cols)
        ),
        "Seed": SEED,
        "Chunk_Size": CHUNK_SIZE,
        "Performer_Sequence_Length": int(
            model.snp_performer.sequence_length
        ),
        "Padding_Size": int(
            model.snp_performer.pad_n
        ),
        "Performer_Readout_Dim": (
            PERFORMER_READOUT_DIM
        ),
        "Decoder_Performer_Dim": (
            DECODER_PERFORMER_DIM
        ),
        "Total_Latent_Dim": (
            TOTAL_LATENT_DIM
        ),
        "Performer_Latent_Dim": (
            PERFORMER_LATENT_DIM
        ),
        "Linear_Latent_Dim": (
            LINEAR_LATENT_DIM
        ),
        "Predictor_Head": (
            "48->64(BN+GELU)->1; "
            "no predictor dropout"
        ),
        "Dropout_Upstream": DROPOUT,
        "Epochs": MAX_EPOCHS,
        "Batch_Size_Used": train_batch,
        "Optimizer": "Adam",
        "Learning_Rate": LEARNING_RATE,
        "Weight_Decay": WEIGHT_DECAY,
        "Gradient_Clip_MaxNorm": GRAD_CLIP,
        "Pred_Weight": PRED_WEIGHT,
        "Pearson_Weight": PEARSON_WEIGHT,
        "Recon_Weight": RECON_WEIGHT,
        "KL_Weight": KL_WEIGHT,
        "SNP_Scaling": "None",
        "Missing_Value_Rule": (
            "True NaN -> most frequent; "
            "-1 retained as valid genotype"
        ),
        "Phenotype_Scaling": "StandardScaler",
        "Validation": "None",
        "Early_Stopping": "None",
        "Scheduler": "None",
        "Training_Time_s": training_time,
        "Peak_GPU_Memory_MB": peak_gpu_mb,
        "Total_Params": total_params,
        "Trainable_Params": trainable_params,
        "PH_Q90": q90,
        "PH_Q95": q95,
        "PH_Q97_5": q975,
        "Elite_Q95_N": int(
            elite_mask.sum()
        ),
        "Mean_Reconstruction_MSE_InSample": float(
            np.mean(
                recon_mse_per_sample
            )
        ),
        **metrics,
        "Metric_Warning": (
            "In-sample metrics are sanity checks only "
            "and must not be interpreted as "
            "generalization performance."
        ),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        f"{prefix}_Summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # H. Reproducibility metadata JSON
    # -------------------------------------------------------------------------
    config = {
        "dataset": "Rice299",
        "trait": TRAIT,
        "purpose": (
            "full-data post-hoc model for "
            "computational PH-directed genotype analysis"
        ),
        "architecture": {
            "chunk_size": CHUNK_SIZE,
            "performer_readout_dim": (
                PERFORMER_READOUT_DIM
            ),
            "decoder_performer_dim": (
                DECODER_PERFORMER_DIM
            ),
            "total_latent_dim": (
                TOTAL_LATENT_DIM
            ),
            "performer_latent_dim": (
                PERFORMER_LATENT_DIM
            ),
            "linear_latent_dim": (
                LINEAR_LATENT_DIM
            ),
            "upstream_dropout": DROPOUT,
            "predictor": (
                "48->64 BatchNorm GELU ->1, "
                "no predictor dropout"
            ),
        },
        "training": {
            "seed": SEED,
            "epochs": MAX_EPOCHS,
            "batch_size_preferred": (
                BATCH_SIZE
            ),
            "batch_size_used": (
                train_batch
            ),
            "optimizer": "Adam",
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "grad_clip_max_norm": (
                GRAD_CLIP
            ),
            "validation": False,
            "early_stopping": False,
            "scheduler": None,
        },
        "loss": {
            "prediction_weight": (
                PRED_WEIGHT
            ),
            "pearson_weight": (
                PEARSON_WEIGHT
            ),
            "reconstruction_weight": (
                RECON_WEIGHT
            ),
            "kl_weight": (
                KL_WEIGHT
            ),
            "kl_reduction": (
                "mean over batch and latent dimensions"
            ),
        },
        "preprocessing": {
            "genotype_scaling": None,
            "minus_one_is_valid_genotype": True,
            "missing_values": (
                "Only true NaN values are imputed "
                "with per-SNP most frequent values"
            ),
            "phenotype_scaling": (
                "StandardScaler fit on all aligned PH samples"
            ),
        },
        "posthoc_reference": {
            "primary_target_suggestion": "PH empirical Q95",
            "sensitivity_targets": [
                "Q90",
                "Q95",
                "Q97.5",
            ],
            "elite_definition": (
                "Observed PH >= empirical Q95"
            ),
        },
    }

    with open(
        f"{prefix}_Config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # I. Checkpoint for round-trip / latent optimization
    # -------------------------------------------------------------------------
    checkpoint = {
        "model_state_dict": model.state_dict(),

        "dataset": "Rice299",
        "trait": TRAIT,
        "analysis_role": (
            "Final full-data post-hoc model only"
        ),

        "snp_names": list(
            map(
                str,
                snp_cols,
            )
        ),
        "aligned_entries": list(
            map(
                str,
                entries,
            )
        ),

        # Store preprocessing numerically so next script can reproduce it
        # without depending on a pickled sklearn object.
        "imputer_statistics": torch.tensor(
            np.asarray(
                imputer.statistics_,
                dtype=np.float32,
            )
        ),
        "scaler_y_mean": float(
            scaler_y.mean_[0]
        ),
        "scaler_y_scale": float(
            scaler_y.scale_[0]
        ),

        "phenotype_reference": {
            "mean": float(
                np.mean(y_flat)
            ),
            "sd_ddof1": float(
                np.std(
                    y_flat,
                    ddof=1,
                )
            ),
            "min": float(
                np.min(y_flat)
            ),
            "max": float(
                np.max(y_flat)
            ),
            "q90": q90,
            "q95": q95,
            "q97_5": q975,
            "elite_q95_indices_0based": (
                elite_indices.tolist()
            ),
        },

        "model_config": {
            "snp_dim": int(
                len(snp_cols)
            ),
            "chunk_size": CHUNK_SIZE,
            "performer_readout_dim": (
                PERFORMER_READOUT_DIM
            ),
            "decoder_performer_dim": (
                DECODER_PERFORMER_DIM
            ),
            "latent_dim": (
                TOTAL_LATENT_DIM
            ),
            "performer_latent_dim": (
                PERFORMER_LATENT_DIM
            ),
            "linear_latent_dim": (
                LINEAR_LATENT_DIM
            ),
            "dropout": DROPOUT,
        },

        "training_config": {
            "seed": SEED,
            "epochs": MAX_EPOCHS,
            "batch_size": train_batch,
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "grad_clip": (
                GRAD_CLIP
            ),
            "pred_weight": (
                PRED_WEIGHT
            ),
            "pearson_weight": (
                PEARSON_WEIGHT
            ),
            "recon_weight": (
                RECON_WEIGHT
            ),
            "kl_weight": (
                KL_WEIGHT
            ),
        },
    }

    torch.save(
        checkpoint,
        f"{prefix}.pth",
    )

    print("\nSaved outputs:")
    for path in sorted(
        output_dir.glob(
            f"{OUTPUT_BASENAME}*"
        )
    ):
        print("  ", path.name)


# =============================================================================
# 7. Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the final full-data Rice299-PH GenoPerfVAE "
            "post-hoc model."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Folder containing the genotype CSV and rice299_PH.csv. "
            "If omitted, common project folders are searched automatically."
        ),
    )

    parser.add_argument(
        "--genotype",
        type=str,
        default=None,
        help=(
            "Optional explicit genotype CSV path. "
            "Overrides --data-dir genotype lookup."
        ),
    )

    parser.add_argument(
        "--phenotype",
        type=str,
        default=None,
        help=(
            "Optional explicit rice299_PH.csv path. "
            "Overrides --data-dir phenotype lookup."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: a results folder "
            "beside this script."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed_fixed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print("GenoPerfVAE Rice299-PH final post-hoc training")
    print("=" * 90)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device)
        )

    print(
        "Architecture:",
        f"C{CHUNK_SIZE}, "
        f"readout={PERFORMER_READOUT_DIM}, "
        f"decoder={DECODER_PERFORMER_DIM}, "
        f"latent={PERFORMER_LATENT_DIM}+{LINEAR_LATENT_DIM}"
        f"={TOTAL_LATENT_DIM}"
    )

    print(
        "Training:",
        f"Adam(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}), "
        f"epochs={MAX_EPOCHS}, "
        f"pred={PRED_WEIGHT}, "
        f"recon={RECON_WEIGHT}, "
        f"KL={KL_WEIGHT}, "
        f"seed={SEED}"
    )

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    if args.genotype is not None:
        genotype_file = Path(
            args.genotype
        )
    else:
        data_dir = resolve_data_dir(
            script_dir,
            args.data_dir,
        )
        genotype_file = (
            data_dir
            / GENOTYPE_FILENAME
        )

    if args.phenotype is not None:
        phenotype_file = Path(
            args.phenotype
        )
    else:
        if args.genotype is not None and args.data_dir is None:
            # If only genotype is explicitly provided, try its folder first.
            candidate = (
                genotype_file.parent
                / PHENOTYPE_FILENAME
            )
            if candidate.exists():
                phenotype_file = candidate
            else:
                data_dir = resolve_data_dir(
                    script_dir,
                    args.data_dir,
                )
                phenotype_file = (
                    data_dir
                    / PHENOTYPE_FILENAME
                )
        else:
            if "data_dir" not in locals():
                data_dir = resolve_data_dir(
                    script_dir,
                    args.data_dir,
                )
            phenotype_file = (
                data_dir
                / PHENOTYPE_FILENAME
            )

    if not genotype_file.exists():
        raise FileNotFoundError(
            f"Genotype file not found: {genotype_file}"
        )

    if not phenotype_file.exists():
        raise FileNotFoundError(
            f"Phenotype file not found: {phenotype_file}"
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (
            script_dir
            / "results_Rice_PH_Final_PostHoc"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        X_genotype,
        genotype_entries,
        snp_cols,
    ) = load_genotype(
        genotype_file
    )

    (
        X_all,
        y_all,
        entries,
    ) = align_ph(
        X_genotype,
        genotype_entries,
        phenotype_file,
    )

    (
        X_imputed,
        y_scaled,
        imputer,
        scaler_y,
    ) = fit_full_data_preprocessing(
        X_all,
        y_all,
    )

    print("\nPreprocessing")
    print("-------------")
    print(
        "True NaN before imputation:",
        int(
            np.isnan(X_all).sum()
        )
    )
    print(
        "NaN after imputation:",
        int(
            np.isnan(X_imputed).sum()
        )
    )
    print(
        "-1 count after imputation:",
        int(
            np.sum(
                X_imputed == -1
            )
        )
    )
    print(
        "No SNP scaling is applied."
    )
    print(
        "PH scaler mean:",
        float(
            scaler_y.mean_[0]
        )
    )
    print(
        "PH scaler scale:",
        float(
            scaler_y.scale_[0]
        )
    )

    model = EnhancedGeneticModel(
        snp_dim=X_imputed.shape[1],
        latent_dim=TOTAL_LATENT_DIM,
        performer_readout_dim=(
            PERFORMER_READOUT_DIM
        ),
        decoder_performer_dim=(
            DECODER_PERFORMER_DIM
        ),
        performer_latent_dim=(
            PERFORMER_LATENT_DIM
        ),
        linear_latent_dim=(
            LINEAR_LATENT_DIM
        ),
        dropout=DROPOUT,
        chunk_size=CHUNK_SIZE,
    ).to(device)

    total_params, trainable_params = count_parameters(
        model
    )

    print("\nModel")
    print("-----")
    print(
        "Sequence length:",
        model.snp_performer.sequence_length
    )
    print(
        "Padding:",
        model.snp_performer.pad_n
    )
    print(
        "Total params:",
        f"{total_params:,}"
    )
    print(
        "Trainable params:",
        f"{trainable_params:,}"
    )

    (
        history_df,
        training_time,
        peak_gpu_mb,
        train_batch,
    ) = train_full_data_model(
        model,
        X_imputed,
        y_scaled,
        device,
    )

    (
        pred_scaled,
        pred_real,
        mu,
        logvar,
        recon_mse_per_sample,
    ) = collect_full_data_outputs(
        model,
        X_imputed,
        y_scaled,
        scaler_y,
        device,
    )

    sanity = regression_metrics(
        y_all.reshape(-1),
        pred_real,
    )

    print("\n" + "=" * 90)
    print("Post-training sanity check (IN-SAMPLE ONLY)")
    print("=" * 90)

    for key, value in sanity.items():
        print(
            key,
            "=",
            f"{value:.6f}"
            if np.isfinite(value)
            else "NaN"
        )

    print(
        "Mean reconstruction MSE =",
        f"{np.mean(recon_mse_per_sample):.6f}"
    )
    print(
        "WARNING: these in-sample metrics are NOT "
        "generalization-performance estimates."
    )

    save_outputs(
        output_dir=output_dir,
        model=model,
        X_all_raw=X_all,
        X_imputed=X_imputed,
        y_all=y_all,
        y_scaled=y_scaled,
        entries=entries,
        snp_cols=snp_cols,
        imputer=imputer,
        scaler_y=scaler_y,
        history_df=history_df,
        training_time=training_time,
        peak_gpu_mb=peak_gpu_mb,
        train_batch=train_batch,
        pred_scaled=pred_scaled,
        pred_real=pred_real,
        mu=mu,
        logvar=logvar,
        recon_mse_per_sample=recon_mse_per_sample,
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(
        "Primary checkpoint:",
        output_dir
        / f"{OUTPUT_BASENAME}.pth"
    )
    print(
        "Next step: use this checkpoint for PH-directed "
        "latent optimization and target-genotype "
        "round-trip validation."
    )


if __name__ == "__main__":
    main()
