# -*- coding: utf-8 -*-
"""
GenoPerfVAE_LGCG: standalone full-model implementation for Wheat2000.

Architecture:
    SNP input
      -> LGCG chunk-context recalibration
      -> Performer encoder (256-D readout)
      -> Performer posterior branch (mu/logvar: 32-D)
      -> Linear Shortcut (D -> 2048 -> 512 -> 64)
      -> Linear posterior branch (mu/logvar: 16-D)
      -> concatenate posterior parameters (32 + 16 = 48)
      -> one 48-D VAE reparameterization during training
      -> Predictor (48 -> 64 -> 1)
      -> SNP Decoder (48 -> SNP dimension; training reconstruction only)

Loss:
    1.50 * phenotype prediction MSE
  + 0.30 * SNP reconstruction MSE
  + 0.80 * KL divergence

Wheat2000 protocol:
  - genotype file: data/2000gene.pkl
  - six phenotype files in the same data/ directory
  - -1 is retained as a valid genotype value; only true NaN is imputed
  - genotype imputation and phenotype standardization are fitted on each
    outer training fold only
  - genotype/phenotype row counts must match; no silent truncation
  - 10-fold KFold, fixed 200 epochs, no early stopping or scheduler
  - test prediction and inference timing do not execute the auxiliary Decoder

Usage:
    python GenoPerfVAE_LGCG.py
    python GenoPerfVAE_LGCG.py --traits TKW PROT --seed 42
    python GenoPerfVAE_LGCG.py --traits TKW --max-folds 1 --epochs 1

Dependencies:
    pip install performer-pytorch pandas numpy scipy scikit-learn torch
"""

import argparse
import gc
import json
import math
import os
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from performer_pytorch import Performer
except ImportError as exc:
    raise ImportError(
        "Install performer-pytorch in the active environment: "
        "pip install performer-pytorch"
    ) from exc


# =============================================================================
# Fixed model and training parameters
# =============================================================================
SEED = 42
N_SPLITS = 10

CHUNK_SIZE = 32
LGCG_HIDDEN_DIM = 16
LGCG_ALPHA = 0.5
LGCG_KERNEL_SIZE = 3

PERFORMER_READOUT_DIM = 256
DECODER_PERFORMER_DIM = 256
ENCODER_DEPTH = 2
ENCODER_HEADS = 4
DECODER_DEPTH = 2
DECODER_HEADS = 4

PERFORMER_LATENT_DIM = 32
LINEAR_LATENT_DIM = 16
TOTAL_LATENT_DIM = 48

SHORTCUT_HIDDEN_DIM_1 = 2048
SHORTCUT_HIDDEN_DIM_2 = 512
SHORTCUT_OUTPUT_DIM = 64
PREDICTOR_HIDDEN_DIM = 64

MODEL_DROPOUT = 0.3
SHORTCUT_DROPOUT = 0.3
PREDICTOR_DROPOUT = 0.0

BATCH_SIZE = 48
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

PRED_WEIGHT = 1.50
RECON_WEIGHT = 0.30
KL_WEIGHT = 0.80

TRAIT_FILES = {
    "TKW": "2000_1_phe_tkw.txt",
    "TESTW": "2000_2_phe_testw.txt",
    "LENGTH": "2000_3_phe_length.txt",
    "WIDTH": "2000_4_phe_width.txt",
    "HARD": "2000_5_phe_hardness.txt",
    "PROT": "2000_6_phe_Prot.txt",
}


class LocalGlobalChunkGate(nn.Module):
    """Local-global context-gated reweighting of fixed-size SNP chunks.

    Input/output: [batch, num_chunks, chunk_size].
    Independent sigmoid gate per chunk; the gate is NOT a softmax distribution.
    Complexity: O(B * N * (C*H + C)) for fixed H and kernel size.
    """

    def __init__(self, chunk_size, num_chunks, pad_n=0, hidden_dim=16,
                 dropout=0.3, alpha=0.5, kernel_size=3):
        super().__init__()
        C = int(chunk_size)
        N = int(num_chunks)
        H = int(hidden_dim)
        if C <= 0 or N <= 0 or H <= 0:
            raise ValueError("chunk_size, num_chunks and hidden_dim must be positive")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1] to keep residual scaling nonnegative")
        if not (0 <= pad_n < C):
            raise ValueError("pad_n must be between 0 and chunk_size-1")
        if int(kernel_size) < 1 or int(kernel_size) % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")

        self.alpha = float(alpha)  # fixed hyperparameter, zero parameters
        self.norm = nn.LayerNorm(C)
        # Depthwise convolution along the chunk axis.
        self.local_context = nn.Conv1d(
            C, C, kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2, groups=C, bias=False
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * C, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, 1),
        )
        # Sigmoid gate = 0.5 at initialization -> residual scale exactly 1.
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        fraction = torch.ones(N, dtype=torch.float32)
        if pad_n:
            fraction[-1] = (C - pad_n) / C
        self.register_buffer("chunk_valid_fraction", fraction, persistent=False)
        self.last_chunk_gates = None

    def forward(self, chunks):
        if chunks.ndim != 3 or chunks.size(-1) != self.norm.normalized_shape[0]:
            raise ValueError("Expected [batch, num_chunks, chunk_size] input")

        z = self.norm(chunks)
        B, N, C = z.shape
        if N != self.chunk_valid_fraction.numel():
            raise ValueError("Unexpected number of chunks")

        # Adjacent-chunk context. SNP order must be meaningful; see model notes.
        local = self.local_context(z.transpose(1, 2)).transpose(1, 2)

        # The last partially padded chunk contributes proportionally fewer SNPs.
        valid = self.chunk_valid_fraction.to(dtype=z.dtype).view(1, N, 1)
        global_context = (z * valid).sum(dim=1, keepdim=True) / valid.sum()
        global_context = global_context.expand(B, N, C)

        fusion = torch.cat((z, local, global_context), dim=-1)
        gate = torch.sigmoid(self.gate_mlp(fusion))  # [B,N,1] in (0,1)
        self.last_chunk_gates = gate.detach()
        scale = 1.0 + self.alpha * (2.0 * gate - 1.0)
        return z * scale

    def get_chunk_gates(self):
        """Return gates [B,N], or None until forward; NOT attention weights."""
        return (None if self.last_chunk_gates is None
                else self.last_chunk_gates.squeeze(-1))

class LGCGChunkedSNP_Performer(nn.Module):
    """LGCG fixed-SNP-chunk encoding followed by unchanged Performer/P2 readout.

    SNP input -> zero pad -> fixed chunks -> LayerNorm/local-global gate
    -> original learnable positional embeddings -> Performer -> full-token
    flatten -> Linear(N*C, 256).

    This prototype does not automatically enforce chromosome boundaries;
    do not interpret depthwise-convolution adjacency as biological linkage
    across chromosome junctions without providing boundary-aware metadata.
    """

    def __init__(
        self, snp_dim, target_chunk_size=32, performer_dim=128,
        dropout=0.3, lgcg_hidden_dim=16, lgcg_alpha=0.5,
        lgcg_kernel_size=3,
    ):
        super().__init__()

        self.snp_dim = int(snp_dim)
        self.chunk_size = int(target_chunk_size)
        if self.snp_dim <= 0 or self.chunk_size <= 0:
            raise ValueError("snp_dim and target_chunk_size must be positive")
        if self.chunk_size % 4 != 0:
            raise ValueError("chunk_size must be divisible by 4 (Performer heads)")

        self.pad_n = (-self.snp_dim) % self.chunk_size
        self.num_chunks = (self.snp_dim + self.pad_n) // self.chunk_size
        self.sequence_length = self.num_chunks
        self.flatten_readout_dim = self.num_chunks * self.chunk_size
        self.padding_layer = (nn.ZeroPad1d((0, self.pad_n))
                              if self.pad_n else None)

        # LGCG replaces chunk_norm + chunk_weight_net + chunk_scale.
        # Keep initialization reproducible so that the
        # positional embedding, Performer and downstream shared modules receive
        # matched random initializations under the same seed as the baseline.
        with torch.random.fork_rng(devices=[]):
            self.lgcg = LocalGlobalChunkGate(
                chunk_size=self.chunk_size,
                num_chunks=self.num_chunks,
                pad_n=self.pad_n,
                hidden_dim=lgcg_hidden_dim,
                dropout=dropout,
                alpha=lgcg_alpha,
                kernel_size=lgcg_kernel_size,
            )
        original_hidden_dim = max(self.chunk_size // 4, 32)
        _original_chunk_mlp_rng_advance = nn.Sequential(
            nn.Linear(self.chunk_size, original_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(original_hidden_dim, 1),
        )
        del _original_chunk_mlp_rng_advance

        # Keep the learnable positional encoding fixed across all ablation variants.
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_chunks, self.chunk_size)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        num_heads = ENCODER_HEADS
        head_dim = self.chunk_size // num_heads
        self.performer = Performer(
            dim=self.chunk_size, depth=ENCODER_DEPTH, heads=num_heads,
            dim_head=head_dim, ff_dropout=dropout,
            attn_dropout=dropout, causal=False,
        )

        # Preserve exactly the original P2 readout architecture (R256).
        with torch.random.fork_rng(devices=[]):
            self.output_projection = nn.Sequential(
                nn.Linear(self.flatten_readout_dim, performer_dim),
                nn.LayerNorm(performer_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        _original_rng_advance = nn.Linear(self.chunk_size * 2, performer_dim)
        del _original_rng_advance

    def forward(self, x):
        if x.ndim != 2 or x.size(1) != self.snp_dim:
            raise ValueError(f"Expected genotype [B,{self.snp_dim}]")
        if self.padding_layer is not None:
            x = self.padding_layer(x.unsqueeze(1)).squeeze(1)
        x = x.reshape(x.size(0), self.num_chunks, self.chunk_size)

        x = self.lgcg(x)
        x = self.pos_dropout(x + self.pos_embedding)
        x = self.performer(x)               # [B,N,C]
        x_flat = x.flatten(start_dim=1)     # [B,N*C]
        return self.output_projection(x_flat)

    def get_chunk_gates(self):
        return self.lgcg.get_chunk_gates()

    def get_chunk_weights(self):
        """Compatibility only: returns sigmoid gates, NOT softmax probabilities.

        Existing analysis scripts must NOT interpret this as normalized
        attribution across chunks; use get_chunk_gates() instead.
        """
        return self.get_chunk_gates()

class AdaptiveChunkedSNP_Performer_Decoder(nn.Module):
    def __init__(self, latent_dim, snp_dim, target_chunk_size=32, performer_dim=128, dropout=0.3):
        super().__init__()
        self.snp_dim = snp_dim
        remainder = snp_dim % target_chunk_size
        self.pad_n = target_chunk_size - remainder if remainder > 0 else 0
        self.num_chunks = (snp_dim + self.pad_n) // target_chunk_size
        self.chunk_size = target_chunk_size

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, performer_dim),
            nn.BatchNorm1d(performer_dim),
            nn.GELU()
        )
        self.to_seq = nn.Sequential(
            nn.Linear(performer_dim, self.num_chunks * self.chunk_size),
            nn.BatchNorm1d(self.num_chunks * self.chunk_size),
            nn.GELU()
        )
        num_heads = DECODER_HEADS

        head_dim = (
                self.chunk_size // num_heads
        )

        self.performer = Performer(
            dim=self.chunk_size,
            depth=DECODER_DEPTH,
            heads=num_heads,
            dim_head=head_dim,
            ff_dropout=dropout,
            attn_dropout=dropout,
            causal=False
        )
        self.final_head = nn.Linear(self.chunk_size, self.chunk_size)

    def forward(self, z):
        batch_size = z.size(0)
        h = self.latent_proj(z)
        h_seq = self.to_seq(h).view(batch_size, self.num_chunks, self.chunk_size)
        h_decoded = self.performer(h_seq)
        x_flat = self.final_head(h_decoded).view(batch_size, -1)
        return x_flat[:, :self.snp_dim]

def resolve_wheat2000_data_dir(base_dir):
    base_dir = Path(base_dir)

    candidates = [
        base_dir / "data",
        base_dir.parent / "data",
        base_dir.parent / "data" / "wheat2000",
        base_dir / "SNP_origin",
        base_dir.parent / "SNP_origin",
        Path(r"E:\VGMP\wheat2000"),
        Path(r"E:\VGMP\wheat2000\SNP_origin"),
    ]

    for data_dir in candidates:
        if (data_dir / "2000gene.pkl").exists():
            print("Resolved Wheat2000 data directory:", data_dir)
            return data_dir

    raise FileNotFoundError(
        "Could not locate Wheat2000 genotype file 2000gene.pkl."
    )

def resolve_trait_file(data_dir, trait, preferred_name):
    data_dir = Path(data_dir)
    preferred = data_dir / preferred_name

    if preferred.exists():
        return preferred

    trait_index = {
        "TKW": 1,
        "TESTW": 2,
        "LENGTH": 3,
        "WIDTH": 4,
        "HARD": 5,
        "PROT": 6,
    }[trait]

    pattern = f"2000_{trait_index}_phe_*.txt"
    matches = sorted(data_dir.glob(pattern))

    if len(matches) == 1:
        print(f"{trait}: using discovered phenotype file {matches[0].name}")
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

def load_wheat2000_genotype(genotype_file):
    obj = pd.read_pickle(genotype_file)

    if isinstance(obj, pd.DataFrame):
        X = obj.values
    elif isinstance(obj, pd.Series):
        X = obj.to_numpy()
    else:
        X = np.asarray(obj)

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D Wheat2000 genotype matrix, got {X.shape}."
        )

    sample_rows = np.arange(X.shape[0], dtype=int)

    print("=" * 90)
    print("Wheat2000 genotype")
    print("=" * 90)
    print("Samples:", X.shape[0])
    print("SNPs:", X.shape[1])
    print("True NaN:", int(np.isnan(X).sum()))
    print(
        "Genotype range:",
        float(np.nanmin(X)),
        "to",
        float(np.nanmax(X)),
    )
    print("IMPORTANT: -1 is retained as a VALID Wheat2000 genotype value.")

    return (
        np.ascontiguousarray(X, dtype=np.float32),
        sample_rows,
    )

def first_numeric_from_line(line):
    line = line.strip()
    if not line:
        return None

    for token in re.split(r"[\s,]+", line):
        if not token:
            continue
        try:
            return float(token)
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
            value = first_numeric_from_line(line)
            if value is not None:
                values.append(value)

    if not values:
        raise ValueError(
            f"No numeric phenotype values found in {phenotype_file}."
        )

    y = np.asarray(values, dtype=np.float32).reshape(-1, 1)

    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"Phenotype contains NaN/Inf: {phenotype_file}"
        )

    return y

class GenoPerfVAE_LGCG_Model(nn.Module):
    """The complete B5 model: LGCG + Performer + Shortcut + VAE + Decoder."""

    def __init__(self, snp_dim, dropout=MODEL_DROPOUT, init_seed=SEED):
        super().__init__()
        self.snp_dim = int(snp_dim)
        self.latent_dim = TOTAL_LATENT_DIM

        # Keep initialization reproducible and source-compatible.
        set_seed_fixed(init_seed)

        # 1) LGCG + Performer genomic encoder: [B, D] -> [B, 256].
        self.snp_performer = LGCGChunkedSNP_Performer(
            self.snp_dim,
            target_chunk_size=CHUNK_SIZE,
            performer_dim=PERFORMER_READOUT_DIM,
            dropout=dropout,
            lgcg_hidden_dim=LGCG_HIDDEN_DIM,
            lgcg_alpha=LGCG_ALPHA,
            lgcg_kernel_size=LGCG_KERNEL_SIZE,
        )

        # 2) Dense Linear Shortcut: D -> 2048 -> 512 -> 64.
        self.linear_shortcut = nn.Sequential(
            nn.Linear(self.snp_dim, SHORTCUT_HIDDEN_DIM_1),
            nn.BatchNorm1d(SHORTCUT_HIDDEN_DIM_1),
            nn.GELU(),
            nn.Linear(SHORTCUT_HIDDEN_DIM_1, SHORTCUT_HIDDEN_DIM_2),
            nn.BatchNorm1d(SHORTCUT_HIDDEN_DIM_2),
            nn.GELU(),
            nn.Dropout(SHORTCUT_DROPOUT),
            nn.Linear(SHORTCUT_HIDDEN_DIM_2, SHORTCUT_OUTPUT_DIM),
        )

        # 3) Branch-specific Gaussian posterior heads:
        #    Performer: 256 -> 32; Shortcut: 64 -> 16.
        #    Keep the posterior heads RNG-local to reproduce the original
        #    predictor/decoder initialization sequence.
        with torch.random.fork_rng(devices=[]):
            self.fc_mu_performer = nn.Linear(
                PERFORMER_READOUT_DIM, PERFORMER_LATENT_DIM
            )
            self.fc_logvar_performer = nn.Linear(
                PERFORMER_READOUT_DIM, PERFORMER_LATENT_DIM
            )
            self.fc_mu_linear = nn.Linear(
                SHORTCUT_OUTPUT_DIM, LINEAR_LATENT_DIM
            )
            self.fc_logvar_linear = nn.Linear(
                SHORTCUT_OUTPUT_DIM, LINEAR_LATENT_DIM
            )

        # 4) Predictor: 48 -> 64 -> BatchNorm -> GELU -> 1.
        #    No predictor dropout.
        self.predictor = nn.Sequential(
            nn.Linear(TOTAL_LATENT_DIM, PREDICTOR_HIDDEN_DIM),
            nn.BatchNorm1d(PREDICTOR_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(PREDICTOR_HIDDEN_DIM, 1),
        )

        # 5) SNP reconstruction Decoder: 48-D -> original SNP dimension.
        self.decoder = AdaptiveChunkedSNP_Performer_Decoder(
            latent_dim=TOTAL_LATENT_DIM,
            snp_dim=self.snp_dim,
            target_chunk_size=CHUNK_SIZE,
            performer_dim=DECODER_PERFORMER_DIM,
            dropout=dropout,
        )

    def forward(self, x, compute_reconstruction=True):
        h_performer = self.snp_performer(x)
        h_linear = self.linear_shortcut(x)

        mu = torch.cat(
            (
                self.fc_mu_performer(h_performer),
                self.fc_mu_linear(h_linear),
            ),
            dim=1,
        )
        logvar = torch.cat(
            (
                self.fc_logvar_performer(h_performer),
                self.fc_logvar_linear(h_linear),
            ),
            dim=1,
        )

        # One concatenated 48-D reparameterization.
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            # Deterministic inference uses the posterior mean.
            z = mu

        y_pred = self.predictor(z)
        x_recon = self.decoder(z) if compute_reconstruction else None

        return {
            "y_pred": y_pred,
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }


class GenoPerfVAE_LGCG_Loss:
    """Prediction MSE + SNP reconstruction MSE + VAE KL divergence."""

    def __call__(self, outputs, x, y):
        if outputs["x_recon"] is None:
            raise RuntimeError(
                "SNP reconstruction is required during training; "
                "call the model with compute_reconstruction=True."
            )
        if outputs["mu"] is None or outputs["logvar"] is None:
            raise RuntimeError("VAE posterior parameters are missing.")

        pred = F.mse_loss(outputs["y_pred"], y)
        recon = F.mse_loss(outputs["x_recon"], x)
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

        total = (
            PRED_WEIGHT * pred
            + RECON_WEIGHT * recon
            + KL_WEIGHT * kl
        )
        return {
            "total": total,
            "pred": pred,
            "recon": recon,
            "kl": kl,
        }




def set_seed_fixed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cuda_sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

def safe_batch_size(n_samples, preferred=BATCH_SIZE):
    batch = min(int(preferred), int(n_samples))
    while batch > 2 and n_samples % batch == 1:
        batch -= 1
    return max(batch, 2)

def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)

def safe_corr(func, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    try:
        return float(func(y_true, y_pred)[0])
    except Exception:
        return np.nan

def topk_metrics(y_true, y_pred, frac):
    """Higher-is-better ranking convention; raw predictions are also saved for re-analysis."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    pred_top = np.argsort(y_pred)[-k:]
    true_top = np.argsort(y_true)[-k:]
    overlap = int(len(set(pred_top.tolist()).intersection(true_top.tolist())))
    sd = float(np.std(y_true, ddof=1)) if n > 1 else np.nan
    selected_mean = float(np.mean(y_true[pred_top]))
    rs = (selected_mean - float(np.mean(y_true))) / sd if np.isfinite(sd) and sd > 1e-12 else np.nan
    return {
        f"SelAcc_Top{int(frac*100)}": overlap / k,
        f"Top{int(frac*100)}_Overlap_Count": overlap,
        f"RS_Top{int(frac*100)}_SD": rs,
        f"SelectedMean_Top{int(frac*100)}": selected_mean,
    }

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan
    result = {
        "Pearson": safe_corr(stats.pearsonr, y_true, y_pred),
        "Spearman": safe_corr(stats.spearmanr, y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": mse,
        "RMSE": rmse,
        "nRMSE": float(rmse / sd) if np.isfinite(sd) and sd > 1e-12 else np.nan,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias_PredMinusObs": float(np.mean(y_pred - y_true)),
        "Observed_Mean": float(np.mean(y_true)),
        "Predicted_Mean": float(np.mean(y_pred)),
        "Observed_SD": sd,
        "Predicted_SD": float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else np.nan,
    }
    result.update(topk_metrics(y_true, y_pred, 0.10))
    result.update(topk_metrics(y_true, y_pred, 0.20))
    return result

def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def upsert_csv(path, new_df, key_cols):
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df.copy()
    if not merged.empty:
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
    atomic_csv(merged, path)
    return merged

class DeviceResidentBatcher:
    """Move one fold's X/y and shuffled epoch orders to the device once."""
    def __init__(self, X, y, batch_size, shuffle, seed, device, num_epochs=1):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.n_samples = int(len(X))
        self._epoch = 0
        self.X = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=self.device)
        self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32), device=self.device)
        self.epoch_orders = (
            precompute_epoch_orders(self.n_samples, self.batch_size, seed, num_epochs).to(self.device)
            if self.shuffle else None
        )

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            if self._epoch >= self.epoch_orders.shape[0]:
                raise RuntimeError("Training batcher exhausted its precomputed epoch orders.")
            order = self.epoch_orders[self._epoch]
            self._epoch += 1
        else:
            order = None
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            if order is None:
                yield self.X[start:end], self.y[start:end]
            else:
                idx = order[start:end]
                yield self.X.index_select(0, idx), self.y.index_select(0, idx)

def precompute_epoch_orders(n_samples, batch_size, seed, num_epochs):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    orders = [torch.randperm(n_samples, generator=generator) for _ in range(num_epochs)]
    return torch.stack(orders, dim=0) if orders else torch.empty((0, n_samples), dtype=torch.long)

def train_one_epoch(model, batcher, optimizer, loss_fn, clip_parameters):
    model.train()
    sums = torch.zeros(4, dtype=torch.float32, device=batcher.device)
    n_batches = 0
    for bx, by in batcher:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(bx)
        losses = loss_fn(outputs, bx, by)
        losses["total"].backward()
        try:
            torch.nn.utils.clip_grad_norm_(clip_parameters, GRAD_CLIP, foreach=True)
        except (TypeError, RuntimeError):
            torch.nn.utils.clip_grad_norm_(clip_parameters, GRAD_CLIP)
        optimizer.step()
        sums += torch.stack((losses["total"].detach(), losses["pred"].detach(),
                             losses["recon"].detach(), losses["kl"].detach())).float()
        n_batches += 1
    vals = (sums / max(n_batches, 1)).detach().cpu().tolist()
    return {"total": vals[0], "pred": vals[1], "recon": vals[2], "kl": vals[3]}

def evaluate_predictions(model, batcher):
    model.eval()
    preds, trues = [], []
    for bx, by in batcher:
        outputs = model(bx, compute_reconstruction=False)
        preds.append(outputs["y_pred"])
        trues.append(by)
    return torch.cat(preds, dim=0).detach().cpu().numpy(), torch.cat(trues, dim=0).detach().cpu().numpy()

def benchmark_inference(model, batcher, device):
    model.eval()
    for bx, _ in batcher:
        _ = model(bx, compute_reconstruction=False)
        break
    cuda_sync(device)
    start = time.perf_counter()
    n = 0
    for bx, _ in batcher:
        _ = model(bx, compute_reconstruction=False)
        n += int(bx.size(0))
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    return elapsed, elapsed / max(n, 1) * 1000.0

def build_summary(fold_df):
    if fold_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "Pearson", "Spearman", "R2", "MSE", "RMSE", "nRMSE", "MAE",
        "Bias_PredMinusObs", "SelAcc_Top10", "SelAcc_Top20",
        "RS_Top10_SD", "RS_Top20_SD", "Top10_Overlap_Count",
        "Top20_Overlap_Count", "Train_Time_s", "Epoch_Time_Mean_s",
        "Peak_GPU_Memory_MB", "Inference_ms_per_sample",
        "Total_Params_M", "Trainable_Params_M", "TrainLoss_Final",
        "PredLoss_Final", "ReconLoss_Final", "KLLoss_Final",
        "WeightedPredLoss_Final", "WeightedReconLoss_Final",
        "WeightedKLLoss_Final",
    ]
    rows = []
    for trait, group in fold_df.groupby("Trait", dropna=False):
        row = {
            "Dataset": "Wheat2000",
            "Trait": trait,
            "Model": "GenoPerfVAE_LGCG",
            "Seed": int(group["Seed"].iloc[0]),
            "Completed_Folds": int(group["Fold"].nunique()),
        }
        for col in metric_cols:
            if col in group:
                row[f"{col}_Mean"] = float(group[col].mean())
                row[f"{col}_SD"] = (
                    float(group[col].std(ddof=1)) if len(group) > 1 else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone full GenoPerfVAE-LGCG model for Wheat2000."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed; default 42. Seed is included in output filenames."
    )
    parser.add_argument(
        "--traits", nargs="+", type=str.upper,
        default=list(TRAIT_FILES.keys()), choices=list(TRAIT_FILES.keys()),
        help="Traits to run; default: all six Wheat2000 traits."
    )
    parser.add_argument(
        "--max-folds", type=int, default=N_SPLITS,
        help="Use 10 for full CV; smaller values are for smoke tests."
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS,
        help="Training epochs per fold; default 200."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Optional directory containing 2000gene.pkl and phenotype files."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Optional output directory. Seed is still included in output filenames."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = int(args.seed)
    epochs = int(args.epochs)
    n_folds = int(args.max_folds)

    if not (1 <= n_folds <= N_SPLITS):
        raise ValueError(f"--max-folds must be between 1 and {N_SPLITS}.")
    if epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    set_seed_fixed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    base_dir = Path(__file__).resolve().parent
    data_dir = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else resolve_wheat2000_data_dir(base_dir)
    )
    if not (data_dir / "2000gene.pkl").exists():
        raise FileNotFoundError(
            f"Could not find genotype file: {data_dir / '2000gene.pkl'}"
        )

    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    else:
        run_suffix = (
            ""
            if (n_folds == N_SPLITS and epochs == MAX_EPOCHS)
            else f"_Pilot_F{n_folds}_E{epochs}"
        )
        output_root = (
            base_dir
            / f"results_GenoPerfVAE_LGCG_Wheat2000_FullModel{run_suffix}_Seed{seed}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    tag = f"Seed{seed}"
    fold_path = output_root / f"GenoPerfVAE_LGCG_FoldMetrics_{tag}_CHECKPOINT.csv"
    pred_path = output_root / f"GenoPerfVAE_LGCG_Predictions_{tag}_CHECKPOINT.csv"
    manifest_path = output_root / f"GenoPerfVAE_LGCG_SplitManifest_{tag}.csv"
    summary_path = output_root / f"GenoPerfVAE_LGCG_Summary_{tag}_CHECKPOINT.csv"
    config_path = output_root / f"GenoPerfVAE_LGCG_RunConfig_{tag}.json"
    snp_order_path = output_root / "GenoPerfVAE_LGCG_SNP_Order.csv"

    config = {
        "Code_Version": "GenoPerfVAE_LGCG_Standalone_FullModel_v1",
        "Dataset": "Wheat2000",
        "Model": "GenoPerfVAE_LGCG",
        "Seed": seed,
        "Traits": args.traits,
        "N_Splits": N_SPLITS,
        "Max_Folds_This_Run": n_folds,
        "Epochs": epochs,
        "Chunk_Size": CHUNK_SIZE,
        "LGCG_Hidden_Dim": LGCG_HIDDEN_DIM,
        "LGCG_Alpha": LGCG_ALPHA,
        "LGCG_Kernel_Size": LGCG_KERNEL_SIZE,
        "Performer_Readout_Dim": PERFORMER_READOUT_DIM,
        "Encoder_Depth": ENCODER_DEPTH,
        "Encoder_Heads": ENCODER_HEADS,
        "Encoder_Head_Dim": CHUNK_SIZE // ENCODER_HEADS,
        "Shortcut_Hidden_Dimensions": [
            SHORTCUT_HIDDEN_DIM_1, SHORTCUT_HIDDEN_DIM_2, SHORTCUT_OUTPUT_DIM
        ],
        "Performer_Latent_Dim": PERFORMER_LATENT_DIM,
        "Linear_Latent_Dim": LINEAR_LATENT_DIM,
        "Total_Latent_Dim": TOTAL_LATENT_DIM,
        "Decoder_Performer_Dim": DECODER_PERFORMER_DIM,
        "Decoder_Depth": DECODER_DEPTH,
        "Decoder_Heads": DECODER_HEADS,
        "Decoder_Head_Dim": CHUNK_SIZE // DECODER_HEADS,
        "Predictor": "48 -> 64 -> BatchNorm -> GELU -> 1; no predictor dropout",
        "Model_Dropout": MODEL_DROPOUT,
        "Shortcut_Dropout": SHORTCUT_DROPOUT,
        "Predictor_Dropout": PREDICTOR_DROPOUT,
        "Batch_Size": BATCH_SIZE,
        "Optimizer": "Adam",
        "Learning_Rate": LEARNING_RATE,
        "Weight_Decay": WEIGHT_DECAY,
        "Gradient_Clip": GRAD_CLIP,
        "Prediction_Weight": PRED_WEIGHT,
        "Reconstruction_Weight": RECON_WEIGHT,
        "KL_Weight": KL_WEIGHT,
        "VAE": "branch-specific posterior: Performer 32-D + Shortcut 16-D; one 48-D reparameterization during training; posterior mean during evaluation",
        "Decoder": "SNP reconstruction during training; disabled during test prediction and inference timing",
        "CV": "10-fold KFold(shuffle=True, random_state=seed)",
        "Preprocessing": (
            "Most-frequent genotype imputation fitted on outer training fold only; "
            "phenotype StandardScaler fitted on outer training fold only; -1 retained "
            "as a valid genotype value; no SNP scaling; genotype/phenotype row counts "
            "must match; no automatic truncation."
        ),
        "Selection_Metrics": (
            "Top 10%/20% assume larger phenotype values are preferred; raw predictions "
            "are saved for re-analysis."
        ),
        "Data_Dir": str(data_dir),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    X_genotype, sample_rows = load_wheat2000_genotype(data_dir / "2000gene.pkl")
    pd.DataFrame({
        "SNP_Index_0Based": np.arange(X_genotype.shape[1], dtype=int)
    }).to_csv(snp_order_path, index=False)

    print(
        f"Traits={args.traits}; seed={seed}; folds={n_folds}; epochs={epochs}"
    )
    print("Data directory:", data_dir)
    print("Output directory:", output_root)
    print(
        f"Full model: LGCG(C={CHUNK_SIZE}, H={LGCG_HIDDEN_DIM}, "
        f"alpha={LGCG_ALPHA}) + Performer(256) + Shortcut(64) + "
        f"VAE(32+16=48) + Decoder(256)."
    )
    print(
        f"Loss weights: prediction={PRED_WEIGHT}, reconstruction={RECON_WEIGHT}, "
        f"KL={KL_WEIGHT}; Adam lr={LEARNING_RATE}; batch={BATCH_SIZE}; "
        f"epochs={epochs}; grad_clip={GRAD_CLIP}."
    )

    fold_df = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
    pred_df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    manifest_df = (
        pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    )
    completed = set()
    if not fold_df.empty:
        completed = {
            (str(r.Trait), int(r.Fold), int(r.Seed))
            for r in fold_df.itertuples()
        }

    for trait in args.traits:
        phenotype_file = resolve_trait_file(
            data_dir, trait, TRAIT_FILES[trait]
        )
        y_loaded = load_wheat2000_phenotype(phenotype_file)

        if len(y_loaded) != len(X_genotype):
            raise ValueError(
                f"{trait}: genotype N={len(X_genotype)} but phenotype "
                f"N={len(y_loaded)}. Automatic truncation is disabled; "
                "verify sample row alignment."
            )

        X_all = np.ascontiguousarray(X_genotype, dtype=np.float32)
        y_all = np.ascontiguousarray(y_loaded, dtype=np.float32)
        rows = sample_rows

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_all), start=1):
            if fold > n_folds:
                break
            if (trait, fold, seed) in completed:
                print(
                    f"[SKIP] {trait} fold {fold}: checkpoint already exists "
                    f"for {tag}."
                )
                continue

            print("\\n" + "=" * 96)
            print(
                f"Trait={trait} | Fold={fold}/{n_folds} | Seed={seed} | "
                "Model=GenoPerfVAE_LGCG"
            )
            print("=" * 96)

            # Leakage-safe preprocessing: fit on outer training fold only.
            imputer = SimpleImputer(
                strategy="most_frequent", keep_empty_features=True
            )
            X_train = imputer.fit_transform(X_all[train_idx]).astype(np.float32)
            X_test = imputer.transform(X_all[test_idx]).astype(np.float32)

            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(
                y_all[train_idx]
            ).astype(np.float32)
            y_test_scaled = scaler_y.transform(
                y_all[test_idx]
            ).astype(np.float32)
            y_test_raw = y_all[test_idx].reshape(-1).astype(np.float64)

            train_bs = safe_batch_size(len(train_idx), BATCH_SIZE)
            train_batcher = DeviceResidentBatcher(
                X_train, y_train_scaled, train_bs, True, seed, device,
                num_epochs=epochs
            )
            test_batcher = DeviceResidentBatcher(
                X_test, y_test_scaled, BATCH_SIZE, False, seed, device,
                num_epochs=1
            )

            # Save the exact split manifest for later reproducibility.
            new_manifest = []
            for idx in train_idx:
                new_manifest.append({
                    "Dataset": "Wheat2000", "Trait": trait, "Fold": fold,
                    "Sample_Row_0Based": int(rows[idx]),
                    "Aligned_Row_0Based": int(idx), "Set": "Train", "Seed": seed,
                })
            for idx in test_idx:
                new_manifest.append({
                    "Dataset": "Wheat2000", "Trait": trait, "Fold": fold,
                    "Sample_Row_0Based": int(rows[idx]),
                    "Aligned_Row_0Based": int(idx), "Set": "Test", "Seed": seed,
                })
            manifest_df = pd.concat(
                [manifest_df, pd.DataFrame(new_manifest)], ignore_index=True
            )
            manifest_df = manifest_df.drop_duplicates(
                ["Trait", "Fold", "Sample_Row_0Based", "Set", "Seed"],
                keep="last"
            )
            atomic_csv(manifest_df, manifest_path)

            # Build the complete full-model branch. Reset the training RNG
            # after construction to make each fold reproducible.
            model = GenoPerfVAE_LGCG_Model(
                snp_dim=X_all.shape[1],
                dropout=MODEL_DROPOUT,
                init_seed=seed,
            ).to(device)
            set_seed_fixed(seed)

            loss_fn = GenoPerfVAE_LGCG_Loss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
            clip_parameters = [
                p for p in model.parameters() if p.requires_grad
            ]
            total_params, trainable_params = count_model_parameters(model)

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            cuda_sync(device)
            train_start = time.perf_counter()
            epoch_times = []
            final_train_stats = {
                "total": np.nan, "pred": np.nan, "recon": np.nan, "kl": np.nan
            }

            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()
                final_train_stats = train_one_epoch(
                    model, train_batcher, optimizer, loss_fn, clip_parameters
                )
                cuda_sync(device)
                epoch_times.append(time.perf_counter() - t0)

                if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                    print(
                        f"Epoch {epoch:03d}/{epochs} | "
                        f"total={final_train_stats['total']:.5f} | "
                        f"pred={final_train_stats['pred']:.5f} | "
                        f"recon={final_train_stats['recon']:.5f} | "
                        f"kl={final_train_stats['kl']:.5f}",
                        flush=True,
                    )

            cuda_sync(device)
            train_time = time.perf_counter() - train_start

            # Test prediction uses posterior mean and does not run Decoder.
            pred_scaled, true_scaled_check = evaluate_predictions(
                model, test_batcher
            )
            pred_raw = scaler_y.inverse_transform(
                pred_scaled
            ).reshape(-1).astype(np.float64)
            true_from_scaled = scaler_y.inverse_transform(
                true_scaled_check
            ).reshape(-1)

            if not np.allclose(
                true_from_scaled, y_test_raw, atol=1e-4, rtol=1e-5
            ):
                raise RuntimeError(
                    "Test phenotype inverse-transform mismatch; "
                    "check row alignment."
                )

            metrics = calc_metrics(y_test_raw, pred_raw)
            infer_total, infer_ms = benchmark_inference(
                model, test_batcher, device
            )
            peak_gpu_mb = (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                if device.type == "cuda" else np.nan
            )

            weighted_pred = PRED_WEIGHT * final_train_stats["pred"]
            weighted_recon = RECON_WEIGHT * final_train_stats["recon"]
            weighted_kl = KL_WEIGHT * final_train_stats["kl"]
            weighted_total = (
                weighted_pred + weighted_recon + weighted_kl
            )

            fold_row = {
                "Dataset": "Wheat2000",
                "Trait": trait,
                "Model": "GenoPerfVAE_LGCG",
                "Fold": fold,
                "Seed": seed,
                "Use_LGCG": True,
                "Use_Linear": True,
                "Use_VAE": True,
                "Use_Decoder": True,
                "SNP_Dimension": int(X_all.shape[1]),
                "Chunk_Size_C": CHUNK_SIZE,
                "LGCG_Hidden_Dim": LGCG_HIDDEN_DIM,
                "LGCG_Alpha": LGCG_ALPHA,
                "LGCG_Kernel_Size": LGCG_KERNEL_SIZE,
                "Encoder_Depth": ENCODER_DEPTH,
                "Encoder_Heads": ENCODER_HEADS,
                "Encoder_Head_Dim": CHUNK_SIZE // ENCODER_HEADS,
                "Performer_Readout_Dim": PERFORMER_READOUT_DIM,
                "Performer_Latent_Dim": PERFORMER_LATENT_DIM,
                "Linear_Shortcut_Output_Dim": SHORTCUT_OUTPUT_DIM,
                "Linear_Latent_Dim": LINEAR_LATENT_DIM,
                "Total_Latent_Dim": TOTAL_LATENT_DIM,
                "Decoder_Performer_Dim": DECODER_PERFORMER_DIM,
                "Decoder_Depth": DECODER_DEPTH,
                "Decoder_Heads": DECODER_HEADS,
                "Decoder_Head_Dim": CHUNK_SIZE // DECODER_HEADS,
                "Decoder_Input_Dim": TOTAL_LATENT_DIM,
                "Predictor_Hidden_Dim": PREDICTOR_HIDDEN_DIM,
                "Model_Dropout": MODEL_DROPOUT,
                "Shortcut_Dropout": SHORTCUT_DROPOUT,
                "Predictor_Dropout": PREDICTOR_DROPOUT,
                "Train_N": int(len(train_idx)),
                "Test_N": int(len(test_idx)),
                "Epochs": epochs,
                "Batch_Size": train_bs,
                "Optimizer": "Adam",
                "Learning_Rate": LEARNING_RATE,
                "Weight_Decay": WEIGHT_DECAY,
                "Grad_Clip": GRAD_CLIP,
                "Pred_Weight": PRED_WEIGHT,
                "Pearson_Weight": 0.0,
                "Recon_Weight": RECON_WEIGHT,
                "KL_Weight": KL_WEIGHT,
                "TrainLoss_Final": final_train_stats["total"],
                "PredLoss_Final": final_train_stats["pred"],
                "ReconLoss_Final": final_train_stats["recon"],
                "KLLoss_Final": final_train_stats["kl"],
                "WeightedPredLoss_Final": weighted_pred,
                "WeightedReconLoss_Final": weighted_recon,
                "WeightedKLLoss_Final": weighted_kl,
                "WeightedLossSum_Final": weighted_total,
                "Train_Time_s": float(train_time),
                "Epoch_Time_Mean_s": float(np.mean(epoch_times)),
                "Epoch_Time_SD_s": (
                    float(np.std(epoch_times, ddof=1))
                    if len(epoch_times) > 1 else 0.0
                ),
                "Peak_GPU_Memory_MB": peak_gpu_mb,
                "Inference_Total_s": float(infer_total),
                "Inference_ms_per_sample": float(infer_ms),
                "Total_Params": total_params,
                "Trainable_Params": trainable_params,
                "Total_Params_M": total_params / 1e6,
                "Trainable_Params_M": trainable_params / 1e6,
                "Train_Y_Mean": float(scaler_y.mean_[0]),
                "Train_Y_Scale": float(scaler_y.scale_[0]),
                "Imputer_Fit": "OuterTrainOnly_MostFrequent",
                "GenotypeMinus1": "RetainedAsValidValue",
                "Y_Scaling": "OuterTrainOnlyStandardScaler_MetricsOnRawScale",
                **metrics,
            }

            # Save sample-level predictions first; fold metrics marks completion.
            prediction_rows = []
            for local_i, idx in enumerate(test_idx):
                pred_std = float(pred_scaled.reshape(-1)[local_i])
                obs_std = float(y_test_scaled.reshape(-1)[local_i])
                prediction_rows.append({
                    "Dataset": "Wheat2000",
                    "Trait": trait,
                    "Model": "GenoPerfVAE_LGCG",
                    "Fold": fold,
                    "Seed": seed,
                    "Aligned_Row_0Based": int(idx),
                    "Sample_Row_0Based": int(rows[idx]),
                    "Observed_Raw": float(y_test_raw[local_i]),
                    "Predicted_Raw": float(pred_raw[local_i]),
                    "Residual_Raw_PredMinusObs": float(
                        pred_raw[local_i] - y_test_raw[local_i]
                    ),
                    "Observed_Standardized_TrainScaler": obs_std,
                    "Predicted_Standardized_TrainScaler": pred_std,
                    "Residual_Standardized_TrainScaler": float(
                        pred_std - obs_std
                    ),
                    "Train_Y_Mean": float(scaler_y.mean_[0]),
                    "Train_Y_Scale": float(scaler_y.scale_[0]),
                })

            pred_df = upsert_csv(
                pred_path,
                pd.DataFrame(prediction_rows),
                ["Trait", "Model", "Fold", "Sample_Row_0Based", "Seed"],
            )
            fold_df = upsert_csv(
                fold_path,
                pd.DataFrame([fold_row]),
                ["Trait", "Model", "Fold", "Seed"],
            )
            completed.add((trait, fold, seed))

            summary_df = build_summary(fold_df)
            atomic_csv(summary_df, summary_path)

            print(
                f"Result: Pearson={metrics['Pearson']:.5f} | "
                f"R2={metrics['R2']:.5f} | RMSE={metrics['RMSE']:.5f} | "
                f"MAE={metrics['MAE']:.5f} | Params={total_params:,} | "
                f"PeakGPU={peak_gpu_mb:.1f} MB"
            )

            del model, optimizer, train_batcher, test_batcher
            del X_train, X_test, y_train_scaled, y_test_scaled
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # Rebuild the final summary from the checkpoint file.
    fold_df = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
    summary_df = build_summary(fold_df)
    atomic_csv(summary_df, summary_path)

    print("\\n" + "=" * 96)
    print("GenoPerfVAE_LGCG full-model experiment completed.")
    print("Output directory:", output_root)
    print("Files:")
    for path in [
        fold_path, pred_path, manifest_path, summary_path,
        config_path, snp_order_path
    ]:
        print(" -", path.name)
    if not summary_df.empty:
        show_cols = [
            c for c in [
                "Trait", "Completed_Folds", "Pearson_Mean", "Pearson_SD",
                "R2_Mean", "R2_SD", "RMSE_Mean", "RMSE_SD",
                "MAE_Mean", "MAE_SD", "SelAcc_Top20_Mean", "RS_Top20_SD_Mean"
            ] if c in summary_df.columns
        ]
        print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
