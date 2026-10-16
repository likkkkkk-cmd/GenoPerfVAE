# -*- coding: utf-8 -*-
"""
GenoPerfVAE Wheat2000 P2+M8B-Latent48 (32+16) R256 + Predictor64 GELU/NoDropout
================================================================================

Controlled 10-fold predictive benchmark for Wheat2000.

Model / training fixed to the current GELU+NoDropout configuration:
- P2 full-token Performer readout: [B,N,32] -> flatten -> Linear(N*32,256)
- Dense shortcut: D -> 2048 -> 512 -> 64
- Branch-specific posterior: Performer 32 + Linear 16 = latent 48
- Predictor: 48 -> 64 -> 1, BatchNorm + GELU, NO predictor dropout
- Decoder Performer width = 256
- chunk_size = 32 only
- Loss = 1.5*prediction + 0*Pearson + 0.5*reconstruction + 0.8*KL
- Adam(lr=1e-4), weight_decay=0, grad clip=1.0
- fixed 200 epochs, no validation, no early stopping, no scheduler
- 10-fold KFold(shuffle=True, random_state=42)

Wheat2000 data protocol:
- genotype: 2000gene.pkl
- phenotype files:
    2000_1_phe_tkw.txt
    2000_2_phe_testw.txt
    2000_3_phe_length.txt
    2000_4_phe_width.txt
    2000_5_phe_hardness.txt (fallback: unique 2000_5_phe_*.txt)
    2000_6_phe_Prot.txt
- phenotype parser: first numeric value from each non-empty line
- row-order alignment between genotype and phenotype
- true NaN only -> most-frequent imputation fitted on OUTER TRAIN only
- -1 is retained as a valid Wheat2000 genotype value
- no additional SNP scaling
- phenotype StandardScaler fitted on OUTER TRAIN only; metrics are reported
  after inverse transformation to the original phenotype scale

Dependencies:
    pip install performer-pytorch
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy import stats
import os
import re
import warnings
from tqdm import tqdm
import copy
from sklearn.model_selection import KFold, train_test_split
import random
warnings.filterwarnings('ignore')
import time
import gc

# 确保安装 Performer
try:
    from performer_pytorch import Performer
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "performer-pytorch"])
    from performer_pytorch import Performer


# ==========================================
# 1. 核心组件：SwiGLU 与 Pearson 损失
# ==========================================
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w12 = nn.Linear(in_features, hidden_features * 2)

    def forward(self, x):
        x, gate = self.w12(x).chunk(2, dim=-1)
        return x * F.silu(gate)



class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        mu_p, mu_t = torch.mean(y_pred), torch.mean(y_true)
        diff_p, diff_t = y_pred - mu_p, y_true - mu_t
        num = torch.sum(diff_p * diff_t)
        den = torch.sqrt(torch.sum(diff_p ** 2) * torch.sum(diff_t ** 2) + 1e-8)
        return 1 - (num / den)


class GeneticAwareLoss(nn.Module):
    def __init__(
        self,
        pred_weight=1.5,
        pearson_weight=1.0,
        recon_weight=1.0,
        kl_weight=0.4
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pearson = PearsonLoss()
        self.p_weight = pearson_weight
        self.m_weight = pred_weight
        self.r_weight = recon_weight
        self.k_weight = kl_weight

    def forward(self, x_recon, x, y_pred, y, mu, logvar):
        mse_l = self.mse(y_pred, y)
        pea_l = self.pearson(y_pred, y)
        rec_l = self.mse(x_recon, x)

        # KL FIX:
        # Average over both batch and latent dimensions.
        kl_l = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        total = (
            self.m_weight * mse_l
            + self.p_weight * pea_l
            + self.r_weight * rec_l
            + self.k_weight * kl_l
        )

        return total, rec_l, pea_l, mse_l, kl_l


# ==========================================
# 2. 模型架构
# ==========================================
class AdaptiveChunkedSNP_Performer(nn.Module):
    """
    ASCM + Performer encoder with P2 no-pooling full-token flatten readout.

    Workflow:
        SNP input
        -> dynamic padding
        -> contiguous SNP chunking
        -> chunk normalization
        -> sample-specific adaptive chunk weighting
        -> learnable positional encoding
        -> Performer over genomic chunk tokens
        -> flatten ALL token outputs [B,N*C]
        -> Linear(N*C, performer_dim)

    No AvgPool, MaxPool, regional pooling, or summary tokens are used.
    """

    def __init__(
        self,
        snp_dim,
        target_chunk_size=32,
        performer_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.snp_dim = int(snp_dim)
        self.chunk_size = int(target_chunk_size)

        remainder = self.snp_dim % self.chunk_size
        self.pad_n = (
            self.chunk_size - remainder
            if remainder > 0
            else 0
        )
        self.num_chunks = (
            self.snp_dim + self.pad_n
        ) // self.chunk_size
        self.sequence_length = self.num_chunks
        self.flatten_readout_dim = self.num_chunks * self.chunk_size

        if self.pad_n > 0:
            self.padding_layer = nn.ZeroPad1d((0, self.pad_n))
        else:
            self.padding_layer = None

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

        # P2 full-token flatten projection. Keep this new parameter block RNG-local
        # so every module created afterwards receives exactly the Original RNG state.
        with torch.random.fork_rng(devices=[]):
            self.output_projection = nn.Sequential(
                nn.Linear(self.flatten_readout_dim, performer_dim),
                nn.LayerNorm(performer_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Advance global RNG exactly as Original Avg+Max readout's Linear(2*C,256).
        # The dummy layer is local and is not registered in this model.
        _original_rng_advance = nn.Linear(
            self.chunk_size * 2,
            performer_dim,
        )
        del _original_rng_advance

        self.last_chunk_weights = None

    def forward(self, x):
        if self.padding_layer is not None:
            x = self.padding_layer(x.unsqueeze(1)).squeeze(1)

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

        # Same Performer sequence as Original.
        x = self.performer(x)                         # [B,N,C]

        # P2: NO pooling. Preserve every token and every token channel.
        x_flat = x.flatten(start_dim=1)              # [B,N*C]
        output = self.output_projection(x_flat)      # [B,performer_dim]
        return output

    def get_chunk_weights(self):
        if self.last_chunk_weights is None:
            return None
        return self.last_chunk_weights.squeeze(-1)


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
        num_heads = 4

        head_dim = (
                self.chunk_size // num_heads
        )

        self.performer = Performer(
            dim=self.chunk_size,
            depth=2,
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


class EnhancedGeneticModel(nn.Module):
    """
    GenoPerfVAE P2+M8B-Latent48 with fixed 32+16 branch-specific latent allocation
    and a controlled Predictor64 GELU/no-dropout phenotype head.

    Fixed representation paths:
        Performer tokens [B,N,32]
        -> flatten ALL tokens [B,N*32]
        -> Linear(N*32, 256)

        Linear shortcut:
        D -> 2048 -> 512 -> 64

    Configurable posterior allocation:
        Performer256 -> mu/logvar performer_latent_dim
        Shortcut64   -> mu/logvar linear_latent_dim
        performer_latent_dim + linear_latent_dim = 48
        -> concat posterior parameters -> one reparameterization -> z48

    Predictor keeps hidden width 64, BN, and GELU, but removes predictor dropout.
    Decoder and all upstream representation paths are unchanged.
    """

    def __init__(
        self,
        snp_dim,
        latent_dim=48,
        performer_readout_dim=256,
        decoder_performer_dim=256,
        performer_latent_dim=32,
        linear_latent_dim=16,
        dropout=0.5,
        chunk_size=256,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.performer_readout_dim = int(performer_readout_dim)
        self.decoder_performer_dim = int(decoder_performer_dim)
        self.performer_latent_dim = int(performer_latent_dim)
        self.linear_latent_dim = int(linear_latent_dim)

        if self.performer_readout_dim != 256:
            raise ValueError(
                "This controlled experiment fixes the P2 Performer readout at 256. "
                f"Got {self.performer_readout_dim}."
            )

        if self.decoder_performer_dim != 256:
            raise ValueError(
                "This controlled experiment fixes the decoder Performer width at 256. "
                f"Got {self.decoder_performer_dim}."
            )

        if self.performer_latent_dim <= 0 or self.linear_latent_dim <= 0:
            raise ValueError(
                "Both branch latent dimensions must be positive; got "
                f"Performer={self.performer_latent_dim}, Linear={self.linear_latent_dim}."
            )

        if self.performer_latent_dim + self.linear_latent_dim != self.latent_dim:
            raise ValueError(
                "Branch latent dimensions must sum to total latent_dim. "
                f"Got {self.performer_latent_dim}+{self.linear_latent_dim} "
                f"!= {self.latent_dim}."
            )

        # P2-R256 nonlinear branch: preserve ALL post-Performer token information.
        self.snp_performer = AdaptiveChunkedSNP_Performer(
            snp_dim,
            target_chunk_size=chunk_size,
            performer_dim=self.performer_readout_dim,
            dropout=dropout,
        )

        # Original direct/global dense shortcut remains unchanged.
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

        # ------------------------------------------------------------------
        # Branch-specific posterior heads for the Latent48 32+16 design.
        # Keep their initialization RNG-local for deterministic controlled runs.
        # ------------------------------------------------------------------
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

        # Controlled predictor-head ablation:
        # keep 48 -> 64 -> 1, BatchNorm, and GELU unchanged; remove ONLY
        # the predictor Dropout(0.3).
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Original P2/M8B decoder width unchanged at 256.
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

    def forward(self, x):
        h_performer = self.snp_performer(x)  # [B,256]
        h_linear = self.linear_shortcut(x)   # [B,64]

        mu_performer = self.fc_mu_performer(h_performer)
        logvar_performer = self.fc_logvar_performer(h_performer)
        mu_linear = self.fc_mu_linear(h_linear)
        logvar_linear = self.fc_logvar_linear(h_linear)

        mu = torch.cat([mu_performer, mu_linear], dim=1)
        logvar = torch.cat([logvar_performer, logvar_linear], dim=1)

        # Safety check: allocation must produce the requested total z48.
        if mu.shape[1] != self.latent_dim or logvar.shape[1] != self.latent_dim:
            raise RuntimeError(
                f"Posterior width mismatch: mu={mu.shape[1]}, "
                f"logvar={logvar.shape[1]}, expected={self.latent_dim}."
            )

        z = self.reparameterize(mu, logvar)
        y_p = self.predictor(z)
        x_r = self.decoder(z)
        return x_r, y_p, mu, logvar


# ==========================================
# 3. 训练与数据处理逻辑
# ==========================================
class GeneticAwareTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = GeneticAwareLoss(
            pred_weight=PRED_WEIGHT,
            pearson_weight=PEARSON_WEIGHT,
            recon_weight=RECON_WEIGHT,
            kl_weight=KL_WEIGHT
        )

    def train_step(self, loader, optimizer, scheduler=None):
        self.model.train()

        sum_total = 0.0
        sum_pred = 0.0
        sum_recon = 0.0
        sum_pearson = 0.0
        sum_kl = 0.0
        n_batches = 0

        for bx, by in loader:
            bx = bx.to(self.device, non_blocking=True)
            by = by.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            xr, yp, mu, lv = self.model(bx)

            loss, rec_l, pea_l, mse_l, kl_l = self.loss_fn(
                xr, bx, yp, by, mu, lv
            )

            loss.backward()

            # Original GenoPerfVAE training protocol: global gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=GRAD_CLIP
            )
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            sum_total += float(loss.detach().item())
            sum_pred += float(mse_l.detach().item())
            sum_recon += float(rec_l.detach().item())
            sum_pearson += float(pea_l.detach().item())
            sum_kl += float(kl_l.detach().item())
            n_batches += 1

        n_batches = max(n_batches, 1)

        return {
            "total": sum_total / n_batches,
            "pred": sum_pred / n_batches,
            "recon": sum_recon / n_batches,
            "pearson": sum_pearson / n_batches,
            "kl": sum_kl / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, loader, scaler_y):
        self.model.eval()
        total_loss = 0
        preds, trues = [], []
        for bx, by in loader:
            bx, by = bx.to(self.device), by.to(self.device)
            xr, yp, mu, lv = self.model(bx)
            loss, _, _, _, _ = self.loss_fn(xr, bx, yp, by, mu, lv)
            total_loss += loss.item()
            preds.append(yp.cpu().numpy())
            trues.append(by.cpu().numpy())

        y_p_scaled = np.concatenate(preds)
        y_t_scaled = np.concatenate(trues)
        y_p = scaler_y.inverse_transform(y_p_scaled).flatten()
        y_t = scaler_y.inverse_transform(y_t_scaled).flatten()

        avg_loss = total_loss / len(loader)
        if np.std(y_p) < 1e-6:
            return avg_loss, 0.0, -1.0, 0.0

        r, _ = stats.pearsonr(y_t, y_p)
        r2_val = r2_score(y_t, y_p)
        pearson_sq = r ** 2
        return avg_loss, r, r2_val, pearson_sq



# =============================================================================
# Wheat2000 GenoPerfVAE P2: Flatten All Performer Tokens, No Pooling
# Seed42 | 10-fold | fixed 200 epochs
# =============================================================================

from pathlib import Path
from sklearn.metrics import mean_absolute_error

SEED = 42
N_SPLITS = 10

# Controlled training strategy:
# use all outer-training samples, no internal validation split.
INTERNAL_VALIDATION = False

TRAIT_FILES = {
    "tkw": "2000_1_phe_tkw.txt",
    "testw": "2000_2_phe_testw.txt",
    "length": "2000_3_phe_length.txt",
    "width": "2000_4_phe_width.txt",
    "hardness": "2000_5_phe_hardness.txt",
    "Prot": "2000_6_phe_Prot.txt",
}

# Run all six Wheat2000 traits.
RUN_ALL_6_TRAITS = True
SCREEN_TRAITS = ["tkw", "testw", "length", "width"]

# Latent48 experiment: Performer 32 + Linear 16 = total latent 48.
LATENT_ALLOCATIONS = [
    (32, 16),
]

PERFORMER_READOUT_DIM = 256
DECODER_PERFORMER_DIM = 256
TOTAL_LATENT_DIM = 48

CHUNK_SIZES = [32,]

BATCH_SIZE = 48
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

# Original GenoPerfVAE joint objective.
PRED_WEIGHT = 1.50
PEARSON_WEIGHT = 0.0
RECON_WEIGHT = 0.30
# KL reduction remains mean over batch and latent dimensions.
KL_WEIGHT = 0.80


def set_seed_fixed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_sync_fixed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_model_parameters_fixed(model):
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


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
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
            y_pred
        )[0]
    )


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    ).reshape(-1)

    mse = float(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    y_sd = (
        float(
            np.std(
                y_true,
                ddof=1
            )
        )
        if len(y_true) > 1
        else np.nan
    )

    return {
        "Pearson":
            safe_pearson(
                y_true,
                y_pred
            ),
        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred
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
                    np.isfinite(y_sd)
                    and y_sd > 1e-12
                )
                else np.nan
            ),
        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred
                )
            ),
    }


def safe_batch_size(
    n_samples,
    preferred=48
):
    """
    The model contains BatchNorm layers.
    Avoid a final training batch of size 1.
    """
    batch = min(
        preferred,
        n_samples
    )

    while (
        batch > 2
        and n_samples % batch == 1
    ):
        batch -= 1

    return batch


def make_loader(
    X,
    y,
    batch_size,
    shuffle,
    seed
):
    dataset = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(
                X,
                dtype=np.float32
            )
        ),
        torch.from_numpy(
            np.ascontiguousarray(
                y,
                dtype=np.float32
            )
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


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
        "tkw": 1,
        "testw": 2,
        "length": 3,
        "width": 4,
        "hardness": 5,
        "Prot": 6,
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


@torch.no_grad()
def evaluate_collect(
    model,
    loader,
    loss_fn,
    device
):
    model.eval()

    total_loss = 0.0
    preds = []
    trues = []

    for bx, by in loader:
        bx = bx.to(
            device,
            non_blocking=True
        )

        by = by.to(
            device,
            non_blocking=True
        )

        xr, yp, mu, lv = model(bx)

        loss, _, _, _, _ = loss_fn(
            xr,
            bx,
            yp,
            by,
            mu,
            lv
        )

        total_loss += float(
            loss.item()
        )

        preds.append(
            yp.detach()
            .cpu()
            .numpy()
        )

        trues.append(
            by.detach()
            .cpu()
            .numpy()
        )

    return (
        total_loss
        / max(
            len(loader),
            1
        ),
        np.concatenate(
            preds,
            axis=0
        ),
        np.concatenate(
            trues,
            axis=0
        ),
    )


@torch.no_grad()
def benchmark_inference_fixed(
    model,
    loader,
    device
):
    model.eval()

    # Warm-up
    for bx, _ in loader:
        bx = bx.to(
            device,
            non_blocking=True
        )

        _ = model(bx)
        break

    cuda_sync_fixed()

    total_samples = 0
    start = time.perf_counter()

    for bx, _ in loader:
        bx = bx.to(
            device,
            non_blocking=True
        )

        _ = model(bx)

        total_samples += (
            bx.size(0)
        )

    cuda_sync_fixed()

    total_time = (
        time.perf_counter()
        - start
    )

    return (
        total_time,
        total_time
        / max(
            total_samples,
            1
        )
        * 1000.0
    )


def run_setting(
    X_all,
    y_all,
    sample_rows,
    trait,
    chunk_size,
    performer_latent_dim,
    linear_latent_dim,
    output_dir,
    device
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    performer_latent_dim = int(performer_latent_dim)
    linear_latent_dim = int(linear_latent_dim)
    if performer_latent_dim + linear_latent_dim != TOTAL_LATENT_DIM:
        raise ValueError(
            f"Invalid latent allocation: {performer_latent_dim}+{linear_latent_dim} "
            f"!= {TOTAL_LATENT_DIM}."
        )

    allocation_tag = f"P{performer_latent_dim}_L{linear_latent_dim}"
    variant_name = (
        f"P2_M8B_Latent48_{allocation_tag}_R256_Predictor64_GELU_NoDropout_FullToken_BranchSpecificLatent_"
        "OriginalTraining_Seed42"
    )

    snp_dim = int(
        X_all.shape[1]
    )

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_rows = []
    prediction_rows = []
    membership_rows = []

    print(
        "\n"
        + "#" * 105
    )
    print(
        f"GenoPerfVAE P2+M8B-Latent48 R256 + Predictor64 GELU/NoDropout | Wheat2000 | "
        f"{trait} | C={chunk_size} | latent={performer_latent_dim}+{linear_latent_dim} | "
        "flatten-all-token readout; NO pooling | Original loss | Adam | grad clip=1.0"
    )
    print(
        "#" * 105
    )

    for fold, (
        outer_train_idx,
        test_idx
    ) in enumerate(
        kf.split(
            X_all
        ),
        start=1
    ):
        set_seed_fixed(
            SEED
        )

        print(
            "\n"
            + "-" * 105
        )
        print(
            f"{trait} | C={chunk_size} | latent={performer_latent_dim}+{linear_latent_dim} | Fold {fold}/10"
        )
        print(
            "-" * 105
        )

        # -----------------------------------------------------
        # Original controlled training protocol:
        # use ALL outer-training samples.
        # No internal validation split and no early stopping.
        # The held-out outer test fold remains completely untouched.
        # -----------------------------------------------------
        train_idx = outer_train_idx

        # -----------------------------------------------------
        # True NaN only.
        # Wheat2000 -1 remains a valid genotype.
        # Fit imputer on the complete OUTER TRAIN only.
        # -----------------------------------------------------
        imputer = SimpleImputer(
            strategy="most_frequent",
            keep_empty_features=True
        )

        X_train = (
            imputer
            .fit_transform(
                X_all[
                    train_idx
                ]
            )
            .astype(
                np.float32
            )
        )

        X_test = (
            imputer
            .transform(
                X_all[
                    test_idx
                ]
            )
            .astype(
                np.float32
            )
        )

        # No additional SNP scaling:
        # preserve the Wheat2000 genotype representation.
        scaler_y = StandardScaler()

        y_train = (
            scaler_y
            .fit_transform(
                y_all[
                    train_idx
                ]
            )
            .astype(
                np.float32
            )
        )

        y_test_scaled = (
            scaler_y
            .transform(
                y_all[
                    test_idx
                ]
            )
            .astype(
                np.float32
            )
        )

        y_test = (
            y_all[
                test_idx
            ]
            .reshape(-1)
            .astype(
                np.float32
            )
        )

        train_batch = safe_batch_size(
            len(
                train_idx
            ),
            BATCH_SIZE
        )

        train_loader = make_loader(
            X_train,
            y_train,
            train_batch,
            True,
            SEED
        )

        test_loader = make_loader(
            X_test,
            y_test_scaled,
            BATCH_SIZE,
            False,
            SEED
        )

        # -----------------------------------------------------
        # P2 full-token flatten readout experiment.
        # Shortcut64, R256 readout, decoder width256, and training remain fixed; total latent is 48.
        # -----------------------------------------------------
        model = EnhancedGeneticModel(
            snp_dim=snp_dim,
            latent_dim=TOTAL_LATENT_DIM,
            performer_readout_dim=PERFORMER_READOUT_DIM,
            decoder_performer_dim=DECODER_PERFORMER_DIM,
            performer_latent_dim=performer_latent_dim,
            linear_latent_dim=linear_latent_dim,
            dropout=0.3,
            chunk_size=chunk_size
        ).to(
            device
        )

        trainer = GeneticAwareTrainer(
            model,
            device
        )

        num_chunks = (
            model
            .snp_performer
            .num_chunks
        )

        performer_sequence_length = (
            model
            .snp_performer
            .sequence_length
        )

        padding_size = (
            model
            .snp_performer
            .pad_n
        )

        actual_chunk_size = (
            model
            .snp_performer
            .chunk_size
        )

        total_params, trainable_params = (
            count_model_parameters_fixed(
                model
            )
        )

        # -----------------------------------------------------
        # Single-variable Performer-readout experiment.
        # The ONLY intended change from Original GenoPerfVAE is the P2 full-token flatten readout.
        # Adam is used at 1e-4 with grad clip=1.0 for fixed 200 epochs.
        # No validation, no early stopping, no LR scheduler.
        # -----------------------------------------------------
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        epoch_times = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(
                device
            )

        cuda_sync_fixed()
        training_start = time.perf_counter()

        # Fixed 200 epochs.
        # No validation, no early stopping, no checkpoint selection.
        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):
            cuda_sync_fixed()
            epoch_start = time.perf_counter()

            train_stats = trainer.train_step(
                train_loader,
                optimizer,
                scheduler=None
            )

            cuda_sync_fixed()

            epoch_times.append(
                time.perf_counter()
                - epoch_start
            )

            if (
                epoch == 1
                or epoch % 20 == 0
                or epoch == MAX_EPOCHS
            ):
                weighted_pred = (
                    trainer.loss_fn.m_weight
                    * train_stats["pred"]
                )
                weighted_recon = (
                    trainer.loss_fn.r_weight
                    * train_stats["recon"]
                )
                weighted_kl = (
                    trainer.loss_fn.k_weight
                    * train_stats["kl"]
                )

                print(
                    f"Epoch "
                    f"{epoch:03d}/"
                    f"{MAX_EPOCHS} | "
                    f"Total={train_stats['total']:.4f} | "
                    f"Pred={train_stats['pred']:.4f} "
                    f"(w={weighted_pred:.4f}) | "
                    f"Recon={train_stats['recon']:.4f} "
                    f"(w={weighted_recon:.4f}) | "
                    f"KL={train_stats['kl']:.4f} "
                    f"(w={weighted_kl:.4f})"
                )

        cuda_sync_fixed()

        training_time = (
            time.perf_counter()
            - training_start
        )

        actual_epochs = len(
            epoch_times
        )

        best_epoch = MAX_EPOCHS

        # -----------------------------------------------------
        # Held-out outer test exactly once.
        # -----------------------------------------------------
        (
            test_loss,
            test_pred_scaled,
            test_true_scaled
        ) = evaluate_collect(
            model,
            test_loader,
            trainer.loss_fn,
            device
        )

        test_pred = (
            scaler_y
            .inverse_transform(
                test_pred_scaled
            )
            .reshape(-1)
        )

        test_true_check = (
            scaler_y
            .inverse_transform(
                test_true_scaled
            )
            .reshape(-1)
        )

        if not np.allclose(
            test_true_check,
            y_test,
            atol=1e-4,
            rtol=1e-5
        ):
            raise RuntimeError(
                "Inverse-transformed y_test "
                "does not match original phenotype."
            )

        metrics = calc_metrics(
            y_test,
            test_pred
        )

        (
            inference_total,
            inference_ms_sample
        ) = benchmark_inference_fixed(
            model,
            test_loader,
            device
        )

        peak_gpu_mb = (
            float(
                torch.cuda
                .max_memory_allocated(
                    device
                )
                / 1024**2
            )
            if torch.cuda.is_available()
            else np.nan
        )

        fold_row = {
            "Model": "GenoPerfVAE",
            "Variant": variant_name,
            "Dataset": "Wheat2000",
            "Pooling_Method": "None_FlattenAllPerformerTokens",
            "Flatten_Readout_Dim": int(model.snp_performer.flatten_readout_dim),
            "Performer_Readout_Dim": int(model.performer_readout_dim),
            "Decoder_Performer_Dim": int(model.decoder_performer_dim),
            "Readout_Init_RNG_Stable": True,
            "Latent_Total_Dim": TOTAL_LATENT_DIM,
            "Performer_Latent_Dim": performer_latent_dim,
            "Linear_Latent_Dim": linear_latent_dim,
            "Latent_Allocation": allocation_tag,
            "Posterior_Init_RNG_Stable": True,
            "Trait": trait,
            "Chunk_Size_C":
                int(
                    actual_chunk_size
                ),
            "Attention_Length_N":
                int(
                    performer_sequence_length
                ),
            "Genomic_Chunk_Length_N":
                int(
                    num_chunks
                ),
            "SNP_Dimension_D":
                int(
                    snp_dim
                ),
            "Padding_P":
                int(
                    padding_size
                ),
            "Fold": fold,
            "Outer_Train_N":
                len(
                    outer_train_idx
                ),
            "Train_N":
                len(
                    train_idx
                ),
            "Validation_N":
                0,
            "Test_N":
                len(
                    test_idx
                ),
            "Best_Epoch":
                best_epoch,
            "Actual_Epochs":
                actual_epochs,
            "Internal_Validation":
                False,
            "Optimizer":
                "Adam",
            "Learning_Rate":
                LEARNING_RATE,
            "Weight_Decay":
                WEIGHT_DECAY,
            "Gradient_Clipping":
                True,
            "Grad_Clip_Max_Norm":
                GRAD_CLIP,
            "Batch_Size":
                BATCH_SIZE,
            "Pred_Weight":
                PRED_WEIGHT,
            "Pearson_Weight":
                PEARSON_WEIGHT,
            "Recon_Weight":
                RECON_WEIGHT,
            "KL_Weight":
                KL_WEIGHT,
            "Predictor_Input":
                "z",
            "Shortcut_Output_Dim":
                64,
            "Fusion_Method":
                f"BranchSpecificPosterior_Performer256to{performer_latent_dim}_Linear64to{linear_latent_dim}",
            "PreLatent_Feature_Fusion":
                "Separate_Performer256_and_Shortcut64_PosteriorHeads",
            "Total_Latent_Dim":
                TOTAL_LATENT_DIM,
            "Predictor_Head":
                "Latent48_48to64_BN_GELU_NoDropout_to1",
            "Predictor_Hidden_Dim":
                64,
            "Predictor_Dropout":
                0.0,
            "Predictor_Activation":
                "GELU",
            "KL_Reduction":
                "mean_over_batch_and_latent",
            "Training_Strategy":
                (
                    "AllOuterTrain"
                    "+Fixed200Epochs"
                    "+NoValidation"
                    "+NoEarlyStopping"
                    "+Adam1e-4"
                    "+OriginalLoss_1.5_0_0.5_0.8"
                    f"+GradClip1+FlattenAllTokens+NoPooling+M8BBranchSpecificLatent"
                    f"+R256Readout+Latent48P{performer_latent_dim}L{linear_latent_dim}"
                    "+Predictor64_BN_GELU_NoDropout"
                ),
            **metrics,
            "Test_Loss":
                float(
                    test_loss
                ),
            "Epoch_Time_Mean_s":
                float(
                    np.mean(
                        epoch_times
                    )
                ),
            "Epoch_Time_SD_s":
                (
                    float(
                        np.std(
                            epoch_times,
                            ddof=1
                        )
                    )
                    if len(
                        epoch_times
                    ) > 1
                    else 0.0
                ),
            "Training_Time_s":
                float(
                    training_time
                ),
            "Peak_GPU_Memory_MB":
                peak_gpu_mb,
            "Inference_Total_s":
                float(
                    inference_total
                ),
            "Inference_ms_per_sample":
                float(
                    inference_ms_sample
                ),
            "Total_Params_M":
                total_params / 1e6,
            "Trainable_Params_M":
                trainable_params / 1e6,
            "Observed_Mean":
                float(
                    np.mean(
                        y_test
                    )
                ),
            "Prediction_Mean":
                float(
                    np.mean(
                        test_pred
                    )
                ),
            "Mean_Bias_PredMinusObs":
                float(
                    np.mean(
                        test_pred
                        - y_test
                    )
                ),
            "Observed_SD":
                float(
                    np.std(
                        y_test,
                        ddof=1
                    )
                ),
            "Prediction_SD":
                float(
                    np.std(
                        test_pred,
                        ddof=1
                    )
                ),
            "Seed":
                SEED,
            "X_Processing":
                (
                    "Wheat2000_Minus1Valid"
                    "+TrainOnlyTrueNaNImputer"
                    "+NoSNPScaling"
                ),
            "Y_Processing":
                (
                    "OuterTrainOnlyStandardScaler"
                    "+InverseTransformForMetrics"
                ),
        }

        fold_rows.append(
            fold_row
        )

        for (
            idx,
            obs,
            pred
        ) in zip(
            test_idx,
            y_test,
            test_pred
        ):
            prediction_rows.append({
                "Trait":
                    trait,
                "Chunk_Size_C":
                    actual_chunk_size,
                "Fold":
                    fold,
                "Aligned_Row_0Based":
                    int(
                        idx
                    ),
                "Sample_Row_0Based":
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
                "Chunk_Size_C":
                    actual_chunk_size,
                "Fold":
                    fold,
                "Aligned_Row_0Based":
                    int(
                        idx
                    ),
                "Sample_Row_0Based":
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
                "Chunk_Size_C":
                    actual_chunk_size,
                "Fold":
                    fold,
                "Aligned_Row_0Based":
                    int(
                        idx
                    ),
                "Sample_Row_0Based":
                    int(
                        sample_rows[
                            idx
                        ]
                    ),
                "Set":
                    "Test",
            })

        print(
            f"Fold {fold}: "
            f"r="
            f"{metrics['Pearson']:.4f}, "
            f"R2="
            f"{metrics['R2']:.4f}, "
            f"RMSE="
            f"{metrics['RMSE']:.4f}, "
            f"nRMSE="
            f"{metrics['nRMSE']:.4f}, "
            f"MAE="
            f"{metrics['MAE']:.4f}"
        )

        pd.DataFrame(
            fold_rows
        ).to_csv(
            output_dir
            / (
                f"GenoPerfVAE_Wheat2000_P2_"
                f"{trait}_C{chunk_size}_"
                "FoldResults_CHECKPOINT.csv"
            ),
            index=False
        )

        del (
            model,
            trainer,
            optimizer,
            train_loader,
            test_loader,
            X_train,
            X_test,
            y_train,
            y_test_scaled,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            f"GenoPerfVAE_Wheat2000_"
            f"{trait}_C{chunk_size}_"
            "FoldResults.csv"
        ),
        index=False
    )

    pred_df.to_csv(
        output_dir
        / (
            f"GenoPerfVAE_Wheat2000_"
            f"{trait}_C{chunk_size}_"
            "Predictions.csv"
        ),
        index=False
    )

    membership_df.to_csv(
        output_dir
        / (
            f"GenoPerfVAE_Wheat2000_"
            f"{trait}_C{chunk_size}_"
            "10FoldMembership.csv"
        ),
        index=False
    )

    summary = {
        "Model": "GenoPerfVAE",
        "Variant": variant_name,
        "Dataset": "Wheat2000",
        "Pooling_Method": "None_FlattenAllPerformerTokens",
        "Flatten_Readout_Dim": int(fold_df["Flatten_Readout_Dim"].iloc[0]),
        "Performer_Readout_Dim": int(fold_df["Performer_Readout_Dim"].iloc[0]),
        "Decoder_Performer_Dim": int(fold_df["Decoder_Performer_Dim"].iloc[0]),
        "Readout_Init_RNG_Stable": True,
        "Posterior_Init_RNG_Stable": True,
        "Latent_Total_Dim": TOTAL_LATENT_DIM,
        "Performer_Latent_Dim": performer_latent_dim,
        "Linear_Latent_Dim": linear_latent_dim,
        "Latent_Allocation": allocation_tag,
        "Trait": trait,
        "N": len(
            X_all
        ),
        "Chunk_Size_C":
            int(
                chunk_size
            ),
        "Attention_Length_N":
            int(
                fold_df[
                    "Attention_Length_N"
                ].iloc[0]
            ),
        "Genomic_Chunk_Length_N":
            int(
                fold_df[
                    "Genomic_Chunk_Length_N"
                ].iloc[0]
            ),
        "SNP_Dimension_D":
            int(
                snp_dim
            ),
        "Padding_P":
            int(
                fold_df[
                    "Padding_P"
                ].iloc[0]
            ),
        "Pred_Weight":
            PRED_WEIGHT,
        "Pearson_Weight":
            PEARSON_WEIGHT,
        "Recon_Weight":
            RECON_WEIGHT,
        "KL_Weight":
            KL_WEIGHT,
        "Optimizer":
            "Adam",
        "Learning_Rate":
            LEARNING_RATE,
        "Weight_Decay":
            WEIGHT_DECAY,
        "Gradient_Clipping":
            True,
        "Grad_Clip_Max_Norm":
            GRAD_CLIP,
        "Predictor_Input":
            "z",
        "Shortcut_Output_Dim":
            64,
        "Fusion_Method":
            f"BranchSpecificPosterior_Performer256to{performer_latent_dim}_Linear64to{linear_latent_dim}",
        "PreLatent_Feature_Fusion":
            "Separate_Performer256_and_Shortcut64_PosteriorHeads",
        "Total_Latent_Dim":
            TOTAL_LATENT_DIM,
        "Predictor_Head":
            "Latent48_48to64_BN_GELU_NoDropout_to1",
        "Predictor_Hidden_Dim":
            64,
        "Predictor_Dropout":
            0.0,
        "Predictor_Activation":
            "GELU",
        "KL_Reduction":
            "mean_over_batch_and_latent",
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

    oof = calc_metrics(
        pred_df[
            "Observed"
        ],
        pred_df[
            "Predicted"
        ]
    )

    for metric, value in (
        oof.items()
    ):
        summary[
            f"OOF_{metric}"
        ] = value

    summary.update({
        "Best_Epoch_Mean":
            fold_df[
                "Best_Epoch"
            ].mean(),
        "Best_Epoch_SD":
            fold_df[
                "Best_Epoch"
            ].std(
                ddof=1
            ),
        "Epoch_Time_Mean_s":
            fold_df[
                "Epoch_Time_Mean_s"
            ].mean(),
        "Epoch_Time_SD_s":
            fold_df[
                "Epoch_Time_Mean_s"
            ].std(
                ddof=1
            ),
        "Training_Time_Mean_s":
            fold_df[
                "Training_Time_s"
            ].mean(),
        "Training_Time_SD_s":
            fold_df[
                "Training_Time_s"
            ].std(
                ddof=1
            ),
        "Peak_GPU_Memory_Mean_MB":
            fold_df[
                "Peak_GPU_Memory_MB"
            ].mean(),
        "Peak_GPU_Memory_Max_MB":
            fold_df[
                "Peak_GPU_Memory_MB"
            ].max(),
        "Inference_Mean_ms_sample":
            fold_df[
                "Inference_ms_per_sample"
            ].mean(),
        "Total_Params_M":
            fold_df[
                "Total_Params_M"
            ].iloc[0],
        "Trainable_Params_M":
            fold_df[
                "Trainable_Params_M"
            ].iloc[0],
    })

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        output_dir
        / (
            f"GenoPerfVAE_Wheat2000_"
            f"{trait}_C{chunk_size}_"
            "Summary.csv"
        ),
        index=False
    )

    print(
        "\n"
        + "=" * 95
    )
    print(
        f"{trait} | C={chunk_size} | latent={performer_latent_dim}+{linear_latent_dim} | 10-fold summary"
    )
    print(
        "=" * 95
    )
    print(
        f"Pearson: "
        f"{summary['Pearson_Mean']:.4f} ± "
        f"{summary['Pearson_SD']:.4f}"
    )
    print(
        f"R2     : "
        f"{summary['R2_Mean']:.4f} ± "
        f"{summary['R2_SD']:.4f}"
    )
    print(
        f"RMSE   : "
        f"{summary['RMSE_Mean']:.4f} ± "
        f"{summary['RMSE_SD']:.4f}"
    )
    print(
        f"nRMSE  : "
        f"{summary['nRMSE_Mean']:.4f} ± "
        f"{summary['nRMSE_SD']:.4f}"
    )
    print(
        f"MAE    : "
        f"{summary['MAE_Mean']:.4f} ± "
        f"{summary['MAE_SD']:.4f}"
    )

    return (
        fold_df,
        pred_df,
        summary_df
    )


def main_wheat2000():
    set_seed_fixed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print(
        "P2+M8B-Latent48: NO pooling; flatten every post-Performer genomic token "
        "before Linear(N*C->256); branch allocation=32+16; total latent=48; "
        "decoder width=256; Predictor64 BN+GELU NoDropout."
    )
    print(
        "Loss configuration:",
        f"pred={PRED_WEIGHT}, pearson={PEARSON_WEIGHT}, "
        f"recon={RECON_WEIGHT}, kl={KL_WEIGHT}, "
        f"optimizer=Adam(lr={LEARNING_RATE}), weight_decay=0, grad_clip=1.0, "
        "full_outer_train=True, fixed_epochs=200, validation=False, "
        "early_stopping=False, scheduler=None, "
        "KL_reduction=mean_over_batch_and_latent, "
        "chunk_size=32 only."
    )

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    base_dir = Path(__file__).resolve().parent
    data_dir = resolve_wheat2000_data_dir(base_dir)
    genotype_file = data_dir / "2000gene.pkl"

    X_genotype, sample_rows = load_wheat2000_genotype(
        genotype_file
    )

    output_root = (
        base_dir
        / "results_genoperfvae_wheat2000_P2_M8B_Latent48_P32L16_R256_"
          "Predictor64_GELU_NoDropout_C32_OriginalTraining_Seed42"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Preserve the exact genotype column order used by the model.
    pd.DataFrame({
        "SNP_Index_0Based": np.arange(X_genotype.shape[1], dtype=int),
    }).to_csv(
        output_root / "Wheat2000_SNP_Order.csv",
        index=False,
    )

    all_summaries = []
    failed = []

    selected_traits = (
        list(TRAIT_FILES.keys())
        if RUN_ALL_6_TRAITS
        else SCREEN_TRAITS
    )

    print(
        "Traits to run:",
        selected_traits,
        "| RUN_ALL_6_TRAITS=",
        RUN_ALL_6_TRAITS,
    )
    print(
        "Latent allocations:",
        LATENT_ALLOCATIONS,
        "| Performer readout=",
        PERFORMER_READOUT_DIM,
        "| total latent=",
        TOTAL_LATENT_DIM,
        "| chunk sizes=",
        CHUNK_SIZES,
    )

    for trait in selected_traits:
        try:
            phenotype_file = resolve_trait_file(
                data_dir=data_dir,
                trait=trait,
                preferred_name=TRAIT_FILES[trait],
            )

            y_loaded = load_wheat2000_phenotype(
                phenotype_file
            )

            # Keep the established Wheat2000 row-order protocol.  Normally both
            # sources contain the same number of samples.  If a historical file
            # differs, use the common leading row-aligned subset and report it.
            if y_loaded.shape[0] != X_genotype.shape[0]:
                n_common = min(
                    y_loaded.shape[0],
                    X_genotype.shape[0],
                )
                print(
                    f"WARNING [{trait}]: genotype N={X_genotype.shape[0]}, "
                    f"phenotype N={y_loaded.shape[0]}; using first "
                    f"{n_common} row-aligned samples."
                )
                X_all = np.ascontiguousarray(
                    X_genotype[:n_common],
                    dtype=np.float32,
                )
                y_all = np.ascontiguousarray(
                    y_loaded[:n_common],
                    dtype=np.float32,
                )
                trait_rows = sample_rows[:n_common]
            else:
                X_all = X_genotype
                y_all = y_loaded
                trait_rows = sample_rows

            print(
                f"\n{trait}: N={len(X_all)}, SNPs={X_all.shape[1]}, "
                f"y mean={float(np.mean(y_all)):.6f}, "
                f"y sd={float(np.std(y_all)):.6f}"
            )

            trait_dir = output_root / trait
            trait_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({
                "Aligned_Row_0Based": np.arange(len(X_all), dtype=int),
                "Sample_Row_0Based": trait_rows,
                "Observed": y_all.reshape(-1),
            }).to_csv(
                trait_dir / f"Wheat2000_{trait}_AlignedSamples.csv",
                index=False,
            )

            for performer_latent_dim, linear_latent_dim in LATENT_ALLOCATIONS:
                for chunk_size in CHUNK_SIZES:
                    allocation_tag = (
                        f"P{performer_latent_dim}_L{linear_latent_dim}"
                    )
                    setting_dir = (
                        trait_dir
                        / allocation_tag
                        / f"chunk_{chunk_size}_flatten_all_tokens_R256_"
                          "Latent48_Predictor64_GELU_NoDropout"
                    )

                    _, _, summary_df = run_setting(
                        X_all=X_all,
                        y_all=y_all,
                        sample_rows=trait_rows,
                        trait=trait,
                        chunk_size=chunk_size,
                        performer_latent_dim=performer_latent_dim,
                        linear_latent_dim=linear_latent_dim,
                        output_dir=setting_dir,
                        device=device,
                    )

                    summary_df["Phenotype_File"] = str(phenotype_file)
                    all_summaries.append(summary_df)

                    pd.concat(
                        all_summaries,
                        ignore_index=True,
                    ).to_csv(
                        output_root
                        / "GenoPerfVAE_Wheat2000_P2_M8B_Latent48_P32L16_R256_"
                          "Predictor64_GELU_NoDropout_C32_OriginalTraining_"
                          "Fixed200_Seed42_Summary_CHECKPOINT.csv",
                        index=False,
                    )

            del y_loaded, X_all, y_all, trait_rows
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as exc:
            print(
                f"[ERROR] {trait}: "
                f"{type(exc).__name__}: {exc}"
            )
            failed.append({
                "Trait": trait,
                "Reason": f"{type(exc).__name__}: {exc}",
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(
            output_root
            / "GenoPerfVAE_Wheat2000_P2_M8B_Latent48_P32L16_R256_"
              "Predictor64_GELU_NoDropout_C32_OriginalTraining_"
              "Fixed200_Seed42_Summary.csv",
            index=False,
        )

        display_cols = [
            "Trait",
            "N",
            "Flatten_Readout_Dim",
            "Performer_Readout_Dim",
            "Decoder_Performer_Dim",
            "Performer_Latent_Dim",
            "Linear_Latent_Dim",
            "Latent_Allocation",
            "Chunk_Size_C",
            "Attention_Length_N",
            "Padding_P",
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
            "Epoch_Time_Mean_s",
            "Training_Time_Mean_s",
            "Peak_GPU_Memory_Max_MB",
            "Inference_Mean_ms_sample",
            "Total_Params_M",
        ]

        print("\n" + "=" * 185)
        print(
            "GenoPerfVAE Wheat2000 P2+M8B-Latent48 Full-Token Flatten + "
            "Branch-Specific Latent (32+16) + Predictor64 GELU/NoDropout | "
            f"traits={selected_traits} | C=32 | "
            "loss 1.5/0/0.5/0.8 | Adam lr=1e-4 | "
            "grad clip=1.0 | fixed 200 epochs"
        )
        print("=" * 185)
        print(combined[display_cols].to_string(index=False))

    if failed:
        pd.DataFrame(failed).to_csv(
            output_root / "GenoPerfVAE_Wheat2000_FailedTraits.csv",
            index=False,
        )


if __name__ == "__main__":
    main_wheat2000()
