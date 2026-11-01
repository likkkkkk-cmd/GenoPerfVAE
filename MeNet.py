# -*- coding: utf-8 -*-
"""
MeNet_Wheat2000_7Traits_10Fold_FairComparison.py

Standalone adaptation of the uploaded MeNet Version2.0 repository for the
Wheat2000 benchmark used in the GenoPerfVAE manuscript.

Wheat2000 inputs
--------------
Genotype:
    2000gene.pkl

Expected genotype shape:
    approximately 2000 samples x 33,709 markers

Important:
    -1 is a VALID genotype state and is NOT missing.
    Continuous values already present in the genotype CSV (e.g. 0.78698...)
    are retained unchanged.
    The first column "Entry" is the sample identifier and is never used as SNP.

Phenotypes:
    wheat2000_PH.csv
    wheat2000_PnN.csv
    wheat2000_LA.csv
    wheat2000_GL.csv
    wheat2000_GW.csv
    wheat2000_YPP.csv
    wheat2000_YLD.csv

Each phenotype CSV must contain:
    Entry, <trait>

Benchmark design
----------------
- Outer 10-fold CV:
      KFold(n_splits=10, shuffle=True, random_state=42)
- Within every outer-training fold:
      85% model-training / 15% validation, random_state=42
- The outer test fold is never used for:
      preprocessing,
      phenotype-aware contrastive learning,
      genetic-relatedness construction,
      checkpoint selection.

MeNet two-stage workflow
------------------------
Stage 1: RepGeno trait-specific encoder
    genotype -> trait-aware triplet contrastive embedding

Stage 2: MeNet
    VE branch: raw SNP vector
    RepGeno branch: phenotype-specific genetic-relatedness vector
    Fusion -> phenotype prediction

Anti-leakage redesign
---------------------
The original repository computes a genetic-relatedness matrix after training the
trait-specific encoder. For strict CV, this script uses ONLY the inner training
samples as the relatedness reference set:

    GR_train = relatedness(train, train)
    GR_val   = relatedness(val, train)
    GR_test  = relatedness(test, train)

Thus validation/test genotypes do not define the reference basis used by the
training samples, and outer-test phenotypes are never used.

Repository Version2.0 defaults retained
---------------------------------------
Contrastive encoder:
    stride=2
    out_dim=128
    margin=0.1
    epochs=5
    batch_size=32
    AdamW lr=0.01, weight_decay=0.01
    ReduceLROnPlateau patience=5

MeNet:
    batch_size=8
    epochs=20
    lr=0.01
    AdamW weight_decay=0.01
    loss=L1

    VE:
        conv=[8,11,2]
        res1=[16,11,2]
        res2=[32,11,2]
        adaptive=4
        dropout=0.6
        embedding=[4096,1024]

    RepGeno:
        embedding=[4096,1024]
        dropout=0.6

    output:
        VE/RepGeno embedding dim=1024
        hidden dim=1024
        dropout=0.6

Chromosome windows
------------------
The repository enables chromosome windows when explicitly requested or when
marker count exceeds 100,000. Wheat2000 has ~33.7k markers, so this script uses
the repository's default non-window MeNet path. This also avoids changing the model
architecture solely for this benchmark.

Integrated gradients
--------------------
The repository calculates IG every epoch for interpretability reporting.
IG does not contribute to the training loss or parameter updates. It is omitted
from the benchmark training loop to avoid large extra runtime. This does NOT
change the optimization objective or checkpoint criterion.

Outputs
-------
For every trait:
    exact Entry alignment
    outer/inner fold membership
    fold metrics
    predictions
    RepGeno/MeNet best epochs
    runtime / GPU memory / parameter counts

Final:
    MeNet_Wheat2000_AllTraits_Summary.csv
"""

import gc
import copy
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# Global settings
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 10
INNER_VAL_FRAC = 0.15

# ---------------- Stage 1: trait-specific RepGeno encoder ----
REPGENO_STRIDE = 2
REPGENO_OUT_DIM = 128
REPGENO_MARGIN = 0.1
REPGENO_EPOCHS = 5
REPGENO_BATCH_SIZE = 32
REPGENO_LR = 0.01

# ---------------- Stage 2: MeNet -----------------------------
MENET_EPOCHS = 20
MENET_BATCH_SIZE = 8
MENET_LR = 0.01

# PyTorch AdamW default, made explicit for reproducibility.
ADAMW_WEIGHT_DECAY = 0.01

MENET_PARAM = {
    "VE": {
        "conv": [8, 11, 2],
        "res1": [16, 11, 2],
        "res2": [32, 11, 2],
        "adaptive": 4,
        "dropout": 0.6,
        "embedding_dim": [4096, 1024],
    },
    "RepGeno": {
        "embedding_dim": [4096, 1024],
        "dropout": 0.6,
    },
    "output": {
        "VE_and_RepGeno_embedding_dim": 1024,
        "embedding_dim": 1024,
        "dropout": 0.6,
    },
}


# ============================================================
# Reproducibility
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


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# Wheat2000 data loading
# Same row-order protocol used in the GenoPerfVAE / VMGP-S scripts
# ============================================================

def load_wheat2000_genotype(genotype_file):
    """
    Load Wheat2000 genotype from the .pkl file used in the existing
    GenoPerfVAE / VMGP-S comparison.

    Expected shape:
        approximately 2000 samples x 33,709 markers
    """
    genotype_file = Path(genotype_file)
    if not genotype_file.exists():
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")

    obj = pd.read_pickle(genotype_file)

    if isinstance(obj, pd.DataFrame):
        X = obj.to_numpy(dtype=np.float32)
        marker_names = [str(c) for c in obj.columns]
    elif isinstance(obj, pd.Series):
        arr = obj.to_numpy()
        if arr.ndim == 1 and len(arr) > 0 and hasattr(arr[0], '__len__'):
            X = np.vstack(arr).astype(np.float32)
        else:
            X = np.asarray(arr, dtype=np.float32)
        marker_names = [f'SNP_{i}' for i in range(X.shape[1])]
    else:
        X = np.asarray(obj, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f'Wheat2000 genotype object must be 2-D. Got shape: {X.shape}')
        marker_names = [f'SNP_{i}' for i in range(X.shape[1])]

    if X.ndim != 2:
        raise ValueError(f'Genotype matrix must be 2-D, got {X.shape}')

    print('=' * 88)
    print('Wheat2000 genotype loaded')
    print('Shape:', X.shape)
    print('Genotype range:', float(np.nanmin(X)), float(np.nanmax(X)))
    print('True missing genotype values:', int(np.isnan(X).sum()))
    print('=' * 88)

    if X.shape[0] != 2000:
        warnings.warn(
            'Wheat2000 is expected to contain about 2000 samples, '
            f'but the loaded genotype has {X.shape[0]} rows.'
        )

    return X, marker_names


def load_wheat2000_phenotype(phenotype_file):
    """
    Load a Wheat2000 phenotype text file using the same rule as the
    existing GenoPerfVAE / VMGP-S Wheat2000 scripts: take the first numeric
    token from each non-empty line.
    """
    phenotype_file = Path(phenotype_file)
    if not phenotype_file.exists():
        raise FileNotFoundError(f'Phenotype file not found: {phenotype_file}')

    values = []
    with open(phenotype_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.replace(',', ' ').replace(';', ' ').split()
            value = None
            for token in tokens:
                try:
                    value = float(token)
                    break
                except ValueError:
                    continue
            if value is not None:
                values.append(value)

    if len(values) == 0:
        raise ValueError(f'No numeric phenotype values were found in {phenotype_file}')

    return np.asarray(values, dtype=np.float32)


def align_wheat2000_by_row_order(X, y):
    """
    Align genotype and phenotype by row order, matching the existing
    Wheat2000 benchmark scripts.
    """
    n = min(X.shape[0], len(y))
    if X.shape[0] != len(y):
        warnings.warn(
            'Genotype and phenotype sample counts differ. '
            f'Genotype={X.shape[0]}, phenotype={len(y)}. '
            f'Both will be truncated to n={n} by row order.'
        )

    X2 = X[:n]
    y2 = y[:n]
    valid = np.isfinite(y2)
    removed = int(np.sum(~valid))
    X2 = X2[valid]
    y2 = y2[valid]
    sample_ids = np.arange(n, dtype=int)[valid]

    print('Aligned samples:', len(y2))
    print('Phenotype NA removed:', removed)
    return X2, y2, sample_ids


# ============================================================
# Stage 1 dataset: source-compatible triplet sampling
# ============================================================

class TraitTripletDataset(Dataset):
    """
    Wheat2000 uses flag=0 behavior from the uploaded repository:
    all samples belong to one population label, two random candidates
    are sampled, and TripletLoss decides phenotype-nearer vs farther.
    """

    def __init__(self, X, y):
        self.X = torch.as_tensor(
            X,
            dtype=torch.float32
        ).unsqueeze(1)  # [N,1,D]

        self.y = torch.as_tensor(
            y,
            dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def _random_other(self, index):
        if len(self) <= 1:
            raise RuntimeError(
                "Triplet sampling requires at least 2 samples."
            )

        j = index
        while j == index:
            j = random.randrange(
                len(self)
            )
        return j

    def __getitem__(self, index):
        j = self._random_other(index)
        k = self._random_other(index)

        return (
            self.X[index],
            self.y[index],
            self.X[j],
            self.y[j],
            self.X[k],
            self.y[k],
        )


# ============================================================
# Source-compatible adaptive triplet loss
# ============================================================

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor,
        anchor_phenotype,
        s,
        s_phenotype,
        t,
        t_phenotype
    ):
        # Same flag=False behavior from uploaded loss.py:
        # closer phenotype candidate -> positive.
        pos_mask = (
            torch.abs(
                anchor_phenotype
                - s_phenotype
            )
            <=
            torch.abs(
                anchor_phenotype
                - t_phenotype
            )
        )

        positive = torch.where(
            pos_mask,
            s,
            t
        )
        positive_phenotype = torch.where(
            pos_mask,
            s_phenotype,
            t_phenotype
        )

        negative = torch.where(
            pos_mask,
            t,
            s
        )
        negative_phenotype = torch.where(
            pos_mask,
            t_phenotype,
            s_phenotype
        )

        r = (
            torch.abs(
                anchor_phenotype
                - negative_phenotype
            )
            /
            (
                torch.abs(
                    anchor_phenotype
                    - positive_phenotype
                )
                + 1e-2
            )
        )

        margin = torch.clip(
            self.margin * r,
            0.1,
            0.9
        )

        anchor = F.normalize(
            anchor,
            p=2,
            dim=1
        )
        positive = F.normalize(
            positive,
            p=2,
            dim=1
        )
        negative = F.normalize(
            negative,
            p=2,
            dim=1
        )

        d_pos = F.pairwise_distance(
            anchor,
            positive,
            p=2
        )
        d_neg = F.pairwise_distance(
            anchor,
            negative,
            p=2
        )

        losses = F.softplus(
            d_pos
            - d_neg
            + margin
        )

        return losses.mean()


# ============================================================
# Stage 1 model: uploaded TraitSpecificEncoderForRepGeno
# ============================================================

class ContrastiveBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        pool_size
    ):
        super().__init__()

        self.resnet = nn.Sequential(
            nn.BatchNorm1d(
                in_channels
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride
            ),
            nn.BatchNorm1d(
                out_channels
            ),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride
            ),
            nn.BatchNorm1d(
                out_channels
            )
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=pool_size,
            stride=pool_size
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.resnet(x)
        x = x + identity
        x = self.max_pool(x)
        return x


class TraitSpecificEncoderForRepGeno(nn.Module):
    def __init__(
        self,
        snp_size,
        stride=2,
        out_dim=128
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            ContrastiveBackbone(
                1, 8, 5,
                stride,
                stride
            ),
            ContrastiveBackbone(
                8, 16, 5,
                stride,
                stride
            ),
            ContrastiveBackbone(
                16, 32, 5,
                stride,
                stride
            ),
            nn.Flatten(),
        )

        # Use a dummy tensor only to determine the exact flattened size.
        # This preserves the uploaded architecture while avoiding shape
        # assumptions for Wheat2000 marker count.
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()

            dummy = torch.ones(
                2,
                1,
                snp_size
            )

            flat_dim = self.backbone(
                dummy
            ).shape[-1]

            self.backbone.train(
                was_training
            )

        self.embedding = nn.Linear(
            flat_dim,
            out_dim
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x

    def forward(
        self,
        anchor,
        positive,
        negative
    ):
        return (
            self.forward_once(anchor),
            self.forward_once(positive),
            self.forward_once(negative)
        )


# ============================================================
# Stage 2 MeNet: uploaded non-window architecture
# ============================================================

class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        pool_size
    ):
        super().__init__()

        self.resnet = nn.Sequential(
            nn.BatchNorm1d(
                in_channels
            ),
            nn.LeakyReLU(
                negative_slope=0.05
            ),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(
                out_channels
            ),
            nn.LeakyReLU(
                negative_slope=0.05
            ),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            ),
            nn.BatchNorm1d(
                out_channels
            )
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=pool_size,
            stride=pool_size
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.resnet(x)
        x = x + identity
        x = self.max_pool(x)
        return x


class VE(nn.Module):
    def __init__(
        self,
        snp_size,
        config
    ):
        super().__init__()

        self.resnet = nn.Sequential(
            nn.Conv1d(
                1,
                config["conv"][0],
                kernel_size=config["conv"][1],
                padding=config["conv"][1] // 2,
                stride=config["conv"][2]
            ),
            ResNetBlock(
                config["conv"][0],
                config["res1"][0],
                config["res1"][1],
                config["res1"][2]
            ),
            ResNetBlock(
                config["res1"][0],
                config["res2"][0],
                config["res2"][1],
                config["res2"][2]
            )
        )

        self.adaptive_pool = (
            nn.AdaptiveAvgPool1d(
                config["adaptive"]
            )
        )

        # Determine the flattened dimension from the actual architecture.
        # The uploaded model computes this analytically; dynamic inference is
        # equivalent but safer when adapting to a new SNP count.
        with torch.no_grad():
            was_training = self.resnet.training
            self.resnet.eval()

            dummy = torch.ones(
                2,
                1,
                snp_size
            )

            z = self.resnet(
                dummy
            )
            z = torch.permute(
                z,
                (0, 2, 1)
            )
            z = self.adaptive_pool(
                z
            )

            flat_dim = int(
                np.prod(
                    z.shape[1:]
                )
            )

            self.resnet.train(
                was_training
            )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                flat_dim,
                config["embedding_dim"][0]
            ),
            nn.BatchNorm1d(
                config["embedding_dim"][0]
            ),
            nn.LeakyReLU(
                negative_slope=0.05
            ),
            nn.Dropout(
                config["dropout"]
            ),
            nn.Linear(
                config["embedding_dim"][0],
                config["embedding_dim"][1]
            )
        )

    def forward(self, x):
        x = self.resnet(x)
        x = torch.permute(
            x,
            (0, 2, 1)
        )
        x = self.adaptive_pool(x)
        x = self.linear(x)
        return x


class RepGeno(nn.Module):
    def __init__(
        self,
        rep_size,
        config
    ):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                rep_size,
                config["embedding_dim"][0]
            ),
            nn.BatchNorm1d(
                config["embedding_dim"][0]
            ),
            nn.LeakyReLU(
                negative_slope=0.05
            ),
            nn.Dropout(
                config["dropout"]
            ),
            nn.Linear(
                config["embedding_dim"][0],
                config["embedding_dim"][1]
            ),
        )

    def forward(self, x):
        return self.linear(x)


class CrossInformationFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.field = nn.Sequential(
            nn.Linear(
                2 * dim,
                4 * dim
            ),
            nn.BatchNorm1d(
                4 * dim
            ),
            nn.LeakyReLU(
                negative_slope=0.05
            ),
            nn.Dropout(
                0.6
            ),
            nn.Linear(
                4 * dim,
                dim
            ),
        )

    def forward(self, x):
        return self.field(x)


class Fusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config[
            "VE_and_RepGeno_embedding_dim"
        ]

        self.f_ve = (
            CrossInformationFeatureFusion(
                dim
            )
        )

        self.f_repGeno = (
            CrossInformationFeatureFusion(
                dim
            )
        )

        self.output = nn.Sequential(
            nn.Linear(
                2 * dim,
                config["embedding_dim"]
            ),
            nn.BatchNorm1d(
                config["embedding_dim"]
            ),
            nn.LeakyReLU(
                negative_slope=0.05
            ),
            nn.Dropout(
                config["dropout"]
            ),
            nn.Linear(
                config["embedding_dim"],
                1
            ),
        )

    def forward(self, x, y):
        flow_field = torch.cat(
            [x, y],
            dim=1
        )

        x_f = (
            self.f_ve(
                flow_field
            )
            + x
        )

        y_f = (
            self.f_repGeno(
                flow_field
            )
            + y
        )

        return self.output(
            torch.cat(
                (x_f, y_f),
                dim=1
            )
        )


class MeNet(nn.Module):
    def __init__(
        self,
        snp_size,
        rep_size,
        config
    ):
        super().__init__()

        self.ve = VE(
            snp_size,
            config["VE"]
        )

        self.repGeno = RepGeno(
            rep_size,
            config["RepGeno"]
        )

        self.fusion = Fusion(
            config["output"]
        )

    def forward(self, snp, gr):
        x = self.ve(snp)
        y = self.repGeno(gr)
        return self.fusion(
            x,
            y
        )


# ============================================================
# Utility functions
# ============================================================

def count_parameters(model):
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


def state_dict_to_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v
        in model.state_dict().items()
    }


def safe_pearson(y_true, y_pred):
    if (
        len(y_true) < 2
        or np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    return float(
        pearsonr(
            y_true,
            y_pred
        )[0]
    )


# ============================================================
# Stage 1 training
# ============================================================

def evaluate_triplet_loss(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            (
                anchor,
                anchor_y,
                s,
                s_y,
                t,
                t_y
            ) = [
                x.to(
                    device,
                    non_blocking=True
                )
                for x in batch
            ]

            a, p, q = model(
                anchor,
                s,
                t
            )

            loss = criterion(
                a,
                anchor_y,
                p,
                s_y,
                q,
                t_y
            )

            total += float(
                loss.item()
            )
            n += 1

    return total / max(
        n,
        1
    )


def train_repgeno_encoder(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed
):
    set_seed(seed)

    train_ds = TraitTripletDataset(
        X_train,
        y_train
    )
    val_ds = TraitTripletDataset(
        X_val,
        y_val
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=REPGENO_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=REPGENO_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = TraitSpecificEncoderForRepGeno(
        snp_size=X_train.shape[1],
        stride=REPGENO_STRIDE,
        out_dim=REPGENO_OUT_DIM
    ).to(device)

    criterion = TripletLoss(
        margin=REPGENO_MARGIN
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=REPGENO_LR,
        weight_decay=ADAMW_WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None

    cuda_sync()
    start = time.perf_counter()

    for epoch in range(
        REPGENO_EPOCHS
    ):
        model.train()

        total = 0.0
        n = 0

        for batch in train_loader:
            (
                anchor,
                anchor_y,
                s,
                s_y,
                t,
                t_y
            ) = [
                x.to(
                    device,
                    non_blocking=True
                )
                for x in batch
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            a, p, q = model(
                anchor,
                s,
                t
            )

            loss = criterion(
                a,
                anchor_y,
                p,
                s_y,
                q,
                t_y
            )

            loss.backward()
            optimizer.step()

            total += float(
                loss.item()
            )
            n += 1

        train_loss = (
            total
            / max(n, 1)
        )

        val_loss = (
            evaluate_triplet_loss(
                model,
                val_loader,
                criterion,
                device
            )
        )

        scheduler.step(
            val_loss
        )

        print(
            f"    RepGeno epoch "
            f"{epoch + 1:02d}/{REPGENO_EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = state_dict_to_cpu(
                model
            )

    cuda_sync()
    elapsed = (
        time.perf_counter()
        - start
    )

    if best_state is None:
        raise RuntimeError(
            "RepGeno encoder did not produce a checkpoint."
        )

    model.load_state_dict(
        best_state
    )
    model.to(
        device
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    return (
        model,
        best_epoch,
        best_val,
        elapsed,
        total_params,
        trainable_params
    )


# ============================================================
# Genetic relatedness
# ============================================================

def encode_genotypes(
    encoder,
    X,
    device,
    batch_size=64
):
    ds = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32
        ).unsqueeze(1)
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    encoder.eval()

    outputs = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(
                device,
                non_blocking=True
            )

            z = encoder.forward_once(
                xb
            )

            outputs.append(
                z.cpu()
            )

    return torch.cat(
        outputs,
        dim=0
    ).float()


def normalize_rows(x):
    norm = torch.linalg.vector_norm(
        x,
        ord=2,
        dim=1,
        keepdim=True
    ).clamp_min(
        1e-12
    )
    return x / norm


def build_inductive_relatedness(
    encoder,
    X_train,
    X_val,
    X_test,
    device
):
    """
    Repository distance definition:
        normalize embedding rows
        Euclidean distance
        GR = 1 - distance / max_distance

    Strict CV modification:
        reference columns are TRAIN samples only.
    """

    z_train = normalize_rows(
        encode_genotypes(
            encoder,
            X_train,
            device
        )
    )

    z_val = normalize_rows(
        encode_genotypes(
            encoder,
            X_val,
            device
        )
    )

    z_test = normalize_rows(
        encode_genotypes(
            encoder,
            X_test,
            device
        )
    )

    # CPU cdist is sufficient for Wheat2000 and avoids extra GPU memory.
    d_train = torch.cdist(
        z_train,
        z_train,
        p=2
    )

    max_distance = float(
        d_train.max().item()
    )

    if max_distance <= 0:
        max_distance = 1.0

    gr_train = (
        1.0
        - d_train
        / max_distance
    )

    gr_val = (
        1.0
        - torch.cdist(
            z_val,
            z_train,
            p=2
        )
        / max_distance
    )

    gr_test = (
        1.0
        - torch.cdist(
            z_test,
            z_train,
            p=2
        )
        / max_distance
    )

    return (
        gr_train.numpy().astype(
            np.float32
        ),
        gr_val.numpy().astype(
            np.float32
        ),
        gr_test.numpy().astype(
            np.float32
        ),
        max_distance
    )


# ============================================================
# Stage 2 MeNet
# ============================================================

def make_menet_loader(
    X,
    GR,
    y,
    batch_size,
    shuffle,
    drop_last,
    seed
):
    snp_tensor = torch.as_tensor(
        X,
        dtype=torch.float32
    ).unsqueeze(1)

    gr_tensor = torch.as_tensor(
        GR,
        dtype=torch.float32
    ).unsqueeze(1)

    y_tensor = torch.as_tensor(
        y,
        dtype=torch.float32
    ).unsqueeze(1)

    ds = TensorDataset(
        snp_tensor,
        gr_tensor,
        y_tensor
    )

    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(
            seed
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def evaluate_menet(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    count = 0
    preds = []
    trues = []

    with torch.no_grad():
        for snp, gr, y in loader:
            snp = snp.to(
                device,
                non_blocking=True
            )
            gr = gr.to(
                device,
                non_blocking=True
            )
            y = y.to(
                device,
                non_blocking=True
            )

            pred = model(
                snp,
                gr
            )

            loss = criterion(
                pred,
                y
            )

            batch_n = snp.shape[0]

            total_loss += (
                float(loss.item())
                * batch_n
            )
            count += batch_n

            preds.append(
                pred.view(-1)
                .cpu()
                .numpy()
            )
            trues.append(
                y.view(-1)
                .cpu()
                .numpy()
            )

    pred = np.concatenate(
        preds
    )
    true = np.concatenate(
        trues
    )

    loss = (
        total_loss
        / max(count, 1)
    )

    r2 = (
        r2_score(
            true,
            pred
        )
        if len(true) >= 2
        else np.nan
    )

    return (
        loss,
        float(r2),
        pred,
        true
    )


def train_menet_model(
    X_train,
    GR_train,
    y_train,
    X_val,
    GR_val,
    y_val,
    device,
    seed
):
    set_seed(
        seed
    )

    train_loader = make_menet_loader(
        X_train,
        GR_train,
        y_train,
        MENET_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        seed=seed
    )

    val_loader = make_menet_loader(
        X_val,
        GR_val,
        y_val,
        MENET_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        seed=seed
    )

    model = MeNet(
        snp_size=X_train.shape[1],
        rep_size=GR_train.shape[1],
        config=MENET_PARAM
    ).to(
        device
    )

    criterion = nn.L1Loss().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MENET_LR,
        weight_decay=ADAMW_WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5
    )

    best_val_r2 = -np.inf
    best_val_loss = np.inf
    best_epoch = 0
    best_state = None

    epoch_times = []

    cuda_sync()
    start = time.perf_counter()

    for epoch in range(
        MENET_EPOCHS
    ):
        cuda_sync()
        epoch_start = time.perf_counter()

        model.train()

        train_total = 0.0
        train_n = 0

        for snp, gr, y in train_loader:
            snp = snp.to(
                device,
                non_blocking=True
            )
            gr = gr.to(
                device,
                non_blocking=True
            )
            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                snp,
                gr
            )

            loss = criterion(
                pred,
                y
            )

            loss.backward()
            optimizer.step()

            batch_n = snp.shape[0]

            train_total += (
                float(loss.item())
                * batch_n
            )
            train_n += batch_n

        train_loss = (
            train_total
            / max(train_n, 1)
        )

        (
            val_loss,
            val_r2,
            _,
            _
        ) = evaluate_menet(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_loss
        )

        cuda_sync()
        epoch_time = (
            time.perf_counter()
            - epoch_start
        )
        epoch_times.append(
            epoch_time
        )

        print(
            f"    MeNet epoch "
            f"{epoch + 1:02d}/{MENET_EPOCHS} | "
            f"train L1={train_loss:.6f} | "
            f"val L1={val_loss:.6f} | "
            f"val R2={val_r2:.6f}"
        )

        # Uploaded training code selects the checkpoint by validation R².
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = state_dict_to_cpu(
                model
            )

    cuda_sync()
    elapsed = (
        time.perf_counter()
        - start
    )

    if best_state is None:
        # Defensive fallback for pathological all-NaN validation R².
        best_state = state_dict_to_cpu(
            model
        )
        best_epoch = MENET_EPOCHS
        best_val_r2 = np.nan

    model.load_state_dict(
        best_state
    )
    model.to(
        device
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    return {
        "model":
            model,
        "best_epoch":
            best_epoch,
        "best_val_r2":
            float(best_val_r2),
        "best_val_loss":
            float(best_val_loss),
        "training_time_s":
            elapsed,
        "epoch_time_mean_s":
            float(
                np.mean(
                    epoch_times
                )
            ),
        "epoch_time_sd_s":
            float(
                np.std(
                    epoch_times,
                    ddof=1
                )
            )
            if len(epoch_times) > 1
            else 0.0,
        "total_params":
            total_params,
        "trainable_params":
            trainable_params,
    }


def predict_menet(
    model,
    X_test,
    GR_test,
    y_test,
    device
):
    loader = make_menet_loader(
        X_test,
        GR_test,
        y_test,
        MENET_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        seed=RANDOM_STATE
    )

    criterion = nn.L1Loss().to(
        device
    )

    cuda_sync()
    start = time.perf_counter()

    (
        loss,
        r2,
        pred,
        true
    ) = evaluate_menet(
        model,
        loader,
        criterion,
        device
    )

    cuda_sync()
    elapsed = (
        time.perf_counter()
        - start
    )

    ms_per_sample = (
        elapsed
        / len(pred)
        * 1000.0
    )

    return (
        pred,
        true,
        loss,
        r2,
        ms_per_sample
    )


# ============================================================
# One trait: strict 10-fold evaluation
# ============================================================

def run_trait_10fold(
    X,
    y,
    entries,
    trait_name,
    output_dir,
    random_state=42
):
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    n_samples, n_snps = X.shape

    outer_cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state
    )

    outer_splits = list(
        outer_cv.split(
            np.arange(
                n_samples
            )
        )
    )

    fold_membership = []
    fold_results = []
    all_predictions = []

    for fold_no, (
        outer_train_idx,
        test_idx
    ) in enumerate(
        outer_splits,
        start=1
    ):
        print(
            "\n"
            + "=" * 88
        )
        print(
            f"MeNet | Wheat2000 | "
            f"{trait_name} | "
            f"Fold {fold_no}/{N_SPLITS}"
        )
        print(
            f"Outer train={len(outer_train_idx)}, "
            f"test={len(test_idx)}"
        )
        print(
            "=" * 88
        )

        # ----------------------------------------------------
        # Inner validation split from outer training only
        # ----------------------------------------------------
        train_idx, val_idx = (
            train_test_split(
                outer_train_idx,
                test_size=INNER_VAL_FRAC,
                random_state=random_state,
                shuffle=True
            )
        )

        for idx in train_idx:
            fold_membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_no,
                "Sample_Index_0Based":
                    int(idx),
                "Original_Row_Index_0Based":
                    int(entries[idx]),
                "Set":
                    "Train"
            })

        for idx in val_idx:
            fold_membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_no,
                "Sample_Index_0Based":
                    int(idx),
                "Original_Row_Index_0Based":
                    int(entries[idx]),
                "Set":
                    "Validation"
            })

        for idx in test_idx:
            fold_membership.append({
                "Trait":
                    trait_name,
                "Fold":
                    fold_no,
                "Sample_Index_0Based":
                    int(idx),
                "Original_Row_Index_0Based":
                    int(entries[idx]),
                "Set":
                    "Test"
            })

        # ----------------------------------------------------
        # Fold-safe preprocessing
        # Fit only on INNER TRAIN, not validation/test.
        # ----------------------------------------------------
        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        X_train_imp = imputer.fit_transform(
            X[train_idx]
        ).astype(
            np.float32
        )

        X_val_imp = imputer.transform(
            X[val_idx]
        ).astype(
            np.float32
        )

        X_test_imp = imputer.transform(
            X[test_idx]
        ).astype(
            np.float32
        )

        # Match the existing Wheat2000 fair-comparison preprocessing:
        # fold-wise SNP scaling to [-1, 1]. Fit on inner train only.
        x_scaler = MinMaxScaler(
            feature_range=(-1, 1)
        )

        X_train = x_scaler.fit_transform(
            X_train_imp
        ).astype(
            np.float32
        )

        X_val = x_scaler.transform(
            X_val_imp
        ).astype(
            np.float32
        )

        X_test = x_scaler.transform(
            X_test_imp
        ).astype(
            np.float32
        )

        y_scaler = StandardScaler()

        y_train = (
            y_scaler.fit_transform(
                y[train_idx]
                .reshape(-1, 1)
            )
            .reshape(-1)
            .astype(np.float32)
        )

        y_val = (
            y_scaler.transform(
                y[val_idx]
                .reshape(-1, 1)
            )
            .reshape(-1)
            .astype(np.float32)
        )

        y_test = (
            y_scaler.transform(
                y[test_idx]
                .reshape(-1, 1)
            )
            .reshape(-1)
            .astype(np.float32)
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        set_seed(
            random_state
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(
                device
            )

        fold_start = time.perf_counter()

        # ----------------------------------------------------
        # Stage 1: phenotype-aware RepGeno encoder
        # ----------------------------------------------------
        print(
            "  Stage 1: trait-aware RepGeno encoder"
        )

        (
            encoder,
            repgeno_best_epoch,
            repgeno_best_val_loss,
            repgeno_time,
            repgeno_params,
            repgeno_trainable
        ) = train_repgeno_encoder(
            X_train,
            y_train,
            X_val,
            y_val,
            device,
            random_state
        )

        # ----------------------------------------------------
        # Build strict inductive genetic relatedness
        # ----------------------------------------------------
        print(
            "  Building train-reference genetic relatedness..."
        )

        gr_start = time.perf_counter()

        (
            GR_train,
            GR_val,
            GR_test,
            gr_max_distance
        ) = build_inductive_relatedness(
            encoder,
            X_train,
            X_val,
            X_test,
            device
        )

        gr_time = (
            time.perf_counter()
            - gr_start
        )

        # Encoder is no longer needed on GPU.
        del encoder
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"  GR shapes: "
            f"train={GR_train.shape}, "
            f"val={GR_val.shape}, "
            f"test={GR_test.shape}"
        )

        # ----------------------------------------------------
        # Stage 2: MeNet
        # ----------------------------------------------------
        print(
            "  Stage 2: MeNet prediction model"
        )

        train_info = train_menet_model(
            X_train,
            GR_train,
            y_train,
            X_val,
            GR_val,
            y_val,
            device,
            random_state
        )

        model = train_info[
            "model"
        ]

        (
            pred_scaled,
            true_scaled,
            test_l1,
            _,
            inference_ms
        ) = predict_menet(
            model,
            X_test,
            GR_test,
            y_test,
            device
        )

        # ----------------------------------------------------
        # Back to original phenotype units
        # ----------------------------------------------------
        y_pred = (
            y_scaler.inverse_transform(
                pred_scaled.reshape(
                    -1,
                    1
                )
            )
            .reshape(-1)
        )

        y_true = (
            y_scaler.inverse_transform(
                true_scaled.reshape(
                    -1,
                    1
                )
            )
            .reshape(-1)
        )

        pearson = safe_pearson(
            y_true,
            y_pred
        )

        r2 = float(
            r2_score(
                y_true,
                y_pred
            )
        )

        mae = float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        )

        total_fold_time = (
            time.perf_counter()
            - fold_start
        )

        peak_gpu_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 2)
            if torch.cuda.is_available()
            else np.nan
        )

        result = {
            "Trait":
                trait_name,
            "Fold":
                fold_no,
            "Train_N":
                len(train_idx),
            "Validation_N":
                len(val_idx),
            "Test_N":
                len(test_idx),
            "SNPs":
                n_snps,
            "RepGeno_Reference_N":
                GR_train.shape[1],
            "Pearson":
                pearson,
            "R2":
                r2,
            "MAE":
                mae,
            "RepGeno_Best_Epoch":
                repgeno_best_epoch,
            "RepGeno_Best_Val_Loss":
                repgeno_best_val_loss,
            "RepGeno_Training_Time_s":
                repgeno_time,
            "Relatedness_Time_s":
                gr_time,
            "MeNet_Best_Epoch":
                train_info[
                    "best_epoch"
                ],
            "MeNet_Best_Val_R2":
                train_info[
                    "best_val_r2"
                ],
            "MeNet_Best_Val_L1":
                train_info[
                    "best_val_loss"
                ],
            "MeNet_Training_Time_s":
                train_info[
                    "training_time_s"
                ],
            "MeNet_Epoch_Time_Mean_s":
                train_info[
                    "epoch_time_mean_s"
                ],
            "MeNet_Epoch_Time_SD_s":
                train_info[
                    "epoch_time_sd_s"
                ],
            "Total_Fold_Time_s":
                total_fold_time,
            "Peak_GPU_Memory_MB":
                peak_gpu_mb,
            "Inference_ms_per_sample":
                inference_ms,
            "RepGeno_Params_M":
                repgeno_params / 1e6,
            "RepGeno_Trainable_Params_M":
                repgeno_trainable / 1e6,
            "MeNet_Params_M":
                train_info[
                    "total_params"
                ] / 1e6,
            "MeNet_Trainable_Params_M":
                train_info[
                    "trainable_params"
                ] / 1e6,
            "GR_Max_Train_Distance":
                gr_max_distance,
        }

        fold_results.append(
            result
        )

        all_predictions.append(
            pd.DataFrame({
                "Trait":
                    trait_name,
                "Fold":
                    fold_no,
                "Sample_Index_0Based":
                    test_idx,
                "Original_Row_Index_0Based":
                    entries[test_idx],
                "Observed":
                    y_true,
                "Predicted":
                    y_pred,
            })
        )

        print(
            f"  TEST | "
            f"Pearson={pearson:.6f} | "
            f"R2={r2:.6f} | "
            f"MAE={mae:.6f}"
        )

        # ----------------------------------------------------
        # Per-fold checkpoints
        # ----------------------------------------------------
        pd.DataFrame(
            fold_membership
        ).to_csv(
            output_dir
            / "Wheat2000_10Fold_Membership_CHECKPOINT.csv",
            index=False
        )

        pd.DataFrame(
            fold_results
        ).to_csv(
            output_dir
            / (
                f"MeNet_Wheat2000_"
                f"{trait_name}_"
                "FoldResults_CHECKPOINT.csv"
            ),
            index=False
        )

        pd.concat(
            all_predictions,
            ignore_index=True
        ).to_csv(
            output_dir
            / (
                f"MeNet_Wheat2000_"
                f"{trait_name}_"
                "Predictions_CHECKPOINT.csv"
            ),
            index=False
        )

        del model
        del GR_train
        del GR_val
        del GR_test

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Final per-trait outputs
    # ========================================================
    fold_df = pd.DataFrame(
        fold_results
    )

    pred_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    membership_df = pd.DataFrame(
        fold_membership
    )

    def mean_sd(col):
        vals = pd.to_numeric(
            fold_df[col],
            errors="coerce"
        )
        return (
            float(vals.mean()),
            float(vals.std(ddof=1))
        )

    p_mean, p_sd = mean_sd(
        "Pearson"
    )
    r2_mean, r2_sd = mean_sd(
        "R2"
    )
    mae_mean, mae_sd = mean_sd(
        "MAE"
    )
    time_mean, time_sd = mean_sd(
        "Total_Fold_Time_s"
    )
    epoch_mean, epoch_sd = mean_sd(
        "MeNet_Epoch_Time_Mean_s"
    )
    gpu_mean, gpu_sd = mean_sd(
        "Peak_GPU_Memory_MB"
    )
    inf_mean, inf_sd = mean_sd(
        "Inference_ms_per_sample"
    )

    summary_df = pd.DataFrame([{
        "Model":
            "MeNet",
        "Dataset":
            "Wheat2000",
        "Trait":
            trait_name,
        "N":
            n_samples,
        "SNPs":
            n_snps,
        "Folds":
            N_SPLITS,
        "Pearson_Mean":
            p_mean,
        "Pearson_SD":
            p_sd,
        "R2_Mean":
            r2_mean,
        "R2_SD":
            r2_sd,
        "MAE_Mean":
            mae_mean,
        "MAE_SD":
            mae_sd,
        "Total_Fold_Time_Mean_s":
            time_mean,
        "Total_Fold_Time_SD_s":
            time_sd,
        "MeNet_Epoch_Time_Mean_s":
            epoch_mean,
        "MeNet_Epoch_Time_SD_s":
            epoch_sd,
        "Peak_GPU_Memory_Mean_MB":
            gpu_mean,
        "Peak_GPU_Memory_SD_MB":
            gpu_sd,
        "Inference_Mean_ms_per_sample":
            inf_mean,
        "Inference_SD_ms_per_sample":
            inf_sd,
        "Random_State":
            random_state,
        "Inner_Validation_Fraction":
            INNER_VAL_FRAC,
        "RepGeno_Epochs":
            REPGENO_EPOCHS,
        "RepGeno_Batch_Size":
            REPGENO_BATCH_SIZE,
        "RepGeno_LR":
            REPGENO_LR,
        "RepGeno_Margin":
            REPGENO_MARGIN,
        "MeNet_Epochs":
            MENET_EPOCHS,
        "MeNet_Batch_Size":
            MENET_BATCH_SIZE,
        "MeNet_LR":
            MENET_LR,
        "MeNet_Loss":
            "L1",
        "Windows":
            False,
        "IG_During_Training":
            False,
    }])

    membership_df.to_csv(
        output_dir
        / "Wheat2000_10Fold_Membership.csv",
        index=False
    )

    fold_df.to_csv(
        output_dir
        / (
            f"MeNet_Wheat2000_"
            f"{trait_name}_"
            "FoldResults.csv"
        ),
        index=False
    )

    pred_df.to_csv(
        output_dir
        / (
            f"MeNet_Wheat2000_"
            f"{trait_name}_"
            "Predictions.csv"
        ),
        index=False
    )

    summary_df.to_csv(
        output_dir
        / (
            f"MeNet_Wheat2000_"
            f"{trait_name}_"
            "Summary.csv"
        ),
        index=False
    )

    print(
        "\n"
        + "-" * 88
    )
    print(
        f"{trait_name} FINAL"
    )
    print(
        f"Pearson r = "
        f"{p_mean:.6f} ± {p_sd:.6f}"
    )
    print(
        f"R2        = "
        f"{r2_mean:.6f} ± {r2_sd:.6f}"
    )
    print(
        f"MAE       = "
        f"{mae_mean:.6f} ± {mae_sd:.6f}"
    )
    print(
        "-" * 88
    )

    return fold_df, summary_df


# ============================================================
# All seven Wheat2000 traits
# ============================================================

def run_all_traits(
    genotype_file,
    phenotype_files,
    output_root,
    random_state=42
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(random_state)

    X_full, marker_names = load_wheat2000_genotype(genotype_file)

    all_summaries = []
    failed = []

    for trait_i, (trait_name, phenotype_file) in enumerate(phenotype_files.items(), start=1):
        print('\n\n' + '#' * 88)
        print(f'Trait {trait_i}/{len(phenotype_files)}: {trait_name}')
        print(f'Phenotype file: {phenotype_file}')
        print('#' * 88)

        try:
            y_raw = load_wheat2000_phenotype(phenotype_file)
            X, y, sample_ids = align_wheat2000_by_row_order(X_full, y_raw)

            trait_dir = output_root / trait_name
            trait_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({
                'Sample_Index_0Based': np.arange(len(sample_ids)),
                'Original_Row_Index_0Based': sample_ids,
            }).to_csv(
                trait_dir / f'Wheat2000_{trait_name}_AlignedSamples.csv',
                index=False
            )

            _, summary_df = run_trait_10fold(
                X=X,
                y=y,
                entries=sample_ids,
                trait_name=trait_name,
                output_dir=trait_dir,
                random_state=random_state
            )

            summary_df['Phenotype_File'] = str(phenotype_file)
            all_summaries.append(summary_df)

            pd.concat(all_summaries, ignore_index=True).to_csv(
                output_root / 'MeNet_Wheat2000_AllTraits_Summary_CHECKPOINT.csv',
                index=False
            )

        except Exception as exc:
            print(f'[ERROR] {trait_name}: {type(exc).__name__}: {exc}')
            failed.append({
                'Trait': trait_name,
                'Phenotype_File': str(phenotype_file),
                'Reason': f'{type(exc).__name__}: {exc}',
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(
            output_root / 'MeNet_Wheat2000_AllTraits_Summary.csv',
            index=False
        )
        print('\n' + '=' * 88)
        print('MeNet Wheat2000 ALL-TRAIT SUMMARY')
        print('=' * 88)
        print(combined[[
            'Trait','Pearson_Mean','Pearson_SD','R2_Mean','R2_SD','MAE_Mean','MAE_SD'
        ]].to_string(index=False))
    else:
        combined = pd.DataFrame()

    if failed:
        pd.DataFrame(failed).to_csv(
            output_root / 'MeNet_Wheat2000_FailedTraits.csv',
            index=False
        )

    return combined


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    WHEAT_DIR = Path(
        r"E:\VGMP\wheat2000\SNP_origin"
    )

    GENOTYPE_FILE = WHEAT_DIR / "2000gene.pkl"

    # Verify local names for testw/length/width/hardness before a full run.
    # tkw and Prot follow the filenames used in the existing Wheat2000 scripts.
    PHENOTYPE_FILES = {
        # "tkw": WHEAT_DIR / "2000_1_phe_tkw.txt",
        # "testw": WHEAT_DIR / "2000_2_phe_testw.txt",
        # "length": WHEAT_DIR / "2000_3_phe_length.txt",
        # "width": WHEAT_DIR / "2000_4_phe_width.txt",
        "hardness": WHEAT_DIR / "2000_5_phe_Hard.txt",
        # "Prot": WHEAT_DIR / "2000_6_phe_Prot.txt",
    }

    OUTPUT_ROOT = (
        Path(__file__).resolve().parent
        / "results_menet_wheat2000_all_traits"
    )

    if not GENOTYPE_FILE.exists():
        raise FileNotFoundError(
            "Wheat2000 genotype file not found:\n"
            f"{GENOTYPE_FILE}"
        )

    run_all_traits(
        genotype_file=GENOTYPE_FILE,
        phenotype_files=PHENOTYPE_FILES,
        output_root=OUTPUT_ROOT,
        random_state=RANDOM_STATE
    )
