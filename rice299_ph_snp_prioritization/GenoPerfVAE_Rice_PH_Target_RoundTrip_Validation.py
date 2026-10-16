#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GenoPerfVAE_Rice_PH_Target_RoundTrip_Validation.py

Purpose
-------
Use the final full-data Rice299-PH post-hoc GenoPerfVAE checkpoint to perform
PH-directed latent-space target optimization and strict round-trip validation:

    elite latent reference
        -> optimized latent target z*
        -> decoder continuous genotype
        -> per-marker projection to an OBSERVED genotype state
        -> re-encode with the full GenoPerfVAE
        -> final predicted PH

This script is designed specifically for computational validation of the
post-hoc target-genotype analysis. It does NOT estimate generalization
performance and does NOT claim biological breedability, haplotype preservation,
causality, or functional SNP effects.

Default primary setting
-----------------------
- Elite reference: observed PH >= empirical Q95.
- Target phenotype: empirical PH Q95.
- Start latent vector: elite posterior-mean centroid.
- Target-directed objective in standardized phenotype space.
- Latent constraint: soft penalty outside +/- 3 elite latent SD.
- Final discrete genotype: each SNP is projected to the closest genotype state
  actually observed at that SNP in the aligned/imputed Rice299 reference data.

Why projection instead of round(0,1,2)?
---------------------------------------
The current Rice299 pipeline retains -1 as a VALID genotype value. Therefore,
global rounding/clipping to 0/1/2 would be inconsistent with the final training
representation. Per-marker projection avoids generating a state that was never
observed at that marker in the reference population.

Recommended interpretation
--------------------------
"unconstrained in silico target genotype" or "computational target genotype",
NOT "ideal genotype", "breedable genotype", or "functional variant design".
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
import torch.optim as optim

warnings.filterwarnings("ignore")

try:
    from performer_pytorch import Performer
except ImportError as exc:
    raise ImportError(
        "performer-pytorch is required. Install it with:\n"
        "    pip install performer-pytorch"
    ) from exc


# =============================================================================
# 0. File names / defaults
# =============================================================================

CHECKPOINT_BASENAME = "GenoPerfVAE_Rice_PH_Final_PostHoc.pth"
GENOTYPE_FILENAME = (
    "MET_crfilt_.90_outliers_removed_for_RRBlup_line_corrected.csv"
)
PHENOTYPE_FILENAME = "rice299_PH.csv"

DEFAULT_SEED = 42
DEFAULT_ELITE_QUANTILE = 0.95
DEFAULT_TARGET_MODE = "q95"

DEFAULT_OPT_STEPS = 500
DEFAULT_OPT_LR = 0.01

# Target-directed latent objective:
# L = target_loss + lambda_dist * distance_penalty
#                 + lambda_out * out_of_manifold_penalty
DEFAULT_LAMBDA_DIST = 0.05
DEFAULT_LAMBDA_OUT = 0.50
DEFAULT_LATENT_Z_LIMIT = 3.0

OUTPUT_PREFIX = "GenoPerfVAE_Rice_PH_Target_RoundTrip"


# =============================================================================
# 1. Reproducibility helpers
# =============================================================================

def set_seed_fixed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. Model architecture: MUST match the final post-hoc training script
# =============================================================================

class AdaptiveChunkedSNP_Performer(nn.Module):
    def __init__(
        self,
        snp_dim,
        target_chunk_size,
        performer_dim,
        dropout,
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
        self.flatten_readout_dim = (
            self.num_chunks * self.chunk_size
        )

        self.padding_layer = (
            nn.ZeroPad1d((0, self.pad_n))
            if self.pad_n > 0
            else None
        )

        self.chunk_norm = nn.LayerNorm(
            self.chunk_size
        )

        hidden_dim = max(
            self.chunk_size // 4,
            32,
        )

        self.chunk_weight_net = nn.Sequential(
            nn.Linear(
                self.chunk_size,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        self.chunk_scale = nn.Parameter(
            torch.tensor(0.1)
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_chunks,
                self.chunk_size,
            )
        )

        nn.init.trunc_normal_(
            self.pos_embedding,
            std=0.02,
        )

        self.pos_dropout = nn.Dropout(
            dropout
        )

        num_heads = 4

        if self.chunk_size % num_heads != 0:
            raise ValueError(
                f"chunk_size={self.chunk_size} must be divisible "
                f"by num_heads={num_heads}."
            )

        head_dim = (
            self.chunk_size
            // num_heads
        )

        self.performer = Performer(
            dim=self.chunk_size,
            depth=2,
            heads=num_heads,
            dim_head=head_dim,
            ff_dropout=dropout,
            attn_dropout=dropout,
            causal=False,
        )

        # Match the final training script exactly.
        with torch.random.fork_rng(
            devices=[]
        ):
            self.output_projection = nn.Sequential(
                nn.Linear(
                    self.flatten_readout_dim,
                    performer_dim,
                ),
                nn.LayerNorm(
                    performer_dim
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
            )

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

        chunk_scores = self.chunk_weight_net(
            x
        )

        chunk_weights = torch.softmax(
            chunk_scores,
            dim=1,
        )

        scaled_weights = (
            chunk_weights
            * self.num_chunks
        )

        self.last_chunk_weights = (
            chunk_weights.detach()
        )

        centered_weights = (
            scaled_weights
            - 1.0
        )

        x = x * (
            1.0
            + self.chunk_scale
            * centered_weights
        )

        x = (
            x
            + self.pos_embedding
        )

        x = self.pos_dropout(
            x
        )

        x = self.performer(
            x
        )

        x_flat = x.flatten(
            start_dim=1
        )

        return self.output_projection(
            x_flat
        )


class AdaptiveChunkedSNP_Performer_Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        snp_dim,
        target_chunk_size,
        performer_dim,
        dropout,
    ):
        super().__init__()

        self.snp_dim = int(
            snp_dim
        )

        self.chunk_size = int(
            target_chunk_size
        )

        remainder = (
            self.snp_dim
            % self.chunk_size
        )

        self.pad_n = (
            self.chunk_size
            - remainder
            if remainder > 0
            else 0
        )

        self.num_chunks = (
            self.snp_dim
            + self.pad_n
        ) // self.chunk_size

        self.latent_proj = nn.Sequential(
            nn.Linear(
                latent_dim,
                performer_dim,
            ),
            nn.BatchNorm1d(
                performer_dim
            ),
            nn.GELU(),
        )

        self.to_seq = nn.Sequential(
            nn.Linear(
                performer_dim,
                self.num_chunks
                * self.chunk_size,
            ),
            nn.BatchNorm1d(
                self.num_chunks
                * self.chunk_size
            ),
            nn.GELU(),
        )

        num_heads = 4

        if self.chunk_size % num_heads != 0:
            raise ValueError(
                f"chunk_size={self.chunk_size} must be divisible "
                f"by num_heads={num_heads}."
            )

        head_dim = (
            self.chunk_size
            // num_heads
        )

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
            self.chunk_size,
        )

    def forward(self, z):
        batch_size = (
            z.size(0)
        )

        h = self.latent_proj(
            z
        )

        h_seq = self.to_seq(
            h
        ).view(
            batch_size,
            self.num_chunks,
            self.chunk_size,
        )

        h_decoded = self.performer(
            h_seq
        )

        x_flat = self.final_head(
            h_decoded
        ).view(
            batch_size,
            -1,
        )

        return x_flat[
            :,
            :self.snp_dim,
        ]


class EnhancedGeneticModel(nn.Module):
    def __init__(
        self,
        snp_dim,
        latent_dim,
        performer_readout_dim,
        decoder_performer_dim,
        performer_latent_dim,
        linear_latent_dim,
        dropout,
        chunk_size,
    ):
        super().__init__()

        self.latent_dim = int(
            latent_dim
        )

        self.performer_readout_dim = int(
            performer_readout_dim
        )

        self.decoder_performer_dim = int(
            decoder_performer_dim
        )

        self.performer_latent_dim = int(
            performer_latent_dim
        )

        self.linear_latent_dim = int(
            linear_latent_dim
        )

        if (
            self.performer_latent_dim
            + self.linear_latent_dim
            != self.latent_dim
        ):
            raise ValueError(
                "Branch latent dimensions do not sum to "
                "total latent dimension."
            )

        self.snp_performer = (
            AdaptiveChunkedSNP_Performer(
                snp_dim=snp_dim,
                target_chunk_size=chunk_size,
                performer_dim=self.performer_readout_dim,
                dropout=dropout,
            )
        )

        self.linear_shortcut = nn.Sequential(
            nn.Linear(
                snp_dim,
                2048,
            ),
            nn.BatchNorm1d(
                2048
            ),
            nn.GELU(),

            nn.Linear(
                2048,
                512,
            ),
            nn.BatchNorm1d(
                512
            ),
            nn.GELU(),

            nn.Dropout(
                dropout
            ),
            nn.Linear(
                512,
                64,
            ),
        )

        with torch.random.fork_rng(
            devices=[]
        ):
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
            nn.Linear(
                self.latent_dim,
                64,
            ),
            nn.BatchNorm1d(
                64
            ),
            nn.GELU(),
            nn.Linear(
                64,
                1,
            ),
        )

        self.decoder = (
            AdaptiveChunkedSNP_Performer_Decoder(
                latent_dim=self.latent_dim,
                snp_dim=snp_dim,
                target_chunk_size=chunk_size,
                performer_dim=self.decoder_performer_dim,
                dropout=dropout,
            )
        )

    def reparameterize(
        self,
        mu,
        logvar,
    ):
        if self.training:
            return (
                mu
                + torch.randn_like(
                    mu
                )
                * torch.exp(
                    0.5
                    * logvar
                )
            )

        return mu

    def encode(
        self,
        x,
    ):
        h_performer = (
            self.snp_performer(
                x
            )
        )

        h_linear = (
            self.linear_shortcut(
                x
            )
        )

        mu_performer = (
            self.fc_mu_performer(
                h_performer
            )
        )

        logvar_performer = (
            self.fc_logvar_performer(
                h_performer
            )
        )

        mu_linear = (
            self.fc_mu_linear(
                h_linear
            )
        )

        logvar_linear = (
            self.fc_logvar_linear(
                h_linear
            )
        )

        mu = torch.cat(
            [
                mu_performer,
                mu_linear,
            ],
            dim=1,
        )

        logvar = torch.cat(
            [
                logvar_performer,
                logvar_linear,
            ],
            dim=1,
        )

        return (
            mu,
            logvar,
        )

    def forward(
        self,
        x,
    ):
        (
            mu,
            logvar,
        ) = self.encode(
            x
        )

        z = self.reparameterize(
            mu,
            logvar,
        )

        y_p = self.predictor(
            z
        )

        x_r = self.decoder(
            z
        )

        return (
            x_r,
            y_p,
            mu,
            logvar,
        )


# =============================================================================
# 3. Checkpoint / path utilities
# =============================================================================

def torch_load_checkpoint(
    path,
    device,
):
    """
    Explicit weights_only=False is appropriate here because this is the
    user's own locally generated checkpoint containing metadata plus tensors.
    """
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def resolve_checkpoint(
    script_dir,
    explicit_checkpoint=None,
):
    if explicit_checkpoint is not None:
        path = Path(
            explicit_checkpoint
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        return path

    candidates = [
        script_dir
        / "results_Rice_PH_Final_PostHoc"
        / CHECKPOINT_BASENAME,

        script_dir
        / CHECKPOINT_BASENAME,

        Path(
            r"E:\VGMP\Extended Comparison\GenoPerfVAE"
        )
        / "results_Rice_PH_Final_PostHoc"
        / CHECKPOINT_BASENAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        "Could not locate the post-hoc checkpoint.\n"
        f"Checked:\n{checked}\n"
        "Use --checkpoint to specify it explicitly."
    )


def resolve_data_files(
    script_dir,
    explicit_data_dir=None,
    explicit_genotype=None,
    explicit_phenotype=None,
):
    if explicit_genotype is not None:
        genotype_file = Path(
            explicit_genotype
        )
    else:
        genotype_file = None

    if explicit_phenotype is not None:
        phenotype_file = Path(
            explicit_phenotype
        )
    else:
        phenotype_file = None

    if (
        genotype_file is not None
        and phenotype_file is not None
    ):
        if not genotype_file.exists():
            raise FileNotFoundError(
                f"Genotype file not found: {genotype_file}"
            )

        if not phenotype_file.exists():
            raise FileNotFoundError(
                f"Phenotype file not found: {phenotype_file}"
            )

        return (
            genotype_file,
            phenotype_file,
        )

    if explicit_data_dir is not None:
        candidates = [
            Path(
                explicit_data_dir
            )
        ]
    else:
        candidates = [
            script_dir / "data",
            script_dir.parent / "data",
            script_dir.parent
            / "data"
            / "rice299",
            Path(
                r"E:\VGMP\rice299"
            ),
            script_dir,
        ]

    for folder in candidates:
        g = (
            genotype_file
            if genotype_file is not None
            else (
                folder
                / GENOTYPE_FILENAME
            )
        )

        p = (
            phenotype_file
            if phenotype_file is not None
            else (
                folder
                / PHENOTYPE_FILENAME
            )
        )

        if (
            g.exists()
            and p.exists()
        ):
            return (
                g,
                p,
            )

    checked = "\n".join(
        f"  - {folder}"
        for folder in candidates
    )

    raise FileNotFoundError(
        "Could not locate the Rice299 genotype/PH files.\n"
        f"Checked folders:\n{checked}\n"
        "Use --data-dir or explicit --genotype / --phenotype."
    )


# =============================================================================
# 4. Restore data exactly in checkpoint SNP/sample order
# =============================================================================

def load_reference_data(
    genotype_file,
    phenotype_file,
    checkpoint,
):
    snp_names = list(
        map(
            str,
            checkpoint[
                "snp_names"
            ],
        )
    )

    aligned_entries = list(
        map(
            str,
            checkpoint[
                "aligned_entries"
            ],
        )
    )

    imputer_stats = (
        checkpoint[
            "imputer_statistics"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
        .reshape(-1)
    )

    if len(
        imputer_stats
    ) != len(
        snp_names
    ):
        raise RuntimeError(
            "Checkpoint imputer statistics and SNP names "
            "have different lengths."
        )

    geno_df = pd.read_csv(
        genotype_file
    )

    if "Entry" not in geno_df.columns:
        raise KeyError(
            "Genotype file must contain 'Entry'."
        )

    geno_df[
        "Entry"
    ] = (
        geno_df[
            "Entry"
        ]
        .astype(str)
        .str.strip()
    )

    if geno_df[
        "Entry"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Entry IDs found in genotype data."
        )

    missing_snps = [
        snp
        for snp in snp_names
        if snp not in geno_df.columns
    ]

    if missing_snps:
        raise KeyError(
            "The current genotype file does not contain "
            "all checkpoint SNPs. Examples: "
            f"{missing_snps[:10]}"
        )

    geno_df = (
        geno_df
        .set_index(
            "Entry"
        )
    )

    missing_entries = [
        entry
        for entry in aligned_entries
        if entry not in geno_df.index
    ]

    if missing_entries:
        raise KeyError(
            "The current genotype file does not contain "
            "all checkpoint-aligned samples. Examples: "
            f"{missing_entries[:10]}"
        )

    X_raw = (
        geno_df
        .loc[
            aligned_entries,
            snp_names,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    # IMPORTANT:
    # Only true NaN is imputed. -1 remains valid.
    X_imputed = X_raw.copy()

    nan_rows, nan_cols = np.where(
        np.isnan(
            X_imputed
        )
    )

    if len(
        nan_rows
    ) > 0:
        X_imputed[
            nan_rows,
            nan_cols,
        ] = imputer_stats[
            nan_cols
        ]

    if np.isnan(
        X_imputed
    ).any():
        raise RuntimeError(
            "NaN remains after restoring checkpoint imputation."
        )

    phe_df = pd.read_csv(
        phenotype_file
    )

    if (
        "Entry"
        not in phe_df.columns
        or "PH"
        not in phe_df.columns
    ):
        raise KeyError(
            "Phenotype file must contain 'Entry' and 'PH'."
        )

    phe_df[
        "Entry"
    ] = (
        phe_df[
            "Entry"
        ]
        .astype(str)
        .str.strip()
    )

    phe_df[
        "PH"
    ] = pd.to_numeric(
        phe_df[
            "PH"
        ],
        errors="coerce",
    )

    if phe_df[
        "Entry"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Entry IDs found in phenotype data."
        )

    phe_map = (
        phe_df
        .set_index(
            "Entry"
        )[
            "PH"
        ]
    )

    missing_ph = [
        entry
        for entry in aligned_entries
        if (
            entry
            not in phe_map.index
            or not np.isfinite(
                phe_map.loc[
                    entry
                ]
            )
        )
    ]

    if missing_ph:
        raise KeyError(
            "Missing PH values for checkpoint-aligned samples. "
            f"Examples: {missing_ph[:10]}"
        )

    y = (
        phe_map
        .loc[
            aligned_entries
        ]
        .to_numpy(
            dtype=np.float32
        )
        .reshape(
            -1,
            1,
        )
    )

    # Verify phenotype reference saved during training.
    ref = checkpoint.get(
        "phenotype_reference",
        {},
    )

    if "q95" in ref:
        q95_now = float(
            np.quantile(
                y.reshape(-1),
                0.95,
            )
        )

        if not np.isclose(
            q95_now,
            float(
                ref[
                    "q95"
                ]
            ),
            atol=1e-5,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "Current phenotype data do not reproduce checkpoint Q95. "
                "Please verify that the same Rice299 release is being used."
            )

    return (
        X_raw,
        X_imputed,
        y,
        np.asarray(
            aligned_entries,
            dtype=object,
        ),
        snp_names,
    )


# =============================================================================
# 5. Model / scaling helpers
# =============================================================================

def build_model_from_checkpoint(
    checkpoint,
    device,
):
    cfg = checkpoint[
        "model_config"
    ]

    model = EnhancedGeneticModel(
        snp_dim=int(
            cfg[
                "snp_dim"
            ]
        ),
        latent_dim=int(
            cfg[
                "latent_dim"
            ]
        ),
        performer_readout_dim=int(
            cfg[
                "performer_readout_dim"
            ]
        ),
        decoder_performer_dim=int(
            cfg[
                "decoder_performer_dim"
            ]
        ),
        performer_latent_dim=int(
            cfg[
                "performer_latent_dim"
            ]
        ),
        linear_latent_dim=int(
            cfg[
                "linear_latent_dim"
            ]
        ),
        dropout=float(
            cfg[
                "dropout"
            ]
        ),
        chunk_size=int(
            cfg[
                "chunk_size"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def scaled_to_real(
    y_scaled,
    scaler_mean,
    scaler_scale,
):
    return (
        np.asarray(
            y_scaled,
            dtype=np.float64,
        )
        * float(
            scaler_scale
        )
        + float(
            scaler_mean
        )
    )


def real_to_scaled(
    y_real,
    scaler_mean,
    scaler_scale,
):
    return (
        np.asarray(
            y_real,
            dtype=np.float64,
        )
        - float(
            scaler_mean
        )
    ) / float(
        scaler_scale
    )


@torch.no_grad()
def encode_all_mu(
    model,
    X_imputed,
    device,
    batch_size=64,
):
    mus = []
    logvars = []

    for start in range(
        0,
        len(
            X_imputed
        ),
        batch_size,
    ):
        stop = min(
            start
            + batch_size,
            len(
                X_imputed
            ),
        )

        bx = torch.tensor(
            X_imputed[
                start:stop
            ],
            dtype=torch.float32,
            device=device,
        )

        (
            mu,
            logvar,
        ) = model.encode(
            bx
        )

        mus.append(
            mu.detach()
            .cpu()
            .numpy()
        )

        logvars.append(
            logvar.detach()
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(
            mus,
            axis=0,
        ),
        np.concatenate(
            logvars,
            axis=0,
        ),
    )


# =============================================================================
# 6. Target / elite definitions
# =============================================================================

def resolve_target_value(
    y,
    mode,
):
    y_flat = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    mode = mode.lower()

    if mode == "q90":
        return float(
            np.quantile(
                y_flat,
                0.90,
            )
        )

    if mode == "q95":
        return float(
            np.quantile(
                y_flat,
                0.95,
            )
        )

    if mode in {
        "q97.5",
        "q975",
    }:
        return float(
            np.quantile(
                y_flat,
                0.975,
            )
        )

    if mode == "max":
        return float(
            np.max(
                y_flat
            )
        )

    raise ValueError(
        f"Unknown target mode: {mode}"
    )


def define_elite_indices(
    y,
    elite_quantile,
):
    y_flat = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    threshold = float(
        np.quantile(
            y_flat,
            elite_quantile,
        )
    )

    idx = np.where(
        y_flat
        >= threshold
    )[0]

    if len(
        idx
    ) < 2:
        raise ValueError(
            "Elite definition produced fewer than two samples. "
            "Use a lower elite quantile."
        )

    return (
        idx,
        threshold,
    )


# =============================================================================
# 7. Latent optimization
# =============================================================================

def optimize_latent_target(
    model,
    mu_all,
    elite_indices,
    target_real,
    scaler_mean,
    scaler_scale,
    device,
    steps,
    lr,
    lambda_dist,
    lambda_out,
    latent_z_limit,
):
    elite_mu_np = mu_all[
        elite_indices
    ]

    elite_mu_mean_np = (
        elite_mu_np.mean(
            axis=0,
            keepdims=True,
        )
        .astype(
            np.float32
        )
    )

    elite_mu_std_np = (
        elite_mu_np.std(
            axis=0,
            ddof=1,
            keepdims=True,
        )
        .astype(
            np.float32
        )
    )

    # Numerical floor only; does not change well-varying latent dimensions.
    elite_mu_std_np = np.maximum(
        elite_mu_std_np,
        1e-6,
    )

    z_start = torch.tensor(
        elite_mu_mean_np,
        dtype=torch.float32,
        device=device,
    )

    elite_mu_mean = torch.tensor(
        elite_mu_mean_np,
        dtype=torch.float32,
        device=device,
    )

    elite_mu_std = torch.tensor(
        elite_mu_std_np,
        dtype=torch.float32,
        device=device,
    )

    target_scaled_value = float(
        real_to_scaled(
            target_real,
            scaler_mean,
            scaler_scale,
        )
    )

    target_scaled = torch.tensor(
        [[
            target_scaled_value
        ]],
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        start_pred_scaled = (
            model.predictor(
                z_start
            )
        )

        start_pred_real = float(
            scaled_to_real(
                start_pred_scaled
                .detach()
                .cpu()
                .numpy()
                .item(),
                scaler_mean,
                scaler_scale,
            )
        )

    z_opt = (
        z_start
        .clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    optimizer = optim.Adam(
        [
            z_opt
        ],
        lr=lr,
    )

    best = None
    trace_rows = []

    for step in range(
        1,
        steps + 1,
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        y_pred_scaled = (
            model.predictor(
                z_opt
            )
        )

        # Target-directed rather than open-ended maximization:
        # controls overshoot and makes the target explicit.
        target_loss = torch.mean(
            (
                y_pred_scaled
                - target_scaled
            ) ** 2
        )

        # Mean squared displacement from the elite latent centroid.
        distance_penalty = torch.mean(
            (
                z_opt
                - z_start
            ) ** 2
        )

        z_score = (
            z_opt
            - elite_mu_mean
        ) / elite_mu_std

        # Soft squared penalty only beyond +/- latent_z_limit SD.
        excess = torch.relu(
            torch.abs(
                z_score
            )
            - latent_z_limit
        )

        out_penalty = torch.mean(
            excess ** 2
        )

        total_loss = (
            target_loss
            + lambda_dist
            * distance_penalty
            + lambda_out
            * out_penalty
        )

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_pred_scaled = (
                model.predictor(
                    z_opt
                )
            )

            current_pred_real = float(
                scaled_to_real(
                    current_pred_scaled
                    .detach()
                    .cpu()
                    .numpy()
                    .item(),
                    scaler_mean,
                    scaler_scale,
                )
            )

            max_abs_zscore = float(
                torch.max(
                    torch.abs(
                        z_score
                    )
                )
                .detach()
                .cpu()
                .item()
            )

            row = {
                "Step": step,
                "Total_Loss": float(
                    total_loss
                    .detach()
                    .cpu()
                    .item()
                ),
                "Target_Loss": float(
                    target_loss
                    .detach()
                    .cpu()
                    .item()
                ),
                "Distance_Penalty": float(
                    distance_penalty
                    .detach()
                    .cpu()
                    .item()
                ),
                "Out_Of_Manifold_Penalty": float(
                    out_penalty
                    .detach()
                    .cpu()
                    .item()
                ),
                "Predicted_PH": current_pred_real,
                "Target_PH": float(
                    target_real
                ),
                "Absolute_Target_Error": abs(
                    current_pred_real
                    - float(
                        target_real
                    )
                ),
                "Max_Abs_Elite_ZScore": max_abs_zscore,
            }

            trace_rows.append(
                row
            )

            score_tuple = (
                row[
                    "Absolute_Target_Error"
                ],
                row[
                    "Out_Of_Manifold_Penalty"
                ],
                row[
                    "Distance_Penalty"
                ],
            )

            if (
                best is None
                or score_tuple
                < best[
                    "score_tuple"
                ]
            ):
                best = {
                    "score_tuple": (
                        score_tuple
                    ),
                    "step": step,
                    "z": (
                        z_opt
                        .detach()
                        .clone()
                    ),
                    "pred_real": (
                        current_pred_real
                    ),
                    "pred_scaled": float(
                        current_pred_scaled
                        .detach()
                        .cpu()
                        .numpy()
                        .item()
                    ),
                    "max_abs_zscore": (
                        max_abs_zscore
                    ),
                    "target_loss": (
                        row[
                            "Target_Loss"
                        ]
                    ),
                    "distance_penalty": (
                        row[
                            "Distance_Penalty"
                        ]
                    ),
                    "out_penalty": (
                        row[
                            "Out_Of_Manifold_Penalty"
                        ]
                    ),
                }

    if best is None:
        raise RuntimeError(
            "Latent optimization did not produce a valid result."
        )

    trace_df = pd.DataFrame(
        trace_rows
    )

    return {
        "z_start": (
            z_start
            .detach()
            .clone()
        ),
        "z_best": (
            best[
                "z"
            ]
        ),
        "start_pred_real": (
            start_pred_real
        ),
        "best_pred_real": (
            best[
                "pred_real"
            ]
        ),
        "best_pred_scaled": (
            best[
                "pred_scaled"
            ]
        ),
        "best_step": (
            best[
                "step"
            ]
        ),
        "best_max_abs_zscore": (
            best[
                "max_abs_zscore"
            ]
        ),
        "best_target_loss": (
            best[
                "target_loss"
            ]
        ),
        "best_distance_penalty": (
            best[
                "distance_penalty"
            ]
        ),
        "best_out_penalty": (
            best[
                "out_penalty"
            ]
        ),
        "elite_mu_mean_np": (
            elite_mu_mean_np
        ),
        "elite_mu_std_np": (
            elite_mu_std_np
        ),
        "trace_df": (
            trace_df
        ),
    }


# =============================================================================
# 8. Genotype projection / round-trip
# =============================================================================

def observed_states_by_marker(
    X_reference,
):
    states = []

    for j in range(
        X_reference.shape[
            1
        ]
    ):
        col = X_reference[
            :,
            j,
        ]

        marker_states = np.unique(
            col[
                np.isfinite(
                    col
                )
            ]
        )

        if len(
            marker_states
        ) == 0:
            raise RuntimeError(
                f"No valid observed states for SNP index {j}."
            )

        states.append(
            marker_states.astype(
                np.float32
            )
        )

    return states


def project_to_observed_states(
    x_continuous,
    marker_states,
):
    x_continuous = np.asarray(
        x_continuous,
        dtype=np.float32,
    ).reshape(-1)

    if len(
        x_continuous
    ) != len(
        marker_states
    ):
        raise ValueError(
            "Continuous genotype length does not match "
            "number of SNP state sets."
        )

    x_discrete = np.empty_like(
        x_continuous,
        dtype=np.float32,
    )

    projection_distance = np.empty_like(
        x_continuous,
        dtype=np.float32,
    )

    for j, states in enumerate(
        marker_states
    ):
        distances = np.abs(
            states
            - x_continuous[
                j
            ]
        )

        idx = int(
            np.argmin(
                distances
            )
        )

        x_discrete[
            j
        ] = states[
            idx
        ]

        projection_distance[
            j
        ] = distances[
            idx
        ]

    return (
        x_discrete,
        projection_distance,
    )


@torch.no_grad()
def roundtrip_prediction(
    model,
    x_vector,
    scaler_mean,
    scaler_scale,
    device,
):
    x_tensor = torch.tensor(
        np.asarray(
            x_vector,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        ),
        dtype=torch.float32,
        device=device,
    )

    (
        x_recon,
        y_pred_scaled,
        mu,
        logvar,
    ) = model(
        x_tensor
    )

    y_pred_real = float(
        scaled_to_real(
            y_pred_scaled
            .detach()
            .cpu()
            .numpy()
            .item(),
            scaler_mean,
            scaler_scale,
        )
    )

    return {
        "prediction_real": (
            y_pred_real
        ),
        "prediction_scaled": float(
            y_pred_scaled
            .detach()
            .cpu()
            .numpy()
            .item()
        ),
        "mu": (
            mu.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        ),
        "logvar": (
            logvar.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        ),
        "reconstruction": (
            x_recon.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        ),
    }


# =============================================================================
# 9. Elite reference / distances / state composition
# =============================================================================

def marker_mode(
    values,
):
    values = np.asarray(
        values
    )

    unique_values, counts = np.unique(
        values,
        return_counts=True,
    )

    # np.unique is sorted; ties are resolved deterministically
    # toward the numerically smallest state.
    return unique_values[
        np.argmax(
            counts
        )
    ]


def elite_reference_genotypes(
    X_reference,
    elite_indices,
):
    elite_X = X_reference[
        elite_indices
    ]

    elite_mean = elite_X.mean(
        axis=0
    )

    elite_mode = np.asarray(
        [
            marker_mode(
                elite_X[
                    :,
                    j,
                ]
            )
            for j in range(
                elite_X.shape[
                    1
                ]
            )
        ],
        dtype=np.float32,
    )

    return (
        elite_X,
        elite_mean.astype(
            np.float32
        ),
        elite_mode,
    )


def genotype_distances(
    target,
    X_reference,
):
    target = np.asarray(
        target,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    abs_diff = np.abs(
        X_reference
        - target
    )

    mean_abs = abs_diff.mean(
        axis=1
    )

    mismatch_fraction = (
        X_reference
        != target
    ).mean(
        axis=1
    )

    rmse_state = np.sqrt(
        (
            (
                X_reference
                - target
            ) ** 2
        ).mean(
            axis=1
        )
    )

    return (
        mean_abs,
        mismatch_fraction,
        rmse_state,
    )


def state_composition(
    vector,
):
    values, counts = np.unique(
        np.asarray(
            vector
        ),
        return_counts=True,
    )

    total = int(
        np.sum(
            counts
        )
    )

    rows = []

    for value, count in zip(
        values,
        counts,
    ):
        rows.append({
            "State": float(
                value
            ),
            "Count": int(
                count
            ),
            "Fraction": float(
                count
                / total
            ),
        })

    return pd.DataFrame(
        rows
    )


def empirical_percentile(
    value,
    observed,
):
    observed = np.asarray(
        observed,
        dtype=np.float64,
    ).reshape(-1)

    return float(
        100.0
        * np.mean(
            observed
            <= float(
                value
            )
        )
    )


# =============================================================================
# 10. Output generation
# =============================================================================

def save_analysis(
    output_dir,
    checkpoint_path,
    genotype_file,
    phenotype_file,
    checkpoint,
    model,
    X_imputed,
    y,
    entries,
    snp_names,
    marker_states,
    elite_indices,
    elite_threshold,
    target_mode,
    target_real,
    optimization,
    x_cont,
    x_discrete,
    projection_distance,
    cont_rt,
    disc_rt,
    elite_mean,
    elite_mode,
    args,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        output_dir
        / OUTPUT_PREFIX
    )

    y_flat = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    z_best = (
        optimization[
            "z_best"
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    z_start = (
        optimization[
            "z_start"
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # -------------------------------------------------------------------------
    # A. Optimization trace
    # -------------------------------------------------------------------------
    optimization[
        "trace_df"
    ].to_csv(
        f"{prefix}_OptimizationTrace.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # B. Target genotype continuous / discrete
    # -------------------------------------------------------------------------
    target_genotype_df = pd.DataFrame({
        "SNP_Index_0Based": np.arange(
            len(
                snp_names
            ),
            dtype=int,
        ),
        "SNP": snp_names,
        "Decoded_Continuous": x_cont,
        "Projected_Discrete": x_discrete,
        "Projection_Abs_Distance": projection_distance,
    })

    target_genotype_df.to_csv(
        f"{prefix}_TargetGenotype.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # C. Candidate-prioritization-ready comparison with elite reference
    #    NOTE: this is NOT called functional effect.
    # -------------------------------------------------------------------------
    signed_deviation = (
        x_discrete
        - elite_mean
    )

    magnitude = np.abs(
        signed_deviation
    )

    changed_vs_elite_mode = (
        x_discrete
        != elite_mode
    )

    comparison_rows = []

    for j, snp in enumerate(
        snp_names
    ):
        states_string = ",".join(
            str(
                float(
                    state
                )
            )
            for state in marker_states[
                j
            ]
        )

        comparison_rows.append({
            "SNP_Index_0Based": j,
            "SNP": snp,
            "Observed_States": states_string,
            "Elite_Mean": float(
                elite_mean[
                    j
                ]
            ),
            "Elite_Mode": float(
                elite_mode[
                    j
                ]
            ),
            "Target_Continuous": float(
                x_cont[
                    j
                ]
            ),
            "Target_Discrete": float(
                x_discrete[
                    j
                ]
            ),
            "Signed_Target_Minus_EliteMean": float(
                signed_deviation[
                    j
                ]
            ),
            "Magnitude_Abs_Deviation": float(
                magnitude[
                    j
                ]
            ),
            "Changed_vs_Elite_Mode": bool(
                changed_vs_elite_mode[
                    j
                ]
            ),
            "Projection_Abs_Distance": float(
                projection_distance[
                    j
                ]
            ),
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        [
            "Magnitude_Abs_Deviation",
            "SNP_Index_0Based",
        ],
        ascending=[
            False,
            True,
        ],
    )

    comparison_df.insert(
        0,
        "Optimization_Rank",
        np.arange(
            1,
            len(
                comparison_df
            )
            + 1,
            dtype=int,
        )
    )

    comparison_df.to_csv(
        f"{prefix}_Target_vs_Elite.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # D. Distance from observed population / elite
    # -------------------------------------------------------------------------
    (
        mean_abs,
        mismatch_fraction,
        rmse_state,
    ) = genotype_distances(
        x_discrete,
        X_imputed,
    )

    neighbor_df = pd.DataFrame({
        "Aligned_Row_0Based": np.arange(
            len(
                entries
            ),
            dtype=int,
        ),
        "Entry": entries,
        "Observed_PH": y_flat,
        "Is_Elite": np.isin(
            np.arange(
                len(
                    entries
                )
            ),
            elite_indices,
        ),
        "Mean_Absolute_State_Distance": mean_abs,
        "Mismatch_Fraction": mismatch_fraction,
        "State_RMSE": rmse_state,
    }).sort_values(
        [
            "Mean_Absolute_State_Distance",
            "Mismatch_Fraction",
        ],
        ascending=[
            True,
            True,
        ],
    )

    neighbor_df.head(
        50
    ).to_csv(
        f"{prefix}_NearestObserved.csv",
        index=False,
    )

    elite_neighbor_df = (
        neighbor_df[
            neighbor_df[
                "Is_Elite"
            ]
        ]
        .copy()
        .head(
            50
        )
    )

    elite_neighbor_df.to_csv(
        f"{prefix}_NearestElite.csv",
        index=False,
    )

    nearest_observed = (
        neighbor_df
        .iloc[
            0
        ]
    )

    nearest_elite = (
        elite_neighbor_df
        .iloc[
            0
        ]
    )

    # -------------------------------------------------------------------------
    # E. State composition
    # -------------------------------------------------------------------------
    target_state_df = (
        state_composition(
            x_discrete
        )
    )

    target_state_df.to_csv(
        f"{prefix}_TargetStateComposition.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # F. Latent round-trip
    # -------------------------------------------------------------------------
    cont_mu = (
        cont_rt[
            "mu"
        ]
    )

    disc_mu = (
        disc_rt[
            "mu"
        ]
    )

    latent_rows = []

    for k in range(
        len(
            z_best
        )
    ):
        latent_rows.append({
            "Latent_Dim": k,
            "Start_Z": float(
                z_start[
                    k
                ]
            ),
            "Optimized_Z": float(
                z_best[
                    k
                ]
            ),
            "Continuous_RoundTrip_Mu": float(
                cont_mu[
                    k
                ]
            ),
            "Discrete_RoundTrip_Mu": float(
                disc_mu[
                    k
                ]
            ),
            "Elite_Mu_Mean": float(
                optimization[
                    "elite_mu_mean_np"
                ][
                    0,
                    k,
                ]
            ),
            "Elite_Mu_SD": float(
                optimization[
                    "elite_mu_std_np"
                ][
                    0,
                    k,
                ]
            ),
        })

    pd.DataFrame(
        latent_rows
    ).to_csv(
        f"{prefix}_LatentRoundTrip.csv",
        index=False,
    )

    latent_cont_euclidean = float(
        np.linalg.norm(
            z_best
            - cont_mu
        )
    )

    latent_disc_euclidean = float(
        np.linalg.norm(
            z_best
            - disc_mu
        )
    )

    # -------------------------------------------------------------------------
    # G. Summary / validation table
    # -------------------------------------------------------------------------
    latent_pred = float(
        optimization[
            "best_pred_real"
        ]
    )

    continuous_pred = float(
        cont_rt[
            "prediction_real"
        ]
    )

    discrete_pred = float(
        disc_rt[
            "prediction_real"
        ]
    )

    decode_loss = (
        latent_pred
        - continuous_pred
    )

    discretization_loss = (
        continuous_pred
        - discrete_pred
    )

    total_roundtrip_loss = (
        latent_pred
        - discrete_pred
    )

    changed_count = int(
        np.sum(
            changed_vs_elite_mode
        )
    )

    changed_fraction = float(
        np.mean(
            changed_vs_elite_mode
        )
    )

    mean_projection_distance = float(
        np.mean(
            projection_distance
        )
    )

    max_projection_distance = float(
        np.max(
            projection_distance
        )
    )

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

    observed_max = float(
        np.max(
            y_flat
        )
    )

    elite_y = y_flat[
        elite_indices
    ]

    summary = {
        "Dataset": "Rice299",
        "Trait": "PH",
        "Analysis": (
            "PH-directed computational target "
            "round-trip validation"
        ),

        "Checkpoint": str(
            checkpoint_path
        ),
        "Genotype_File": str(
            genotype_file
        ),
        "Phenotype_File": str(
            phenotype_file
        ),

        "N_Samples": int(
            len(
                y_flat
            )
        ),
        "N_SNPs": int(
            len(
                snp_names
            )
        ),

        "PH_Mean": float(
            np.mean(
                y_flat
            )
        ),
        "PH_SD_ddof1": float(
            np.std(
                y_flat,
                ddof=1,
            )
        ),
        "PH_Q90": q90,
        "PH_Q95": q95,
        "PH_Q97_5": q975,
        "PH_Observed_Max": (
            observed_max
        ),

        "Elite_Quantile": float(
            args.elite_quantile
        ),
        "Elite_PH_Threshold": float(
            elite_threshold
        ),
        "Elite_N": int(
            len(
                elite_indices
            )
        ),
        "Elite_PH_Mean": float(
            np.mean(
                elite_y
            )
        ),
        "Elite_PH_Min": float(
            np.min(
                elite_y
            )
        ),
        "Elite_PH_Max": float(
            np.max(
                elite_y
            )
        ),

        "Target_Mode": (
            target_mode
        ),
        "Target_PH": float(
            target_real
        ),

        "Starting_Latent_Prediction_PH": float(
            optimization[
                "start_pred_real"
            ]
        ),
        "Optimized_Latent_Prediction_PH": (
            latent_pred
        ),
        "Optimization_Best_Step": int(
            optimization[
                "best_step"
            ]
        ),
        "Optimization_Target_Abs_Error": abs(
            latent_pred
            - target_real
        ),
        "Optimization_Max_Abs_Elite_ZScore": float(
            optimization[
                "best_max_abs_zscore"
            ]
        ),

        "Continuous_RoundTrip_Prediction_PH": (
            continuous_pred
        ),
        "Discrete_RoundTrip_Prediction_PH": (
            discrete_pred
        ),

        "Decode_Loss_PH": float(
            decode_loss
        ),
        "Discretization_Loss_PH": float(
            discretization_loss
        ),
        "Total_RoundTrip_Loss_PH": float(
            total_roundtrip_loss
        ),

        "Discrete_RoundTrip_Target_Abs_Error": abs(
            discrete_pred
            - target_real
        ),

        "Discrete_RoundTrip_Empirical_Percentile": (
            empirical_percentile(
                discrete_pred,
                y_flat,
            )
        ),

        "Changed_SNPs_vs_Elite_Mode": (
            changed_count
        ),
        "Changed_SNP_Fraction_vs_Elite_Mode": (
            changed_fraction
        ),

        "Mean_Projection_Distance_to_Observed_State": (
            mean_projection_distance
        ),
        "Max_Projection_Distance_to_Observed_State": (
            max_projection_distance
        ),

        "Nearest_Observed_Entry": str(
            nearest_observed[
                "Entry"
            ]
        ),
        "Nearest_Observed_PH": float(
            nearest_observed[
                "Observed_PH"
            ]
        ),
        "Nearest_Observed_MeanAbs_State_Distance": float(
            nearest_observed[
                "Mean_Absolute_State_Distance"
            ]
        ),
        "Nearest_Observed_Mismatch_Fraction": float(
            nearest_observed[
                "Mismatch_Fraction"
            ]
        ),

        "Nearest_Elite_Entry": str(
            nearest_elite[
                "Entry"
            ]
        ),
        "Nearest_Elite_PH": float(
            nearest_elite[
                "Observed_PH"
            ]
        ),
        "Nearest_Elite_MeanAbs_State_Distance": float(
            nearest_elite[
                "Mean_Absolute_State_Distance"
            ]
        ),
        "Nearest_Elite_Mismatch_Fraction": float(
            nearest_elite[
                "Mismatch_Fraction"
            ]
        ),

        "Latent_Euclidean_Optimized_to_ContinuousRT_Mu": (
            latent_cont_euclidean
        ),
        "Latent_Euclidean_Optimized_to_DiscreteRT_Mu": (
            latent_disc_euclidean
        ),

        "Optimization_Steps": int(
            args.steps
        ),
        "Optimization_LR": float(
            args.lr
        ),
        "Lambda_Distance": float(
            args.lambda_dist
        ),
        "Lambda_OutOfManifold": float(
            args.lambda_out
        ),
        "Latent_Z_Limit_SD": float(
            args.latent_z_limit
        ),

        "Interpretation_Note": (
            "Computational target only; does not establish "
            "breedability, haplotype feasibility, causal SNP effects, "
            "or biological functional validity."
        ),
    }

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        f"{prefix}_Summary.csv",
        index=False,
    )

    # A concise reviewer-facing table.
    reviewer_table = pd.DataFrame([
        {
            "Quantity": "Observed PH maximum",
            "Value": observed_max,
        },
        {
            "Quantity": f"Optimization target ({target_mode})",
            "Value": float(
                target_real
            ),
        },
        {
            "Quantity": "Starting latent prediction",
            "Value": float(
                optimization[
                    "start_pred_real"
                ]
            ),
        },
        {
            "Quantity": "Optimized latent prediction",
            "Value": latent_pred,
        },
        {
            "Quantity": "Continuous genotype round-trip prediction",
            "Value": continuous_pred,
        },
        {
            "Quantity": "Discrete genotype round-trip prediction",
            "Value": discrete_pred,
        },
        {
            "Quantity": "Decode loss",
            "Value": float(
                decode_loss
            ),
        },
        {
            "Quantity": "Discretization loss",
            "Value": float(
                discretization_loss
            ),
        },
        {
            "Quantity": "Changed SNP fraction vs elite modal genotype",
            "Value": float(
                changed_fraction
            ),
        },
        {
            "Quantity": "Nearest observed mean absolute state distance",
            "Value": float(
                nearest_observed[
                    "Mean_Absolute_State_Distance"
                ]
            ),
        },
    ])

    reviewer_table.to_csv(
        f"{prefix}_ReviewerTable.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # H. Full configuration metadata
    # -------------------------------------------------------------------------
    config = {
        "analysis": (
            "Rice299 PH-directed target round-trip validation"
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "data": {
            "genotype": str(
                genotype_file
            ),
            "phenotype": str(
                phenotype_file
            ),
            "n_samples": int(
                len(
                    y_flat
                )
            ),
            "n_snps": int(
                len(
                    snp_names
                )
            ),
            "minus_one_is_valid_genotype": True,
            "snp_scaling": None,
        },
        "elite": {
            "quantile": float(
                args.elite_quantile
            ),
            "threshold": float(
                elite_threshold
            ),
            "n": int(
                len(
                    elite_indices
                )
            ),
        },
        "target": {
            "mode": target_mode,
            "value": float(
                target_real
            ),
        },
        "optimization": {
            "seed": int(
                args.seed
            ),
            "steps": int(
                args.steps
            ),
            "lr": float(
                args.lr
            ),
            "lambda_dist": float(
                args.lambda_dist
            ),
            "lambda_out": float(
                args.lambda_out
            ),
            "latent_z_limit_sd": float(
                args.latent_z_limit
            ),
            "objective": (
                "standardized phenotype target squared error "
                "+ distance penalty + out-of-manifold penalty"
            ),
            "start": (
                "mean posterior mu of elite samples"
            ),
        },
        "discretization": {
            "method": (
                "per-marker nearest state observed in "
                "the aligned/imputed reference population"
            ),
            "global_round_0_1_2": False,
        },
        "interpretation": {
            "term": (
                "unconstrained in silico target genotype"
            ),
            "not_claimed": [
                "direct breedability",
                "parental haplotype feasibility",
                "phase preservation",
                "recombination feasibility",
                "causal SNP effect",
                "functional variant discovery",
            ],
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
    # I. Save optimized latent vector for follow-up sensitivity / perturbation
    # -------------------------------------------------------------------------
    np.savez_compressed(
        f"{prefix}_LatentVectors.npz",
        z_start=z_start.astype(
            np.float32
        ),
        z_optimized=z_best.astype(
            np.float32
        ),
        elite_mu_mean=optimization[
            "elite_mu_mean_np"
        ].astype(
            np.float32
        ),
        elite_mu_std=optimization[
            "elite_mu_std_np"
        ].astype(
            np.float32
        ),
        continuous_roundtrip_mu=cont_mu.astype(
            np.float32
        ),
        discrete_roundtrip_mu=disc_mu.astype(
            np.float32
        ),
    )

    print("\nSaved outputs:")
    for path in sorted(
        output_dir.glob(
            f"{OUTPUT_PREFIX}*"
        )
    ):
        print(
            "  ",
            path.name,
        )

    return summary


# =============================================================================
# 11. Command-line interface / main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "GenoPerfVAE Rice299-PH target genotype "
            "round-trip validation."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to GenoPerfVAE_Rice_PH_Final_PostHoc.pth. "
            "If omitted, common locations are searched."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Folder containing the Rice299 genotype CSV "
            "and rice299_PH.csv."
        ),
    )

    parser.add_argument(
        "--genotype",
        type=str,
        default=None,
        help=(
            "Explicit genotype CSV path."
        ),
    )

    parser.add_argument(
        "--phenotype",
        type=str,
        default=None,
        help=(
            "Explicit rice299_PH.csv path."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: "
            "results_Rice_PH_Target_RoundTrip beside this script."
        ),
    )

    parser.add_argument(
        "--elite-quantile",
        type=float,
        default=DEFAULT_ELITE_QUANTILE,
        help=(
            "Elite threshold quantile. Default 0.95."
        ),
    )

    parser.add_argument(
        "--target-mode",
        type=str,
        default=DEFAULT_TARGET_MODE,
        choices=[
            "q90",
            "q95",
            "q97.5",
            "q975",
            "max",
        ],
        help=(
            "Observed-data target definition. Default q95."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_OPT_STEPS,
        help=(
            "Latent optimization steps. Default 500."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_OPT_LR,
        help=(
            "Latent optimization learning rate. Default 0.01."
        ),
    )

    parser.add_argument(
        "--lambda-dist",
        type=float,
        default=DEFAULT_LAMBDA_DIST,
        help=(
            "Weight of squared distance-to-start penalty. "
            "Default 0.05."
        ),
    )

    parser.add_argument(
        "--lambda-out",
        type=float,
        default=DEFAULT_LAMBDA_OUT,
        help=(
            "Weight of out-of-elite-manifold penalty. "
            "Default 0.50."
        ),
    )

    parser.add_argument(
        "--latent-z-limit",
        type=float,
        default=DEFAULT_LATENT_Z_LIMIT,
        help=(
            "Soft per-dimension elite latent SD boundary. "
            "Default 3.0."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Optimization seed. Default 42."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not (
        0.0
        < args.elite_quantile
        < 1.0
    ):
        raise ValueError(
            "--elite-quantile must be between 0 and 1."
        )

    if args.steps < 1:
        raise ValueError(
            "--steps must be >= 1."
        )

    if args.lr <= 0:
        raise ValueError(
            "--lr must be > 0."
        )

    if args.latent_z_limit <= 0:
        raise ValueError(
            "--latent-z-limit must be > 0."
        )

    set_seed_fixed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    checkpoint_path = resolve_checkpoint(
        script_dir,
        args.checkpoint,
    )

    (
        genotype_file,
        phenotype_file,
    ) = resolve_data_files(
        script_dir=script_dir,
        explicit_data_dir=args.data_dir,
        explicit_genotype=args.genotype,
        explicit_phenotype=args.phenotype,
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir is not None
        else (
            script_dir
            / "results_Rice_PH_Target_RoundTrip"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 95)
    print("GenoPerfVAE Rice299-PH target round-trip validation")
    print("=" * 95)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Genotype:", genotype_file)
    print("Phenotype:", phenotype_file)
    print("Output:", output_dir)

    checkpoint = torch_load_checkpoint(
        checkpoint_path,
        device,
    )

    if (
        str(
            checkpoint.get(
                "trait",
                ""
            )
        ).upper()
        != "PH"
    ):
        raise RuntimeError(
            "Checkpoint trait is not PH."
        )

    model = build_model_from_checkpoint(
        checkpoint,
        device,
    )

    (
        X_raw,
        X_imputed,
        y,
        entries,
        snp_names,
    ) = load_reference_data(
        genotype_file,
        phenotype_file,
        checkpoint,
    )

    print("\nReference data")
    print("--------------")
    print(
        "Aligned samples:",
        len(
            entries
        )
    )
    print(
        "SNPs:",
        len(
            snp_names
        )
    )
    print(
        "True NaN in aligned raw genotype:",
        int(
            np.isnan(
                X_raw
            ).sum()
        )
    )
    print(
        "NaN after restored imputation:",
        int(
            np.isnan(
                X_imputed
            ).sum()
        )
    )
    print(
        "-1 count after imputation:",
        int(
            np.sum(
                X_imputed
                == -1
            )
        )
    )

    scaler_mean = float(
        checkpoint[
            "scaler_y_mean"
        ]
    )

    scaler_scale = float(
        checkpoint[
            "scaler_y_scale"
        ]
    )

    (
        mu_all,
        logvar_all,
    ) = encode_all_mu(
        model,
        X_imputed,
        device,
    )

    (
        elite_indices,
        elite_threshold,
    ) = define_elite_indices(
        y,
        args.elite_quantile,
    )

    target_real = resolve_target_value(
        y,
        args.target_mode,
    )

    print("\nPH reference")
    print("------------")
    print(
        "Observed mean:",
        f"{float(np.mean(y)):.6f}"
    )
    print(
        "Observed max:",
        f"{float(np.max(y)):.6f}"
    )
    print(
        "Q90:",
        f"{float(np.quantile(y, 0.90)):.6f}"
    )
    print(
        "Q95:",
        f"{float(np.quantile(y, 0.95)):.6f}"
    )
    print(
        "Q97.5:",
        f"{float(np.quantile(y, 0.975)):.6f}"
    )
    print(
        "Elite quantile:",
        args.elite_quantile,
        "| threshold:",
        f"{elite_threshold:.6f}",
        "| N:",
        len(
            elite_indices
        )
    )
    print(
        "Target mode:",
        args.target_mode,
        "| Target PH:",
        f"{target_real:.6f}"
    )

    print("\nLatent optimization")
    print("-------------------")
    print(
        "Objective: standardized target squared error "
        "+ distance penalty + out-of-manifold penalty"
    )
    print(
        f"steps={args.steps}, lr={args.lr}, "
        f"lambda_dist={args.lambda_dist}, "
        f"lambda_out={args.lambda_out}, "
        f"z_limit={args.latent_z_limit} SD"
    )

    start_time = time.perf_counter()

    optimization = optimize_latent_target(
        model=model,
        mu_all=mu_all,
        elite_indices=elite_indices,
        target_real=target_real,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
        steps=args.steps,
        lr=args.lr,
        lambda_dist=args.lambda_dist,
        lambda_out=args.lambda_out,
        latent_z_limit=args.latent_z_limit,
    )

    optimization_time = (
        time.perf_counter()
        - start_time
    )

    print(
        "Starting latent prediction:",
        f"{optimization['start_pred_real']:.6f}"
    )
    print(
        "Optimized latent prediction:",
        f"{optimization['best_pred_real']:.6f}"
    )
    print(
        "Best optimization step:",
        optimization[
            "best_step"
        ]
    )
    print(
        "Absolute target error:",
        f"{abs(optimization['best_pred_real'] - target_real):.6f}"
    )
    print(
        "Max abs elite latent z-score:",
        f"{optimization['best_max_abs_zscore']:.6f}"
    )
    print(
        "Optimization time (s):",
        f"{optimization_time:.3f}"
    )

    z_best = (
        optimization[
            "z_best"
        ]
    )

    # Decode continuous genotype.
    with torch.no_grad():
        x_cont = (
            model.decoder(
                z_best
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(
                np.float32
            )
        )

    marker_states = (
        observed_states_by_marker(
            X_imputed
        )
    )

    (
        x_discrete,
        projection_distance,
    ) = project_to_observed_states(
        x_cont,
        marker_states,
    )

    # Strict round-trip through the entire model.
    cont_rt = roundtrip_prediction(
        model=model,
        x_vector=x_cont,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
    )

    disc_rt = roundtrip_prediction(
        model=model,
        x_vector=x_discrete,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
    )

    (
        elite_X,
        elite_mean,
        elite_mode,
    ) = elite_reference_genotypes(
        X_imputed,
        elite_indices,
    )

    latent_pred = float(
        optimization[
            "best_pred_real"
        ]
    )

    cont_pred = float(
        cont_rt[
            "prediction_real"
        ]
    )

    disc_pred = float(
        disc_rt[
            "prediction_real"
        ]
    )

    print("\n" + "=" * 95)
    print("ROUND-TRIP VALIDATION")
    print("=" * 95)
    print(
        "Target PH:",
        f"{target_real:.6f}"
    )
    print(
        "Optimized latent prediction:",
        f"{latent_pred:.6f}"
    )
    print(
        "Continuous genotype round-trip prediction:",
        f"{cont_pred:.6f}"
    )
    print(
        "Discrete genotype round-trip prediction:",
        f"{disc_pred:.6f}"
    )
    print(
        "Decode loss (latent - continuous RT):",
        f"{latent_pred - cont_pred:.6f}"
    )
    print(
        "Discretization loss (continuous RT - discrete RT):",
        f"{cont_pred - disc_pred:.6f}"
    )
    print(
        "Total round-trip loss (latent - discrete RT):",
        f"{latent_pred - disc_pred:.6f}"
    )

    changed_count = int(
        np.sum(
            x_discrete
            != elite_mode
        )
    )

    changed_fraction = float(
        np.mean(
            x_discrete
            != elite_mode
        )
    )

    print(
        "Changed SNPs vs elite modal genotype:",
        f"{changed_count}/{len(snp_names)} "
        f"({100.0 * changed_fraction:.2f}%)"
    )

    summary = save_analysis(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        genotype_file=genotype_file,
        phenotype_file=phenotype_file,
        checkpoint=checkpoint,
        model=model,
        X_imputed=X_imputed,
        y=y,
        entries=entries,
        snp_names=snp_names,
        marker_states=marker_states,
        elite_indices=elite_indices,
        elite_threshold=elite_threshold,
        target_mode=args.target_mode,
        target_real=target_real,
        optimization=optimization,
        x_cont=x_cont,
        x_discrete=x_discrete,
        projection_distance=projection_distance,
        cont_rt=cont_rt,
        disc_rt=disc_rt,
        elite_mean=elite_mean,
        elite_mode=elite_mode,
        args=args,
    )

    print("\n" + "=" * 95)
    print("DONE")
    print("=" * 95)
    print(
        "Final discrete target PH prediction:",
        f"{summary['Discrete_RoundTrip_Prediction_PH']:.6f}"
    )
    print(
        "Empirical PH percentile of that prediction:",
        f"{summary['Discrete_RoundTrip_Empirical_Percentile']:.2f}"
    )
    print(
        "Nearest observed entry:",
        summary[
            "Nearest_Observed_Entry"
        ],
        "| PH:",
        f"{summary['Nearest_Observed_PH']:.6f}",
        "| mean abs state distance:",
        f"{summary['Nearest_Observed_MeanAbs_State_Distance']:.6f}",
    )
    print(
        "\nInterpretation reminder:"
    )
    print(
        "This is an unconstrained computational target genotype. "
        "The analysis does NOT demonstrate direct breedability, "
        "haplotype/recombination feasibility, or causal SNP effects."
    )


if __name__ == "__main__":
    main()
