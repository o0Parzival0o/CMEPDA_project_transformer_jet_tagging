"""
model.py
========
GN2 Transformer architecture for jet flavour tagging.

Architecture (from Nature Communications 2026, doi:10.1038/s41467-025-65059-6):

  Input:
    - jet features   (n_jet_vars,)         copied for each track → concat with track
    - track features (n_tracks, n_track_vars)
    - padding mask   (n_tracks,)           True = real track, False = padded

  Per-track initialiser:
    - Linear(n_jet_vars + n_track_vars → 256) + ReLU   [1 hidden layer]
    - Linear(256 → 256)                                [output layer]

  Transformer Encoder:
    - 4 layers, 8 attention heads
    - embedding dim = 256, feed-forward dim = 512
    - Pre-LayerNorm (as in the paper, §Methods)

  Pooling:
    - Attention pooling → global jet representation (dim 128)
    - Per-track projection → track embeddings       (dim 128)

  Task-specific heads  [3 hidden layers: 128 → 64 → 32]:
    1. Jet classification     (primary)    : 4 classes  (b, c, light, tau)
    2. Track origin           (auxiliary)  : 7 classes
    3. Track-pair vertex      (auxiliary)  : 2 classes  (binary per pair)

Loss:
    L = w_jet * CE_jet + w_origin * CE_origin + w_vertex * CE_vertex
    Default weights from the paper: w_jet=1, w_origin=0.5, w_vertex=0.5
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("GN2.model")

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PreLNTransformerLayer(nn.Module):
    """
    Transformer encoder layer with Pre-LayerNorm (before attention & FFN),
    as specified in the paper (§Methods, ref [88]: Xiong et al. 2020).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x                : (B, T, d_model)
            key_padding_mask : (B, T) — True = position to IGNORE (padding convention
                               used by nn.MultiheadAttention)

        Returns:
            (B, T, d_model)
        """
        # --- self-attention with pre-LN ---
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.drop(x) + residual

        # --- feed-forward with pre-LN ---
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + residual

        return x


class AttentionPooling(nn.Module):
    """
    Attention pooling (§Methods, ref [89]) to produce a single global
    representation from the set of track embeddings.

    Learns a query vector that attends over all (real) tracks and outputs
    a weighted sum — equivalent to a learned, context-aware mean.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.query    = nn.Linear(d_in, 1)
        self.proj_out = nn.Linear(d_in, d_out)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x            : (B, T, d_in)  — track embeddings from transformer
            padding_mask : (B, T)        — True = real track, False = padding
        
        Returns:
            (B, d_out)   — pooled jet representation
        """
        scores = self.query(x).squeeze(-1)              # (B, T)
        if padding_mask is not None:
            scores = scores.masked_fill(~padding_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)         # (B, T)
        pooled  = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, d_in)
        return self.proj_out(pooled)                    # (B, d_out)


def _mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """Helper: build an MLP with ReLU activations."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# GN2 model
# ---------------------------------------------------------------------------

class GN2(nn.Module):
    """
    GN2 transformer-based jet flavour tagger.

    Attributes:
        n_jet_vars     : number of jet-level input features.
        n_track_vars   : number of track-level input features.
        n_classes      : number of jet flavour classes (default 4: b, c, light, tau).
        n_track_origin : number of track-origin classes for the auxiliary task (default 7).
        embed_dim      : transformer embedding dimension (default 256).
        n_heads        : number of attention heads (default 8).
        n_layers       : number of transformer encoder layers (default 4).
        ff_dim         : feed-forward inner dimension (default 512).
        pool_dim       : output dimension of attention pooling (default 128).
        dropout        : dropout rate (default 0.0).
    """

    def __init__(
        self,
        n_jet_vars    : int,
        n_track_vars  : int,
        n_classes     : int  = 4,
        n_track_origin: int  = 7,
        embed_dim     : int  = 256,
        n_heads       : int  = 8,
        n_layers      : int  = 4,
        ff_dim        : int  = 512,
        pool_dim      : int  = 128,
        dropout       : float = 0.0,
    ):
        super().__init__()

        self.n_jet_vars      = n_jet_vars
        self.n_track_vars    = n_track_vars
        self.n_classes       = n_classes
        self.n_track_origin  = n_track_origin
        self.embed_dim       = embed_dim
        self.pool_dim        = pool_dim

        in_dim = n_jet_vars + n_track_vars   # combined per-track input

        # ---- Per-track initialiser  (1 hidden layer + output, both size 256) ----
        self.track_init = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ---- Transformer encoder  (4 layers, 8 heads, pre-LN) ----
        self.transformer = nn.ModuleList([
            PreLNTransformerLayer(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)   # post-encoder norm (pre-LN style)

        # ---- Per-track projection  embed_dim → pool_dim ----
        self.track_proj = nn.Linear(embed_dim, pool_dim)

        # ---- Attention pooling  embed_dim → pool_dim ----
        self.pool = AttentionPooling(embed_dim, pool_dim)

        # ---- Task heads  [3 hidden layers: 128 → 64 → 32] ----
        # Primary: jet classification
        self.jet_head = _mlp(pool_dim, [128, 64, 32], n_classes)

        # Auxiliary 1: per-track origin (uses track embed + global rep)
        self.origin_head = _mlp(pool_dim + pool_dim, [128, 64, 32], n_track_origin)

        # Auxiliary 2: track-pair vertex compatibility (binary per pair)
        # Input: concatenation of two track embeddings + global rep
        self.vertex_head = _mlp(pool_dim + pool_dim + pool_dim, [128, 64, 32], 2)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"GN2 initialised — embed={embed_dim}, layers={n_layers}, "
            f"heads={n_heads}, ff={ff_dim}  |  params: {n_params:,}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        jet_features  : torch.Tensor,
        track_features: torch.Tensor,
        mask          : torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            jet_features   : (B, n_jet_vars)
            track_features : (B, T, n_track_vars)
            mask           : (B, T)  True = real track, False = padding

        Returns:
            dict with keys:
              'jet_logits'    : (B, n_classes)             — primary task
              'origin_logits' : (B, T, n_track_origin)     — auxiliary task 1
              'vertex_logits' : (B, T, T, 2)               — auxiliary task 2
        """
        B, T, _ = track_features.shape

        # 1. Concatenate jet features (broadcast) with track features
        jet_expanded = jet_features.unsqueeze(1).expand(-1, T, -1)   # (B, T, J)
        combined     = torch.cat([jet_expanded, track_features], dim=-1)  # (B, T, J+K)

        # 2. Per-track initialisation
        x = self.track_init(combined)   # (B, T, embed_dim)

        # 3. Transformer encoder
        # nn.MultiheadAttention expects key_padding_mask: True = IGNORE
        attn_mask = ~mask   # (B, T) — True where padded
        for layer in self.transformer:
            x = layer(x, key_padding_mask=attn_mask)
        x = self.final_norm(x)          # (B, T, embed_dim)

        # 4. Project track embeddings down to pool_dim
        track_emb = self.track_proj(x)  # (B, T, pool_dim)

        # 5. Attention pooling → global jet representation
        jet_rep = self.pool(x, padding_mask=mask)   # (B, pool_dim)

        # 6. Primary head: jet classification
        jet_logits = self.jet_head(jet_rep)          # (B, n_classes)

        # 7. Auxiliary head 1: track origin
        jet_rep_exp    = jet_rep.unsqueeze(1).expand(-1, T, -1)  # (B, T, pool_dim)
        origin_input   = torch.cat([track_emb, jet_rep_exp], dim=-1)  # (B, T, 2*pool_dim)
        origin_logits  = self.origin_head(origin_input)              # (B, T, n_track_origin)

        # 8. Auxiliary head 2: track-pair vertex compatibility
        # Build all (T, T) pairs efficiently without Python loops
        ti = track_emb.unsqueeze(2).expand(-1, -1, T, -1)  # (B, T, T, pool_dim)
        tj = track_emb.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, T, pool_dim)
        gr = jet_rep.unsqueeze(1).unsqueeze(1).expand(-1, T, T, -1)  # (B, T, T, pool_dim)
        vertex_input  = torch.cat([ti, tj, gr], dim=-1)     # (B, T, T, 3*pool_dim)
        vertex_logits = self.vertex_head(vertex_input)       # (B, T, T, 2)

        return {
            "jet_logits"    : jet_logits,
            "origin_logits" : origin_logits,
            "vertex_logits" : vertex_logits,
        }

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self,
        jet_features  : torch.Tensor,
        track_features: torch.Tensor,
        mask          : torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities for jet classification. Shape: (B, n_classes)."""
        self.eval()
        out = self.forward(jet_features, track_features, mask)
        return torch.softmax(out["jet_logits"], dim=-1)

    @torch.no_grad()
    def discriminant_db(
        self,
        jet_features  : torch.Tensor,
        track_features: torch.Tensor,
        mask          : torch.Tensor,
        fc            : float = 0.2,
        ftau          : float = 0.05,
    ) -> torch.Tensor:
        """
        Compute the b-tagging discriminant D_b (Eq. 1 in the paper):
            D_b = log( p_b / (fc*p_c + ftau*p_tau + (1-fc-ftau)*p_light) )

        Returns: (B,) tensor of discriminant values.
        """
        proba = self.predict_proba(jet_features, track_features, mask)
        # class order: light=0, c=1, b=2, tau=3  (from JET_FLAVOUR_MAP in dataset.py)
        pb   = proba[:, 2]
        pc   = proba[:, 1]
        pu   = proba[:, 0]
        ptau = proba[:, 3]
        denom = fc * pc + ftau * ptau + (1 - fc - ftau) * pu
        return torch.log(pb / denom.clamp(min=1e-9))