"""
model.py
========
GN2 Transformer architecture for jet flavour tagging.

Architecture:

  Input:
    - jet features   (n_jet_vars,)
    - track features (n_tracks, n_track_vars)
    - padding mask   (n_tracks,):              True = real track, False = padded

  Per-track initialiser:
    - Linear(n_jet_vars + n_track_vars -> 256) + ReLU   (1 hidden layer)
    - Linear(256 -> 256)                                (output layer)

  Transformer Encoder:
    - 4 layers, 8 attention heads
    - embedding dim = 256, feed-forward dim = 512
    - Pre-LayerNorm

  Pooling:
    - Attention pooling: global jet representation (dim 128)
    - Per-track projection: track embeddings       (dim 128)

  Task-specific heads  (3 hidden layers: 128 -> 64 -> 32):
    1. Jet classification     (primary)    : 4 classes  (b, c, light, tau)

Loss:
    L = w_jet * CE_jet + w_origin * CE_origin + w_vertex * CE_vertex
    Default weights (from the paper): w_jet=1, w_origin=0.5, w_vertex=0.5
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("GN2.model")

# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class TransformerLayer(nn.Module):
    """
    Transformer encoder layer with Pre-LayerNorm (before attention & FFN).

    Attributes:
        d_model (int) : embedding dimension
        n_heads (int) : number of attention heads
        d_ff (int) : feed-forward inner dimension
        dropout (float) : dropout rate
    """
    def __init__(
        self,
        dim_emb: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.0
    ):
        """
        Initialize the transformer layer.

        Args:
            d_model (int): embedding dimension (default 256)
            n_heads (int): number of attention heads (default 8)
            d_ff (int): feed-forward inner dimension (default 512)
            dropout (float): dropout rate (default 0.0)
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_emb)      # normalization before attention
        self.attn  = nn.MultiheadAttention(dim_emb, n_heads, dropout=dropout, batch_first=True)     # note: batch_first=True for (B, T, d_model) input
        self.norm2 = nn.LayerNorm(dim_emb)      # normalization before feed-forward
        self.ff    = nn.Sequential(
            nn.Linear(dim_emb, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_emb),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer layer.

        Args:
            inputs           (torch.Tensor): shape (B, T, d_model)
            key_padding_mask (torch.Tensor, optional): shape (B, T), False = position to IGNORE

        Returns:
            x (torch.Tensor): shape (B, T, d_model) transformed output
        """
        # self-attention with pre-norm
        residual = inputs           # save inputs for skip connection
        x = self.norm1(inputs)
        x, _ = self.attn(x, x, x, key_padding_mask=~key_padding_mask)
        x = self.drop(x) + residual

        # feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + residual

        return x


class AttentionPooling(nn.Module):
    """
    Attention pooling to produce a single global representation from
    the set of track embeddings.

    Attributes:
        d_in (int): input embedding dimension (from transformer)
        d_out (int): output embedding dimension (for jet representation)
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int
    ):
        """
        Initialize the attention pooling layer.

        Args:
            d_in (int): input embedding dimension (from transformer)
            d_out (int): output embedding dimension (for jet representation)
        """
        super().__init__()
        self.query    = nn.Linear(dim_in, 1)        # "score" for each track embedding
        self.proj_out = nn.Linear(dim_in, dim_out)  # final projection of pooled representation

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the attention pooling layer.

        Args:
            x            (torch.Tensor): shape (B, T, d_in), track embeddings from transformer
            padding_mask (torch.Tensor, optional): shape (B, T), True = real track, False = padding
        
        Returns:
            (torch.Tensor): shape (B, d_out), pooled jet representation
        """
        scores = self.query(x).squeeze(-1)                  # (B, T), attention scores for each track
        if padding_mask is not None:
            scores = scores.masked_fill(~padding_mask, float('-inf'))       # mask padded positions to -inf so that softmax gives zero weight to ignored tracks
        weights = torch.softmax(scores, dim=-1)             # (B, T)
        pooled  = (weights.unsqueeze(-1) * x).sum(dim=1)    # (B, d_in)
        return self.proj_out(pooled)                        # (B, d_out)


def _mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """
    Build an MLP with ReLU activations.
    
    Args:
        in_dim (int): input dimension
        hidden_dims (list[int]): list of hidden layer dimensions
        out_dim (int): output dimension

    Returns:
        nn.Sequential: the MLP model
    """
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
        n_jet_vars (int): number of jet-level input features.
        n_track_vars (int): number of track-level input features.
        n_classes (int): number of jet flavour classes (default 4: b, c, light, tau).
        embed_dim (int): transformer embedding dimension (default 256).
        n_heads (int): number of attention heads (default 8).
        n_layers (int): number of transformer encoder layers (default 4).
        ff_dim (int): feed-forward inner dimension (default 512).
        pool_dim (int): output dimension of attention pooling (default 128).
        dropout (float): dropout rate (default 0.0).
    """

    def __init__(
        self,
        n_jet_vars   : int,
        n_track_vars : int,
        n_classes    : int  = 4,
        embed_dim    : int  = 256,
        n_heads      : int  = 8,
        n_layers     : int  = 4,
        ff_dim       : int  = 512,
        pool_dim     : int  = 128,
        dropout      : float = 0.0,
    ):
        """
        Initialize the GN2 model.

        Args:
            n_jet_vars (int): number of jet-level input features.
            n_track_vars (int): number of track-level input features.
            n_classes (int, optional): number of jet flavour classes (default 4: b, c, light, tau).
            embed_dim (int, optional): transformer embedding dimension (default 256).
            n_heads (int, optional): number of attention heads (default 8).
            n_layers (int, optional): number of transformer encoder layers (default 4).
            ff_dim (int, optional): feed-forward inner dimension (default 512).
            pool_dim (int, optional): output dimension of attention pooling (default 128).
            dropout (float, optional): dropout rate (default 0.0).
        """
        super().__init__()

        self.n_jet_vars   = n_jet_vars
        self.n_track_vars = n_track_vars
        self.n_classes    = n_classes
        self.embed_dim    = embed_dim
        self.pool_dim     = pool_dim

        in_dim = n_jet_vars + n_track_vars   # combined per-track input

        # 1. Per-track initialiser (1 hidden layer + output, both size "embed_dim")
        self.track_init = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. Transformer encoder ("n_layers" layers, "n_heads" heads, pre-norm)
        self.transformer = nn.ModuleList([
            TransformerLayer(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)   # post-encoder norm

        # # 3. Per-track projection ("embed_dim" -> "pool_dim") TODO: for auxiliary tasks that require track-level outputs (origin, vertex)
        # self.track_proj = nn.Linear(embed_dim, pool_dim)

        # 4. Attention pooling ("embed_dim" -> "pool_dim")
        self.pool = AttentionPooling(embed_dim, pool_dim)

        # 5. Task heads (3 hidden layers: 128 -> 64 -> 32)
        self.jet_head = _mlp(pool_dim, [128, 64, 32], n_classes)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)     # count all trainable parameters
        logger.info(
            f"GN2 initialised - embed={embed_dim}, layers={n_layers}, heads={n_heads}, ff={ff_dim}  |  params: {n_params:,}"
        )

    def forward(
        self,
        jet_features   : torch.Tensor,
        track_features : torch.Tensor,
        mask           : torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the GN2 model.

        Args:
            jet_features   (torch.Tensor): shape (B, n_jet_vars)
            track_features (torch.Tensor): shape (B, T, n_track_vars)
            mask           (torch.Tensor): shape (B, T),  True = real track, False = padding

        Returns:
            dict with keys:
              'jet_logits' : (torch.Tensor): shape (B, n_classes)
        """
        _, T, _ = track_features.shape

        # 1. Concatenate jet features (broadcast) with track features
        jet_expanded = jet_features.unsqueeze(1).expand(-1, T, -1)          # (B, T, J), unsqueeze jet features to (B, 1, J) and expand to (B, T, J) to concatenate with track features (-1 means "keep original size")
        combined     = torch.cat([jet_expanded, track_features], dim=-1)    # (B, T, J+K), concatenate along feature dimension

        # 2. Per-track initialisation
        x = self.track_init(combined)   # (B, T, embed_dim)

        # 3. Transformer encoder
        for layer in self.transformer:
            x = layer(x, key_padding_mask=mask)
        x = self.final_norm(x)          # (B, T, embed_dim)

        # 4. Attention pooling: global jet representation
        jet_rep = self.pool(x, padding_mask=mask)   # (B, pool_dim)

        # 5. Primary head: jet classification
        jet_logits = self.jet_head(jet_rep)         # (B, n_classes)

        return {
            "jet_logits" : jet_logits,
        }

    @torch.no_grad()
    def predict_proba(
        self,
        jet_features  : torch.Tensor,
        track_features: torch.Tensor,
        mask          : torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class probabilities for the jet classification head.

        Args:
            jet_features   (torch.Tensor): shape (B, n_jet_vars)
            track_features (torch.Tensor): shape (B, T, n_track_vars)
            mask           (torch.Tensor): shape (B, T),  True = real track, False = padding
        
        Returns:
            (torch.Tensor): shape (B, n_classes), predicted class probabilities
        """
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
        Compute the b-tagging discriminant D_b:
            D_b = log( p_b / (fc*p_c + ftau*p_tau + (1-fc-ftau)*p_light) )

        Args:
            jet_features   (torch.Tensor): shape (B, n_jet_vars)
            track_features (torch.Tensor): shape (B, T, n_track_vars)
            mask           (torch.Tensor): shape (B, T),  True = real track, False = padding
            fc             (float): fraction of c-jets
            ftau           (float): fraction of tau-jets

        Returns:
            (torch.Tensor): shape (B,), the b-tagging discriminant D_b
        """
        proba = self.predict_proba(jet_features, track_features, mask)
        # class order: light=0, c=1, b=2, tau=3     TODO: make this more robust by using label_map instead of hardcoding class order
        pb   = proba[:, 2]
        pc   = proba[:, 1]
        pu   = proba[:, 0]
        ptau = proba[:, 3]
        denom = fc * pc + ftau * ptau + (1 - fc - ftau) * pu
        return torch.log(pb / denom.clamp(min=1e-9))
    
    @torch.no_grad()
    def discriminant_dc(
        self,
        jet_features  : torch.Tensor,
        track_features: torch.Tensor,
        mask          : torch.Tensor,
        fb            : float = 0.3,
        ftau          : float = 0.01,
    ) -> torch.Tensor:
        """
        Compute the c-tagging discriminant D_c:
            D_c = log( p_c / (fb*p_b + ftau*p_tau + (1-fb-ftau)*p_light) )

        Args:
            jet_features   (torch.Tensor): shape (B, n_jet_vars)
            track_features (torch.Tensor): shape (B, T, n_track_vars)
            mask           (torch.Tensor): shape (B, T),  True = real track, False = padding
            fb             (float): fraction of b-jets
            ftau           (float): fraction of tau-jets

        Returns:
            (torch.Tensor): shape (B,), the c-tagging discriminant D_c
        """
        proba = self.predict_proba(jet_features, track_features, mask)
        # class order: light=0, c=1, b=2, tau=3     TODO: make this more robust by using label_map instead of hardcoding class order
        pb   = proba[:, 2]
        pc   = proba[:, 1]
        pu   = proba[:, 0]
        ptau = proba[:, 3]
        denom = fb * pb + ftau * ptau + (1 - fb - ftau) * pu
        return torch.log(pc / denom.clamp(min=1e-9))