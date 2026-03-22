"""
GN2-Lite: versione semplificata di GN2 per jet flavour tagging.

Differenze rispetto a GN2 originale (paper ATLAS 2026):
  - Nessun auxiliary task (track origin, vertex grouping)
  - 2 layer transformer invece di 4
  - 4 attention head invece di 8
  - Embedding dim 128 invece di 256
  - FFN dim 256 invece di 512
  - ~0.3M parametri invece di 2.3M

Invariato rispetto a GN2:
  - Stesse 19 track features + 2 jet features
  - Pre-LayerNorm nel transformer (più stabile)
  - Attention pooling per aggregare le tracce
  - Padding mask per le tracce non valide
  - Ordinamento per |d0 significance| (gestito nel DataLoader)
  - 4 classi: b-jet, c-jet, light-jet, tau-jet
  - Discriminante Db come definita nel paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── Costanti (da gn2_dataloader.py) ─────────────────────────────────────────

N_JET_FEAT   = 2   # pt_btagJes, eta_btagJes
N_TRACK_FEAT = 19  # le 19 variabili GN2 del notebook ATLAS
N_CLASSES    = 4   # b=0, c=1, light=2, tau=3
MAX_TRACKS   = 40


# ─── Blocco Transformer con pre-LayerNorm ─────────────────────────────────────

class PreNormTransformerLayer(nn.Module):
    """
    Blocco Transformer con pre-LayerNorm (come in GN2 e Vision Transformer).
    Schema: LN → MHA → residual, poi LN → FFN → residual.
    Il pre-LN è più stabile numericamente del post-LN durante il training.
    """

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,  # input shape: (B, T, E) — più leggibile
        )
        self.norm2 = nn.FFN_norm = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x                : (B, T, E)  — T = n. tracce, E = embed_dim
        key_padding_mask : (B, T)     — True dove la traccia è padding (invalida)
        """
        # Multi-head self-attention con pre-LayerNorm
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attn_out

        # Feed-forward con pre-LayerNorm
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


# ─── Attention Pooling ────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Aggrega le rappresentazioni delle tracce in un unico vettore jet.
    Usa un vettore query apprendibile (come in GN2 e BERT [CLS]).
    Le tracce con padding vengono escluse dalla softmax tramite la mask.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query  = nn.Parameter(torch.randn(embed_dim))
        self.linear = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x    : (B, T, E)
        mask : (B, T) — True = traccia valida
        restituisce: (B, E)
        """
        # Punteggi di attenzione: (B, T)
        scores = self.linear(torch.tanh(x + self.query)).squeeze(-1)

        if mask is not None:
            # Maschera le tracce invalide con -inf prima della softmax
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)          # (B, T)
        pooled  = (weights.unsqueeze(-1) * x).sum(dim=1) # (B, E)
        return pooled


# ─── GN2-Lite ─────────────────────────────────────────────────────────────────

class GN2Lite(nn.Module):
    """
    GN2 semplificato: solo il task primario di classificazione del jet.

    Architettura:
      1. Track initialiser: proietta ogni traccia in embed_dim con 1 hidden layer
      2. Transformer encoder: 2 layer × 4 head (pre-LN)
      3. Attention pooling: tracce → singolo vettore jet
      4. Concatenazione con jet features
      5. Classifier MLP: 3 hidden layer → N_CLASSES
    """

    def __init__(
        self,
        n_jet_feat:   int   = N_JET_FEAT,
        n_track_feat: int   = N_TRACK_FEAT,
        n_classes:    int   = N_CLASSES,
        embed_dim:    int   = 128,
        n_heads:      int   = 4,
        n_layers:     int   = 2,
        ffn_dim:      int   = 256,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── 1. Track initialiser ──────────────────────────────────────────
        # (n_track_feat + n_jet_feat) → embed_dim
        # Le jet features vengono concatenate a ogni traccia (come in GN2)
        combined_feat = n_track_feat + n_jet_feat
        self.track_init = nn.Sequential(
            nn.Linear(combined_feat, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # ── 2. Transformer encoder ────────────────────────────────────────
        self.transformer = nn.ModuleList([
            PreNormTransformerLayer(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.proj_out = nn.Linear(embed_dim, embed_dim)  # proiezione finale tracce

        # ── 3. Attention pooling ──────────────────────────────────────────
        self.pooling = AttentionPooling(embed_dim)

        # ── 4. Final LayerNorm sul jet repr. concatenato ──────────────────
        self.jet_norm = nn.LayerNorm(embed_dim + n_jet_feat)

        # ── 5. Classifier MLP ─────────────────────────────────────────────
        # Schema: (embed_dim + n_jet_feat) → 128 → 64 → 32 → n_classes
        cls_in = embed_dim + n_jet_feat
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(
        self,
        jet_features:   torch.Tensor,
        track_features: torch.Tensor,
        track_mask:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Parametri
        ----------
        jet_features   : (B, N_JET_FEAT)         — features del jet
        track_features : (B, MAX_TRACKS, N_TRACK_FEAT) — features delle tracce
        track_mask     : (B, MAX_TRACKS)  bool   — True = traccia valida

        Restituisce
        -----------
        logits : (B, N_CLASSES)  — logit non normalizzati (da passare a CrossEntropy)
        """
        B, T, _ = track_features.shape

        # ── 1. Concatena jet features a ogni traccia (come fa GN2) ────────
        # jet_features: (B, F_j) → (B, T, F_j)
        jet_expanded = jet_features.unsqueeze(1).expand(B, T, -1)
        combined     = torch.cat([track_features, jet_expanded], dim=-1)  # (B, T, F_j+F_t)

        # ── 2. Track initialiser ──────────────────────────────────────────
        x = self.track_init(combined)  # (B, T, embed_dim)

        # ── 3. Transformer encoder ────────────────────────────────────────
        # key_padding_mask: True dove la traccia è INVALIDA (opposto di track_mask)
        padding_mask = ~track_mask  # (B, T)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.proj_out(x)  # (B, T, embed_dim)

        # ── 4. Attention pooling → jet representation ─────────────────────
        jet_repr = self.pooling(x, mask=track_mask)  # (B, embed_dim)

        # ── 5. Concatena jet features al jet repr. ────────────────────────
        jet_combined = torch.cat([jet_repr, jet_features], dim=-1)  # (B, embed+F_j)
        jet_combined = self.jet_norm(jet_combined)

        # ── 6. Classificazione ────────────────────────────────────────────
        logits = self.classifier(jet_combined)  # (B, N_CLASSES)
        return logits

    # ── Utility: discriminante Db come nel paper ──────────────────────────

    @staticmethod
    def discriminant_db(
        logits: torch.Tensor,
        fc:     float = 0.2,
        ftau:   float = 0.05,
    ) -> torch.Tensor:
        """
        Calcola Db = log(pb / (fc*pc + ftau*ptau + (1-fc-ftau)*pu))
        come nella eq. (1) del paper GN2.

        logits : (B, 4) — ordine: b=0, c=1, u=2, tau=3
        """
        probs = torch.softmax(logits, dim=-1)
        pb, pc, pu, ptau = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
        num = pb
        den = fc * pc + ftau * ptau + (1.0 - fc - ftau) * pu
        return torch.log(num / den.clamp(min=1e-9))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Esempio di utilizzo ──────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # Instanzia il modello
    model = GN2Lite(
        embed_dim=128,
        n_heads=4,
        n_layers=2,
        ffn_dim=256,
        dropout=0.1,
    )
    print(f"Parametri totali: {model.count_parameters():,}")
    print(model)

    # Batch sintetico
    B = 32
    jet_feat   = torch.randn(B, N_JET_FEAT)
    track_feat = torch.randn(B, MAX_TRACKS, N_TRACK_FEAT)

    # Simula tracce valide: ogni jet ha tra 3 e 40 tracce valide
    n_valid = torch.randint(3, MAX_TRACKS + 1, (B,))
    mask = torch.zeros(B, MAX_TRACKS, dtype=torch.bool)
    for i, n in enumerate(n_valid):
        mask[i, :n] = True

    # Forward pass
    with torch.no_grad():
        logits = model(jet_feat, track_feat, mask)
        db     = GN2Lite.discriminant_db(logits)

    print(f"\nInput:")
    print(f"  jet_features   : {jet_feat.shape}")
    print(f"  track_features : {track_feat.shape}")
    print(f"  track_mask     : {mask.shape}  (n_valid tracce: {n_valid[:5].tolist()}...)")
    print(f"\nOutput:")
    print(f"  logits shape   : {logits.shape}")
    print(f"  logits[0]      : {logits[0].tolist()}")
    probs = torch.softmax(logits[0], dim=0)
    print(f"  probs[0]       : b={probs[0]:.3f}  c={probs[1]:.3f}  u={probs[2]:.3f}  tau={probs[3]:.3f}")
    print(f"  Db[0]          : {db[0]:.3f}")