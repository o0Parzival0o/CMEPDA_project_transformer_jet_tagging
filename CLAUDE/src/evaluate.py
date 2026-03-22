"""
evaluate.py
===========
Valutazione del modello GN2 sul test set.

Produce:
  - Tabella degli Operating Points (rejection a 70%, 77%, 85%, ecc.)
  - ROC curves per b-tagging (c-jet rejection e light-jet rejection vs efficienza)
  - Distribuzioni del discriminante D_b
  - Confronto con GN2v01 e DL1d se i loro score sono disponibili nel file HDF5
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Import opzionale matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend (server/cluster)
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib non disponibile — i plot non verranno generati")


# ──────────────────────────────────────────────────────────────────────────────
# Inference sul test set
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_test_set(
    model:       torch.nn.Module,
    test_loader: DataLoader,
    device:      torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Esegue inference su tutto il test set.

    Returns:
        probs  : (N, 4) — probabilità [pb, pc, pu, ptau]
        labels : (N,)   — classi vere {0=b, 1=c, 2=u, 3=tau}
    """
    model.eval()
    all_probs  = []
    all_labels = []

    for jet_feat, track_feat, track_mask, labels in test_loader:
        jet_feat   = jet_feat.to(device)
        track_feat = track_feat.to(device)
        track_mask = track_mask.to(device)

        logits = model(jet_feat, track_feat, track_mask)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Rejection vs efficiency (ROC-style per HEP)
# ──────────────────────────────────────────────────────────────────────────────

def rejection_vs_efficiency(
    scores:      np.ndarray,   # discriminante D_b per tutti i jet
    true_labels: np.ndarray,   # classi vere
    signal_cls:  int = 0,      # b-jet
    bg_cls:      int = 1,      # classe di background (c o light)
    n_points:    int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola rejection del background vs efficienza del segnale
    al variare della soglia.

    Returns:
        eff_signal : array di efficienze del segnale
        rej_bg     : array di rejection del background
    """
    sig_scores = scores[true_labels == signal_cls]
    bg_scores  = scores[true_labels == bg_cls]

    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    eff_sig = np.array([(sig_scores > t).mean() for t in thresholds])
    mistagging = np.array([(bg_scores > t).mean() for t in thresholds])

    # Evita divisione per zero
    mistagging = np.maximum(mistagging, 1e-10)
    rej_bg = 1.0 / mistagging

    # Ordina per efficienza crescente
    order   = np.argsort(eff_sig)
    return eff_sig[order], rej_bg[order]


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_rejection_curves(
    probs:      np.ndarray,   # (N, 4)
    labels:     np.ndarray,   # (N,)
    save_dir:   str,
    fc:         float = 0.2,
    ftau:       float = 0.05,
    title_suffix: str = "",
):
    """
    Plotta c-jet rejection e light-jet rejection vs b-jet tagging efficiency.
    Stessa visualizzazione di Fig. 2 e Fig. 4 dell'articolo.
    """
    if not HAS_MPL:
        return

    from .discriminant import compute_db

    pb, pc, pu, ptau = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
    db = compute_db(pb, pc, pu, ptau, fc=fc, ftau=ftau)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"GN2 b-tagging performance{title_suffix}", fontsize=13)

    colors = {"c-jet": "#e6194b", "light-jet": "#3cb44b"}
    bg_configs = [
        (1, "c-jet",    axes[0], "c-jet rejection"),
        (2, "light-jet", axes[1], "light-jet rejection"),
    ]

    for bg_cls, bg_name, ax, ylabel in bg_configs:
        eff, rej = rejection_vs_efficiency(db, labels,
                                           signal_cls=0, bg_cls=bg_cls)
        ax.semilogy(eff, rej, color=colors[bg_name], lw=2, label="GN2 (questo modello)")

        # Linee verticali per gli operating point standard
        for op in [0.70, 0.77, 0.85]:
            ax.axvline(op, color="gray", lw=0.8, linestyle="--", alpha=0.6)
            ax.text(op + 0.005, ax.get_ylim()[0] * 1.5,
                    f"{int(op*100)}%", fontsize=8, color="gray")

        ax.set_xlabel("b-jet tagging efficiency", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(0.5, 1.0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "rejection_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot salvato: {out}")


def plot_discriminant_distribution(
    probs:    np.ndarray,
    labels:   np.ndarray,
    save_dir: str,
    fc:       float = 0.2,
    ftau:     float = 0.05,
):
    """Distribuzioni del discriminante D_b per classe."""
    if not HAS_MPL:
        return

    from .discriminant import compute_db

    pb, pc, pu, ptau = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
    db = compute_db(pb, pc, pu, ptau, fc=fc, ftau=ftau)

    fig, ax = plt.subplots(figsize=(8, 5))

    class_cfg = [
        (0, "b-jet",     "#2171b5", "solid"),
        (1, "c-jet",     "#e6194b", "dashed"),
        (2, "light-jet", "#3cb44b", "dotted"),
        (3, "tau-jet",   "#ff7f00", "dashdot"),
    ]
    for cls, name, color, ls in class_cfg:
        mask = (labels == cls)
        if mask.sum() == 0:
            continue
        ax.hist(db[mask], bins=100, density=True,
                histtype="step", color=color, linestyle=ls,
                lw=1.5, label=f"{name} (n={mask.sum():,})")

    ax.set_xlabel(r"$D_b$ discriminant", fontsize=12)
    ax.set_ylabel("Density (normalised)", fontsize=12)
    ax.set_title("GN2 discriminant distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = Path(save_dir) / "discriminant_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot salvato: {out}")


def plot_training_history(history: dict, save_dir: str):
    """Plotta train/val loss e learning rate nel tempo."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train loss", color="#2171b5")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss",   color="#e6194b")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, history["lr"], color="#3cb44b")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "training_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot salvato: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Valutazione completa
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model:       torch.nn.Module,
    test_loader: DataLoader,
    cfg:         dict,
    save_dir:    str,
    history:     Optional[dict] = None,
):
    """
    Pipeline di valutazione completa sul test set.
    Stampa la tabella degli Operating Points e salva i plot.
    """
    from .discriminant import compute_db, evaluate_operating_points

    ocfg = cfg.get("output", {})
    dcfg = cfg.get("discriminant", {})

    device_str = cfg["training"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("=" * 60)
    logger.info("Valutazione sul test set...")
    logger.info("=" * 60)

    # Inference
    probs, labels = predict_test_set(model, test_loader, device)

    # Calcola D_b
    pb, pc, pu, ptau = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
    fc   = dcfg.get("fc_btag",   0.2)
    ftau = dcfg.get("ftau_btag", 0.05)
    db   = compute_db(pb, pc, pu, ptau, fc=fc, ftau=ftau)

    # Operating Points
    ops = dcfg.get("operating_points", [0.65, 0.70, 0.77, 0.85, 0.90])
    evaluate_operating_points(db, labels, b_efficiencies=ops)

    # Accuracy per classe
    preds = probs.argmax(axis=-1)
    class_names = ["b-jet", "c-jet", "light-jet", "tau-jet"]
    logger.info("\nAccuracy per classe:")
    for cls in range(4):
        mask = (labels == cls)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == cls).mean()
        logger.info(f"  {class_names[cls]:12s}: {acc*100:.2f}% "
                    f"(n={mask.sum():,})")

    # Plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if ocfg.get("plot_rejection_vs_efficiency", True):
        plot_rejection_curves(probs, labels, save_dir, fc=fc, ftau=ftau)

    plot_discriminant_distribution(probs, labels, save_dir, fc=fc, ftau=ftau)

    if history is not None:
        plot_training_history(history, save_dir)

    # Salva probs e labels come numpy (utile per analisi successive)
    np.save(Path(save_dir) / "test_probs.npy",  probs)
    np.save(Path(save_dir) / "test_labels.npy", labels)
    logger.info(f"Probabilità e label salvati in {save_dir}")