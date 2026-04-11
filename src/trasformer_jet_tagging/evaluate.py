"""
evaluate.py
===========
Evaluation script for GN2.

Produces:
  1. ROC curves  — b-jet efficiency vs c/light/tau-jet rejection
  2. D_b discriminant distributions per flavour
  3. Rejection at standard operating points (65, 70, 77, 85, 90% b-eff)
  4. Confusion matrix on jet classification
  5. Per-flavour accuracy

All plots are saved under config["output"]["eval_dir"].

Usage:
    python evaluate.py --config configs/config.json --checkpoint outputs/checkpoints/best_model.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from torch.utils.data import DataLoader

import src.trasformer_jet_tagging.utils as utils
from src.trasformer_jet_tagging.dataset import GN2Dataset
from src.trasformer_jet_tagging.model import GN2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GN2.evaluate")

# Standard b-tagging operating points used by ATLAS
B_TAG_OPS = [0.65, 0.70, 0.77, 0.85, 0.90]

# class index → flavour name  (matches JET_FLAVOUR_MAP in dataset.py)
FLAVOUR_NAMES = {0: "light", 1: "c", 2: "b", 3: "tau"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[GN2, dict]:
    """Load GN2 from a checkpoint saved by train.py."""
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    jet_vars   = config["data"]["jet_features"]
    track_vars = config["data"]["track_features"]
    model_cfg  = config.get("model", {})

    model = GN2(
        n_jet_vars    = len(jet_vars),
        n_track_vars  = len(track_vars),
        n_classes     = model_cfg.get("n_classes", 4),
        n_track_origin= model_cfg.get("n_track_origin", 7),
        embed_dim     = model_cfg.get("embed_dim", 256),
        n_heads       = model_cfg.get("n_heads", 8),
        n_layers      = model_cfg.get("n_layers", 4),
        ff_dim        = model_cfg.get("ff_dim", 512),
        pool_dim      = model_cfg.get("pool_dim", 128),
        dropout       = 0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model, config


def collect_predictions(
    model  : GN2,
    loader : DataLoader,
    device : torch.device,
    fc     : float = 0.2,
    ftau   : float = 0.05,
) -> dict:
    """
    Run inference on the full loader and collect:
      - proba      : (N, 4)   softmax probabilities
      - db         : (N,)     b-tagging discriminant D_b
      - labels     : (N,)     true jet flavour labels
    """
    all_proba  = []
    all_db     = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            jet_f  = batch["jet_features"].to(device)
            trk_f  = batch["track_features"].to(device)
            mask   = batch["mask"].to(device)
            labels = batch["label"]

            proba = model.predict_proba(jet_f, trk_f, mask)   # (B, 4)
            db    = model.discriminant_db(jet_f, trk_f, mask, fc=fc, ftau=ftau)

            all_proba.append(proba.cpu().numpy())
            all_db.append(db.cpu().numpy())
            all_labels.append(labels.numpy())

    return {
        "proba" : np.concatenate(all_proba,  axis=0),
        "db"    : np.concatenate(all_db,     axis=0),
        "labels": np.concatenate(all_labels, axis=0),
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_discriminant(db: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
    """D_b distributions per flavour."""
    fig, ax = plt.subplots(figsize=(7, 5))
    flavours = [(2, "b-jets", "royalblue"), (1, "c-jets", "darkorange"), (0, "light-jets", "green")]
    for cls, name, color in flavours:
        mask = labels == cls
        ax.hist(db[mask], bins=60, range=(-10, 15), histtype="step",
                lw=2, label=name, color=color, density=True)
    ax.set_xlabel(r"$D_b$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(fontsize=12)
    ax.set_title("GN2 b-tagging discriminant")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved discriminant plot → {save_path}")


def plot_roc(db: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
    """
    Background rejection vs b-jet efficiency (ROC) for c-jets and light-jets.
    Mirrors Fig. 2 of the paper.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    b_mask = labels == 2
    for bg_cls, name, color, ls in [
        (1, "c-jets",     "darkorange", "-"),
        (0, "light-jets", "green",      "-."),
    ]:
        bg_mask = labels == bg_cls
        y_true  = np.concatenate([np.ones(b_mask.sum()), np.zeros(bg_mask.sum())])
        y_score = np.concatenate([db[b_mask], db[bg_mask]])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # rejection = 1 / fpr  (avoid division by zero)
        valid = fpr > 0
        ax.plot(tpr[valid], 1.0 / fpr[valid], color=color, ls=ls,
                lw=2, label=f"{name} rejection")

    # Mark standard operating points on the b-axis
    for op in B_TAG_OPS:
        ax.axvline(op, color="grey", lw=0.8, ls=":")

    ax.set_xlabel("b-jet tagging efficiency", fontsize=13)
    ax.set_ylabel("Background rejection", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.0)
    ax.legend(fontsize=12)
    ax.set_title("GN2 ROC curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC plot → {save_path}")


def plot_confusion(proba: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
    """Normalised confusion matrix."""
    preds  = proba.argmax(axis=1)
    names  = [FLAVOUR_NAMES[i] for i in range(4)]
    cm     = confusion_matrix(labels, preds, normalize="true")
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title("GN2 confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {save_path}")


def compute_rejections_at_ops(
    db        : np.ndarray,
    labels    : np.ndarray,
    b_eff_ops : list = B_TAG_OPS,
) -> dict:
    """
    For each operating point (b-jet efficiency), compute the threshold
    and return c-jet and light-jet rejection.
    """
    b_mask = labels == 2
    c_mask = labels == 1
    l_mask = labels == 0

    results = {}
    for op in b_eff_ops:
        # threshold: the (1-op) percentile of the b-jet discriminant
        thr = np.percentile(db[b_mask], (1 - op) * 100)
        c_rej = 1.0 / max((db[c_mask] > thr).mean(), 1e-9)
        l_rej = 1.0 / max((db[l_mask] > thr).mean(), 1e-9)
        results[op] = {"threshold": thr, "c_rejection": c_rej, "light_rejection": l_rej}
        logger.info(
            f"  OP {op*100:.0f}%  thr={thr:.3f}  "
            f"c-rej={c_rej:.1f}  light-rej={l_rej:.1f}"
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, checkpoint_path: str, fc: float, ftau: float) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Load model
    model, config = load_model(checkpoint_path, device)

    # 2. Load preprocessing artifacts
    preprocess_dir = Path(config["output"]["preprocess_dir"])
    idx_dir   = preprocess_dir / "indices"
    norm_path = preprocess_dir / "norm_stats.json"

    test_idx   = np.load(idx_dir / "test_indices.npy")
    with open(norm_path) as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    logger.info(f"Test set: {len(test_idx):,} jets")

    # 3. DataLoader
    test_dataset = GN2Dataset(
        file_path       = config["data"]["h5_path"],
        indices         = test_idx,
        n_tracks        = config["data"].get("max_tracks", 40),
        jet_vars        = config["data"]["jet_features"],
        track_vars      = config["data"]["track_features"],
        jet_flavour     = config["data"]["label"],
        jet_flavour_map = config["data"]["label_map"],
        norm_stats      = norm_stats,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config["training"].get("batch_size", 1024),
        shuffle     = False,
        num_workers = config["training"].get("num_workers", 4),
        pin_memory  = torch.cuda.is_available(),
    )

    # 4. Collect predictions
    logger.info("Running inference on test set …")
    preds = collect_predictions(model, test_loader, device, fc=fc, ftau=ftau)

    # 5. Metrics
    logger.info("--- Rejection at standard operating points ---")
    rejections = compute_rejections_at_ops(preds["db"], preds["labels"])

    acc = (preds["proba"].argmax(1) == preds["labels"]).mean()
    logger.info(f"Overall jet classification accuracy: {acc:.4f}")

    for cls in range(4):
        mask = preds["labels"] == cls
        cls_acc = (preds["proba"][mask].argmax(1) == cls).mean()
        logger.info(f"  {FLAVOUR_NAMES[cls]}-jet accuracy: {cls_acc:.4f}")

    # 6. Save plots
    eval_dir = Path(config["output"].get("eval_dir", "outputs/eval"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    plot_discriminant(preds["db"],    preds["labels"], eval_dir / "discriminant_Db.png")
    plot_roc(         preds["db"],    preds["labels"], eval_dir / "roc_curves.png")
    plot_confusion(   preds["proba"], preds["labels"], eval_dir / "confusion_matrix.png")

    # 7. Save rejection table to JSON
    rej_path = eval_dir / "rejections_at_ops.json"
    with open(rej_path, "w") as f:
        json.dump(
            {f"{int(op*100)}pct": v for op, v in rejections.items()},
            f, indent=2
        )
    logger.info(f"Saved rejection table → {rej_path}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GN2 evaluation")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--fc",   type=float, default=0.2,
                        help="fc parameter for D_b discriminant (default 0.2)")
    parser.add_argument("--ftau", type=float, default=0.05,
                        help="ftau parameter for D_b discriminant (default 0.05)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.fc, args.ftau)