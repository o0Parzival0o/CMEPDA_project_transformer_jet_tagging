"""
evaluate.py
===========
Evaluation script for the GN2 jet flavour tagging pipeline.

Computes classification metrics, ROC curves (D_b and D_c), and optionally
saves a confusion matrix and per-class score distributions.

Outputs:
  Directory specified in ``config["output"]["eval_dir"]`` (default: ``outputs/eval``):

  .. code-block:: text

      outputs/eval/
      ├── metrics.json            - accuracy, per-class precision/recall/F1
      ├── confusion_matrix.pdf    - normalised confusion matrix
      ├── score_distributions.pdf - softmax score distributions per class
      ├── roc_db.pdf              - b-tagging ROC (D_b)
      └── roc_dc.pdf              - c-tagging ROC (D_c)
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from transformer_jet_tagging import utils
from transformer_jet_tagging.constants import FLAVOUR_COLORS
from transformer_jet_tagging.dataset import GN2Dataset, gn2_dataloader
from transformer_jet_tagging.model import GN2
from transformer_jet_tagging.plotting import plot_roc_db, plot_roc_dc

matplotlib.use("Agg")
hep.style.use(hep.style.ATLAS)##########################################################################################################################3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GN2.evaluate")



@torch.no_grad()
def run_inference(
    model: GN2,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over the full loader and collect outputs.

    Args:
        model (GN2): trained GN2 model in eval mode.
        loader (DataLoader): test set DataLoader.
        device (torch.device): torch device.

    Returns:
        proba (np.ndarray): shape (N, n_classes), softmax probabilities.
        preds (np.ndarray): shape (N,), argmax class predictions.
        labels (np.ndarray): shape (N,), true class labels.
    """
    model.eval()

    all_prob, all_labels = [], []

    for batch in loader:
        jet_feats   = batch["jet_features"].to(device)
        track_feats = batch["track_features"].to(device)
        mask        = batch["mask"].to(device)

        prob = model.predict_proba(jet_feats, track_feats, mask)
        all_prob.append(prob.cpu().numpy())
        all_labels.append(batch["label"].numpy())

    prob   = np.concatenate(all_prob,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds  = prob.argmax(axis=1)

    # drop jets with label == -1 (unmapped)
    valid = labels != -1
    if not valid.all():
        n_dropped = int((~valid).sum())
        logger.warning("Dropping %s jets with unmapped label (-1).", n_dropped)
    prob   = prob[valid]
    preds  = preds[valid]
    labels = labels[valid]

    return prob, preds, labels


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    label_map: dict[int, str],
    output_dir: Path,
) -> dict:
    """
    Compute and save classification metrics (accuracy, per-class P/R/F1).

    Args:
        preds (np.ndarray): shape (N,), predicted class indices.
        labels (np.ndarray): shape (N,), true class indices.
        label_map (dict): mapping from class index (int) to class name (str).
        output_dir (Path): directory where metrics.json is saved.

    Returns:
        dict: metrics dictionary (also written to metrics.json).
    """
    # invert label_map: index -> name
    idx_to_name = {v: k for k, v in label_map.items()}
    class_names = [idx_to_name.get(i, str(i)) for i in sorted(set(labels))]

    acc = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(acc),
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall":    report[cls]["recall"],
                "f1-score":  report[cls]["f1-score"],
                "support":   int(report[cls]["support"]),
            }
            for cls in class_names if cls in report
        },
        "macro avg":    report.get("macro avg",    {}),
        "weighted avg": report.get("weighted avg", {}),
    }

    out_path = output_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Test accuracy: %.4f", acc)
    logger.info("Metrics saved to %s", out_path)

    for cls in class_names:
        m = metrics["per_class"].get(cls, {})
        logger.info(
            "  %-15s  prec=%.4f  rec=%.4f  f1=%.4f  n=%s",
            cls,
            m.get("precision", 0.),
            m.get("recall",    0.),
            m.get("f1-score",  0.),
            f"{m.get('support', 0):,}",
        )

    return metrics


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    label_map: dict[int, str],
    output_dir: Path,
) -> None:
    """
    Plot and save the normalised confusion matrix.

    Args:
        preds (np.ndarray): shape (N,), predicted class indices.
        labels (np.ndarray): shape (N,), true class indices.
        label_map (dict): mapping from class index (int) to class name (str).
        output_dir (Path): directory where the PDF is saved.
    """
    idx_to_name = {v: k for k, v in label_map.items()}
    class_indices = sorted(set(labels))
    class_names   = [idx_to_name.get(i, str(i)) for i in class_indices]

    cm = confusion_matrix(labels, preds, labels=class_indices, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted class", fontsize=13)
    ax.set_ylabel("True class", fontsize=13)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val   = cm[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color)

    hep.atlas.label(ax=ax, data=False, loc=0, rlabel="GN2 evaluation")
    fig.tight_layout()

    out = output_dir / "confusion_matrix.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_score_distributions(
    proba: np.ndarray,
    labels: np.ndarray,
    label_map: dict[int, str],
    output_dir: Path,
) -> None:
    """
    Plot softmax score distributions for each class, split by true flavour.

    Args:
        proba (np.ndarray): shape (N, n_classes), softmax probabilities.
        labels (np.ndarray): shape (N,), true class labels.
        label_map (dict): mapping from class index (int) to class name (str).
        output_dir (Path): directory where the PDF is saved.
    """
    idx_to_name = {v: k for k, v in label_map.items()}
    class_indices = sorted(set(labels))
    class_names   = [idx_to_name.get(i, str(i)) for i in class_indices]
    n_classes     = len(class_indices)

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=False)
    if n_classes == 1:
        axes = [axes]

    for col, (cls_idx, cls_name) in enumerate(zip(class_indices, class_names, strict=False)):
        ax   = axes[col]
        bins = np.linspace(0, 1, 50)
        score_col = class_indices.index(cls_idx)   # column in proba for this class

        for true_idx, true_name in zip(class_indices, class_names, strict=False):
            mask   = labels == true_idx
            scores = proba[mask, score_col]
            color  = FLAVOUR_COLORS.get(true_idx, None)
            ax.hist(
                scores,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=color,
                label=true_name,
            )

        ax.set_xlabel(f"P({cls_name})", fontsize=13)
        ax.set_ylabel("Entries (normalised)", fontsize=12)
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        hep.atlas.label(
            ax=ax,
            data=False,
            loc=1,
            rlabel=r"$\sqrt{s}=13.6$ TeV, $t\bar{t}$ simulation",
        )

    fig.tight_layout()
    out = output_dir / "score_distributions.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def evaluate(
    config: dict,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    debug_frac: float = 1.0,
) -> dict:
    """
    Run the full evaluation pipeline on the test set.

    Args:
        config (dict): full configuration dictionary.
        checkpoint_path (Path): path to the best model checkpoint (.pt).
        output_dir (Path): directory where all evaluation outputs are saved.
        device (torch.device): torch device.
        debug_frac (float): fraction of test data to use (default 1.0 = all).

    Returns:
        dict: computed metrics (also written to metrics.json in output_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_dir = Path(config["output"]["preprocess_dir"])
    idx_dir        = preprocess_dir / "indices"
    norm_path      = preprocess_dir / "norm_stats.json"

    # load test indices
    test_indices = np.sort(np.load(idx_dir / "test_indices.npy"))

    if debug_frac < 1.0:
        rng          = np.random.default_rng(seed=0)
        test_indices = np.sort(rng.choice(
            test_indices,
            size    = int(len(test_indices) * debug_frac),
            replace = False,
        ))
        logger.info("Debug mode: using %.1f%% of test set (%s jets).",
                    debug_frac * 100, f"{len(test_indices):,}")
    else:
        logger.info("Test set size: %s jets.", f"{len(test_indices):,}")

    # normalization stats
    with open(norm_path, encoding="utf-8") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    data_config = config["data"]
    label_map   = {int(k): v for k, v in data_config["label_map"].items()}

    # dataset & loader
    test_dataset = GN2Dataset(
        h5_file_path    = data_config["h5_path"],
        jet_indices     = test_indices,
        max_tracks      = data_config.get("max_tracks", 40),
        jet_vars        = data_config["jet_features"],
        track_vars      = data_config["track_features"],
        jet_flavour     = data_config["label"],
        jet_flavour_map = label_map,
        stats           = norm_stats,
    )

    training_config = config.get("training", {})
    test_loader = gn2_dataloader(
        test_dataset,
        batch_size  = training_config.get("batch_size", 1024),
        shuffle     = False,
        num_workers = training_config.get("num_workers", 0),
        pin_memory  = torch.cuda.is_available(),
    )

    # load model from checkpoint
    model = GN2.from_checkpoint(str(checkpoint_path), device)
    model.eval()

    # inference
    logger.info("Running inference on test set ...")
    proba, preds, labels = run_inference(model, test_loader, device)

    # classification metrics
    metrics = compute_metrics(preds, labels, label_map, output_dir)

    # plots
    plot_confusion_matrix(preds, labels, label_map, output_dir)
    plot_score_distributions(proba, labels, label_map, output_dir)

    if config["output"].get("plot_roc", True):
        logger.info("Plotting ROC curves ...")
        plot_roc_db(model=model, loader=test_loader, device=device, output_dir=output_dir)
        plot_roc_dc(model=model, loader=test_loader, device=device, output_dir=output_dir)

    logger.info("Evaluation complete. Outputs saved to '%s/'.", output_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        prog="transformer_jet_tagging.evaluate",
        description="GN2 evaluation on test set",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). "
             "Defaults to <checkpoints_dir>/best_model.pt from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where evaluation outputs are saved. "
             "Defaults to config['output']['eval_dir'] or 'outputs/eval'.",
    )
    parser.add_argument(
        "--debug-frac",
        type=float,
        default=1.0,
        help="Fraction of test data to use (default 1.0 = all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu', 'cuda', 'cuda:0', etc. "
             "Defaults to CUDA if available, else CPU.",
    )

    args = parser.parse_args()

    config = utils.load_config_json(args.config)

    checkpoint_path = Path(
        args.checkpoint
        if args.checkpoint is not None
        else Path(config["output"].get("checkpoints_dir", "outputs/checkpoints")) / "best_model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else config["output"].get("eval_dir", "outputs/eval")
    )

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Device : %s", device)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Output dir: %s", output_dir)

    evaluate(
        config          = config,
        checkpoint_path = checkpoint_path,
        output_dir      = output_dir,
        device          = device,
        debug_frac      = args.debug_frac,
    )


if __name__ == "__main__":
    main()
