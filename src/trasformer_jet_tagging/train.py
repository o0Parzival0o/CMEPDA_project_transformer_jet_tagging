"""
train.py
========
Training logic for GN2.

Provides:
  - GN2Loss         : combined multi-task loss
  - build_scheduler : cosine annealing + linear warmup
  - run_epoch       : single epoch train/val loop
  - train           : full training loop, callable from main.py

Standalone usage (debug):
  python -m src.trasformer_jet_tagging.train --config configs/config.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.trasformer_jet_tagging.utils as utils
from src.trasformer_jet_tagging.dataset import GN2Dataset, GN2DataLoader
from src.trasformer_jet_tagging.model import GN2

logger = logging.getLogger("GN2.train")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class GN2Loss(nn.Module):
    """
    Combined loss for the three GN2 tasks.

    L = w_jet    * CE(jet_logits, jet_labels)
      + w_origin * CE(origin_logits, origin_labels)   [valid tracks only]
      + w_vertex * CE(vertex_logits, vertex_labels)   [valid pairs only]

    Track-origin uses class-weighted CE to handle imbalance (paper §Methods).
    """

    def __init__(
        self,
        w_jet   : float = 1.0,
        w_origin: float = 0.5,
        w_vertex: float = 0.5,
        origin_class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.w_jet    = w_jet
        self.w_origin = w_origin
        self.w_vertex = w_vertex

        self.ce_jet    = nn.CrossEntropyLoss()
        self.ce_origin = nn.CrossEntropyLoss(
            weight=origin_class_weights,
            ignore_index=-1
        )
        self.ce_vertex = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        outputs: dict,
        labels : dict,
        mask   : torch.Tensor,
    ) -> dict:
        """
        Args:
            outputs: dict from GN2.forward()
            labels : dict with keys:
                'jet_label'    : (B,)      long
                'origin_label' : (B, T)    long  (-1 = ignore)
                'vertex_label' : (B, T, T) long  (-1 = ignore)
            mask   : (B, T) bool, True = real track

        Returns:
            dict with 'total', 'jet', 'origin', 'vertex' losses.
        """
        loss_jet = self.ce_jet(outputs["jet_logits"], labels["jet_label"])

        B, T, C = outputs["origin_logits"].shape
        loss_origin = self.ce_origin(
            outputs["origin_logits"].reshape(B * T, C),
            labels["origin_label"].reshape(B * T),
        )

        B, T, _, C2 = outputs["vertex_logits"].shape
        pair_mask  = mask.unsqueeze(2) & mask.unsqueeze(1)
        vtx_logits = outputs["vertex_logits"].reshape(B * T * T, C2)
        vtx_labels = labels["vertex_label"].reshape(B * T * T)
        vtx_labels = vtx_labels.masked_fill(~pair_mask.reshape(B * T * T), -1)
        loss_vertex = self.ce_vertex(vtx_logits, vtx_labels)

        total = (self.w_jet    * loss_jet
               + self.w_origin * loss_origin
               + self.w_vertex * loss_vertex)

        return {"total": total, "jet": loss_jet, "origin": loss_origin, "vertex": loss_vertex}


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimiser, n_total_steps: int, warmup_frac: float = 0.01) -> LambdaLR:
    """
    Cosine annealing with linear warmup.

    Args:
        optimiser     : AdamW instance.
        n_total_steps : total number of optimiser steps (epochs x batches).
        warmup_frac   : fraction of steps used for linear warmup (default 0.01).

    Returns:
        LambdaLR scheduler.
    """
    n_warmup = max(1, int(warmup_frac * n_total_steps))

    def lr_lambda(step: int) -> float:
        if step < n_warmup:
            return step / n_warmup
        progress = (step - n_warmup) / max(1, n_total_steps - n_warmup)
        cosine   = 0.5 * (1 + np.cos(np.pi * progress))
        return max(cosine, 1e-5 / 5e-4)

    return LambdaLR(optimiser, lr_lambda)


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------
def run_epoch(
    model    : GN2,
    loader   : DataLoader,
    criterion: GN2Loss,
    optimiser: AdamW,
    scheduler: LambdaLR,
    device   : torch.device,
    is_train : bool,
    scaler   : torch.cuda.amp.GradScaler = None,
) -> dict:
    """
    Run one full epoch.

    Args:
        model     : GN2 instance.
        loader    : train or val DataLoader.
        criterion : GN2Loss instance.
        optimiser : AdamW instance (unused during validation).
        scheduler : LambdaLR instance (stepped only during training).
        device    : torch device.
        is_train  : True for training, False for validation.
        scaler    : GradScaler for AMP (None = disabled).

    Returns:
        dict with averaged losses: 'total', 'jet', 'origin', 'vertex'.
    """
    model.train() if is_train else model.eval()

    totals    = {"total": 0.0, "jet": 0.0, "origin": 0.0, "vertex": 0.0}
    n_batches = 0
    ctx       = torch.enable_grad if is_train else torch.no_grad

    with ctx():
        for batch in loader:
            jet_f = batch["jet_features"].to(device)
            trk_f = batch["track_features"].to(device)
            mask  = batch["mask"].to(device)

            labels = {
                "jet_label": batch["label"].to(device),
                "origin_label": batch.get(
                    "origin_label",
                    torch.full((jet_f.size(0), trk_f.size(1)), -1, dtype=torch.long)
                ).to(device),
                "vertex_label": batch.get(
                    "vertex_label",
                    torch.full((jet_f.size(0), trk_f.size(1), trk_f.size(1)), -1, dtype=torch.long)
                ).to(device),
            }

            if is_train and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(jet_f, trk_f, mask)
                    losses  = criterion(outputs, labels, mask)
                optimiser.zero_grad()
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
            elif is_train:
                outputs = model(jet_f, trk_f, mask)
                losses  = criterion(outputs, labels, mask)
                optimiser.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                scheduler.step()
            else:
                outputs = model(jet_f, trk_f, mask)
                losses  = criterion(outputs, labels, mask)

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def train(
    model       : GN2,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    config      : dict,
    output_dir  : Path,
    device      : torch.device,
) -> GN2:
    """
    Full training loop. Designed to be called from main.py.

    Handles loss, optimiser, scheduler, AMP, TensorBoard logging,
    and best-checkpoint saving.

    Args:
        model        : GN2 instance (already on device).
        train_loader : DataLoader for training set.
        val_loader   : DataLoader for validation set.
        config       : full config dict (sections: 'training', 'output').
        output_dir   : directory where checkpoints and TB runs are saved.
        device       : torch device.

    Returns:
        GN2 model loaded with the best checkpoint weights.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = config.get("training", {})

    criterion = GN2Loss(
        w_jet    = training_cfg.get("w_jet",    1.0),
        w_origin = training_cfg.get("w_origin", 0.5),
        w_vertex = training_cfg.get("w_vertex", 0.5),
    )

    lr        = training_cfg.get("lr", 5e-4)
    wd        = training_cfg.get("weight_decay", 1e-5)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    n_epochs      = training_cfg.get("epochs", 20)
    n_total_steps = n_epochs * len(train_loader)
    scheduler     = build_scheduler(
        optimiser,
        n_total_steps,
        warmup_frac=training_cfg.get("warmup_frac", 0.01),
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    best_val_loss = float("inf")
    ckpt_path     = output_dir / "best_model.pt"

    for epoch in range(1, n_epochs + 1):
        train_losses = run_epoch(model, train_loader, criterion, optimiser,
                                 scheduler, device, is_train=True,  scaler=scaler)
        val_losses   = run_epoch(model, val_loader,   criterion, optimiser,
                                 scheduler, device, is_train=False)

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train={train_losses['total']:.4f} "
            f"(jet={train_losses['jet']:.4f} "
            f"orig={train_losses['origin']:.4f} "
            f"vtx={train_losses['vertex']:.4f}) | "
            f"val={val_losses['total']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        for k, v in train_losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_losses.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", lr_now, epoch)

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_loss"   : best_val_loss,
                "config"     : config,
            }, ckpt_path)
            logger.info(f"  ↳ New best val_loss={best_val_loss:.4f} — saved to {ckpt_path}")

    writer.close()
    logger.info("Training complete.")

    # reload best weights before returning
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    return model


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

def _build_components_from_config(config_path: str, debug_frac: float = 0.05):
    """
    Build all components needed for training directly from a config file.
    Used only by the __main__ block for standalone debug runs.

    Args:
        config_path : path to JSON config file.
        debug_frac  : fraction of data to use (default 0.05 = 5%).

    Returns:
        tuple: (model, train_loader, val_loader, config, output_dir, device)
    """
    config = utils.load_config_json(config_path)

    preprocess_dir = Path(config["output"]["preprocess_dir"])
    output_dir     = Path(config["output"].get("checkpoints_dir", "outputs/checkpoints"))

    # run preprocessing if artifacts are missing
    idx_dir   = preprocess_dir / "indices"
    norm_path = preprocess_dir / "norm_stats.json"
    artifacts = [idx_dir / "train_indices.npy", idx_dir / "val_indices.npy", norm_path]

    if not all(p.exists() for p in artifacts):
        logger.info("Preprocessing artifacts not found — running preprocess.py …")
        from src.trasformer_jet_tagging import preprocess
        preprocess.main(config_path)

    train_indices = np.load(idx_dir / "train_indices.npy")
    val_indices   = np.load(idx_dir / "val_indices.npy")

    if debug_frac < 1.0:
        rng           = np.random.default_rng(seed=42)
        train_indices = rng.choice(train_indices, size=int(len(train_indices) * debug_frac), replace=False)
        val_indices   = rng.choice(val_indices,   size=int(len(val_indices)   * debug_frac), replace=False)
        logger.info(f"Debug mode: {debug_frac:.0%} of data — "
                    f"train={len(train_indices):,}  val={len(val_indices):,}")

    train_indices = np.sort(train_indices)
    val_indices   = np.sort(val_indices)

    with open(norm_path) as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    file_path  = config["data"]["h5_path"]
    jet_vars   = config["data"]["jet_features"]
    track_vars = config["data"]["track_features"]
    label_var  = config["data"]["label"]
    label_map  = {int(k): v for k, v in config["data"]["label_map"].items()}
    max_tracks = config["data"].get("max_tracks", 40)

    training_cfg = config.get("training", {})
    batch_size   = training_cfg.get("batch_size", 1024)
    num_workers  = training_cfg.get("num_workers", 0)

    common_kwargs = dict(
        file_path       = file_path,
        n_tracks        = max_tracks,
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = label_var,
        jet_flavour_map = label_map,
        norm_stats      = norm_stats,
    )
    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    train_loader = GN2DataLoader(GN2Dataset(indices=train_indices, **common_kwargs),
                                    **loader_kwargs, shuffle=True)
    val_loader   = GN2DataLoader(GN2Dataset(indices=val_indices,   **common_kwargs),
                                    **loader_kwargs, shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config.get("model", {})
    model = GN2(
        n_jet_vars   = len(jet_vars),
        n_track_vars = len(track_vars),
        n_classes    = model_cfg.get("n_classes", 4),
        embed_dim    = model_cfg.get("embed_dim", 256),
        n_heads      = model_cfg.get("n_heads", 8),
        n_layers     = model_cfg.get("n_layers", 4),
        ff_dim       = model_cfg.get("ff_dim", 512),
        pool_dim     = model_cfg.get("pool_dim", 128),
        dropout      = model_cfg.get("dropout", 0.0),
    ).to(device)

    return model, train_loader, val_loader, config, output_dir, device


if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="GN2 training (standalone debug)")
    parser.add_argument("--config",      type=str,   default="configs/config.json")
    parser.add_argument("--debug-frac",  type=float, default=0.05,
                        help="Fraction of data to use for debug (default: 5%%)")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    args = parser.parse_args()

    model, train_loader, val_loader, config, output_dir, device = \
        _build_components_from_config(args.config, debug_frac=args.debug_frac)

    # CLI overrides (only relevant for debug runs)
    if args.epochs     is not None: config["training"]["epochs"]     = args.epochs
    if args.lr         is not None: config["training"]["lr"]         = args.lr
    if args.batch_size is not None: config["training"]["batch_size"] = args.batch_size

    train(model, train_loader, val_loader, config, output_dir, device)