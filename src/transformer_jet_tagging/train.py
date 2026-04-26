"""
train.py
========
Training logic for GN2.

Provides:
  - GN2Loss      : combined multi-task loss
  - lr_scheduler : cosine annealing + linear warmup
  - run_epoch    : single epoch train/val loop
  - train        : full training loop, callable from main.py

Outputs (under the directory specified in config["output"]["checkpoints_dir"]):
  outputs/checkpoints/
  ├── runs/
  │   ├── events.out.tfevents.xxxx
  │   └── …
  └── best_model.pt
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import gn2_dataloader
from .model import GN2

logger = logging.getLogger("GN2.train")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class GN2Loss(nn.Module):
    """
    Jet classification loss only.

    L = CE_jet
    """

    def __init__(self):
        """
        Initialize thr GN2 loss.
        """
        super().__init__()
        self.ce_jet = nn.CrossEntropyLoss(ignore_index = -1)

    def forward(self, outputs: dict, labels: dict) -> dict:
        """
        Forward pass to compute the loss.

        Args:
            outputs (dict): model outputs with keys:
                "jet_outputs" (torch.Tensor, shape (batch_size, n_classes)):
                    raw outputs for jet classification.
            labels (dict): ground truth labels with keys:
                "jet_label" (torch.Tensor, shape (batch_size,)): integer class labels for each jet.
        """
        loss_jet = self.ce_jet(outputs["jet_outputs"], labels["jet_label"])

        return {
            "total": loss_jet,
            "jet": loss_jet,
        }


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------
def lr_scheduler(
    optimizer,
    n_total_steps: int,
    warmup_frac: float = 0.01,
    lr_initial: float = 1.0e-07,
    lr_peak: float = 5.0e-04,
    lr_final: float = 1.0e-05,
) -> LambdaLR:
    """
    Build a learning rate scheduler with linear warmup followed by cosine annealing.

    The schedule consists of two phases:
        1. Linear warmup: the learning rate increases from `lr_initial` to `lr_peak`
        over the first `warmup_frac * n_total_steps` steps.
        2. Cosine decay: the learning rate decreases from `lr_peak` to `lr_final`
        following a cosine schedule for the remaining steps.

    Args:
        optimizer: optimizer instance (e.g. AdamW).
        n_total_steps (int): total number of optimizer steps (epochs × batches).
        warmup_frac (float): fraction of steps used for warmup (default: 0.01).
        lr_initial (float): initial learning rate at step 0.
        lr_peak (float): peak learning rate reached after warmup.
        lr_final (float): minimum learning rate at the end of cosine decay.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Scheduler implementing
        warmup + cosine annealing.
    """
    n_warmup = max(1, int(warmup_frac * n_total_steps))

    base_lr = optimizer.param_groups[0]["lr"]

    start_factor = lr_initial / base_lr
    end_factor   = lr_peak / base_lr

    warmup = LinearLR(
        optimizer,
        start_factor = start_factor,
        end_factor   = end_factor,
        total_iters  = n_warmup
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = n_total_steps - n_warmup,
        eta_min = lr_final
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [n_warmup]
    )

    return scheduler


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------
def run_epoch(
    model: GN2,
    loader: DataLoader,
    loss: GN2Loss,
    optimiser: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    is_train: bool,
    scaler: torch.amp.GradScaler = None,
) -> dict:
    """
    Run one full epoch.

    Args:
        model (GN2): GN2 model instance.
        loader (DataLoader): train or val DataLoader.
        criterion (GN2Loss): GN2Loss instance.
        optimiser (AdamW): AdamW instance (unused during validation).
        scheduler (LambdaLR): LambdaLR instance (stepped only during training).
        device (torch.device): Device to move tensors to.
        is_train (bool): True for training, False for validation.
        scaler (torch.amp.GradScaler): optional GradScaler for mixed precision training.
            (default: None)

    Returns:
        dict with averaged loss.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    totals    = {"total": 0.0, "jet": 0.0}
    n_batches = 0
    ctx       = torch.enable_grad if is_train else torch.no_grad

    with ctx():
        for batch in loader:
            jet_vars   = batch["jet_features"].to(device)
            track_vars = batch["track_features"].to(device)
            mask    = batch["mask"].to(device)

            labels = {"jet_label": batch["label"].to(device)}

            if is_train and scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(jet_vars, track_vars, mask)
                    losses  = loss(outputs, labels)
                optimiser.zero_grad()
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
            elif is_train:
                outputs = model(jet_vars, track_vars, mask)
                losses  = loss(outputs, labels)
                optimiser.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                scheduler.step()
            else:
                outputs = model(jet_vars, track_vars, mask)
                losses  = loss(outputs, labels)

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def train(
    model: GN2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> GN2:
    """
    Full training loop.

    Args:
        model (GN2): GN2 instance (already on device).
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        config (dict): full config dict.
        output_dir (Path): directory where checkpoints and TB runs are saved.
        device (torch.device): torch device.

    Returns:
        model (GN2): GN2 model loaded with the best checkpoint weights.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = config.get("training", {})

    loss = GN2Loss()

    lr        = training_config.get("lr_peak", 5.0e-04)
    wd        = training_config.get("weight_decay", 1.0e-05)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    n_epochs      = training_config.get("max_epochs", 20)
    n_total_steps = n_epochs * len(train_loader)
    lr_decay  = lr_scheduler(                           # TODO: riveredere parametri del lr
        optimiser,
        n_total_steps,
        warmup_frac = training_config.get("warmup_frac", 0.01),
        lr_initial = training_config.get("lr_initial", 1.0e-07),
        lr_peak = lr,
        lr_final = training_config.get("lr_final", 1.0e-05),
    )

    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    best_val_loss   = float("inf")
    checkpoint_path = output_dir / "best_model.pt"

    history = {"train_loss": [], "val_loss": [], "lr": []}
    for epoch in range(1, n_epochs + 1):

        train_losses = run_epoch(model, train_loader, loss, optimiser,
                                 lr_decay, device, is_train=True,  scaler=scaler)
        val_losses   = run_epoch(model, val_loader,   loss, optimiser,
                                 lr_decay, device, is_train=False)

        lr_now = lr_decay.get_last_lr()[0]

        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_losses["total"])
        history["lr"].append(lr_now)

        logger.info("Epoch %s/%s | train loss=%s (jet=%s) | val=%s | lr=%s",
                    f"{epoch:4d}", n_epochs, f"{train_losses['total']:.4f}",
                    f"{train_losses['jet']:.4f}", f"{val_losses['total']:.4f}", f"{lr_now:.2e}")

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
            }, checkpoint_path)
            logger.info("    New best val_loss=%s - saved to %s",
                        f"{best_val_loss:.4f}", checkpoint_path)

    writer.close()
    logger.info("Training complete.")

    # reload best weights before returning
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"]
    )

    return model, history


if __name__ == "__main__":

    import argparse
    import json

    from . import utils
    from .dataset import GN2Dataset

    logging.basicConfig(
        level  = logging.DEBUG,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="GN2 training (standalone debug)")
    parser.add_argument("--config",     type=str,   default="configs/config.json")
    parser.add_argument("--debug-frac", type=float, default=0.05,
                        help="Fraction of data to use for debug (default: 5%%)")
    args = parser.parse_args()

    cfg = utils.load_config_json(args.config)

    preprocess_dir = Path(cfg["output"]["preprocess_dir"])
    ckpt_dir     = Path(cfg["output"].get("checkpoints_dir", "outputs/checkpoints"))

    cfg_jet_vars   = cfg["data"]["jet_features"]
    cfg_track_vars = cfg["data"]["track_features"]
    label_vars = cfg["data"]["label"]
    label_map  = {int(k): v for k, v in cfg["data"]["label_map"].items()}

    # run preprocessing if artifacts are missing
    idx_dir   = preprocess_dir / "indices"
    norm_path = preprocess_dir / "norm_stats.json"
    artifacts = [idx_dir / "train_indices.npy", idx_dir / "val_indices.npy", norm_path]

    if not all(p.exists() for p in artifacts):
        logger.info("Preprocessing artifacts not found - running preprocess.py …")
        from . import preprocess
        preprocess.main(args.config)

    train_indices = np.load(idx_dir / "train_indices.npy")
    val_indices   = np.load(idx_dir / "val_indices.npy")

    if args.debug_frac < 1.0:
        rng = np.random.default_rng(seed=42)
        train_indices = rng.choice(
            train_indices,
            size=int(len(train_indices) * args.debug_frac),
            replace=False
        )
        val_indices = rng.choice(
            val_indices,
            size=int(len(val_indices) * args.debug_frac),
            replace=False
        )
        logger.info("Debug mode: %s of data - train=%s  val=%s",
                    f"{args.debug_frac:.0%}", f"{len(train_indices):,}", f"{len(val_indices):,}")

    train_indices = np.sort(train_indices)
    val_indices   = np.sort(val_indices)

    with open(norm_path, encoding="utf-8") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    common_kwargs = dict(
        h5_file_path    = cfg["data"]["h5_path"],
        max_tracks      = cfg["data"].get("max_tracks", 40),
        jet_vars        = cfg_jet_vars,
        track_vars      = cfg_track_vars,
        jet_flavour     = label_vars,
        jet_flavour_map = label_map,
        stats           = norm_stats,
    )

    cfg_training = cfg.get("training", {})
    batch_size      = cfg_training.get("batch_size", 1024)
    num_workers     = cfg_training.get("num_workers", 0)

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    train_dataset = GN2Dataset(jet_indices=train_indices, **common_kwargs)
    val_dataset   = GN2Dataset(jet_indices=val_indices,   **common_kwargs)

    cfg_train_loader = gn2_dataloader(train_dataset, **loader_kwargs,
                                 shuffle=cfg["data"].get("shuffle", False))
    cfg_val_loader   = gn2_dataloader(val_dataset, **loader_kwargs, shuffle=False)

    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = cfg.get("model", {})
    run_model = GN2(
        n_jet_vars       = len(cfg_jet_vars),
        n_track_vars     = len(cfg_track_vars),
        n_classes        = len(label_map),
        init_hidden_dim  = model_config.get("initialiser_hidden_dim", None),
        init_output_dim  = model_config.get("initialiser_output_dim", None),
        embed_dim        = model_config.get("transformer_embed_dim", None),
        n_heads          = model_config.get("transformer_n_heads", None),
        n_layers         = model_config.get("transformer_n_layers", None),
        ff_dim           = model_config.get("transformer_ff_dim", None),
        pool_dim         = model_config.get("pooling_dim", None),
        dropout          = model_config.get("transformer_dropout", None),
        head_hidden_dims = model_config.get("head_hidden_dims", None),
        activation       = model_config.get("activation", None),
    ).to(run_device)

    train(
        run_model,
        cfg_train_loader,
        cfg_val_loader,
        cfg,
        ckpt_dir,
        run_device
    )
