"""
__main__.py
===========
Entry point for the transformer_jet_tagging package.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from transformer_jet_tagging import __version__, utils
from transformer_jet_tagging.dataset import GN2Dataset, gn2_dataloader
from transformer_jet_tagging.evaluate import evaluate
from transformer_jet_tagging.model import GN2
from transformer_jet_tagging.train import train

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GN2")


def main():
    parser = argparse.ArgumentParser(
        prog="transformer_jet_tagging",
        description="GN2 transformer jet tagging pipeline",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to the JSON configuration file.",
    )

    parser.add_argument(
        "--debug-frac",
        type=float,
        default=1.0,
        help="Fraction of data to use",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    debug_frac  = args.debug_frac

    # load configuration
    config = utils.load_config_json(config_path)

    file_path      = config["data"]["h5_path"]
    preprocess_dir = Path(config["output"]["preprocess_dir"])

    jet_vars   = config["data"]["jet_features"]
    track_vars = config["data"]["track_features"]
    label_vars = config["data"]["label"]
    label_map  = {int(k): v for k, v in config["data"]["label_map"].items()}

    batch_size  = config["training"].get("batch_size", 1024)
    shuffle_var = config["data"].get("shuffle", False)

    # preprocessing
    idx_dir = preprocess_dir / "indices"
    norm_path = preprocess_dir / "norm_stats.json"

    artifacts = [
        idx_dir / "train_indices.npy",
        idx_dir / "val_indices.npy",
        idx_dir / "test_indices.npy",
        norm_path,
    ]

    if not all(p.exists() for p in artifacts):
        logger.info("Preprocessing not found: running preprocess")

        from transformer_jet_tagging.preprocess import run_preprocess

        run_preprocess(config_path=config_path)

    for path in artifacts:
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {path}")

    train_indices = np.load(artifacts[0])
    val_indices   = np.load(artifacts[1])
    test_indices  = np.load(artifacts[2])

    if debug_frac < 1.0:
        rng = np.random.default_rng(seed=42)

        train_indices = rng.choice(
            train_indices,
            size=int(len(train_indices) * debug_frac),
            replace=False,
        )
        val_indices = rng.choice(
            val_indices,
            size=int(len(val_indices) * debug_frac),
            replace=False,
        )
        test_indices = rng.choice(
            test_indices,
            size=int(len(test_indices) * debug_frac),
            replace=False,
        )

        logger.info("Debug mode: %s", f"{debug_frac:.1%}")

    train_indices = np.sort(train_indices)
    val_indices   = np.sort(val_indices)
    test_indices  = np.sort(test_indices)

    with open(norm_path, encoding="utf-8") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    logger.info(
        "Train=%s, Val=%s, Test=%s",
        f"{len(train_indices):,}",
        f"{len(val_indices):,}",
        f"{len(test_indices):,}",
    )

    # datasets and dataloaders
    common_kwargs = dict(
        h5_file_path    = file_path,
        max_tracks      = config["data"].get("max_tracks", 40),
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = label_vars,
        jet_flavour_map = label_map,
        stats           = norm_stats,
    )

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = config["training"].get("num_workers", 0),
        pin_memory  = torch.cuda.is_available(),
        drop_last   = config["data"].get("drop_last", False),
    )

    train_dataset = GN2Dataset(jet_indices=train_indices, **common_kwargs)
    val_dataset   = GN2Dataset(jet_indices=val_indices,   **common_kwargs)
    test_dataset  = GN2Dataset(jet_indices=test_indices,  **common_kwargs)

    train_loader = gn2_dataloader(train_dataset, **loader_kwargs, shuffle=shuffle_var)
    val_loader   = gn2_dataloader(val_dataset,   **loader_kwargs, shuffle=False)
    test_loader  = gn2_dataloader(test_dataset,  **loader_kwargs, shuffle=False)

    if config["output"].get("save_plots", False):
        from src.transformer_jet_tagging.plotting import plot_statistics

        plot_statistics(
            h5_path         = file_path,
            jet_vars        = jet_vars,
            track_vars      = track_vars,
            jet_flavour     = label_vars,
            jet_flavour_map = label_map,
            jet_indices     = train_indices,
            output_dir      = config["output"].get("plot_dir", "outputs/plots"),
            n_jets_track    = int(len(train_indices)*0.1),
        )

    # debug batch
    batch = next(iter(train_loader))
    logger.debug("Jets:   %s", batch["jet_features"].shape)
    logger.debug("Tracks: %s", batch["track_features"].shape)
    logger.debug("Labels: %s", batch["label"].shape)

    # model
    device = torch.device("cpu")

    model_config = config.get("model", {})

    gn2_model = GN2(
        n_jet_vars       = len(jet_vars),
        n_track_vars     = len(track_vars),
        n_classes        = len(label_map),
        init_hidden_dim  = model_config.get("initialiser_hidden_dim"),
        init_output_dim  = model_config.get("initialiser_output_dim"),
        embed_dim        = model_config.get("transformer_embed_dim"),
        n_heads          = model_config.get("transformer_n_heads"),
        n_layers         = model_config.get("transformer_n_layers"),
        ff_dim           = model_config.get("transformer_ff_dim"),
        pool_dim         = model_config.get("pooling_dim"),
        dropout          = model_config.get("transformer_dropout"),
        head_hidden_dims = model_config.get("head_hidden_dims"),
        activation       = model_config.get("activation"),
    ).to(device)

    # training
    gn2_model, history = train(
        model        = gn2_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = config,
        output_dir   = Path(config["output"].get("checkpoints_dir", "outputs/checkpoints")),
        device       = device,
    )

    # plots
    if config["output"].get("plot_roc", False):
        from transformer_jet_tagging.plotting import (
            plot_learning_curves,
            plot_roc_db,
            plot_roc_dc,
        )

        out_plot_dir = Path(config["output"]["plots_dir"])

        plot_learning_curves(history.to_dict(), output_dir=out_plot_dir)
        plot_roc_db(model=gn2_model, loader=val_loader, device=device, output_dir=out_plot_dir)
        plot_roc_dc(model=gn2_model, loader=val_loader, device=device, output_dir=out_plot_dir)

    # evaluate
    eval_output_dir = Path(config["output"].get("eval_dir", "outputs/eval"))

    evaluate(
        config          = config,
        checkpoint_path = Path(config["output"].get("checkpoints_dir",
                            "outputs/checkpoints")) / "best_model.pt",
        output_dir      = eval_output_dir,
        device          = device,
    )


if __name__ == "__main__":
    main()
