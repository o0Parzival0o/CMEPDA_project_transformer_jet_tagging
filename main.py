'''
main.py
=======
Main script for the transformer jet tagging project.
'''

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from src.trasformer_jet_tagging import utils, plotting
from src.trasformer_jet_tagging.dataset import GN2Dataset, GN2DataLoader
from src.trasformer_jet_tagging.model import GN2
from src.trasformer_jet_tagging.train import train

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GN2")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GN2 preprocessing pipeline")
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
        help="Fraction of data to use (es. 0.05 per il 5%%)"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    debug_frac = args.debug_frac
    # load configuration
    config = utils.load_config_json(config_path)

    # extract configuration parameters
    file_path      = config["data"]["h5_path"]
    preprocess_dir = Path(config["output"]["preprocess_dir"])

    jet_vars   = config["data"]["jet_features"]
    track_vars = config["data"]["track_features"]
    label_vars = config["data"]["label"]
    label_map  = {int(k): v for k, v in config["data"]["label_map"].items()}

    training_batch_size = config["training"].get("batch_size", 1024)
    shuffle_var         = config["data"].get("shuffle", False)

    # 1. preprocessing
    idx_dir   = preprocess_dir / "indices"
    norm_path = preprocess_dir / "norm_stats.json"

    artifacts_dir = [
        idx_dir / "train_indices.npy",
        idx_dir / "val_indices.npy",
        idx_dir / "test_indices.npy",
        norm_path,
    ]

    if not all(p.exists() for p in artifacts_dir):
        logger.info("Preprocessing files not found. Running preprocess script ...")
        from src.trasformer_jet_tagging import preprocess
        preprocess.main(config_path=config_path)

    for path in artifacts_dir:
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing artifact not found: {path}\nRun preprocess.py first.")

    train_indices = np.load(artifacts_dir[0])
    val_indices   = np.load(artifacts_dir[1])
    test_indices  = np.load(artifacts_dir[2])

    if debug_frac < 1.0:
        rng = np.random.default_rng(seed=42)
        train_indices = rng.choice(train_indices, size=int(len(train_indices) * debug_frac), replace=False)
        val_indices   = rng.choice(val_indices,   size=int(len(val_indices)   * debug_frac), replace=False)
        test_indices  = rng.choice(test_indices,  size=int(len(test_indices)  * debug_frac), replace=False)

        logger.info(f"Debug mode: {debug_frac:.1%} dei dati")
    
    train_indices = np.sort(train_indices)
    val_indices   = np.sort(val_indices)
    test_indices  = np.sort(test_indices)

    with open(norm_path, "r") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    logger.info(f"Train={len(train_indices):,}, Val={len(val_indices):,}, Test={len(test_indices):,}")

    # 2. initialize datasets and dataloaders
    common_kwargs = dict(
        file_path       = file_path,
        n_tracks        = config["data"].get("max_tracks", 40),
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = label_vars,
        jet_flavour_map = label_map,
        norm_stats      = norm_stats,
    )
    loader_kwargs = dict(
        batch_size  = training_batch_size,
        num_workers = config["training"].get("num_workers", 0),
        pin_memory  = torch.cuda.is_available(),                                       # TODO torch.cuda.is_available(),
    )

    train_dataset = GN2Dataset(indices=train_indices, **common_kwargs)
    val_dataset   = GN2Dataset(indices=val_indices,   **common_kwargs)
    test_dataset  = GN2Dataset(indices=test_indices,  **common_kwargs)

    train_loader = GN2DataLoader(train_dataset, **loader_kwargs, shuffle=True)
    val_loader   = GN2DataLoader(val_dataset,   **loader_kwargs, shuffle=False)
    test_loader  = GN2DataLoader(test_dataset,  **loader_kwargs, shuffle=False)

    batch = next(iter(train_loader))
    logger.debug(f"Jets shape:   {batch['jet_features'].shape}")
    logger.debug(f"Tracks shape: {batch['track_features'].shape}")
    logger.debug(f"Labels shape: {batch['label'].shape}")

    if config["output"].get("save_plots", False):
        plotting.make_all_plots(
            file_path       = file_path,
            jet_vars        = jet_vars,
            track_vars      = track_vars,
            jet_flavour     = label_vars,
            jet_flavour_map = label_map,
            indices         = train_indices,
            output_dir      = config["output"].get("plot_dir", "outputs/plots"),
            n_jets_track    = int(len(train_indices)*0.1),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config.get("model", {})
    GN2_model = GN2(
        n_jet_vars       = len(jet_vars),
        n_track_vars     = len(track_vars),
        n_classes        = model_config.get(len(label_map), None),
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
    ).to(device)

    GN2_model = train(
        model        = GN2_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = config,
        output_dir   = config["output"].get("checkpoints_dir", "outputs/checkpoints"),
        device       = device)