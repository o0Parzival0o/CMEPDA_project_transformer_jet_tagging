"""
preprocess.py
=============
Standalone preprocessing script for the GN2 jet flavour tagging pipeline.

Run this script ONCE before training. It will:
  1. Load jet kinematics from the HDF5 file.
  2. Apply kinematic selection (pT, eta cuts).
  3. Split valid indices into train / val / test sets.
  4. Compute normalization statistics (mu, sigma) on the training set only.
  5. Save indices and norm stats to disk.

Outputs (under the directory specified in config["output"]["preprocess_dir"]):
  preprocess_dir/
  ├── indices/
  │   ├── train_indices.npy
  │   ├── val_indices.npy
  │   └── test_indices.npy
  └── norm_stats.json

Usage:
    python preprocess.py --config configs/config.json
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

import src.trasformer_jet_tagging.utils as utils

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GN2.preprocess")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_indices(output_dir: Path, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    """Save train / val / test index arrays as .npy files."""
    idx_dir = output_dir / "indices"
    idx_dir.mkdir(parents=True, exist_ok=True)

    np.save(idx_dir / "train_indices.npy", train)
    np.save(idx_dir / "val_indices.npy",   val)
    np.save(idx_dir / "test_indices.npy",  test)

    logger.info(f"Indices saved to {idx_dir}")
    logger.info(f"  Train : {len(train):>8,} jets")
    logger.info(f"  Val   : {len(val):>8,} jets")
    logger.info(f"  Test  : {len(test):>8,} jets")


def save_norm_stats(output_dir: Path, norm_stats: dict) -> None:
    """Serialize norm stats (numpy arrays) to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "norm_stats.json"

    with open(out_path, "w") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

    logger.info(f"Normalization stats saved to {out_path}")




def main(config_path: str):

    # 1. load configuration
    config = utils.load_config_json(config_path)

    file_path   = config["data"]["h5_path"]
    pt_min      = config["data"]["pt_min_mev"]
    pt_max      = config["data"]["pt_max_mev"]
    eta_max     = config["data"]["eta_max"]
    split_fracs = (
        config["data"]["train_fraction"],
        config["data"]["val_fraction"],
        config["data"]["test_fraction"],
    )
    shuffle     = config["data"].get("shuffle", False)
    seed        = config["data"].get("split_seed", 42)
    jet_vars    = config["data"]["jet_features"]
    track_vars  = config["data"]["track_features"]
    batch_size  = config["data"].get("batch_size", 10_000)
    output_dir  = Path(config["output"]["preprocess_dir"])

    # 2. kinematic selection
    logger.info(f"Reading kinematics from {file_path} ...")
    with h5py.File(file_path, "r") as f:
        pt  = f["jets"]["pt"][:]
        eta = f["jets"]["eta"][:]

    kinematic_mask = (pt > pt_min) & (pt < pt_max) & (np.abs(eta) < eta_max)
    valid_indices  = np.where(kinematic_mask)[0]
    logger.info(f"Jets passing kinematic selection: {len(valid_indices):,} / {len(pt):,}")

    # 3. train / val / test split
    train_frac = split_fracs[0]
    val_frac   = split_fracs[1]
    test_frac  = split_fracs[2]

    training_indices, test_indices = train_test_split(
        valid_indices,
        train_size   = train_frac + val_frac,
        test_size    = test_frac,
        random_state = seed,
        shuffle      = shuffle,
    )
    train_indices, val_indices = train_test_split(
        training_indices,
        train_size   = train_frac / (train_frac + val_frac),
        random_state = seed,
        shuffle      = shuffle,
    )
    save_indices(output_dir, train_indices, val_indices, test_indices)

    # 4. Normalization statistics - computed on training set ONLY
    logger.info("Computing normalization statistics on training set ...")
    norm_stats = utils.compute_normalization_stats(
        file_path     = file_path,
        train_indices = train_indices,
        jet_vars      = jet_vars,
        track_vars    = track_vars,
        batch_size    = batch_size,
    )
    save_norm_stats(output_dir, norm_stats)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GN2 preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()
    config_path = args.config
    main(config_path)