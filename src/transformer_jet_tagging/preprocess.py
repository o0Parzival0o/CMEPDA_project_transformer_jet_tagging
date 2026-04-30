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

Outputs:
  Directory specified in ``config["output"]["preprocess_dir"]``:

  .. code-block:: text

      preprocess_dir/
      ├── indices/
      │   ├── train_indices.npy
      │   ├── val_indices.npy
      │   └── test_indices.npy
      └── norm_stats.json
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from . import utils

# logging
logger = logging.getLogger("GN2.preprocess")


def save_indices(output_dir: Path, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    """
    Save train / val / test index arrays as .npy files.

    Args:
        output_dir (Path): Base directory to save indices.
        train (np.ndarray): Array of training indices.
        val (np.ndarray): Array of validation indices.
        test (np.ndarray): Array of test indices.
    """
    idx_dir = output_dir / "indices"
    idx_dir.mkdir(parents=True, exist_ok=True)

    np.save(idx_dir / "train_indices.npy", train)
    np.save(idx_dir / "val_indices.npy",   val)
    np.save(idx_dir / "test_indices.npy",  test)

    logger.info("Indices saved to %s", idx_dir)
    logger.info("  Train : %s jets", f"{len(train):>8,}")
    logger.info("  Val   : %s jets", f"{len(val):>8,}")
    logger.info("  Test  : %s jets", f"{len(test):>8,}")


def save_norm_stats(output_dir: Path, norm_stats: dict) -> None:
    """
    Serialize norm stats (numpy arrays) to JSON.
    
    Args:
        output_dir (Path): Directory to save the norm_stats.json file.
        norm_stats (dict): Dictionary of normalization stats
            ({"jet_pt": {"mu": ..., "sigma": ...}, ...}).
            Values should be numpy arrays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "norm_stats.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=4)

    logger.info("Normalization stats saved to %s", out_path)


def run_preprocess(config: dict) -> None:
    """
    Run the preprocessing pipeline.

    Args:
        config_path (str): Path to the JSON configuration file.
    
    Raises:
        FileNotFoundError: If the specified HDF5 file is not found.
        KeyError: If expected datasets are missing from the HDF5 file.
    """
    # 1. load configuration
    data_config = config["data"]

    file_path  = data_config["data"]["h5_path"]
    pt_min     = data_config["data"]["pt_min_mev"]
    pt_max     = data_config["data"]["pt_max_mev"]
    eta_max    = data_config["data"]["eta_max"]
    train_frac, val_frac, test_frac = (
        data_config["data"]["train_fraction"],
        data_config["data"]["val_fraction"],
        data_config["data"]["test_fraction"],
    )
    shuffle    = data_config["data"].get("shuffle", False)
    seed       = data_config["data"].get("split_seed", 42)
    jet_vars   = data_config["data"]["jet_features"]
    track_vars = data_config["data"]["track_features"]
    batch_size = data_config["data"].get("batch_size", 10_000)
    output_dir = Path(data_config["output"]["preprocess_dir"])

    # 2. kinematic selection
    logger.info("Reading kinematics from %s ...", file_path)
    try:
        with h5py.File(file_path, "r") as f:
            pt  = f["jets"]["pt"][:]
            eta = f["jets"]["eta"][:]
    except FileNotFoundError:
        logger.error("HDF5 file not found: %s", file_path)
        raise
    except KeyError as e:
        logger.error("Expected dataset not found in HDF5: %s", e)
        raise

    kinematic_mask = (pt > pt_min) & (pt < pt_max) & (np.abs(eta) < eta_max)
    valid_indices  = np.where(kinematic_mask)[0]
    logger.info("Jets passing kinematic selection: %s / %s",
                f"{len(valid_indices):,}", f"{len(pt):,}")

    train_val_indices, test_indices = train_test_split(
        valid_indices,
        train_size   = train_frac + val_frac,
        test_size    = test_frac,
        random_state = seed,
        shuffle      = shuffle,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        train_size   = train_frac / (train_frac + val_frac),
        random_state = seed,
        shuffle      = shuffle,
    )
    save_indices(output_dir, train_indices, val_indices, test_indices)

    # 3. Normalization statistics - computed on training set ONLY
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

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="GN2 preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()
    cfg_path = args.config
    run_preprocess(cfg_path)
