import json
import logging

import src.trasformer_jet_tagging.utils as utils
from src.trasformer_jet_tagging.dataset import GN2Dataset

import numpy as np
from pathlib import Path
import h5py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger("GN2")

if __name__ == "__main__":
    # Load configuration
    config = utils.load_config_json("/home/lnasini/Desktop/PROGETTO_CMEPDA/CMEPDA_project_transformer_jet_tagging/src/trasformer_jet_tagging/configs/config.json")

    # Extract configuration parameters
    file_path = config["data"]["h5_path"]

    pt_min, pt_max, eta_max = config["data"]["pt_min_mev"], config["data"]["pt_max_mev"], config["data"]["eta_max"]
    splitting_frac = (config["data"]["train_fraction"], config["data"]["val_fraction"], config["data"]["test_fraction"])
    shuffle_var = config["data"].get("shuffle", False)
    shuffle_seed = config["data"].get("split_seed", None)

    jet_vars = config["data"]["jet_features"]
    track_vars = config["data"]["track_features"]
    label_vars = config["data"]["label"]
    label_map  = config["data"]["label_map"]

    data_batch_size = config["data"].get("batch_size", 10_000)
    training_batch_size = config["training"].get("batch_size", 1024)

    # Step 1: Read jet kinematics and apply selection to get valid indices
    with h5py.File(file_path, 'r') as f:
        pt  = f['jets']['pt']
        eta = f['jets']['eta']
    
    kinematic_mask = (pt > pt_min) & (pt < pt_max) & (np.abs(eta) < eta_max)
    valid_indices  = np.where(kinematic_mask)[0]

    # Step 2: Split valid indices into train, val, test sets
    training_indices, test_indices = train_test_split(
        valid_indices, 
        train_size = splitting_frac[0] + splitting_frac[1], 
        random_state = shuffle_seed, 
        shuffle = shuffle_var
    )
    train_indices, val_indices = train_test_split(
        training_indices, 
        train_size = splitting_frac[0] / (splitting_frac[0] + splitting_frac[1]),
        random_state = shuffle_seed, 
        shuffle = shuffle_var
    )
    logger.info(f"Split completed: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Step 3: Compute normalization statistics on the training set
    norm_stats = utils.compute_normalization_stats(
        file_path     = file_path,
        train_indices = train_indices,
        jet_vars      = jet_vars,
        track_vars    = track_vars,
        batch_size    = data_batch_size
    )
    # Save normalization stats to JSON for later use in dataset initialization
    path_norm_stats = Path(config["output"]["save_norm_stats"])
    path_norm_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(path_norm_stats, 'w') as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f)

    # Step 4: Initialize datasets and dataloaders for train, val, test sets
    train_dataset = GN2Dataset(
        file_path       = file_path,
        indices         = train_indices,
        n_tracks        = config["data"].get("max_tracks", 40),
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = label_vars,
        jet_flavour_map = label_map,
        norm_stats      = norm_stats
    )
    val_dataset = GN2Dataset(
        file_path       = file_path,
        indices         = val_indices,
        n_tracks        = config["data"].get("max_tracks", 40),
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = label_vars,
        jet_flavour_map = label_map,
        norm_stats      = norm_stats
    )
    test_dataset = GN2Dataset(
        file_path       = file_path,
        indices         = test_indices,
        n_tracks        = config["data"].get("max_tracks", 40),
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = label_vars,
        jet_flavour_map = label_map,
        norm_stats      = norm_stats
    )

    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = training_batch_size,
        shuffle     = shuffle_var,
        num_workers = config["training"].get("num_workers", 4)
    )
    val_loader = DataLoader(
        dataset     = val_dataset,
        batch_size  = training_batch_size,
        shuffle     = shuffle_var,
        num_workers = config["training"].get("num_workers", 4)
    )
    test_loader = DataLoader(
        dataset     = test_dataset,
        batch_size  = training_batch_size,
        shuffle     = shuffle_var,
        num_workers = config["training"].get("num_workers", 4)
    )
    
    batch = next(iter(test_loader))
    logging.debug(f"Jets shape: {batch['jet_features'].shape}")
    logging.debug(f"Tracks shape: {batch['track_features'].shape}")
    logging.debug(f"Labels shape: {batch['label'].shape}")