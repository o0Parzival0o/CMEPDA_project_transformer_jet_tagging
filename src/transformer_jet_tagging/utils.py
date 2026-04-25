"""
utils.py
========
Utility functions for the transformer jet tagging project.
"""

import json
import logging

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from .constants import JET_VARS_DEFAULT, TRACK_VARS_DEFAULT

logger = logging.getLogger("GN2.utils")


def compute_normalization_stats(
    file_path: str,
    train_indices: np.ndarray,
    jet_vars: list[str] | None = None,
    track_vars: list[str] | None = None,
    batch_size: int | None = 10_000
):
    """
    Compute mean and std normalization statistics on the training set only.

    Uses sklearn's StandardScaler with partial_fit to accumulate statistics
    iterating in batches.
    Statistics are computed exclusively on the training set to prevent
    data leakage into validation and test sets.

    Args:
        file_path (str): Path to the HDF5 file containing jets and tracks data.
        train_indices (np.ndarray): Array of integer indices identifying the training jets within
            the HDF5 file.
        jet_vars (list, optional): List of jet-level variable names to include.
            (defaults JET_VARS_DEFAULT)
        track_vars (list, optional): List of track-level variable names to include.
            (default TRACK_VARS_DEFAULT)
        batch_size (int, optional): Number of jets to process per batch during the partial_fit loop.
            (defaults to 10_000)

    Returns:
        dict: Normalization statistics with the following keys:

            - "jet_mu" (np.ndarray, shape (n_jet_vars,)): Per-feature mean for jet-level variables.
            - "jet_sigma" (np.ndarray, shape (n_jet_vars,)): Per-feature standard deviation
                for jet-level variables.
            - "track_mu" (np.ndarray, shape (n_track_vars,)): Per-feature mean computed over all
                valid tracks in the training set.
            - "track_sigma" (np.ndarray, shape (n_track_vars,)): Per-feature standard deviation over
                all valid tracks in the training set.
    """

    jet_vars   = jet_vars   if jet_vars   is not None else JET_VARS_DEFAULT
    track_vars = track_vars if track_vars is not None else TRACK_VARS_DEFAULT

    # h5py requires indices in strictly increasing order
    sorted_indices = np.sort(train_indices)

    jet_scaler   = StandardScaler()
    track_scaler = StandardScaler()

    with h5py.File(file_path, 'r') as f:

        jets_ds:   h5py.Dataset = f['jets']
        tracks_ds: h5py.Dataset = f['tracks']
        jet_dtype_names   = jets_ds.dtype.names   or ()
        track_dtype_names = tracks_ds.dtype.names or ()

        missing_jet_vars = [v for v in jet_vars if v not in jet_dtype_names]
        for jvar in missing_jet_vars:
            logger.warning(
                "Jet variable '%s' not found in HDF5 file. "
                "Skipping this variable for normalization stats.", jvar
            )
            jet_vars.remove(jvar)

        missing_track_vars = [v for v in track_vars if v not in track_dtype_names]
        for tvar in missing_track_vars:
            logger.warning(
                "Track variable '%s' not found in HDF5 file. "
                "Skipping this variable for normalization stats.", tvar
            )
            track_vars.remove(tvar)

        for start in range(0, len(sorted_indices), batch_size):
            batch_idx = sorted_indices[start : start + batch_size]

            # jet features
            jets_raw  = jets_ds[batch_idx]
            jet_batch = np.empty((len(batch_idx), len(jet_vars)), dtype=np.float32)
            for i, jvar in enumerate(jet_vars):
                jet_col = jets_raw[jvar].astype(np.float32)
                if jvar == 'pt':
                    eps     = 1e-8
                    jet_col = np.log(np.clip(jet_col, eps, None))
                jet_batch[:, i] = jet_col
            jet_scaler.partial_fit(jet_batch)

            # track features
            tracks_raw  = tracks_ds[batch_idx]
            track_batch = np.stack(
                [tracks_raw[tvar] for tvar in track_vars], axis=-1
            ).astype(np.float32).reshape(-1, len(track_vars))

            if "valid" in track_dtype_names:
                valid_mask = np.asarray(tracks_raw['valid'], dtype=bool).reshape(-1)
            else:
                logger.warning("'valid' field not found in tracks dataset. "
                               "Assuming all tracks are valid for normalization stats.")
                valid_mask = np.ones(track_batch.shape[0], dtype=bool)

            valid_tracks = track_batch[valid_mask]
            if valid_tracks.shape[0] == 0:
                valid_tracks = np.zeros((1, len(track_vars)), dtype=np.float32)
            track_scaler.partial_fit(valid_tracks)

            logger.debug("partial_fit: batch %s - %s done.", start, start + len(batch_idx))

    norm_stats = {
        'jet_mu':      jet_scaler.mean_,
        'jet_sigma':   jet_scaler.scale_,
        'track_mu':    track_scaler.mean_,
        'track_sigma': track_scaler.scale_,
    }

    logger.info("Normalization stats computed on %s jets.", f"{len(train_indices):,}")

    return norm_stats


def load_config_json(filepath):
    """
    Loads the configuration from the specified JSON file.

    Args:
        filepath (str): path to the JSON configuration file.

    Returns:
        dict: configuration parameters loaded from the JSON file.
    """
    with open(filepath, encoding="utf-8") as f:
        config = json.load(f)
    return config
