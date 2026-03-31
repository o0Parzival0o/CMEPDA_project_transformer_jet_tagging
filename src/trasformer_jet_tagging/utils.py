"""
utils.py
"""

import logging
import json

import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("GN2DataLoader")

JET_VARS_DEFAULT = ['pt', 'eta']
TRACK_VARS_DEFAULT = [
    # tracks in the perigee repn
    'qOverP', 'deta', 'dphi', 'd0', 'z0SinTheta',
    # diagonal of the track cov matrix (first 3 els)
    'qOverPUncertainty', 'thetaUncertainty', 'phiUncertainty',
    # lifetime signed s(d0) and s(z0*sin(theta))
    'lifetimeSignedD0Significance', 'lifetimeSignedZ0SinThetaSignificance',
    # hit level variables
    'numberOfPixelHits', 'numberOfSCTHits',
    'numberOfInnermostPixelLayerHits', 'numberOfNextToInnermostPixelLayerHits',
    'numberOfInnermostPixelLayerSharedHits', 'numberOfInnermostPixelLayerSplitHits',
    'numberOfPixelSharedHits', 'numberOfPixelSplitHits', 'numberOfSCTSharedHits'
]

def compute_normalization_stats(
    file_path: str,
    train_indices: np.ndarray,
    jet_vars: list = JET_VARS_DEFAULT,
    track_vars: list = TRACK_VARS_DEFAULT,
    batch_size: int = 10_000
):
    """
    Compute mean and std normalization statistics on the training set only.

    Uses sklearn's StandardScaler with partial_fit to accumulate statistics
    iterating in batches.
    Statistics are computed exclusively on the training set to prevent
    data leakage into validation and test sets.

    Args:
        file_path       (str):              Path to the HDF5 file containing jets and tracks data.
        train_indices   (np.ndarray):       Array of integer indices identifying the training jets within the HDF5 file.
        jet_vars        (list, optional):   List of jet-level variable names to include.
                                            Defaults to JET_VARS_DEFAULT if not provided.
        track_vars      (list, optional):   List of track-level variable names to include.
                                            Defaults to TRACK_VARS_DEFAULT if not provided.
        batch_size      (int, optional):    Number of jets to process per batch during the partial_fit loop. Controls peak memory usage.
                                            Defaults to 10_000.

    Returns:
        dict: Normalization statistics with the following keys:
            "jet_mu"        (np.ndarray, shape (n_jet_vars,)):      Per-feature mean for jet-level variables.
            "jet_sigma"     (np.ndarray, shape (n_jet_vars,)):      Per-feature standard deviation for jet-level variables.
            "track_mu"      (np.ndarray, shape (n_track_vars,)):    Per-feature mean computed over all valid tracks in the training set.
            "track_sigma"   (np.ndarray, shape (n_track_vars,)):    Per-feature standard deviation over all valid tracks in the training set.
    """

    # h5py requires indices in strictly increasing order
    sorted_indices = np.sort(train_indices)

    jet_scaler   = StandardScaler()
    track_scaler = StandardScaler()

    with h5py.File(file_path, 'r') as f:

        for jvar, tvar in zip(jet_vars, track_vars):
            if jvar not in f['jets'].dtype.names:
                logger.warning(f"Jet variable '{jvar}' not found in HDF5 file. Skipping this variable for normalization stats.")
                jet_vars.remove(jvar)
            if tvar not in f['tracks'].dtype.names:
                logger.warning(f"Track variable '{tvar}' not found in HDF5 file. Skipping this variable for normalization stats.")
                track_vars.remove(tvar)

        for start in range(0, len(sorted_indices), batch_size):
            batch_idx = sorted_indices[start : start + batch_size]

            # jet features
            jets_raw  = f['jets'][batch_idx]
            jet_batch = np.empty((len(batch_idx), len(jet_vars)), dtype=np.float32)  # pre-allocate array for jet features
            for jvar in jet_vars:
                jet_var = jets_raw[jvar].reshape(-1, 1)             # reshape to (n_jets_in_batch, 1) for StandardScaler`
                if jvar == 'pt':
                    jet_var = np.log(jet_var)
                jet_batch[:, jet_vars.index(jvar)] = jet_var[:, 0]  # fill the pre-allocated array
            jet_scaler.partial_fit(jet_batch)

            # track features
            tracks_raw = f['tracks'][batch_idx]
            if "valid" in tracks_raw.dtype.names:
                valid_mask = tracks_raw['valid'].astype(bool)
            else:
                logger.warning("'valid' field not found in tracks dataset. Assuming all tracks are valid for normalization stats.")
                valid_mask = np.ones(len(tracks_raw), dtype=bool)
            track_batch  = np.stack([tracks_raw[tvar] for tvar in track_vars], axis=-1)
            valid_tracks = track_batch[valid_mask]
            if len(valid_tracks) > 0:
                track_scaler.partial_fit(valid_tracks)

            logger.debug(f"partial_fit: batch {start}–{start + len(batch_idx)} done.")

    norm_stats = {
        'jet_mu':      jet_scaler.mean_,
        'jet_sigma':   jet_scaler.scale_,
        'track_mu':    track_scaler.mean_,
        'track_sigma': track_scaler.scale_,
    }

    logger.info(f"Normalization stats computed on {len(train_indices)} jets.")

    return norm_stats

def load_config_json(filepath):
    """
    Loads the configuration from the specified JSON file.
    
    Args:
        filepath (str): path to the JSON configuration file.

    Returns:
        dict: configuration parameters loaded from the JSON file.
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config