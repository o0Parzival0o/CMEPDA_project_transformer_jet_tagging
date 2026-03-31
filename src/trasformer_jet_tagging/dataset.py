"""
dataset.py
==========
High-performance GN2 data pipeline for HDF5 datasets. 
Optimized for Large-Scale Jet Flavour Tagging at ATLAS.

HDF5 Structure:
  /jets          — Jet-level features (1 row per jet).
  /tracks        — Track-level features (Jagged or fixed-size array indexed by jet).
  /eventwise     — Event-level metadata (e.g., eventNumber, mu).
  /truth_hadrons — Simulation truth for hadron labeling and performance studies.

The pipeline utilizes 'Lazy Loading' via h5py to handle datasets that exceed 
available RAM, using NumPy vectorization for high-speed feature extraction.

Pipeline Workflow:
  1. Index Splitting: Generate Train/Val/Test indices using scikit-learn's 
     train_test_split to ensure zero data leakage.
  2. Data Loading: On-the-fly extraction of jet and track features using 
     global indices to map to the HDF5 file structure.
  3. Track Quality Filtering: Active filtering of track candidates using 
      the 'valid' boolean flag before processing.
  4. Padding & Masking: Enforce a fixed-size track array (default max_tracks=40). 
     Generate a boolean padding mask to inform the Transformer's Self-Attention 
     mechanism which inputs to ignore.
  5. Feature Engineering: 
     - Jet-level: Log-transformation of pT and Z-score standardization.
     - Track-level: Z-score standardization (mu and sigma computed ONLY on training set).
  6. Class Balancing: Optional 2D re-sampling (pT, eta) to flatten the 
     background distributions and match the reference class (c-jets).
  7. Integration: Wraps the logic into a PyTorch DataLoader with 
     multi-process worker support and pinned memory for GPU acceleration.
"""

import logging
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# logging configuration
if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("GN2_dataset")
else:
    logger = logging.getLogger("GN2")

JET_FLAVOUR_MAP = {0: 0, 4: 1, 5: 2, 15: 3}  # light, c, b, tau
JET_FLAVOUR = 'HadronConeExclTruthLabelID'
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

class GN2Dataset(Dataset):
    """
    Dataset for flavour tagger using HDF5 file.
    Lazy loading of data for large datasets with Numpy vectorization.
    Includes filtering of invalid tracks and feature normalization. 

    Attributes:
        file_path (str): path of .h5 file.
        indices (np.ndarray): indices of jets to include in the dataset.
        n_tracks (int, optional): maximum number of tracks for each jet (padding/cropping).
        jet_vars (list, optional): list of jet variables.
        track_vars (list, optional): list of tracks variables.
        jet_flavour (str, optional): name of the jet flavour variable in the HDF5 file.
        jet_flavour_map (dict, optional): mapping from raw hadron labels to target classes.
        norm_stats (dict, optional): normalization statistics for jet and track features.
    """

    def __init__(
        self, 
        file_path: str,
        indices: np.ndarray,
        n_tracks: int = 40,
        jet_vars: Optional[list] = JET_VARS_DEFAULT,
        track_vars: Optional[list] = TRACK_VARS_DEFAULT,
        jet_flavour : Optional[str] = JET_FLAVOUR,
        jet_flavour_map: Optional[Dict[int, int]] = JET_FLAVOUR_MAP,
        norm_stats: Optional[Dict] = None
    ):
        """
        Initialize the dataset.

        Args:
            file_path (str): path of .h5 file.
            indices (np.ndarray): indices of jets to include in the dataset.
            n_tracks (int, optional): maximum number of tracks for each jet (padding/cropping).
            jet_vars (list, optional): list of jet variables. Defaults to JET_VARS_DEFAULT if not provided.
            track_vars (list, optional): list of tracks variables. Defaults to TRACK_VARS_DEFAULT if not provided.
            jet_flavour (str, optional): name of the jet flavour variable in the HDF5 file. Defaults to JET_FLAVOUR if not provided.
            jet_flavour_map (dict, optional): mapping from raw hadron labels to target classes. Defaults to JET_FLAVOUR_MAP if not provided.
            norm_stats (dict, optional): normalization statistics for jet and track features. Defaults to None (no normalization) if not provided.

        Raises:
            FileNotFoundError: if the specified HDF5 file does not exist.
            KeyError: if the expected datasets ('jets', 'tracks') are not found in the HDF5 file.
        """
        self.file_path  = file_path
        self.indices    = indices
        self.n_tracks   = n_tracks
        self.jet_vars   = jet_vars
        self.track_vars = track_vars
        self.jet_flavour = jet_flavour
        self.jet_flavour_map = jet_flavour_map
        self.norm_stats = norm_stats

        # initialize h5py file handler as None; will be opened lazily in _get_handler()
        self.handler = None
        
        # initial check
        try:
            with h5py.File(self.file_path, 'r') as f:
                self.n_jets = len(f['jets'])
                logger.info(f"Success loading {file_path}: {self.n_jets} jets found.")
                logger.debug(f"Original shape 'tracks': {f['tracks'].shape}")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _get_handler(self) -> h5py.File:
        """
        Manage the h5py file handler for multiprocessing.
        
        Returns:
            h5py.File: h5py file object open.
        """
        if self.handler is None:
            self.handler = h5py.File(self.file_path, 'r', swmr=True)        # swmr=True allows multiple readers (for num_workers > 0)
        return self.handler

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Returns the shape of the dataset.

        Returns:
            Tuple[int, int, int]: (n_jets, n_tracks, n_track_features)
        """
        return (self.n_jets, self.n_tracks, len(self.track_vars))

    def __len__(self) -> int:
        """
        Calculate the number of selected jets in the dataset.

        Returns:
            int: number of jets in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract a single jet and its associated tracks.

        Optimization: uses h5py slicing to avoid Python loops.
        The tracks are extracted as a subset and then padded to n_tracks with zeros.
        The mask indicates which tracks are valid (True) or padded (False).

        Args:
            idx: index of the jet to extract.

        Returns:
            dict: {
                "jet_features"      (torch.Tensor, shape (n_jet_vars,)):            normalized jet-level features,
                "track_features"    (torch.Tensor, shape (n_tracks, n_track_vars)): normalized track-level features,
                "mask"              (torch.Tensor, shape (n_tracks,)):              boolean mask indicating valid tracks,
                "label"             (torch.Tensor, shape ()):                       ground truth label
            }
        """
        f = self._get_handler()

        real_idx = self.indices[idx]  # maps the dataset index to the actual jet index in the file

        # 1. Loading Jet Features and normalization
        jet_data = f['jets'][real_idx]
        
        jet_pt  = jet_data['pt']
        jet_eta = jet_data['eta']
        # pt log-trasformation
        jet_pt_log = np.log(jet_pt)

        if self.norm_stats and 'jet_mu' in self.norm_stats and 'jet_sigma' in self.norm_stats:
            if len(self.norm_stats['jet_mu']) < 2 or len(self.norm_stats['jet_sigma']) < 2:
                logger.warning("Normalization stats for jets are incomplete. Expected at least 2 values for 'jet_mu' and 'jet_sigma'. Using raw values.")
            else:
                jet_pt_log = (jet_pt_log - self.norm_stats['jet_mu'][0]) / self.norm_stats['jet_sigma'][0]
                jet_eta    = (jet_eta - self.norm_stats['jet_mu'][1]) / self.norm_stats['jet_sigma'][1]
        else:
            logger.info("Normalization stats not provided for jets. Using raw values.")
        
        jet_features = np.array([jet_pt_log, jet_eta], dtype=np.float32)
        
        # 2. Loading Label
        raw_label = jet_data[self.jet_flavour]
        target = self.jet_flavour_map.get(int(raw_label), 0)

        # 3. Loading Tracks with 'valid' Filter (Optimized with slicing)
        tracks_all = f['tracks'][real_idx]
        # bool mask
        valid_tracks = tracks_all[tracks_all['valid'] == True]
        
        n_available = len(valid_tracks)
        n_to_read   = min(n_available, self.n_tracks)

        # pre-allocate arrays for track features and mask
        track_features = np.zeros((self.n_tracks, len(self.track_vars)), dtype=np.float32)
        padding_mask = np.zeros(self.n_tracks, dtype=bool)

        # vectorized extraction: read features for available tracks
        if n_to_read > 0:
            for i, var in enumerate(self.track_vars):
                raw_values = valid_tracks[var][:n_to_read]
                # normalization
                if self.norm_stats and 'track_mu' in self.norm_stats and 'track_sigma' in self.norm_stats:
                    mu    = self.norm_stats['track_mu'][i]
                    sigma = self.norm_stats['track_sigma'][i]
                    track_features[:n_to_read, i] = (raw_values - mu) / sigma
                else:
                    logger.info(f"Normalization stats not provided for track variable '{var}'. Using raw values.")
                    track_features[:n_to_read, i] = raw_values
            
            padding_mask[:n_to_read] = True

        return {
            'jet_features':   torch.from_numpy(jet_features),
            'track_features': torch.from_numpy(track_features),
            'mask':           torch.from_numpy(padding_mask),
            'label':          torch.tensor(target, dtype=torch.long)
        }

if __name__ == "__main__":

    # Example usage and testing of the dataset
    from utils import compute_normalization_stats

    PATH = "/home/lnasini/Desktop/PROGETTO_CMEPDA/CMEPDA_project_transformer_jet_tagging/dataset/mc-flavtag-ttbar-small.h5"

    with h5py.File(PATH, 'r') as f:
        n_jets = len(f['jets'])
        logger.info(f"Total jets in file: {n_jets}")
        indices = np.arange(n_jets)

    norm_stats = compute_normalization_stats(PATH, indices)

    dataset = GN2Dataset(PATH, indices=indices, norm_stats=norm_stats)

    # Test
    sample = dataset[0]
    
    logger.info(f"Batch loaded.")
    logger.info(f"Shape:                {dataset.shape}")
    logger.info(f"Shape jets:           {sample['jet_features'].shape}")
    logger.info(f"Shape tracks:         {sample['track_features'].shape}")
    logger.info(f"Num. valid tracks:    {sample['mask'].sum()}")
    logger.info(f"Target:               {sample['label']}")