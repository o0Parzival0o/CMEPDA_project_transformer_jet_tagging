"""
dataset.py
==========
Caricamento e preprocessing dei dati GN2 da file HDF5 con struttura:
  /jets          — variabili jet-level (una riga per jet)
  /tracks        — variabili track-level (una riga per traccia)
  /eventwise     — variabili evento
  /truth_hadrons — truth hadron info

La struttura /tracks è un array "piatto" di tracce, raggruppate per jet
tramite un offset/count implicito nell'ordine delle righe: le prime N0
righe appartengono al jet 0, le successive N1 al jet 1, ecc.
Il numero di tracce per jet è variabile; si usa la colonna "valid"
come maschera di qualità.

Pipeline:
  1. Leggi /jets e /tracks
  2. Filtra jet (pT, η, JVT, label valido)
  3. Filtra tracce (colonna valid == True)
  4. Per ogni jet: estrai le sue tracce, applica padding a max_tracks,
     costruisci la maschera booleana
  5. Concatena jet features alle track features (broadcast)
  6. Normalizza (fit solo su train)
  7. Re-sampling 2D (pT, η) per bilanciare le classi
  8. Restituisce DataLoader PyTorch
"""

import logging
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GN2DataLoader")

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
    """

    JET_FLAVOUR_MAP = {0: 0, 4: 1, 5: 2, 15: 3}  # light, c, b, tau
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

    def __init__(
        self, 
        file_path: str,
        indices: np.ndarray,
        n_tracks: int = 40,
        jet_vars: Optional[list] = None,
        track_vars: Optional[list] = None,
        norm_stats: Optional[Dict] = None
    ):
        """
        Initialize the dataset.

        Args:
            file_path (str): path of .h5 file.
            indices (np.ndarray): indices of jets to include in the dataset.
            n_tracks (int, optional): maximum number of tracks for each jet (padding/cropping).
            jet_vars (list, optional): list of jet variables.
            track_vars (list, optional): list of tracks variables.
        """
        self.file_path = file_path
        self.indices = indices
        self.n_tracks = n_tracks
        self.jet_vars = jet_vars or self.JET_VARS_DEFAULT
        self.track_vars = track_vars or self.TRACK_VARS_DEFAULT

        # Inizializziamo l'handler del file come None per il multiprocessing (PyTorch workers)
        self.handler = None

        self.norm_stats = norm_stats
        
        # Initial check
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
            self.handler = h5py.File(self.file_path, 'r', swmr=True)        # swmr=True permette allows multiple readers (for num_workers > 0)
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
                'jet_features': torch.Tensor,
                'track_features': torch.Tensor,
                'mask': torch.Tensor,
                'label': torch.Tensor
            }
        """
        f = self._get_handler()

        real_idx = self.indices[idx]  # maps the dataset index to the actual jet index in the file

        # 1. Loading Jet Features and normalization
        jet_data = f['jets'][real_idx]
        
        # pT log-trasformation
        jet_pt = jet_data['pt']
        jet_pt_log = np.log(jet_pt)
        jet_eta = jet_data['eta']
        
        jet_features = np.array([jet_pt_log, jet_eta], dtype=np.float32)
        
        # 2. Loading Label
        raw_label = jet_data['HadronConeExclTruthLabelID']
        target = self.JET_FLAVOUR_MAP.get(int(raw_label), 0)

        # 3. Loading Tracks with 'valid' Filter (Optimized with slicing)
        tracks_all = f['tracks'][real_idx]
        
        # bool mask
        valid_mask_in_file = tracks_all['valid'] == True
        valid_tracks = tracks_all[valid_mask_in_file]
        
        n_available = len(valid_tracks)
        n_to_read = min(n_available, self.n_tracks)

        # Pre-allocate arrays for track features and mask
        track_features = np.zeros((self.n_tracks, len(self.track_vars)), dtype=np.float32)
        padding_mask = np.zeros(self.n_tracks, dtype=bool)

        # Vectorized extraction: read features for available tracks
        if n_to_read > 0:
            for i, var in enumerate(self.track_vars):
                raw_values = valid_tracks[var][:n_to_read]
                # normalization
                track_features[:n_to_read, i] = (raw_values - self.norm_stats['track_mu']) / self.norm_stats['track_sigma']
            
            padding_mask[:n_to_read] = True

        return {
            'jet_features': torch.from_numpy(jet_features),
            'track_features': torch.from_numpy(track_features),
            'mask': torch.from_numpy(padding_mask),
            'label': torch.tensor(target, dtype=torch.long)
        }

if __name__ == "__main__":

    PATH = '../../dataset/mc-flavtag-ttbar-small.h5'

    # dataset = GN2Dataset(PATH)
    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=1024,
    #     shuffle=True, 
    #     num_workers=4,
    #     pin_memory=True
    # )

    # # Test
    # sample = dataset[0]
    
    # logger.info(f"Batch loaded.")
    # logger.info(f"Shape:\t\t{dataset.shape}")
    # logger.info(f"Shape jets:\t{sample['jet_features'].shape}")
    # logger.info(f"Shape tracks:\t{sample['track_features'].shape}")
    # logger.info(f"Numero tracce valide nel sample: {sample['mask'].sum().item()}")
    # logger.info(f"Esempio jet features (norm): {sample['jet_features']}")
    # logger.info(f"Target:\t{sample['label']}")

    # 1. Prendi tutti gli indici e dividili
    with h5py.File(PATH, 'r') as f:
        total_n = len(f['jets'])
    
    indices = np.arange(total_n)
    np.random.shuffle(indices)
    train_indices = indices[:int(0.7*total_n)]
    
    # 2. Qui calcoleresti le stats solo su train_indices...
    my_norm_stats = { ... } 

    # 3. Crei il dataset di training
    train_dataset = GN2Dataset(PATH, indices=train_indices, norm_stats=my_norm_stats)
    loader = DataLoader(train_dataset, batch_size=1024, num_workers=4)