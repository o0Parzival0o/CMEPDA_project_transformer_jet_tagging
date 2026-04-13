"""
dataset.py
==========
High-performance GN2 data pipeline for HDF5 datasets. 
Optimized for Large-Scale Jet Flavour Tagging at ATLAS.

HDF5 Structure:
  /jets          - Jet-level features (1 row per jet).
  /tracks        - Track-level features (Jagged or fixed-size array indexed by jet).
  /eventwise     - Event-level metadata (e.g., eventNumber, mu).
  /truth_hadrons - Simulation truth for hadron labeling and performance studies.

The pipeline utilises 'Lazy Loading' via h5py to handle datasets that exceed 
available RAM, using NumPy vectorization for high-speed feature extraction.

Pipeline Workflow:

1. Index Splitting:
   Generate Train/Val/Test indices using scikit-learn's train_test_split to ensure zero data leakage.

2. Data Loading:
   Batch extraction of jet and track features using a custom collate_fn that reads an entire batch in a single HDF5 call.

3. Track Quality Filtering:
   Active filtering of track candidates using the 'valid' boolean flag before processing.

4. Padding & Masking:
   Enforce a fixed-size track array (default max_tracks=40).
   Generate a boolean padding mask to inform the Transformer's Self-Attention mechanism.

5. Feature Engineering:
   - Jet-level: Log-transformation of pT and Z-score normalization.
   - Track-level: Z-score normalization (computed only on training set).

6. Class Balancing:
   Optional 2D re-sampling (pT, eta) to flatten distributions.

7. Integration:
   Wraps everything into a PyTorch DataLoader with optimized batching.

Performance notes:
  - Indices passed to GN2Dataset MUST be sorted to ensure
    contiguous HDF5 reads and avoid random seeks on disk.
  - Use build_dataloader() which sets up the BatchCollator automatically.
    This replaces per-item __getitem__ HDF5 access with a single batched
    read per batch, reducing I/O overhead by orders of magnitude.
  - num_workers should be 0 when using BatchCollator (the collate_fn
    already parallelises I/O at the batch level via sorted slice reads).
"""

import logging
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler

from src.trasformer_jet_tagging.constants import JET_VARS_DEFAULT, TRACK_VARS_DEFAULT, JET_FLAVOUR_LABEL, JET_FLAVOUR_MAP

# logging configuration
logger = logging.getLogger("GN2.dataset")


class GN2Dataset(Dataset):
    """
    Dataset for flavour tagger using HDF5 file.
    Lazy loading of data for large datasets with NumPy vectorization.
    Includes filtering of invalid tracks and feature normalization.

    IMPORTANT: indices should be sorted before passing to this class
    to guarantee contiguous reads when the BatchCollator is used.

    Attributes:
        file_path (str): path of .h5 file.
        indices (np.ndarray): sorted indices of jets to include in the dataset.
        n_tracks (int): maximum number of tracks for each jet (padding/cropping).
        jet_vars (list): list of jet variables.
        track_vars (list): list of track variables.
        jet_flavour (str): name of the jet flavour variable in the HDF5 file.
        jet_flavour_map (dict): mapping from raw hadron labels to target classes.
        norm_stats (dict, optional): normalization statistics for jet and track features.
    """

    def __init__(
        self,
        file_path: str,
        indices: np.ndarray,
        n_tracks: int = 40,
        jet_vars: Optional[List[str]] = None,
        track_vars: Optional[List[str]] = None,
        jet_flavour: Optional[str] = None,
        jet_flavour_map: Optional[Dict[int, int]] = None,
        norm_stats: Optional[Dict] = None
    ):
        """
        Initialize the dataset.

        Args:
            file_path (str): path of .h5 file.
            indices (np.ndarray): indices of jets to include in the dataset.
                Should be sorted for optimal performance.
            n_tracks (int): maximum number of tracks per jet (padding/cropping).
            jet_vars (list, optional): list of jet variables.
            track_vars (list, optional): list of track variables.
            jet_flavour (str, optional): name of the jet flavour variable in the HDF5 file.
            jet_flavour_map (dict, optional): mapping from raw hadron labels to target classes.
            norm_stats (dict, optional): normalization statistics. None = no normalization.

        Raises:
            FileNotFoundError: if the specified HDF5 file does not exist.
            KeyError: if the expected datasets ('jets', 'tracks') are not found.
        """
        self.file_path       = file_path
        self.indices         = indices
        self.n_tracks        = n_tracks
        self.jet_vars        = jet_vars   if jet_vars   is not None else JET_VARS_DEFAULT
        self.track_vars      = track_vars if track_vars is not None else TRACK_VARS_DEFAULT
        self.jet_flavour     = jet_flavour if jet_flavour is not None else JET_FLAVOUR_LABEL
        self.jet_flavour_map = jet_flavour_map if jet_flavour_map is not None else JET_FLAVOUR_MAP
        self.norm_stats      = norm_stats

        if norm_stats is None:
            logger.warning("No norm_stats provided - raw values will be used.")

        # warning if indices are not sorted (performance degradation)
        if len(indices) > 1 and not np.all(indices[:-1] <= indices[1:]):
            logger.warning("GN2Dataset: indices are NOT sorted.")

        # h5py file handler: None until opened lazily in _get_handler()
        self.handler = None

        # initial check
        try:
            with h5py.File(self.file_path, 'r') as f:
                self.n_jets = len(f['jets'])
                logger.info(f"Success loading {file_path}: {self.n_jets:,} jets found.")
                logger.debug(f"Original shape 'tracks': {f['tracks'].shape}")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _get_handler(self) -> h5py.File:
        """
        Manage the h5py file handler for multiprocessing.

        Returns:
            h5py.File: open h5py file object.
        """
        if self.handler is None:
            try:
                self.handler = h5py.File(self.file_path, 'r', swmr=True)        # swmr=True allows multiple readers (for num_workers > 0)
            except OSError:
                logger.warning("SWMR mode not supported on this filesystem. Standard read.")
                self.handler = h5py.File(self.file_path, 'r')
        return self.handler

    def _normalize_jet(self, jet_pt: float, jet_eta: float) -> np.ndarray:
        """
        Apply log-transform to pT and optional Z-score normalization.

        Args:
            jet_pt  (float): raw jet transverse momentum in MeV.
            jet_eta (float): raw jet pseudorapidity.

        Returns:
            np.ndarray: shape (2,), normalized [log_pt, eta].
        """
        jet_pt_log = np.log(jet_pt)
        if (self.norm_stats is not None and 'jet_mu' in self.norm_stats and 'jet_sigma' in self.norm_stats):
            mu    = self.norm_stats['jet_mu']
            sigma = self.norm_stats['jet_sigma']

            if len(mu) < 2 or len(sigma) < 2:
                logger.warning("norm_stats 'jet_mu'/'jet_sigma' have fewer than 2 entries. Using raw values for jet features.")
            else:
                jet_pt_log = (jet_pt_log - mu[0]) / sigma[0]
                jet_eta    = (jet_eta    - mu[1]) / sigma[1]

        return np.array([jet_pt_log, jet_eta], dtype=np.float32)

    def _process_tracks(self, tracks_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter valid tracks, crop/pad to n_tracks, and normalize.

        Args:
            tracks_raw (np.ndarray): structured array of all tracks for one jet,
                shape (max_tracks_in_file,).

        Returns:
            track_features (np.ndarray): shape (n_tracks, n_track_vars), float32.
            padding_mask   (np.ndarray): shape (n_tracks,), bool: True = real track.
        """
        track_features = np.zeros((self.n_tracks, len(self.track_vars)), dtype=np.float32)
        padding_mask   = np.zeros(self.n_tracks, dtype=bool)

        # slice only the first n_tracks valid entries.
        if 'valid' in tracks_raw.dtype.names:
            valid_mask           = tracks_raw['valid'].astype(bool)
            valid_local_indices  = np.where(valid_mask)[0][:self.n_tracks]
        else:
            valid_local_indices  = np.arange(min(len(tracks_raw), self.n_tracks))

        n_to_read = len(valid_local_indices)
        if n_to_read > 0:
            track_block = tracks_raw[valid_local_indices]   # only the needed rows
            for i, var in enumerate(self.track_vars):
                raw_values = track_block[var].astype(np.float32)
                if (self.norm_stats is not None and 'track_mu' in self.norm_stats and 'track_sigma' in self.norm_stats):
                    mu    = self.norm_stats['track_mu'][i]
                    sigma = self.norm_stats['track_sigma'][i]
                    track_features[:n_to_read, i] = (raw_values - mu) / sigma
                else:
                    track_features[:n_to_read, i] = raw_values
            padding_mask[:n_to_read] = True

        return track_features, padding_mask

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Returns the shape of the dataset.

        Returns:
            Tuple[int, int, int]: (n_jets, n_tracks, n_track_features).
        """
        return (self.n_jets, self.n_tracks, len(self.track_vars))

    def __len__(self) -> int:
        """
        Number of selected jets in the dataset.

        Returns:
            int: number of jets.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract a single jet and its associated tracks.

        Prefer build_dataloader() with BatchCollator over direct iteration:
        BatchCollator reads the whole batch in one HDF5 call, which is
        faster than individual __getitem__ call.

        Args:
            idx (int): dataset-level index.

        Returns:
            dict with keys:
                "jet_features"   (torch.Tensor, shape (n_jet_vars,))
                "track_features" (torch.Tensor, shape (n_tracks, n_track_vars))
                "mask"           (torch.Tensor, shape (n_tracks,))
                "label"          (torch.Tensor, scalar long)
        """
        f        = self._get_handler()
        real_idx = self.indices[idx]  # maps the dataset index to the actual jet index in the file

        # 1. Loading Jet Features and normalization
        jet_data  = f['jets'][real_idx]
        jet_feats = self._normalize_jet(jet_data['pt'], jet_data['eta'])

        # 2. Loading Label
        raw_label = jet_data[self.jet_flavour]
        target    = self.jet_flavour_map.get(int(raw_label), 0)

        # 3. Loading Tracks with 'valid' Filter (Optimized with slicing)
        tracks_raw = f['tracks'][real_idx]
        track_feats, padding_mask = self._process_tracks(tracks_raw)

        return {
            'jet_features':   torch.from_numpy(jet_feats),
            'track_features': torch.from_numpy(track_feats),
            'mask':           torch.from_numpy(padding_mask),
            'label':          torch.tensor(target, dtype=torch.long),
        }
    
class _IndexDataset(Dataset):
    """
    Thin wrapper around GN2Dataset whose __getitem__ returns the integer
    index instead of the processed sample.

    This is the key to making BatchCollator work correctly with PyTorch's
    DataLoader: the DataLoader calls __getitem__ on each element of a batch,
    collects the results into a list, and passes that list to collate_fn.
    By returning the raw index here, collate_fn receives List[int] - exactly
    what BatchCollator needs to do a single batched HDF5 read.
    """

    def __init__(self, dataset: GN2Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> int:
        return idx  # just pass the index through; BatchCollator does the real work



class BatchCollator:
    """
    Custom collate callable that reads an entire batch from HDF5 in one call.

    Works together with _IndexDataset: the DataLoader fetches indices via
    _IndexDataset.__getitem__ and passes List[int] to this collate_fn, which
    then sorts them for contiguous disk access and reads the whole batch in a
    single HDF5 fancy-index call.

    Usage: prefer build_dataloader() which sets everything up correctly.

    Args:
        dataset (GN2Dataset): the dataset instance to read from.
    """

    def __init__(self, dataset: GN2Dataset):
        self.dataset = dataset

    def __call__(self, local_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Read and process one batch.

        Args:
            local_indices (List[int]): dataset-level indices for this batch,
                returned by _IndexDataset.__getitem__ and collected by DataLoader.

        Returns:
            dict with batched tensors (same keys as GN2Dataset.__getitem__).
        """
        ds  = self.dataset
        f   = ds._get_handler()

        # sort local indices so that the corresponding real_idx values are
        # monotonically increasing → contiguous HDF5 reads
        local_arr  = np.asarray(local_indices, dtype=np.intp)
        real_idx   = ds.indices[local_arr]
        sort_order = np.argsort(real_idx)
        real_idx   = real_idx[sort_order] 

        # --- single batch read ---
        jets_raw   = f['jets'][real_idx]                 # one HDF5 call for all jets
        tracks_raw = f['tracks'][real_idx]               # one HDF5 call for all tracks

        n = len(real_idx)
        n_jet_vars   = len(ds.jet_vars)
        n_track_vars = len(ds.track_vars)

        jet_batch   = np.empty((n, n_jet_vars),              dtype=np.float32)
        track_batch = np.zeros((n, ds.n_tracks, n_track_vars), dtype=np.float32)
        mask_batch  = np.zeros((n, ds.n_tracks),              dtype=bool)
        label_batch = np.empty(n,                             dtype=np.int64)

        # --- jet features (vectorized) ---
        pt_log = np.log(jets_raw['pt'].astype(np.float32))
        eta    = jets_raw['eta'].astype(np.float32)

        if (
            ds.norm_stats is not None
            and 'jet_mu' in ds.norm_stats
            and 'jet_sigma' in ds.norm_stats
            and len(ds.norm_stats['jet_mu']) >= 2
            and len(ds.norm_stats['jet_sigma']) >= 2
        ):
            pt_log = (pt_log - ds.norm_stats['jet_mu'][0]) / ds.norm_stats['jet_sigma'][0]
            eta    = (eta    - ds.norm_stats['jet_mu'][1]) / ds.norm_stats['jet_sigma'][1]

        jet_batch[:, 0] = pt_log
        jet_batch[:, 1] = eta

        # --- labels (vectorized) ---
        raw_labels  = jets_raw[ds.jet_flavour].astype(int)
        label_batch = np.array([ds.jet_flavour_map.get(l, 0) for l in raw_labels], dtype=np.int64)

        # --- track features (per-jet loop, but I/O already done) ---
        has_valid    = 'valid' in tracks_raw.dtype.names
        has_norm     = (
            ds.norm_stats is not None
            and 'track_mu' in ds.norm_stats
            and 'track_sigma' in ds.norm_stats
        )
        track_mu    = ds.norm_stats['track_mu']    if has_norm else None
        track_sigma = ds.norm_stats['track_sigma'] if has_norm else None

        for i in range(n):
            jet_tracks = tracks_raw[i]

            if has_valid:
                valid_idx = np.where(jet_tracks['valid'].astype(bool))[0][:ds.n_tracks]
            else:
                valid_idx = np.arange(min(len(jet_tracks), ds.n_tracks))

            n_valid = len(valid_idx)
            if n_valid == 0:
                continue

            block = jet_tracks[valid_idx]
            for j, var in enumerate(ds.track_vars):
                vals = block[var].astype(np.float32)
                if has_norm:
                    vals = (vals - track_mu[j]) / track_sigma[j]
                track_batch[i, :n_valid, j] = vals

            mask_batch[i, :n_valid] = True

        # restore original batch order (undo sort)
        restore_order          = np.empty_like(sort_order)
        restore_order[sort_order] = np.arange(n)
        jet_batch   = jet_batch[restore_order]
        track_batch = track_batch[restore_order]
        mask_batch  = mask_batch[restore_order]
        label_batch = label_batch[restore_order]

        return {
            'jet_features':   torch.from_numpy(jet_batch),
            'track_features': torch.from_numpy(track_batch),
            'mask':           torch.from_numpy(mask_batch),
            'label':          torch.from_numpy(label_batch),
        }



def GN2DataLoader(
    dataset: GN2Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader that uses BatchCollator for fast batched HDF5 reads.

    num_workers is intentionally kept at 0 by default: the BatchCollator
    already minimises I/O by reading the whole batch in a single HDF5 call,
    so spawning extra processes only adds IPC overhead.  Set num_workers > 0
    only if you have confirmed that track processing (not I/O) is the bottleneck.

    Args:
        dataset     (GN2Dataset): dataset instance with sorted indices.
        batch_size  (int):        number of jets per batch.
        shuffle     (bool):       whether to shuffle the order of batches.
        num_workers (int):        DataLoader worker processes (default 0).
        pin_memory  (bool):       pin CPU tensors for faster GPU transfer.
        drop_last   (bool):       drop incomplete last batch.

    Returns:
        DataLoader: configured loader.
    """
    index_ds      = _IndexDataset(dataset)
    sampler       = RandomSampler(index_ds) if shuffle else SequentialSampler(index_ds)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
    collator      = BatchCollator(dataset)

    return DataLoader(
        index_ds,
        batch_sampler = batch_sampler,
        collate_fn    = collator,
        num_workers   = num_workers,
        pin_memory    = pin_memory,
    )




if __name__ == "__main__":

    import argparse
    import sys
    from src.trasformer_jet_tagging.utils import compute_normalization_stats
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(
        description="Test for GN2Dataset: loads a sample, checks shapes and normalization."
    )
    parser.add_argument("--file",       type=str,   required=True,  help="Path to the HDF5 file.")
    parser.add_argument("--n-tracks",   type=int,   default=40,     help="Maximum tracks per jet (default: 40).")
    parser.add_argument("--n-jets",     type=int,   default=None,   help="Limit test to first N jets (default: all).")
    parser.add_argument("--train-frac", type=float, default=0.7,    help="Training split fraction (default: 0.7).")
    parser.add_argument("--seed",       type=int,   default=42,     help="Random seed (default: 42).")
    parser.add_argument("--batch-size", type=int,   default=1024,   help="Batch size for the loader test (default: 1024).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        with h5py.File(args.file, 'r') as f:
            n_total = len(f['jets'])
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Cannot open HDF5 file: {e}")
        sys.exit(1)

    n_jets  = n_total if args.n_jets is None else min(args.n_jets, n_total)
    indices = np.arange(n_jets)
    logger.info(f"Total jets in file : {n_total:,}")
    logger.info(f"Jets used for test : {n_jets:,}")

    train_indices, test_indices = train_test_split(
        indices,
        train_size   = args.train_frac,
        random_state = args.seed,
        shuffle      = True,
    )
    train_indices = np.sort(train_indices)
    test_indices  = np.sort(test_indices)
    logger.info(f"Split - train: {len(train_indices):,}  test: {len(test_indices):,}")

    logger.info("Computing normalization statistics on training set ...")
    norm_stats = compute_normalization_stats(args.file, train_indices)

    train_dataset = GN2Dataset(
        args.file,
        indices=train_indices,
        n_tracks=args.n_tracks,
        norm_stats=norm_stats
    )
    test_dataset  = GN2Dataset(
        args.file,
        indices=test_indices,
        n_tracks=args.n_tracks,
        norm_stats=norm_stats
    )

    # shape check (single item)
    logger.info("=== Shape check (single item) ===")
    sample   = train_dataset[0]
    expected = {
        "jet_features":   (len(JET_VARS_DEFAULT),),
        "track_features": (args.n_tracks, len(TRACK_VARS_DEFAULT)),
        "mask":           (args.n_tracks,),
        "label":          (),
    }
    all_ok = True
    for key, exp_shape in expected.items():
        got    = tuple(sample[key].shape)
        status = "OK" if got == exp_shape else "FAIL"
        if status == "FAIL":
            all_ok = False
        logger.info(f"  {key:<20} expected {str(exp_shape):<25} got {str(got):<25} [{status}]")

    n_valid = sample["mask"].sum().item()
    logger.info(f"  valid tracks in sample[0] : {n_valid} / {args.n_tracks}")
    logger.info(f"  label in sample[0]        : {sample['label'].item()}"
                f" ({[k for k,v in JET_FLAVOUR_MAP.items() if v == sample['label'].item()]})")

    # batch loader test
    import time
    logger.info("=== Batch loader test (BatchCollator) ===")
    train_loader = GN2DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    t0    = time.time()
    batch = next(iter(train_loader))
    dt    = time.time() - t0
    logger.info(f"  Jets shape   : {batch['jet_features'].shape}")
    logger.info(f"  Tracks shape : {batch['track_features'].shape}")
    logger.info(f"  Labels shape : {batch['label'].shape}")
    logger.info(f"  Time for first batch: {dt:.2f} s")

    # normalization check
    logger.info("=== Normalization sanity check (first 1000 training jets) ===")
    n_check  = min(1_000, len(train_dataset))
    jet_pts  = []
    jet_etas = []
    for i in range(n_check):
        s = train_dataset[i]
        jet_pts.append(s["jet_features"][0].item())
        jet_etas.append(s["jet_features"][1].item())
    jet_pts  = np.array(jet_pts)
    jet_etas = np.array(jet_etas)
    logger.info(f"  jet pt  (normalized) - mean: {jet_pts.mean():.4f}  std: {jet_pts.std():.4f}  (expect ~0, ~1)")
    logger.info(f"  jet eta (normalized) - mean: {jet_etas.mean():.4f}  std: {jet_etas.std():.4f}  (expect ~0, ~1)")

    # length check
    logger.info("=== Length consistency ===")
    assert len(train_dataset) == len(train_indices), "train __len__ mismatch"
    assert len(test_dataset)  == len(test_indices),  "test  __len__ mismatch"
    logger.info(f"  train_dataset.__len__() = {len(train_dataset):,}  OK")
    logger.info(f"  test_dataset.__len__()  = {len(test_dataset):,}  OK")

    if all_ok:
        logger.info("All checks passed.")
    else:
        logger.error("One or more shape checks failed. See above.")
        sys.exit(1)