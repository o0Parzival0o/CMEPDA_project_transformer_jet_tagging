"""
dataset.py
==========
High-performance GN2 data pipeline for HDF5 datasets. 
Optimized for large-scale Jet Flavour Tagging at ATLAS.

HDF5 Structure:
  /jets          - jet-level features (1 row per jet).
  /tracks        - track-level features.
  /eventwise     - event-level metadata.
  /truth_hadrons - simulation truth for hadron labeling and performance studies.

The pipeline utilises "lazy loading" via h5py to handle datasets that exceed 
available RAM, using NumPy vectorization for high-speed feature extraction.

Pipeline Workflow:

1. Data Loading:
   Batch extraction of jet and track features using a custom collate_fn that reads an entire batch in a single HDF5 call.

2. Track Quality Filtering:
   Active filtering of track candidates using the "valid" boolean flag before processing.

3. Padding & Masking:
   Enforce a fixed-size track array (default max_tracks=40).
   Generate a boolean padding mask to inform the Transformer's Self-Attention mechanism.

4. Feature Engineering:
   - Jet-level: log-transformation of pT and Z-score normalization (only on training set).
   - Track-level: Z-score normalization (only on training set).

6. Integration:
   Wraps everything into a PyTorch DataLoader with optimized batching.

Performance notes:
  - Indices passed to GN2Dataset MUST be sorted to ensure
    contiguous HDF5 reads and avoid random seeks on disk.
  - Use GN2DataLoader() which sets up the BatchCollator automatically.
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
        n_tracks (int, optional): maximum number of tracks for each jet (padding/cropping).
        jet_vars (list, optional): list of jet variables.
        track_vars (list, optional): list of track variables.
        jet_flavour (str, optional): name of the jet flavour variable in the HDF5 file.
        jet_flavour_map (dict, optional): mapping from raw hadron labels to target classes.
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
            KeyError: if the expected datasets ("jets", "tracks") are not found.
        """
        self.file_path       = file_path
        self.indices         = indices
        self.n_tracks        = n_tracks
        self.jet_vars        = jet_vars        if jet_vars        is not None else JET_VARS_DEFAULT
        self.track_vars      = track_vars      if track_vars      is not None else TRACK_VARS_DEFAULT
        self.jet_flavour     = jet_flavour     if jet_flavour     is not None else JET_FLAVOUR_LABEL
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
            with h5py.File(self.file_path, "r") as f:
                self.n_jets = len(f["jets"])
                logger.info(f"Success loading {file_path}: {self.n_jets:,} jets found.")
                logger.debug(f'Original shape tracks": {f["tracks"].shape}')
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _get_handler(self) -> h5py.File:
        """
        Manage the h5py file handler for multiprocessing.

        Returns:
            (h5py.File): open h5py file object.

        Raise:
            OSError: TODO
        """
        if self.handler is None:
            try:
                self.handler = h5py.File(self.file_path, "r", swmr=True)        # Single Writer Multiple Reader: allows multiple readers (for num_workers > 0)
            except OSError:
                logger.warning("SWMR mode not supported on this filesystem. Standard read.")
                self.handler = h5py.File(self.file_path, "r")
        return self.handler

    def _process_jet(self, jet_pt: np.ndarray, jet_eta: np.ndarray) -> np.ndarray:
        """
        Apply log-transform to pT and optional Z-score normalization.

        Args:
            jet_pt  (np.ndarray): raw jet transverse momentum in MeV.
            jet_eta (np.ndarray): raw jet pseudorapidity.

        Returns:
            (np.ndarray): shape (n, 2), normalized [log_pt, eta].
        """
        jet_pt_log = np.log(jet_pt.astype(np.float32))
        jet_eta    = jet_eta.astype(np.float32)

        if (
            self.norm_stats is not None
            and "jet_mu" in self.norm_stats
            and "jet_sigma" in self.norm_stats
            and len(self.norm_stats["jet_mu"]) >= 2
            and len(self.norm_stats["jet_sigma"]) >= 2
        ):
            jet_pt_log = (jet_pt_log - self.norm_stats["jet_mu"][0]) / self.norm_stats["jet_sigma"][0]
            jet_eta    = (jet_eta    - self.norm_stats["jet_mu"][1]) / self.norm_stats["jet_sigma"][1]

        return np.stack([jet_pt_log, jet_eta], axis=1)  # (n, 2)

    def _process_tracks(self, tracks_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter valid tracks, crop/pad to n_tracks and normalize.

        Args:
            tracks_raw (np.ndarray): shape (max_tracks_in_file,), structured array of all tracks for one jet.

        Returns:
            track_features (np.ndarray): shape (n_tracks, n_track_vars), float32.
            padding_mask   (np.ndarray): shape (n_tracks,), bool: True = real track.
        """
        track_features = np.zeros((self.n_tracks, len(self.track_vars)), dtype=np.float32)
        padding_mask   = np.zeros(self.n_tracks, dtype=bool)

        # slice only the first n_tracks valid entries.
        if "valid" in tracks_raw.dtype.names:
            valid_mask          = tracks_raw["valid"].astype(bool)
            valid_track_indices = np.where(valid_mask)[0][:self.n_tracks]
        else:
            valid_track_indices = np.arange(min(len(tracks_raw), self.n_tracks))

        n_to_read = len(valid_track_indices)
        if n_to_read > 0:
            track_block = tracks_raw[valid_track_indices]   # only the needed rows
            for i, var in enumerate(self.track_vars):
                raw_values = track_block[var].astype(np.float32)
                if (self.norm_stats is not None and "track_mu" in self.norm_stats and "track_sigma" in self.norm_stats):
                    mu    = self.norm_stats["track_mu"][i]
                    sigma = self.norm_stats["track_sigma"][i]
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
            (tuple[int, int, int]): shape (n_jets, n_tracks, n_track_features).
        """
        return (self.n_jets, self.n_tracks, len(self.track_vars))

    def __len__(self) -> int:
        """
        Number of selected jets in the dataset.

        Returns:
            (int): number of jets.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract a single jet and its associated tracks.

        IMPORTANT:
        This method is NOT used during training when using GN2DataLoader.
        It exists only for debugging and standalone inspection.

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
        real_idx = self.indices[idx]    # maps the dataset index to the actual jet index in the file

        # 1. Loading Jet Features and normalization
        jet_data  = f["jets"][real_idx]
        jet_feats = np.array([jet_data["pt"], jet_data["eta"]])
        jet_feats = self._process_jet(np.array([jet_data["pt"]]), np.array([jet_data["eta"]]))[0]

        # 2. Loading Tracks with "valid" Filter
        tracks_raw = f["tracks"][real_idx]
        track_feats, padding_mask = self._process_tracks(tracks_raw)

        # 3. Loading Label
        raw_label = jet_data[self.jet_flavour]
        target    = self.jet_flavour_map.get(int(raw_label), -1)

        return {
            "jet_features":   torch.from_numpy(jet_feats),
            "track_features": torch.from_numpy(track_feats),
            "mask":           torch.from_numpy(padding_mask),
            "label":          torch.tensor(target, dtype=torch.long),
        }
    

class _IndexDataset(Dataset):
    """
    Wrapper around GN2Dataset whose __getitem__ returns the integer
    index instead of the processed sample.

    DataLoader calls __getitem__ on each element of a batch,
    collects the results into a list, and passes that list to collate_fn.
    By returning the raw index here, collate_fn receives List[int] - exactly
    what BatchCollator needs to do a single batched HDF5 read.

    Attributes:
        dataset (GN2Dataset): the dataset to wrap.
    """

    def __init__(self, dataset: GN2Dataset):
        """
        Initialize the wrapper.

        Args:
            dataset (GN2Dataset): the dataset to wrap.
        """
        self.dataset = dataset

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Returns the shape of the dataset.

        Returns:
            (tuple[int, int, int]): shape (n_jets, n_tracks, n_track_features).
        """
        return self.dataset.shape

    def __len__(self) -> int:
        """
        Number of selected jets in the dataset.

        Returns:
            (int): number of jets.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> int:
        """
        Extract the dataset index for the given position.

        Args:
            idx (int): position in the dataset.

        Returns:
            (int): dataset-level index.
        """
        return idx


class _BatchCollator:
    """
    Custom collate callable that reads an entire batch from HDF5 in one call.

    The DataLoader fetches indices via _IndexDataset.__getitem__ and passes
    List[int] to this collate_fn, which then sorts them for contiguous disk
    access and reads the whole batch in a single HDF5 fancy-index call.

    Parameters:
        dataset (GN2Dataset): the dataset instance to read from.
    """

    def __init__(self, dataset: GN2Dataset):
        """
        Initializer of the collate.

        Args:
            dataset (GN2Dataset): the dataset instance to read from.
        """
        self.dataset = dataset

    def __call__(self, local_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Read and process one batch.

        Args:
            local_indices (List[int]): dataset-level indices for this batch,
                returned by _IndexDataset.__getitem__ and collected by DataLoader.

        Returns:
            dict with keys:
                "jet_features"   (torch.Tensor, shape (n_jet_vars,))
                "track_features" (torch.Tensor, shape (n_tracks, n_track_vars))
                "mask"           (torch.Tensor, shape (n_tracks,))
                "label"          (torch.Tensor, scalar long)

        Raise:
            ValueError: TODO
        """
        dataset = self.dataset
        f       = dataset._get_handler()

        # sort local indices
        local_indices = np.asarray(local_indices)
        real_idx      = dataset.indices[local_indices]
        sort_order    = np.argsort(real_idx)
        real_idx      = real_idx[sort_order] 

        # 1. single batch read
        jets_raw   = f["jets"][real_idx]                 # one HDF5 call for all jets
        tracks_raw = f["tracks"][real_idx]               # one HDF5 call for all tracks

        n_jets       = len(real_idx)
        n_jet_vars   = len(dataset.jet_vars)
        n_track_vars = len(dataset.track_vars)

        jet_batch   = np.empty((n_jets, n_jet_vars), dtype=np.float32)
        track_batch = np.zeros((n_jets, dataset.n_tracks, n_track_vars), dtype=np.float32)
        mask_batch  = np.zeros((n_jets, dataset.n_tracks), dtype=bool)
        label_batch = np.empty(n_jets, dtype=np.int64)

        # 2. jet features
        jet_feats = dataset._process_jet(jets_raw["pt"], jets_raw["eta"])
        jet_batch[:] = jet_feats

        # 3. track features
        has_valid = "valid" in tracks_raw.dtype.names
        has_norm  = (
            dataset.norm_stats is not None
            and "track_mu" in dataset.norm_stats
            and "track_sigma" in dataset.norm_stats
        )
        track_mu    = dataset.norm_stats["track_mu"]    if has_norm else None
        track_sigma = dataset.norm_stats["track_sigma"] if has_norm else None

        for i in range(n_jets):
            jet_tracks = tracks_raw[i]

            if has_valid:
                valid_track_indices = np.where(jet_tracks["valid"].astype(bool))[0][:dataset.n_tracks]
            else:
                valid_track_indices = np.arange(min(len(jet_tracks), dataset.n_tracks))

            n_valid = len(valid_track_indices)
            if n_valid == 0:
                continue

            track_block = jet_tracks[valid_track_indices]
            for j, var in enumerate(dataset.track_vars):
                vals = track_block[var].astype(np.float32)
                if has_norm:
                    vals = (vals - track_mu[j]) / track_sigma[j]
                track_batch[i, :n_valid, j] = vals

            mask_batch[i, :n_valid] = True

        # 4. labels
        raw_labels  = jets_raw[dataset.jet_flavour].astype(int)
        label_batch = np.array([dataset.jet_flavour_map.get(label, -1) for label in raw_labels], dtype=np.int64)
        
        # restore original batch order (undo sort)
        restore_order = np.empty_like(sort_order)
        restore_order[sort_order] = np.arange(n_jets)
        jet_batch   = jet_batch[restore_order]
        track_batch = track_batch[restore_order]
        mask_batch  = mask_batch[restore_order]
        label_batch = label_batch[restore_order]

        return {
            "jet_features":   torch.from_numpy(jet_batch),
            "track_features": torch.from_numpy(track_batch),
            "mask":           torch.from_numpy(mask_batch),
            "label":          torch.from_numpy(label_batch),
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

    Args:
        dataset     (GN2Dataset): dataset instance with sorted indices.
        batch_size  (int): number of jets per batch.
        shuffle     (bool): whether to shuffle the order of batches.
        num_workers (int): DataLoader worker processes.
        pin_memory  (bool): pin CPU tensors for faster GPU transfer.
        drop_last   (bool): drop incomplete last batch.

    Returns:
        (DataLoader): configured loader.
    """
    index_dataset = _IndexDataset(dataset)          # only for iteration purpose (no data)
    sampler       = RandomSampler(index_dataset) if shuffle else SequentialSampler(index_dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
    collator      = _BatchCollator(dataset)         # read data

    return DataLoader(
        index_dataset,
        batch_sampler = batch_sampler,
        collate_fn    = collator,
        num_workers   = num_workers,
        pin_memory    = pin_memory,
    )



if __name__ == "__main__":

    import argparse
    import sys
    from src.trasformer_jet_tagging.utils import compute_normalization_stats, load_config_json
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(
        description="Test for GN2Dataset: loads a sample, checks shapes and normalization."
    )

    parser.add_argument("--config",     type=str,   default="configs/config.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config_json(args.config)

    file_path  = config["data"]["h5_path"]
    n_tracks   = config["data"].get("max_tracks", 40)
    train_frac = config["data"].get("train_fraction", 0.7)
    seed       = config["data"].get("split_seed", 42)
    batch_size = config["data"].get("batch_size", 1024)

    try:
        with h5py.File(file_path, "r") as f:
            n_jets = len(f["jets"])
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Cannot open HDF5 file: {e}")
        sys.exit(1)

    indices = np.arange(n_jets)
    logger.info(f"Jets used for test : {n_jets:,}")

    train_indices, test_indices = train_test_split(
        indices,
        train_size   = train_frac,
        random_state = seed,
        shuffle      = True,
    )
    train_indices = np.sort(train_indices)
    test_indices  = np.sort(test_indices)
    logger.info(f"Split - train: {len(train_indices):,}  test: {len(test_indices):,}")

    logger.info("Computing normalization statistics on training set ...")
    norm_stats = compute_normalization_stats(file_path, train_indices)

    train_dataset = GN2Dataset(
        file_path,
        indices=train_indices,
        n_tracks=n_tracks,
        norm_stats=norm_stats
    )
    test_dataset  = GN2Dataset(
        file_path,
        indices=test_indices,
        n_tracks=n_tracks,
        norm_stats=norm_stats
    )

    # shape check (single item)
    logger.info("=== Shape check (single item) ===")
    sample   = train_dataset[0]
    expected = {
        "jet_features":   (len(config["data"].get("jet_features", JET_VARS_DEFAULT)),),
        "track_features": (n_tracks, len(config["data"].get("track_features", TRACK_VARS_DEFAULT))),
        "mask":           (n_tracks,),
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
    logger.info(f"  valid tracks in sample[0] : {n_valid} / {n_tracks}")
    logger.info(f"  label in sample[0]        : {sample["label"].item()}"
                f" ({[k for k,v in JET_FLAVOUR_MAP.items() if v == sample["label"].item()]})")

    # batch loader test
    import time
    logger.info("=== Batch loader test (BatchCollator) ===")
    train_loader = GN2DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    t0    = time.time()
    batch = next(iter(train_loader))
    dt    = time.time() - t0
    logger.info(f"  Jets shape   : {batch["jet_features"].shape}")
    logger.info(f"  Tracks shape : {batch["track_features"].shape}")
    logger.info(f"  Labels shape : {batch["label"].shape}")
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