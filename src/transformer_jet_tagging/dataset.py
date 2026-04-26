"""
dataset.py
==========
High-performance GN2 data pipeline for HDF5 datasets.
Optimized for large-scale Jet Flavour Tagging at ATLAS.

HDF5 Structure:

  - "/jets"          - jet-level features (1 row per jet).
  - "/tracks"        - track-level features.
  - "/eventwise"     - event-level metadata.
  - "/truth_hadrons" - simulation truth for hadron labeling and performance studies.

The pipeline utilises "lazy loading" via h5py to handle datasets that exceed
available RAM, using NumPy vectorization for high-speed feature extraction.

Pipeline Workflow:

1. Data Loading:
   Batch extraction of jet and track features using a custom collate_fn that reads
   an entire batch in a single HDF5 call.

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

  - Use gn2_dataloader() which sets up the BatchCollator automatically.
    This replaces per-item __getitem__ HDF5 access with a single batched
    read per batch, reducing I/O overhead by orders of magnitude.

  - num_workers should be 0 when using BatchCollator (the collate_fn
    already parallelises I/O at the batch level via sorted slice reads).

  - _get_handler() is thread-safe via a per-instance Lock, but h5py file
    handles are NOT shared across processes: each DataLoader worker opens
    its own handle automatically on first access.
"""

import logging
import threading

import h5py
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from .constants import (
    JET_FLAVOUR_LABEL,
    JET_FLAVOUR_MAP,
    JET_VARS_DEFAULT,
    TRACK_VARS_DEFAULT,
)

logger = logging.getLogger("GN2.dataset")


class GN2Dataset(Dataset):
    """
    Dataset for flavour tagger using HDF5 file.
    Lazy loading of data for large datasets with NumPy vectorization.
    Includes filtering of invalid tracks and feature normalization.

    IMPORTANT: indices should be sorted before passing to this class
    to guarantee contiguous reads when the BatchCollator is used.

    Attributes:
        h5_file_path (str): path of .h5 file.
        jet_indices (np.ndarray): sorted indices of jets to include in the dataset.
        max_tracks (int): maximum number of tracks for each jet (padding/cropping).
        jet_vars (list): list of jet variables.
        track_vars (list): list of track variables.
        jet_flavour (str): name of the jet flavour variable in the HDF5 file.
        jet_flavour_map (dict): mapping from raw hadron labels to target classes.
        stats (dict or None): normalization statistics for jet and track features.
    """

    def __init__(
        self,
        h5_file_path: str,
        jet_indices: np.ndarray,
        max_tracks: int = 40,
        jet_vars: list[str] | None = None,
        track_vars: list[str] | None = None,
        jet_flavour: str | None = None,
        jet_flavour_map: dict[int, int] | None = None,
        stats: dict | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            h5_file_path (str): path of .h5 file.
            jet_indices (np.ndarray): indices of jets to include in the dataset.
                Should be sorted for optimal performance.
            max_tracks (int): maximum number of tracks per jet (padding/cropping).
            jet_vars (list, optional): list of jet variables.
            track_vars (list, optional): list of track variables.
            jet_flavour (str, optional): name of the jet flavour variable in the HDF5 file.
            jet_flavour_map (dict, optional): mapping from raw hadron labels to target classes.
            stats (dict, optional): normalization statistics. None = no normalization.

        Raises:
            FileNotFoundError: if the specified HDF5 file does not exist.
            KeyError: if the expected datasets ("jets", "tracks") are not found in the file.
        """
        self.h5_file_path    = h5_file_path
        self.jet_indices     = np.asarray(jet_indices)
        self.max_tracks      = max_tracks
        self.jet_vars        = jet_vars        if jet_vars        is not None else JET_VARS_DEFAULT
        self.track_vars      = track_vars      if track_vars     is not None else TRACK_VARS_DEFAULT
        self.jet_flavour     = jet_flavour     if jet_flavour     is not None else JET_FLAVOUR_LABEL
        self.jet_flavour_map = jet_flavour_map if jet_flavour_map is not None else JET_FLAVOUR_MAP
        self.stats           = stats

        if stats is None:
            logger.warning("No stats provided - raw values will be used.")

        # warn if jet_indices are not sorted
        if len(self.jet_indices) > 1 and not np.all(self.jet_indices[:-1] <= self.jet_indices[1:]):
            logger.warning("GN2Dataset: jet_indices are NOT sorted. Performance may degrade.")

        # h5py file handler: opened lazily in _get_handler(), one per thread
        self._handler: h5py.File | None = None
        self._handler_lock = threading.Lock()

        # initial check
        try:
            with h5py.File(self.h5_file_path, "r") as h5_handle:
                self.n_total_jets = len(h5_handle["jets"])
                logger.info("Success loading %s: %s jets found.",
                            h5_file_path, f"{self.n_total_jets:,}")
                logger.debug("Original shape tracks: %s", f"{h5_handle["tracks"].shape}")
        except (FileNotFoundError, KeyError) as e:
            logger.error("Error loading file %s: %s", h5_file_path, e)
            raise

    def _get_handler(self) -> h5py.File:
        """
        Manage the h5py file handler for multiprocessing.

        Returns:
            (h5py.File): open file object in read mode.

        Raises:
            OSError: if the file cannot be opened.
        """
        if self._handler is None:
            with self._handler_lock:
                # double-checked locking: re-test after acquiring the lock
                if self._handler is None:
                    try:
                        # Single Writer Multiple Reader: allows multiple readers (num_workers > 0)
                        self._handler = h5py.File(self.h5_file_path, "r", swmr=True)
                    except OSError:
                        logger.warning("SWMR mode not supported on this filesystem. Standard read.")
                        self._handler = h5py.File(self.h5_file_path, "r")
        return self._handler

    def _process_jet(self, jet_pt: np.ndarray, jet_eta: np.ndarray) -> np.ndarray:
        """
        Apply log-transform to pT and optional Z-score normalization.

        Args:
            jet_pt (np.ndarray): shape (n,), raw jet transverse momentum in MeV.
            jet_eta (np.ndarray): shape (n,), raw jet pseudorapidity.

        Returns:
            (np.ndarray): shape (n, 2), normalized [log_pt, eta].
        """
        eps = 1e-8
        jet_pt_log = np.log(np.clip(jet_pt.astype(np.float32), eps, None))
        jet_eta    = jet_eta.astype(np.float32)

        if (
            self.stats is not None
            and "jet_mu"    in self.stats
            and "jet_sigma" in self.stats
            and len(self.stats["jet_mu"])    >= 2
            and len(self.stats["jet_sigma"]) >= 2
        ):
            jet_pt_log = (jet_pt_log - self.stats["jet_mu"][0]) / self.stats["jet_sigma"][0]
            jet_eta    = (jet_eta    - self.stats["jet_mu"][1]) / self.stats["jet_sigma"][1]

        return np.stack([jet_pt_log, jet_eta], axis=1)  # (n, 2)

    def _process_tracks(self, tracks_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter valid tracks, crop/pad to max_tracks, and normalize.

        Args:
            tracks_raw (np.ndarray): shape (max_tracks_in_file,),
                structured array of all tracks for one jet.

        Returns:
            track_features (np.ndarray): shape (max_tracks, n_track_vars), float32.
            padding_mask (np.ndarray): shape (max_tracks,), bool; True = real track.
        """
        track_features = np.zeros((self.max_tracks, len(self.track_vars)), dtype=np.float32)
        padding_mask   = np.zeros(self.max_tracks, dtype=bool)

        # slice only the first max_tracks valid entries.
        if "valid" in tracks_raw.dtype.names:
            valid_idx = np.where(tracks_raw["valid"].astype(bool))[0][:self.max_tracks]
        else:
            valid_idx = np.arange(min(len(tracks_raw), self.max_tracks))

        n_valid_tracks = len(valid_idx)
        if n_valid_tracks > 0:
            track_block = tracks_raw[valid_idx]
            for i, var in enumerate(self.track_vars):
                vals = track_block[var].astype(np.float32)
                if (
                    self.stats is not None
                    and "track_mu"    in self.stats
                    and "track_sigma" in self.stats
                ):
                    vals = (vals - self.stats["track_mu"][i])/self.stats["track_sigma"][i]
                track_features[:n_valid_tracks, i] = vals
            padding_mask[:n_valid_tracks] = True

        return track_features, padding_mask

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Shape of the dataset.

        Returns:
            (tuple[int, int, int]): shape (n_total_jets, max_tracks, n_track_features).
        """
        return (self.n_total_jets, self.max_tracks, len(self.track_vars))

    def __len__(self) -> int:
        """
        Number of selected jets in the dataset.

        Returns:
            (int): number of jets.
        """
        return len(self.jet_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Extract a single jet and its associated tracks.

        IMPORTANT:
        This method is NOT used during training when using gn2_dataloader.
        It exists only for debugging and standalone inspection.

        Args:
            idx (int): dataset-level index.

        Returns:
            dict with keys:
                "jet_features" (torch.Tensor, shape (n_jet_vars,))
                "track_features" (torch.Tensor, shape (max_tracks, n_track_vars))
                "mask" (torch.Tensor, shape (max_tracks,))
                "label" (torch.Tensor, scalar long)
        """
        h5_handle  = self._get_handler()
        real_idx = self.jet_indices[idx]    # maps dataset index to the actual jet index in the file

        # 1. jet features
        jet_data  = h5_handle["jets"][real_idx]
        jet_feats = self._process_jet(
            np.array([jet_data["pt"]]),
            np.array([jet_data["eta"]]),
        )[0]

        # 2. track features
        tracks_raw = h5_handle["tracks"][real_idx]
        track_feats, pad_mask = self._process_tracks(tracks_raw)

        # 3. label
        raw_label = int(jet_data[self.jet_flavour])
        target    = self.jet_flavour_map.get(raw_label, -1)
        if target == -1:
            logger.warning(
                "__getitem__: unmapped jet_flavour label %s at file index %s. Assigned label -1.",
                raw_label, real_idx,
            )

        return {
            "jet_features":   torch.from_numpy(jet_feats),
            "track_features": torch.from_numpy(track_feats),
            "mask":           torch.from_numpy(pad_mask),
            "label":          torch.tensor(target, dtype=torch.long),
        }


class _IndexDataset(Dataset):
    """
    Wrapper around GN2Dataset whose __getitem__ returns the integer
    index instead of the processed sample.

    The DataLoader collects these integers into a list and passes them to
    _BatchCollator.__call__, which performs a single batched HDF5 read.

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
    def shape(self) -> tuple[int, int, int]:
        """
        Returns the shape of the dataset.

        Returns:
            (tuple[int, int, int]): shape (n_total_jets, max_tracks, n_track_features).
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
        dataset (GN2Dataset): dataset instance to read from.
    """

    def __init__(self, dataset: GN2Dataset):
        """
        Initializer of the collate.

        Args:
            dataset (GN2Dataset): the dataset instance to read from.
        """
        self.dataset = dataset

    def __call__(self, jet_indices: list[int]) -> dict[str, torch.Tensor]:
        """
        Read and process one batch.

        Args:
            jet_indices (List[int]): dataset-level indices for this batch,
                as returned by _IndexDataset.__getitem__ and collected by DataLoader.

        Returns:
            dict with keys:
                "jet_features" (torch.Tensor, shape (B, n_jet_vars))
                "track_features" (torch.Tensor, shape (B, max_tracks, n_track_vars))
                "mask" (torch.Tensor, shape (B, max_tracks))
                "label" (torch.Tensor, shape (B,))
        """
        dataset   = self.dataset
        h5_handle = dataset._get_handler()

        # sort "jet_indices" for contiguous HDF5 reads
        jet_indices = np.asarray(jet_indices)
        real_idx    = dataset.jet_indices[jet_indices]
        sort_order  = np.argsort(real_idx)
        real_idx    = real_idx[sort_order]

        n_batch_jets = len(real_idx)
        n_track_vars = len(dataset.track_vars)
        max_tracks     = dataset.max_tracks

        # 1. single batched HDF5 read
        jets_raw   = h5_handle["jets"][real_idx]
        tracks_raw = h5_handle["tracks"][real_idx]

        # 2. jet features
        jet_batch = dataset._process_jet(jets_raw["pt"], jets_raw["eta"]).astype(np.float32)

        # 3. track features
        track_batch = np.zeros((n_batch_jets, max_tracks, n_track_vars), dtype=np.float32)
        mask_batch  = np.zeros((n_batch_jets, max_tracks), dtype=bool)

        has_valid = "valid" in tracks_raw.dtype.names
        has_norm  = (
            dataset.stats is not None
            and "track_mu"    in dataset.stats
            and "track_sigma" in dataset.stats
        )
        track_mu    = dataset.stats["track_mu"]    if has_norm else None
        track_sigma = dataset.stats["track_sigma"] if has_norm else None

        # (n_batch_jets, max_tracks) boolean validity matrix
        if has_valid:
            valid_matrix = tracks_raw["valid"].astype(bool)[:, :max_tracks]
        else:
            max_file_tracks = tracks_raw.shape[1]
            valid_matrix    = np.ones((n_batch_jets, min(max_file_tracks, max_tracks)), dtype=bool)

        # pad/crop to max_tracks columns
        if valid_matrix.shape[1] < max_tracks:
            pad_cols     = max_tracks - valid_matrix.shape[1]
            valid_matrix = np.pad(valid_matrix, ((0, 0), (0, pad_cols)))

        mask_batch[:] = valid_matrix

        for i, var in enumerate(dataset.track_vars):
            raw_block = tracks_raw[var].astype(np.float32)[:, :max_tracks]    # (n_jets, max_tracks)
            if raw_block.shape[1] < max_tracks:
                pad_width = max_tracks - raw_block.shape[1]
                raw_block = np.pad(raw_block, ((0, 0), (0, pad_width)))

            track_batch[:, :, i] = raw_block

            if has_norm:
                raw_block = (raw_block - track_mu[i]) / track_sigma[i]

            # zero out padding positions
            raw_block[~valid_matrix] = 0.
            track_batch[:, :, i]     = raw_block

        # 4. labels
        raw_labels  = jets_raw[dataset.jet_flavour].astype(int)
        label_batch = np.array(
            [dataset.jet_flavour_map.get(label, -1) for label in raw_labels],
            dtype=np.int64,
        )
        n_unknown = int((label_batch == -1).sum())
        if n_unknown > 0:
            logger.warning(
                "BatchCollator: %s/%s jets have unmapped jet_flavour labels "
                "and will carry label -1. Check jet_flavour_map.", n_unknown, n_batch_jets
            )

        # restore original batch order (undo sort)
        restore_order = np.empty_like(sort_order)
        restore_order[sort_order] = np.arange(n_batch_jets)

        return {
            "jet_features":   torch.from_numpy(jet_batch[restore_order]),
            "track_features": torch.from_numpy(track_batch[restore_order]),
            "mask":           torch.from_numpy(mask_batch[restore_order]),
            "label":          torch.from_numpy(label_batch[restore_order]),
        }


def gn2_dataloader(
    dataset: GN2Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader that uses _BatchCollator for fast batched HDF5 reads.

    Args:
        dataset (GN2Dataset): dataset instance with sorted indices.
        batch_size (int): number of jets per batch.
        shuffle (bool): whether to shuffle the order of batches.
        num_workers (int): DataLoader worker processes.
        pin_memory (bool): pin CPU tensors for faster GPU transfer.
        drop_last (bool): drop incomplete last batch.

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
    import time

    from sklearn.model_selection import train_test_split

    from .utils import compute_normalization_stats, load_config_json

    parser = argparse.ArgumentParser(
        description="Test for GN2Dataset: loads a sample, checks shapes and normalization."
    )
    parser.add_argument("--config", type=str, default="configs/config.json")
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
    bs         = config["data"].get("batch_size", 1024)

    try:
        with h5py.File(file_path, "r") as h5_file:
            n_total_jets = len(h5_file["jets"])
    except (FileNotFoundError, KeyError) as e:
        logger.error("Cannot open HDF5 file: %s", e)
        sys.exit(1)

    indices = np.arange(n_total_jets)
    logger.info("Total jets in file: %s", f"{n_total_jets:,}")

    train_indices, test_indices = train_test_split(
        indices,
        train_size   = train_frac,
        random_state = seed,
        shuffle      = True,
    )
    train_indices = np.sort(train_indices)
    test_indices  = np.sort(test_indices)
    logger.info("Split - train: %s  test: %s",
                f"{len(train_indices):,}",
                f"{len(train_indices):,}")

    logger.info("Computing normalization statistics on training set ...")
    norm_stats = compute_normalization_stats(file_path, train_indices)

    train_dataset = GN2Dataset(
        file_path,
        jet_indices=train_indices,
        max_tracks=n_tracks,
        stats=norm_stats
    )
    test_dataset = GN2Dataset(
        file_path,
        jet_indices=test_indices,
        max_tracks=n_tracks,
        stats=norm_stats
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
        logger.info("  %s expected %s got %s [%s]",
                    f"{key:<20}", f"{str(exp_shape):<25}", f"{str(got):<25}", f"{status}")

    n_valid = sample["mask"].sum().item()
    logger.info("  valid tracks in sample[0] : %s / %s", n_valid, n_tracks)
    logger.info("  label in sample[0]        : %s (%s)", sample["label"].item(),
                [k for k, v in JET_FLAVOUR_MAP.items() if v == sample['label'].item()])

    # batch loader speed test
    logger.info("=== Batch loader test (BatchCollator) ===")
    train_loader = gn2_dataloader(train_dataset, batch_size=bs, shuffle=True)
    t0    = time.perf_counter()
    batch = next(iter(train_loader))
    dt    = time.perf_counter() - t0
    logger.info("  Jets shape   : %s", batch["jet_features"].shape)
    logger.info("  Tracks shape : %s", batch["track_features"].shape)
    logger.info("  Labels shape : %s", batch["label"].shape)
    logger.info("  Time for first batch: %s s", f"{dt:.2f}")

    # normalization check
    logger.info("=== Normalization sanity check (first 1000 training jets) ===")
    n_check  = min(1_000, len(train_dataset))
    jet_pts  = np.array([train_dataset[i]["jet_features"][0].item() for i in range(n_check)])
    jet_etas = np.array([train_dataset[i]["jet_features"][1].item() for i in range(n_check)])
    logger.info("  jet pt  (normalized) - mean: %s  std: %s  (expect ~0, ~1)",
                f"{jet_pts.mean():.4f}", f"{jet_pts.std():.4f}")
    logger.info("  jet eta (normalized) - mean: %s  std: %s  (expect ~0, ~1)",
                f"{jet_etas.mean():.4f}", f"{jet_etas.std():.4f}")

    # length check
    logger.info("=== Length consistency ===")
    assert len(train_dataset) == len(train_indices), "train __len__ mismatch"
    assert len(test_dataset)  == len(test_indices),  "test  __len__ mismatch"
    logger.info("  train_dataset.__len__() = %s  OK", f"{len(train_dataset):,}")
    logger.info("  test_dataset.__len__()  = %s  OK", f"{len(test_dataset):,}")

    if all_ok:
        logger.info("All checks passed.")
    else:
        logger.error("One or more shape checks FAILED. See above.")
        sys.exit(1)
