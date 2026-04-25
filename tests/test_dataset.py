"""
test_dataset.py
"""

import numpy as np
from src.transformer_jet_tagging.dataset import GN2Dataset, _BatchCollator


def test_dataset_len(fake_hdf5_file):

    ds = GN2Dataset(
        h5_file_path = fake_hdf5_file,
        jet_indices  = np.arange(10),
        max_tracks   = 4,
        jet_vars     = ["pt", "eta"],
        track_vars   = ["x", "y"],
        jet_flavour  = "label",
    )

    assert len(ds) == 10


def test_getitem_output_shapes(fake_hdf5_file):

    ds = GN2Dataset(
        h5_file_path = fake_hdf5_file,
        jet_indices  = np.arange(10),
        max_tracks   = 4,
        jet_vars     = ["pt", "eta"],
        track_vars   = ["x", "y"],
        jet_flavour  = "label",
    )

    sample = ds[0]

    assert sample["jet_features"].shape == (2,)
    assert sample["track_features"].shape == (4, 2)
    assert sample["mask"].shape == (4,)
    assert "label" in sample


def test_getitem_finite_values(fake_hdf5_file):

    ds = GN2Dataset(
        h5_file_path = fake_hdf5_file,
        jet_indices  = np.arange(10),
        max_tracks   = 4,
        jet_vars     = ["pt", "eta"],
        track_vars   = ["x", "y"],
        jet_flavour  = "label",
    )

    sample = ds[0]

    assert np.isfinite(sample["track_features"].numpy()).all()
    assert np.isfinite(sample["jet_features"].numpy()).all()


def test_process_jet_no_norm():

    ds = GN2Dataset.__new__(GN2Dataset)
    ds.stats = None

    pt  = np.array([10., 15., 20.], dtype=np.float32)
    eta = np.array([2., 3., 4.],  dtype=np.float32)

    out = GN2Dataset._process_jet(ds, pt, eta)

    assert out.shape == (3, 2)
    assert np.isfinite(out).all()


def test_process_tracks_padding_mask():

    ds = GN2Dataset.__new__(GN2Dataset)
    ds.max_tracks    = 5
    ds.track_vars  = ["x", "y"]
    ds.jet_vars    = ["pt", "eta"],
    ds.jet_flavour = "label"
    ds.stats       = None

    tracks = np.array([
        (1., 2., 1),
        (2., 3., 1),
        (3., 4., 0),
    ], dtype=[("x","f4"),("y","f4"),("valid","i1")])

    track_features, mask = GN2Dataset._process_tracks(ds, tracks)

    assert track_features.shape == (5, 2)
    assert mask.shape == (5,)
    assert mask.sum() == 2


def test_label_mapping():

    ds = GN2Dataset.__new__(GN2Dataset)
    ds.jet_flavour_map = {1: 0, 2: 1}

    assert ds.jet_flavour_map.get(1) == 0
    assert ds.jet_flavour_map.get(999, -1) == -1    # test unknown key


def test_batch_collator_shapes(fake_hdf5_file):

    ds = GN2Dataset(
        h5_file_path = fake_hdf5_file,
        jet_indices  = np.arange(10),
        max_tracks   = 5,
        jet_vars     = ["pt", "eta"],
        track_vars   = ["x", "y"],
        jet_flavour  = "label",
    )

    collator = _BatchCollator(ds)
    batch = collator([0, 1, 2])

    assert batch["jet_features"].shape == (3, 2)
    assert batch["track_features"].shape == (3, 5, 2)
    assert batch["mask"].shape == (3, 5)
    assert batch["label"].shape == (3,)


def test_batch_collator_finite(fake_hdf5_file):
    
    ds = GN2Dataset(
        h5_file_path = fake_hdf5_file,
        jet_indices  = np.arange(10),
        max_tracks   = 5,
        jet_vars     = ["pt", "eta"],
        track_vars   = ["x", "y"],
        jet_flavour  = "label",
    )

    collator = _BatchCollator(ds)
    batch = collator([0, 1, 2])

    assert np.isfinite(batch["jet_features"].numpy()).all()
    assert np.isfinite(batch["track_features"].numpy()).all()