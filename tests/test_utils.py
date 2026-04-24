"""
test_utils.py
"""


import h5py
import numpy as np
from src.transformer_jet_tagging.utils import compute_normalization_stats, load_config_json


def test_compute_norm_stats_minimal(fake_hdf5_file):

    with h5py.File(fake_hdf5_file, "a") as f:
        f["jets"]["pt"][:]    = np.arange(1, 11)
        f["jets"]["eta"][:]   = np.ones(10)

        f["tracks"]["x"][:] = np.random.randn(10, 5)
        f["tracks"]["y"][:] = np.random.randn(10, 5)
        f["tracks"]["valid"][:] = np.ones((10, 5))

    stats = compute_normalization_stats(
        file_path     = fake_hdf5_file,
        train_indices = np.arange(10),
        jet_vars      = ["pt", "eta"],
        track_vars    = ["x", "y"]
    )

    assert stats["jet_mu"].shape == (2,)
    assert stats["jet_sigma"].shape == (2,)
    assert stats["track_mu"].shape == (2,)
    assert stats["track_sigma"].shape == (2,)


def test_load_config(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"a": 1}')

    config = load_config_json(str(path))
    assert config["a"] == 1