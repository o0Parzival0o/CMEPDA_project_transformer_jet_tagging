"""
conftest.py
"""

import os
import sys

import h5py
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def fake_hdf5_file(tmp_path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        # Create jets dataset (HDF5)
        jet_dtype = np.dtype([
            ("pt", "f4"),               # f4 = float32
            ("eta", "f4"),
            ("label", "i1")             # i1 = int8
        ])
        f.create_dataset("jets", data=np.zeros(10, dtype=jet_dtype))            # 10 jets
        
        track_dtype = np.dtype([
            ("x", "f4"),
            ("y", "f4"),
            ("valid", "i1")
        ])
        f.create_dataset("tracks", data=np.zeros((10, 5), dtype=track_dtype))   # 10 jets con 5 particles per jet
    return str(path)