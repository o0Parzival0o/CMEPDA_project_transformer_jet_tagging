import numpy as np
import h5py
from pathlib import Path
import pytest

from src.transformer_jet_tagging.plotting import (
    _load_jet_data,
    _load_track_data,
    plot_jet_variables,
    plot_track_variables,
    plot_correlations,
    plot_learning_curves,
)


def fill_fake_data(file_path):

    jet_dtype   = np.dtype([
        ("pt", "f4"),
        ("eta", "f4"),
        ("label", "i1")
    ])
    track_dtype = np.dtype([
        ("x", "f4"),
        ("y", "f4"),
        ("valid", "i1")
    ])

    jets_data   = np.zeros(10, dtype=jet_dtype)
    tracks_data = np.zeros((10, 5), dtype=track_dtype)

    jets_data["pt"]    = np.linspace(10, 100, 10)
    jets_data["eta"]   = np.linspace(-2.5, 2.5, 10)
    jets_data["label"] = np.arange(10) % 3

    n_valid = 0
    for i in range(10):
        for j in range(5):
            tracks_data["x"][i, j] = i * 0.1 + j
            tracks_data["y"][i, j] = i * 0.2 + j
            tracks_data["valid"][i, j] = 1 if j < 3 else 0
            if j < 3:
                n_valid += 1

    # sovrascrive i dataset interni con i dati corretti
    with h5py.File(file_path, "r+") as f:
        f["jets"][:]   = jets_data
        f["tracks"][:] = tracks_data

    return n_valid


def test_load_jet_data(fake_hdf5_file):

    _ = fill_fake_data(fake_hdf5_file)

    data = _load_jet_data(
        h5_path=fake_hdf5_file,
        jet_indices=np.arange(10),
        jet_vars=["pt", "eta"],
        jet_flavour="label",
        jet_flavour_map={0: 0, 1: 1, 2: 2},
    )

    assert tuple(data.keys()) == ("pt", "eta", "label")
    assert data["pt"].shape == (10,)
    assert data["pt"].shape == data["label"].shape
    assert data["label"].dtype == np.int32


def test_load_track_data(fake_hdf5_file):

    n_valid = fill_fake_data(fake_hdf5_file)

    data = _load_track_data(
        h5_path=fake_hdf5_file,
        jet_indices=np.arange(10),
        track_vars=["x", "y"],
        jet_flavour="label",
        jet_flavour_map={0: 0, 1: 1, 2: 2},
    )

    assert tuple(data.keys()) == ("x", "y", "label")
    assert data["x"].shape == (n_valid,)
    assert data["x"].shape == data["label"].shape
    assert data["label"].dtype == np.int32


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # error due to empy bin
def test_plot_jet_variables(tmp_path):

    jet_data = {
        "pt": np.random.rand(100),
        "eta": np.random.rand(100),
        "label": np.random.randint(0, 3, 100),
    }

    plot_jet_variables(jet_data, ["pt", "eta"], tmp_path)

    assert (tmp_path / "jet_variables.pdf").exists()

@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # error due to empy bin
def test_plot_track_variables(tmp_path):

    track_data = {
        "x": np.random.rand(200),
        "y": np.random.rand(200),
        "label": np.random.randint(0, 3, 200),
    }

    plot_track_variables(track_data, ["x", "y"], tmp_path)

    assert len(list(tmp_path.glob("track_variables_page*.pdf"))) > 0


def test_plot_correlations(tmp_path):

    jet_data = {
        "pt": np.random.rand(100),
        "eta": np.random.rand(100),
    }

    track_data = {
        "x": np.random.rand(200),
        "y": np.random.rand(200),
    }

    plot_correlations(
        jet_data,
        track_data,
        jet_vars=["pt", "eta"],
        track_vars=["x", "y"],
        output_dir=tmp_path,
    )

    assert (tmp_path / "correlation_jet.pdf").exists()
    assert (tmp_path / "correlation_track.pdf").exists()


def test_empty_tracks(fake_hdf5_file):

    # all zero / valid=0
    data = _load_track_data(
        h5_path=fake_hdf5_file,
        jet_indices=np.arange(10),
        track_vars=["x"],
        jet_flavour="label",
        jet_flavour_map={0: 0},
    )

    assert "x" in data
    assert "label" in data


def test_empty_jets(fake_hdf5_file):

    # all zero / valid=0
    data = _load_jet_data(
        h5_path=fake_hdf5_file,
        jet_indices=np.arange(10),
        jet_vars=["pt"],
        jet_flavour="label",
        jet_flavour_map={0: 0},
    )

    assert "pt" in data
    assert "label" in data


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # error due to empy bin
def test_nan_handling_in_jet_plot(tmp_path):

    jet_data = {
        "pt": np.array([1.0, np.nan, 3.0, 4.0]),
        "eta": np.array([0.1, 0.2, np.nan, 0.4]),
        "label": np.array([0, 1, 2, 3]),
    }

    plot_jet_variables(jet_data, ["pt", "eta"], tmp_path)

    assert (tmp_path / "jet_variables.pdf").exists()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # error due to empy bin
def test_nan_handling_in_track_plot(tmp_path):

    track_data = {
        "x": np.array([1.0, np.nan, 3.0, 4.0]),
        "y": np.array([0.1, 0.2, np.nan, 0.4]),
        "label": np.array([0, 1, 2, 3]),
    }

    plot_track_variables(track_data, ["x", "y"], tmp_path)

    assert (tmp_path / "track_variables_page1.pdf").exists()


def test_nan_handling_in_correlations(tmp_path):

    jet_data = {
        "pt": np.array([1.0, np.nan, 3.0]),
        "eta": np.array([0.1, 0.2, np.nan]),
    }

    track_data = {
        "x": np.random.rand(10),
        "y": np.random.rand(10),
    }

    plot_correlations(
        jet_data,
        track_data,
        jet_vars=["pt", "eta"],
        track_vars=["x", "y"],
        output_dir=tmp_path,
    )


def test_plot_learning_curves(tmp_path):

    history = {
        "train_loss": [1.0, 0.8, 0.6, 0.4],
        "val_loss":   [1.1, 0.9, 0.7, 0.5],
        "lr":         [1e-3, 1e-3, 1e-4, 1e-4],
    }

    plot_learning_curves(history, tmp_path)

    assert (tmp_path / "learning_curves.pdf").exists()


def test_plot_learning_curves_single_epoch(tmp_path):
    
    history = {"train_loss": [1.0], "val_loss": [1.1], "lr": [1e-3]}

    plot_learning_curves(history, tmp_path)

    assert (tmp_path / "learning_curves.pdf").exists()
