"""
test_model.py
"""

import pytest
import torch
from src.transformer_jet_tagging.model import GN2, _get_activation


@pytest.fixture
def dummy_input():
    batch_size, n_tracks = 2, 5
    return {
        "jet_features": torch.randn(batch_size, 3),
        "track_features": torch.randn(batch_size, n_tracks, 4),
        "mask": torch.ones(batch_size, n_tracks, dtype=torch.bool),
    }


def test_activation_factory():

    assert isinstance(_get_activation("relu"), torch.nn.ReLU)
    with pytest.raises(ValueError):
        _get_activation("not_existing")


def test_model_forward_shape(dummy_input):

    model = GN2(
        n_jet_vars      = 3,
        n_track_vars    = 4,
        n_classes       = 4,
        init_hidden_dim = 64,
        init_output_dim = 64,
        embed_dim       = 32,
        n_heads         = 4,
        n_layers        = 2,
        ff_dim          = 64,
        pool_dim        = 16,
    )

    out = model(**dummy_input)
    assert out["jet_outputs"].shape == (2, 4)


def test_predict_proba(dummy_input):

    model = GN2(
        n_jet_vars   = 3,
        n_track_vars = 4
    )

    p = model.predict_proba(**dummy_input)
    assert p.shape == (2, 4)
    assert torch.allclose(p.sum(dim=1), torch.ones(2), atol=1e-5)   # test softmax


def test_discriminant_db_shape(dummy_input):

    model = GN2(
        n_jet_vars   = 3,
        n_track_vars = 4
    )

    d = model.discriminant_db(**dummy_input,
                              label_map={"b-jet":0, "c-jet":1, "light-jet":2, "tau-jet":3})

    assert d.shape == (2,)
    assert isinstance(d, torch.Tensor)


def test_discriminant_dc_shape(dummy_input):

    model = GN2(
        n_jet_vars   = 3,
        n_track_vars = 4
    )

    d = model.discriminant_dc(**dummy_input,
                              label_map={"b-jet":0, "c-jet":1, "light-jet":2, "tau-jet":3})

    assert d.shape == (2,)
    assert isinstance(d, torch.Tensor)


def test_discriminant_db_finite(dummy_input):

    model = GN2(
        n_jet_vars   = 3,
        n_track_vars = 4
    )

    d = model.discriminant_db(**dummy_input,
                              label_map={"b-jet":0, "c-jet":1, "light-jet":2, "tau-jet":3})

    assert torch.isfinite(d).all()


def test_discriminant_dc_finite(dummy_input):

    model = GN2(
        n_jet_vars   = 3,
        n_track_vars = 4
    )

    d = model.discriminant_dc(**dummy_input,
                              label_map={"b-jet":0, "c-jet":1, "light-jet":2, "tau-jet":3})

    assert torch.isfinite(d).all()


def test_discriminant_db_invalid_label_map(dummy_input):

    model = GN2(
        n_jet_vars   = 3,
        n_track_vars = 4
    )

    bad_map = {"b-jet": 0}

    with pytest.raises(ValueError):
        model.discriminant_db(**dummy_input, label_map=bad_map)