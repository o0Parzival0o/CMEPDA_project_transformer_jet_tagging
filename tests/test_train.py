"""
test_train.py
"""

import torch
from src.transformer_jet_tagging.model import GN2
from src.transformer_jet_tagging.train import GN2Loss, lr_scheduler, run_epoch
from torch.optim import AdamW


def test_loss_forward():

    loss = GN2Loss()

    outputs = {"jet_outputs": torch.randn(4, 4)}
    labels  = {"jet_label": torch.tensor([0, 1, 2, 3])}

    out = loss(outputs, labels)

    assert "total" in out
    assert out["total"].shape == torch.Size([])


def test_loss_ignore_index():
    loss = GN2Loss()

    outputs = {"jet_outputs": torch.randn(3, 4)}
    labels  = {"jet_label": torch.tensor([0, -1, 2])}

    out = loss(outputs, labels)

    assert torch.isfinite(out["total"])


def test_scheduler_steps():
    
    model = GN2(3, 4)
    opt   = AdamW(model.parameters(), lr=1e-3)

    scheduler = lr_scheduler(opt, n_total_steps=100)

    lr_before = opt.param_groups[0]["lr"]
    scheduler.step()
    lr_after = opt.param_groups[0]["lr"]

    assert lr_after != lr_before


def test_run_epoch_train_step():

    model = GN2(3, 4)
    loss  = GN2Loss()
    opt   = AdamW(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler(opt, n_total_steps=10)

    batch = {
        "jet_features": torch.randn(2, 3),
        "track_features": torch.randn(2, 5, 4),
        "mask": torch.ones(2, 5, dtype=torch.bool),
        "label": torch.tensor([0, 1]),
    }

    loader = [batch]        # fake dataloader

    out = run_epoch(
        model, loader, loss, opt, scheduler,
        device=torch.device("cpu"),
        is_train=True
    )

    assert "total" in out
    assert isinstance(out["total"], float)


def test_run_epoch_eval_step():

    model = GN2(3, 4)
    loss  = GN2Loss()
    opt   = AdamW(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler(opt, n_total_steps=10)

    batch = {
        "jet_features": torch.randn(2, 3),
        "track_features": torch.randn(2, 5, 4),
        "mask": torch.ones(2, 5, dtype=torch.bool),
        "label": torch.tensor([0, 1]),
    }

    loader = [batch]

    out = run_epoch(
        model, loader, loss, opt, scheduler,
        device=torch.device("cpu"),
        is_train=False
    )

    assert "total" in out
    assert isinstance(out["total"], float)