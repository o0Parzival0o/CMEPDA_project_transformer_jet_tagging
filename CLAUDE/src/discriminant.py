"""
discriminant.py
===============
Calcolo dei discriminanti D_b e D_c (equazioni 1 e 2 dell'articolo).

    D_b = log[ pb / (fc·pc + fτ·pτ + (1 - fc - fτ)·pu) ]   (eq. 1)
    D_c = log[ pc / (fb·pb + fτ·pτ + (1 - fb - fτ)·pu) ]   (eq. 2)

Parametri GN2:
    D_b: fc = 0.2, fτ = 0.05
    D_c: fb = 0.3, fτ = 0.01

Un jet è b-tagged se D_b > soglia; la soglia definisce l'Operating Point (OP),
caratterizzato dall'efficienza inclusiva di b-tagging su un campione tt̄.
"""

import numpy as np
import torch
from typing import Union, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

_EPS = 1e-10

CLASS_NAMES = ["b-jet", "c-jet", "light-jet", "tau-jet"]
CLASS_IDX   = {"b": 0, "c": 1, "u": 2, "tau": 3}


# ──────────────────────────────────────────────────────────────────────────────
# Discriminanti
# ──────────────────────────────────────────────────────────────────────────────

def compute_db(
    pb:   Union[np.ndarray, torch.Tensor],
    pc:   Union[np.ndarray, torch.Tensor],
    pu:   Union[np.ndarray, torch.Tensor],
    ptau: Union[np.ndarray, torch.Tensor],
    fc:   float = 0.2,
    ftau: float = 0.05,
) -> Union[np.ndarray, torch.Tensor]:
    """
    D_b = log[ pb / (fc·pc + fτ·pτ + (1-fc-fτ)·pu) ]
    """
    if isinstance(pb, torch.Tensor):
        denom = fc * pc + ftau * ptau + (1.0 - fc - ftau) * pu
        return torch.log(torch.clamp(pb, min=_EPS) / torch.clamp(denom, min=_EPS))
    else:
        pb, pc, pu, ptau = [np.asarray(x, dtype=np.float64)
                            for x in (pb, pc, pu, ptau)]
        denom = fc * pc + ftau * ptau + (1.0 - fc - ftau) * pu
        return np.log(np.maximum(pb, _EPS) / np.maximum(denom, _EPS))


def compute_dc(
    pb:   Union[np.ndarray, torch.Tensor],
    pc:   Union[np.ndarray, torch.Tensor],
    pu:   Union[np.ndarray, torch.Tensor],
    ptau: Union[np.ndarray, torch.Tensor],
    fb:   float = 0.3,
    ftau: float = 0.01,
) -> Union[np.ndarray, torch.Tensor]:
    """
    D_c = log[ pc / (fb·pb + fτ·pτ + (1-fb-fτ)·pu) ]
    """
    if isinstance(pb, torch.Tensor):
        denom = fb * pb + ftau * ptau + (1.0 - fb - ftau) * pu
        return torch.log(torch.clamp(pc, min=_EPS) / torch.clamp(denom, min=_EPS))
    else:
        pb, pc, pu, ptau = [np.asarray(x, dtype=np.float64)
                            for x in (pb, pc, pu, ptau)]
        denom = fb * pb + ftau * ptau + (1.0 - fb - ftau) * pu
        return np.log(np.maximum(pc, _EPS) / np.maximum(denom, _EPS))


# ──────────────────────────────────────────────────────────────────────────────
# Operating Points
# ──────────────────────────────────────────────────────────────────────────────

def compute_operating_points(
    db_scores:      np.ndarray,
    true_labels:    np.ndarray,
    b_efficiencies: List[float] = [0.65, 0.70, 0.77, 0.85, 0.90],
) -> Dict[float, float]:
    """Soglie D_b che corrispondono alle efficienze di b-tagging target."""
    b_mask = (true_labels == CLASS_IDX["b"])
    db_b   = db_scores[b_mask]
    return {
        eff: float(np.percentile(db_b, (1.0 - eff) * 100.0))
        for eff in b_efficiencies
    }


def compute_rejection(
    db_scores:   np.ndarray,
    true_labels: np.ndarray,
    threshold:   float,
    bg_class:    int,
) -> float:
    """Rejection = 1 / mis-tagging rate per una data soglia."""
    bg_db = db_scores[true_labels == bg_class]
    if len(bg_db) == 0:
        return float("nan")
    n_tagged = (bg_db > threshold).sum()
    return float("inf") if n_tagged == 0 else float(len(bg_db) / n_tagged)


def evaluate_operating_points(
    db_scores:      np.ndarray,
    true_labels:    np.ndarray,
    b_efficiencies: List[float] = [0.65, 0.70, 0.77, 0.85, 0.90],
) -> List[Dict]:
    """
    Valuta c-rejection, light-rejection e tau-rejection
    a tutti gli operating point standard ATLAS.
    """
    thresholds = compute_operating_points(db_scores, true_labels, b_efficiencies)
    results = []
    logger.info("Operating Points:")
    logger.info(f"  {'OP':>6} | {'threshold':>10} | "
                f"{'c-rej':>8} | {'light-rej':>10} | {'tau-rej':>8}")
    logger.info("  " + "-" * 55)
    for eff, thr in thresholds.items():
        row = {
            "efficiency":   eff,
            "threshold":    thr,
            "c_rejection":   compute_rejection(db_scores, true_labels, thr, CLASS_IDX["c"]),
            "u_rejection":   compute_rejection(db_scores, true_labels, thr, CLASS_IDX["u"]),
            "tau_rejection": compute_rejection(db_scores, true_labels, thr, CLASS_IDX["tau"]),
        }
        results.append(row)
        logger.info(
            f"  {eff*100:.0f}%  | {thr:>10.3f} | "
            f"{row['c_rejection']:>8.1f} | "
            f"{row['u_rejection']:>10.1f} | "
            f"{row['tau_rejection']:>8.1f}"
        )
    return results