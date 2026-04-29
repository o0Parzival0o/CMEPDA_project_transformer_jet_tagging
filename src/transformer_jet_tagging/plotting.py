"""
plotting.py
===========
Visualization module for the GN2 jet flavour tagging pipeline.

Generates plots of input and output variables,
reading directly from the HDF5 file and/or a DataLoader.

Plots produced:
  1. Jet variables      - pt (raw and log), eta per flavour class.
  2. Track variables    - per-variable distribution split by flavour.
  3. Correlations       - jet-level Pearson correlation matrix;
                          track-level correlation matrix (mean over jets).
"""

import logging
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import FLAVOUR_COLORS, FLAVOUR_LABELS
from .model import GN2

matplotlib.use("Agg")               # non-interactive backend (no display needed)
hep.style.use(hep.style.ATLAS)
logger = logging.getLogger("GN2.plotting")


def _load_jet_data(
    h5_path: str,
    jet_indices: np.ndarray,
    jet_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
) -> dict[str, np.ndarray]:
    """
    Load jet-level variables and labels from HDF5 for a subset of jets.

    Args:
        h5_path (str): Path to HDF5 file.
        jet_indices (np.ndarray): Sorted jet indices to read.
        jet_vars (list): Jet variable names.
        jet_flavour (str): Name of the flavour field in HDF5.
        jet_flavour_map (dict): Raw label - class index mapping.

    Returns:
        dict:
            var_name (np.ndarray): shape (n_jets,) for each jet variable.
            "label" (np.ndarray): shape (n_jets,) integer class index for each jet.

    Raises:
        FileNotFoundError: if the specified file does not exist.
        KeyError: if expected datasets or fields are missing in the HDF5 file.
    """
    sorted_idx = np.sort(jet_indices)
    data: dict[str, np.ndarray] = {}

    try:
        with h5py.File(h5_path, "r") as f:
            jets = f["jets"][sorted_idx]
            for var in jet_vars:
                data[var] = jets[var].astype(np.float32)
            raw_labels = jets[jet_flavour].astype(int)
            data["label"] = np.array(
                [jet_flavour_map.get(label, 0) for label in raw_labels], dtype=np.int32
            )
    except FileNotFoundError:
        logger.error("HDF5 file not found: %s", h5_path)
        raise
    except KeyError as e:
        logger.error("Missing dataset in HDF5 file: %s", e)
        raise

    return data


def _load_track_data(
    h5_path: str,
    jet_indices: np.ndarray,
    track_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
    max_jets: int = 50_000,
) -> dict[str, np.ndarray]:
    """
    Load valid track-level variables from HDF5, flattened across jets.

    Args:
        h5_path (str): Path to HDF5 file.
        jet_indices (np.ndarray): Sorted jet indices.
        track_vars (list): Track variable names.
        jet_flavour (str): Flavour field name.
        jet_flavour_map (dict): Raw label - class index.
        max_jets (int): Cap on jets to read (memory guard).

    Returns:
        dict:
            var_name (np.ndarray): shape (n_tracks,) for each track variable.
            "label" (np.ndarray): shape (n_tracks,) integer class index for each track's jet.
        
    Raises:
        FileNotFoundError: if the specified file does not exist.
        KeyError: if expected datasets or fields are missing in the HDF5 file.
    """
    sorted_idx = np.sort(jet_indices[:max_jets])
    data_lists: dict[str, list] = {v: [] for v in track_vars}
    data_lists["label"] = []

    try:
        with h5py.File(h5_path, "r") as f:
            jets_raw   = f["jets"][sorted_idx]
            tracks_raw = f["tracks"][sorted_idx]
            raw_labels = jets_raw[jet_flavour].astype(int)
            labels     = np.array(
                [jet_flavour_map.get(label, 0) for label in raw_labels], dtype=np.int32
            )

            has_valid = "valid" in tracks_raw.dtype.names
            for i in range(len(sorted_idx)):
                if has_valid:
                    valid_mask = tracks_raw["valid"][i].astype(bool)
                else:
                    valid_mask = np.ones(tracks_raw.shape[1], dtype=bool)

                n = valid_mask.sum()
                if n == 0:
                    continue
                for var in track_vars:
                    data_lists[var].append(tracks_raw[var][i][valid_mask].astype(np.float32))
                data_lists["label"].append(np.full(n, labels[i], dtype=np.int32))
    except FileNotFoundError:
        logger.error("HDF5 file not found: %s", h5_path)
        raise
    except KeyError as e:
        logger.error("Missing dataset in HDF5 file: %s", e)
        raise

    return {
        k: np.concatenate(v) if len(v) > 0 else np.array(
            [], dtype=np.int32 if k == "label" else np.float32
        )for k, v in data_lists.items()
    }


def plot_jet_variables(
    jet_data: dict[str, np.ndarray],
    jet_vars: list[str],
    output_dir: Path,
) -> None:
    """
    Plot per-flavour distributions of jet-level variables.
    pt is shown both raw (linear) and log-transformed.

    Args:
        jet_data (dict): Output of _load_jet_data().
        jet_vars (list): Variable names to plot.
        output_dir (Path): Directory where PNGs are saved.
    """
    labels  = jet_data["label"]
    classes = sorted(FLAVOUR_LABELS.keys())

    # build plot list: for pt add a log version
    plot_list = []
    for var in jet_vars:
        plot_list.append((var, False))
        if var == "pt":
            plot_list.append(("pt", True))

    n_cols = 2
    n_rows = (len(plot_list) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (var, do_log) in zip(axes, plot_list, strict=False):
        values = jet_data[var].copy()
        labels_var = labels.copy()

        valid = np.isfinite(values)
        values     = values[valid]
        labels_var = labels_var[valid]

        if do_log:
            values = np.log(np.clip(values, 1e-6, None))
            finite = np.isfinite(values)
            values     = values[finite]
            labels_var = labels_var[finite]

        if values.size == 0:
            continue

        q_lo, q_hi = np.min(values), np.max(values)
        bins = np.linspace(q_lo, q_hi, 60)

        for cls in classes:
            mask = labels_var == cls
            ax.hist(
                values[mask],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=FLAVOUR_COLORS[cls],
                label=FLAVOUR_LABELS[cls],
            )

        ax.set_xlabel(var,loc='center',fontsize=14)
        ax.set_ylabel("Entries")
        ax.set_yscale("log")
        ax.legend()
        hep.atlas.label(
                ax=ax,
                data=True,
                loc=1,
                rlabel=r'$\sqrt{s}=$'+str(13.6)+r' TeV, $t\bar{t}$ simulation'
               )

    # hide unused axes
    for ax in axes[len(plot_list):]:
        ax.set_visible(False)

    fig.tight_layout()
    out = output_dir / "jet_variables.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_track_variables(
    track_data: dict[str, np.ndarray],
    track_vars: list[str],
    output_dir: Path,
    vars_per_page: int = 6,
) -> None:
    """
    Plot per-flavour distributions of track-level variables.
    Variables are split across multiple pages if needed.

    Args:
        track_data (dict): Output of _load_track_data().
        track_vars (list): Variable names to plot.
        output_dir (Path): Directory where PNGs are saved.
        vars_per_page (int): Max variables per figure (default 6).
    """
    classes = sorted(FLAVOUR_LABELS.keys())

    for page, start in enumerate(range(0, len(track_vars), vars_per_page)):
        page_vars = track_vars[start : start + vars_per_page]
        n_cols = 3
        n_rows = (len(page_vars) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, var in zip(axes, page_vars, strict=False):
            values = track_data[var]
            labels_var = track_data["label"].copy()

            valid_mask = np.isfinite(values)
            values     = values[valid_mask]
            labels_var = labels_var[valid_mask]

            if values.size == 0:
                continue
            q_lo, q_hi = np.min(values), np.max(values)
            bins = np.linspace(q_lo, q_hi, 60)

            for cls in classes:
                mask = labels_var == cls
                ax.hist(
                    values[mask],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=FLAVOUR_COLORS[cls],
                    label=FLAVOUR_LABELS[cls],
                )

            ax.set_xlabel(var,loc='center',fontsize=14)
            ax.set_ylabel("Entries")
            ax.set_yscale("log")
            ax.legend()
            hep.atlas.label(
                ax=ax,
                data=True,
                loc=1,
                rlabel=r'$\sqrt{s}=$'+str(13.6)+r' TeV, $t\bar{t}$ simulation'
               )

        for ax in axes[len(page_vars):]:
            ax.set_visible(False)

        fig.tight_layout()
        out = output_dir / f"track_variables_page{page + 1}.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved: %s", out)


def _corr_matrix(data_dict, vars_list):
    """
    Compute correlation matrix for the specified variables.
    (Non-finite values are replaced with the column mean before correlation)

    Args:
        data_dict (dict): dict of variable name to np.ndarray.
        vars_list (list): list of variable names to include in the matrix.
    
    Returns:
        np.ndarray: shape (len(vars_list), len(vars_list)), correlation matrix.
    """
    mat = np.column_stack([data_dict[v].astype(np.float32) for v in vars_list])
    # replace inf/nan with column mean
    col_means = np.nanmean(mat, axis=0)
    inds = np.where(~np.isfinite(mat))
    mat[inds] = col_means[inds[1]]
    return np.corrcoef(mat, rowvar=False)


def _draw_heatmap(ax, corr, labels, title):
    """
    Draw a heatmap of the correlation matrix with annotations.

    Args:
        ax: matplotlib axis to draw on.
        corr: 2D array of correlation coefficients.
        labels: list of variable names for axes.
        title: title of the plot.
    
    Returns:
        im: image object from imshow (for colorbar).
    """
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11)
    # annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    return im


def plot_correlations(
    jet_data: dict[str, np.ndarray],
    track_data: dict[str, np.ndarray],
    jet_vars: list[str],
    track_vars: list[str],
    output_dir: Path,
) -> None:
    """
    Plot Pearson correlation matrices for jet and track variables.

    Args:
        jet_data (dict): Output of _load_jet_data().
        track_data (dict): Output of _load_track_data().
        jet_vars (list): Jet variable names.
        track_vars (list): Track variable names.
        output_dir (Path): Directory where PNGs are saved.
    """

    # jet correlation
    if len(jet_vars) >= 2:
        # add log_pt as an extra column
        jet_data_ext = dict(jet_data)
        jet_data_ext["log(pt)"] = np.log(np.clip(jet_data["pt"], 1e-6, None))
        jet_vars_ext = ["log(pt)"] + [v for v in jet_vars if v != "pt"]

        corr_jet = _corr_matrix(jet_data_ext, jet_vars_ext)
        fig, ax = plt.subplots(figsize=(max(5, len(jet_vars_ext)), max(4, len(jet_vars_ext))))
        im = _draw_heatmap(ax, corr_jet, jet_vars_ext, "Jet variables - Correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = output_dir / "correlation_jet.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved: %s", out)

    # track correlation
    if len(track_vars) >= 2:
        corr_track = _corr_matrix(track_data, track_vars)
        n = len(track_vars)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.55)))
        im = _draw_heatmap(ax, corr_track, track_vars, "Track variables - Correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = output_dir / "correlation_track.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved: %s", out)


def plot_statistics(
    h5_path: str,
    jet_vars: list[str],
    track_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
    jet_indices: np.ndarray,
    output_dir: str = "outputs/plots",
    n_jets_track: int = 50_000,
) -> None:
    """
    Generate all plots and save them to output_dir.

    Args:
        h5_path (str): Path to HDF5 file.
        jet_vars (list): Jet variable names.
        track_vars (list): Track variable names.
        jet_flavour (str): Flavour field name in HDF5.
        jet_flavour_map (dict): Raw label - class index.
        jet_indices (np.ndarray): Jet indices to use (e.g. train_indices).
        output_dir (str): Directory for output PNGs.
        n_jets_track (int): Max jets for track plots (memory guard).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading jet data for %s jets ...", f"{len(jet_indices):,}")
    jet_data = _load_jet_data(h5_path, jet_indices, jet_vars, jet_flavour, jet_flavour_map)

    logger.info("Loading track data (up to %s jets) ...", f"{n_jets_track:,}")
    track_data = _load_track_data(h5_path, jet_indices, track_vars, jet_flavour,
                                  jet_flavour_map, max_jets = n_jets_track)

    logger.info("Plotting jet variables ...")
    plot_jet_variables(jet_data, jet_vars, out)

    logger.info("Plotting track variables ...")
    plot_track_variables(track_data, track_vars, out)

    logger.info("Plotting correlations ...")
    plot_correlations(jet_data, track_data, jet_vars, track_vars, out)

    logger.info("All plots saved to '%s/'", out)


def plot_learning_curves(
    history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """
    Plot training and validation loss curves + LR schedule.

    Args:
        history (dict): keys "train_loss", "val_loss", "lr".
        output_dir (Path): Directory where the PDF is saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # loss
    ax = axes[0]
    ax.plot(history["train_loss"], c='r', ls='-', label="Train", linewidth=1.8)
    ax.plot(history["val_loss"], c='b', ls='--', label="Validation", linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss (CE)", fontsize=14)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    hep.atlas.label(ax=ax, data=False, loc=2, rlabel=r"GN2 training")

    # lr
    ax = axes[1]
    ax.plot(history["lr"], color="darkorange", linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Learning Rate", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    hep.atlas.label(ax=ax, data=False, loc=2, rlabel=r"GN2 training")

    fig.tight_layout()
    out = output_dir / "learning_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def _roc_rejection(scores, labels, signal_class, bg_class, n_points=200):
    """
    Calculate signal efficiency and background rejection for ROC curve.

    Args:
        scores (np.ndarray): Discriminant scores for all jets.
        labels (np.ndarray): True class labels for all jets.
        signal_class (int): Class index of the signal (e.g. b-jets).
        bg_class (int): Class index of the background (e.g. c-jets).
        n_points (int): Number of points on the ROC curve.

    Returns:
        eff (np.ndarray): Signal efficiency values.
        rej (np.ndarray): Background rejection values (1 / bg efficiency).
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    eff, rej = [], []
    for thr in thresholds:
        tagged_sig = ((scores >= thr) & (labels == signal_class)).sum()
        total_sig  = (labels == signal_class).sum()
        tagged_bg  = ((scores >= thr) & (labels == bg_class)).sum()
        total_bg   = (labels == bg_class).sum()
        sig_eff = tagged_sig / total_sig if total_sig > 0 else 0
        bg_eff  = tagged_bg  / total_bg  if total_bg  > 0 else 0
        eff.append(sig_eff)
        rej.append(1.0 / bg_eff if bg_eff > 0 else np.nan)
    return np.array(eff), np.array(rej)


def plot_roc_db(
    model: GN2,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    fc: float = 0.2,
    ftau: float = 0.05,
) -> None:
    """
    ROC curve for b-tagging.

    Args:
        model (GN2): trained GN2 model instance.
        loader (DataLoader): DataLoader (val or test set).
        device (torch.device): torch device.
        output_dir (Path): directory where the PDF is saved.
        fc (float): c-fraction. (default 0.2)
        ftau (float): tau-fraction. (default 0.05)
    """

    all_db, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            jet_feats   = batch["jet_features"].to(device)
            track_feats = batch["track_features"].to(device)
            mask        = batch["mask"].to(device)
            db = model.discriminant_db(jet_feats, track_feats, mask, fc=fc, ftau=ftau)
            all_db.append(db.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    db     = np.concatenate(all_db)
    labels = np.concatenate(all_labels)

    b_class, c_class, light_class, tau_class = 0, 1, 2, 3

    fig, ax = plt.subplots(figsize=(7, 6))

    for bg_class, linestyle, label in [
        (c_class,     "-",   "c-jet rejection"),
        (light_class, "-.",  "light-jet rejection"),
        (tau_class,   "--",  r"$\tau$-jet rejection"),
    ]:
        eff, rej = _roc_rejection(db, labels, b_class, bg_class)
        ax.plot(eff, rej, linestyle=linestyle, linewidth=1.8, label=label)

    ax.set_xlabel("b-jet tagging efficiency", fontsize=14)
    ax.set_ylabel("Background rejection", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.0)
    ax.legend()
    hep.atlas.label(
        ax=ax,
        data=False,
        loc=2,
        rlabel=r"$\sqrt{s}=13.6$ TeV, $t\bar{t}$ simulation" \
               r"$20<p_T<250$ GeV, $\left|\eta\right|<2.5$",
    )

    fig.tight_layout()
    out = output_dir / "roc_db.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_roc_dc(
    model: GN2,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    fb: float = 0.3,
    ftau: float = 0.01,
) -> None:
    """
    ROC curve for c-tagging.

    Args:
        model (GN2): trained GN2 model instance.
        loader (DataLoader): DataLoader (val or test set).
        device (torch.device): torch device.
        output_dir (Path): directory where the PDF is saved.
        fb (float): b-fraction. (default 0.3)
        ftau (float): tau-fraction. (default 0.01)
    """

    all_dc, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            jet_feats   = batch["jet_features"].to(device)
            track_feats = batch["track_features"].to(device)
            mask        = batch["mask"].to(device)
            dc = model.discriminant_dc(jet_feats, track_feats, mask, fb=fb, ftau=ftau)
            all_dc.append(dc.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    dc     = np.concatenate(all_dc)
    labels = np.concatenate(all_labels)

    b_class, c_class, light_class, tau_class = 0, 1, 2, 3

    fig, ax = plt.subplots(figsize=(7, 6))

    for bg_class, linestyle, label in [
        (b_class,     "-",   "b-jet rejection"),
        (light_class, "-.",  "light-jet rejection"),
        (tau_class,   "--",  r"$\tau$-jet rejection"),
    ]:
        eff, rej = _roc_rejection(dc, labels, c_class, bg_class)
        ax.plot(eff, rej, linestyle=linestyle, linewidth=1.8, label=label)

    ax.set_xlabel("c-jet tagging efficiency", fontsize=14)
    ax.set_ylabel("Background rejection", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.0)
    ax.legend()
    hep.atlas.label(
        ax=ax,
        data=False,
        loc=2,
        rlabel=r"$\sqrt{s}=13.6$ TeV, $t\bar{t}$ simulation" \
               r"$20<p_T<250$ GeV, $\left|\eta\right|<2.5$",
    )

    fig.tight_layout()
    out = output_dir / "roc_dc.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


if __name__ == "__main__":
    import argparse
    import sys

    from . import utils

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="GN2 input/output variable plots")
    parser.add_argument("--config",       type=str, default="configs/config.json")
    parser.add_argument("--output-dir",   type=str, default="outputs/plots")
    parser.add_argument("--n-jets",       type=int, default=200_000,
                        help="Max jets for jet plots (default: 200000)")
    parser.add_argument("--n-jets-track", type=int, default=50_000,
                        help="Max jets for track plots (default: 50000)")
    args = parser.parse_args()

    config = utils.load_config_json(args.config)

    file_path      = config["data"]["h5_path"]
    jet_features   = config["data"]["jet_features"]
    track_features = config["data"]["track_features"]
    flavour_field  = config["data"]["label"]
    flavour_map    = {int(k): v for k, v in config["data"]["label_map"].items()}
    preprocess_dir = Path(config["output"]["preprocess_dir"])

    idx_path = preprocess_dir / "indices" / "train_indices.npy"
    if not idx_path.exists():
        logger.error("Index file not found: %s. Run preprocess.py first.", idx_path)
        sys.exit(1)

    indices = np.sort(np.load(idx_path))
    if args.n_jets < len(indices):
        rng     = np.random.default_rng(seed=0)
        indices = np.sort(rng.choice(indices, size=args.n_jets, replace=False))
    logger.info("Using %s jets for plots.", f"{len(indices):,}")

    plot_statistics(
        h5_path         = file_path,
        jet_vars        = jet_features,
        track_vars      = track_features,
        jet_flavour     = flavour_field,
        jet_flavour_map = flavour_map,
        jet_indices     = indices,
        output_dir      = args.output_dir,
        n_jets_track    = args.n_jets_track,
    )
