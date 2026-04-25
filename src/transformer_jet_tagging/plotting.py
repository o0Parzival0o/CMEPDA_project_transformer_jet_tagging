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
import numpy as np
import mplhep as hep

from .constants import FLAVOUR_COLORS, FLAVOUR_LABELS

matplotlib.use("Agg")               # non-interactive backend (no display needed)
hep.style.use(hep.style.ATLAS)
logger = logging.getLogger("GN2.plotting")


def _load_jet_data(
    file_path: str,
    indices: np.ndarray,
    jet_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
) -> dict[str, np.ndarray]:
    """
    Load jet-level variables and labels from HDF5 for a subset of jets.

    Args:
        file_path (str): Path to HDF5 file.
        indices (np.ndarray): Sorted jet indices to read.
        jet_vars (list): Jet variable names.
        jet_flavour (str): Name of the flavour field in HDF5.
        jet_flavour_map (dict): Raw label - class index mapping.

    Returns:
        dict:
            var_name (np.ndarray): shape (n_jets,) for each jet variable.
            "label" (np.ndarray): shape (n_jets,) integer class index for each jet.
    """
    sorted_idx = np.sort(indices)
    data: dict[str, np.ndarray] = {}

    with h5py.File(file_path, "r") as f:
        jets = f["jets"][sorted_idx]
        for var in jet_vars:
            data[var] = jets[var].astype(np.float32)
        raw_labels = jets[jet_flavour].astype(int)
        data["label"] = np.array(
            [jet_flavour_map.get(label, 0) for label in raw_labels], dtype=np.int32
        )

    return data


def _load_track_data(
    file_path: str,
    indices: np.ndarray,
    track_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
    max_jets: int = 50_000,
) -> dict[str, np.ndarray]:
    """
    Load valid track-level variables from HDF5, flattened across jets.

    Args:
        file_path (str): Path to HDF5 file.
        indices (np.ndarray): Sorted jet indices.
        track_vars (list): Track variable names.
        jet_flavour (str): Flavour field name.
        jet_flavour_map (dict): Raw label - class index.
        max_jets (int): Cap on jets to read (memory guard).

    Returns:
        dict:
            var_name (np.ndarray): shape (n_tracks,) for each track variable.
            "label" (np.ndarray): shape (n_tracks,) integer class index for each track's jet.
    """
    sorted_idx = np.sort(indices[:max_jets])
    data_lists: dict[str, list] = {v: [] for v in track_vars}
    data_lists["label"] = []

    with h5py.File(file_path, "r") as f:
        jets_raw   = f["jets"][sorted_idx]
        tracks_raw = f["tracks"][sorted_idx]
        raw_labels = jets_raw[jet_flavour].astype(int)
        labels     = np.array(
            [jet_flavour_map.get(label, 0) for label in raw_labels], dtype=np.int32
        )

        has_valid = "valid" in tracks_raw.dtype.names
        for i in range(len(sorted_idx)):
            jet_tracks = tracks_raw[i]
            if has_valid:
                valid_mask = jet_tracks["valid"].astype(bool)
                jet_tracks = jet_tracks[valid_mask]
            n = len(jet_tracks)
            if n == 0:
                continue
            for var in track_vars:
                data_lists[var].append(jet_tracks[var].astype(np.float32))
            data_lists["label"].append(np.full(n, labels[i], dtype=np.int32))

    return {k: np.concatenate(v) for k, v in data_lists.items()}


# ---------------------------------------------------------------------------
# Plot 1 - Jet variables
# ---------------------------------------------------------------------------
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
        if do_log:
            values = np.log(np.clip(values, 1e-6, None))

        # determine range
        q_lo, q_hi = np.min(values), np.max(values)
        bins = np.linspace(q_lo, q_hi, 60)

        for cls in classes:
            mask = labels == cls
            ax.hist(
                values[mask],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=FLAVOUR_COLORS[cls],
                label=FLAVOUR_LABELS[cls],
            )

        ax.set_xlabel(var,loc='center',fontsize=15)
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


# ---------------------------------------------------------------------------
# Plot 2 - Track variables
# ---------------------------------------------------------------------------
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
    labels  = track_data["label"]
    classes = sorted(FLAVOUR_LABELS.keys())

    for page, start in enumerate(range(0, len(track_vars), vars_per_page)):
        page_vars = track_vars[start : start + vars_per_page]
        n_cols = 3
        n_rows = (len(page_vars) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, var in zip(axes, page_vars, strict=False):
            values = track_data[var]
            q_lo, q_hi = np.min(values), np.max(values)
            bins = np.linspace(q_lo, q_hi, 60)

            for cls in classes:
                mask = labels == cls
                ax.hist(
                    values[mask],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=FLAVOUR_COLORS[cls],
                    label=FLAVOUR_LABELS[cls],
                )

            ax.set_xlabel(var,loc='center',fontsize=15)
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


# ---------------------------------------------------------------------------
# Plot 3 - Correlation matrices
# ---------------------------------------------------------------------------
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

    def _corr_matrix(data_dict, vars_list):
        mat = np.column_stack([data_dict[v].astype(np.float32) for v in vars_list])
        # replace inf/nan with column mean
        col_means = np.nanmean(mat, axis=0)
        inds = np.where(~np.isfinite(mat))
        mat[inds] = col_means[inds[1]]
        return np.corrcoef(mat, rowvar=False)

    def _draw_heatmap(ax, corr, labels, title):
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

    # --- jet correlation ---
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

    # --- track correlation ---
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




def make_all_plots(
    file_path: str,
    jet_vars: list[str],
    track_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
    indices: np.ndarray,
    output_dir: str = "outputs/plots",
    n_jets_track: int = 50_000,
) -> None:
    """
    Generate all plots and save them to output_dir.

    Args:
        file_path (str): Path to HDF5 file.
        jet_vars (list): Jet variable names.
        track_vars (list): Track variable names.
        jet_flavour (str): Flavour field name in HDF5.
        jet_flavour_map (dict): Raw label - class index.
        indices (np.ndarray): Jet indices to use (e.g. train_indices).
        output_dir (str): Directory for output PNGs.
        n_jets_track (int): Max jets for track plots (memory guard).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading jet data for %s jets ...", f"{len(indices):,}")
    jet_data = _load_jet_data(file_path, indices, jet_vars, jet_flavour, jet_flavour_map)

    logger.info("Loading track data (up to %s jets) ...", f"{n_jets_track:,}")
    track_data = _load_track_data(file_path, indices, track_vars, jet_flavour,
                                  jet_flavour_map, max_jets = n_jets_track)

    logger.info("Plotting jet variables ...")
    plot_jet_variables(jet_data, jet_vars, out)

    logger.info("Plotting track variables ...")
    plot_track_variables(track_data, track_vars, out)

    logger.info("Plotting correlations ...")
    plot_correlations(jet_data, track_data, jet_vars, track_vars, out)

    logger.info("All plots saved to '%s/'", out)





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

    file_path       = config["data"]["h5_path"]
    jet_vars        = config["data"]["jet_features"]
    track_vars      = config["data"]["track_features"]
    jet_flavour     = config["data"]["label"]
    jet_flavour_map = {int(k): v for k, v in config["data"]["label_map"].items()}
    preprocess_dir  = Path(config["output"]["preprocess_dir"])

    idx_path = preprocess_dir / "indices" / "train_indices.npy"
    if not idx_path.exists():
        logger.error("Index file not found: %s. Run preprocess.py first.", idx_path)
        sys.exit(1)

    indices = np.sort(np.load(idx_path))
    if args.n_jets < len(indices):
        rng     = np.random.default_rng(seed=0)
        indices = np.sort(rng.choice(indices, size=args.n_jets, replace=False))
    logger.info("Using %s jets for plots.", f"{len(indices):,}")

    make_all_plots(
        file_path       = file_path,
        jet_vars        = jet_vars,
        track_vars      = track_vars,
        jet_flavour     = jet_flavour,
        jet_flavour_map = jet_flavour_map,
        indices         = indices,
        output_dir      = args.output_dir,
        n_jets_track    = args.n_jets_track,
    )
