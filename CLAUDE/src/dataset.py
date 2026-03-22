"""
GN2 DataLoader ottimizzato — lettura vettoriale per batch interi.

Problema del dataloader precedente:
  Con num_workers > 0 e shuffle=True, ogni worker riceve indici sparsi
  da tutto il file. Ogni __getitem__ fa una lettura HDF5 singola → cache
  miss continui → throughput ~10 jet/s.

Soluzione adottata (IterableDataset + chunk sequenziale):
  1. Il file viene letto a blocchi sequenziali (chunk_size jet alla volta).
  2. Ogni chunk viene caricato in RAM con UNA sola lettura vettoriale.
  3. Gli indici dentro il chunk vengono shuffled in RAM (O(1) su numpy).
  4. I batch vengono estratti da questo buffer in memoria.
  5. Più worker lavorano su chunk diversi in parallelo.

Throughput atteso: 50.000 – 200.000 jet/s su SSD.

Uso:
    loader = make_fast_dataloader(
        h5_path    = "dataset.h5",
        batch_size = 2048,
        chunk_size = 100_000,
        num_workers = 4,
        shuffle    = True,
    )
    for batch in loader:
        jet   = batch["jet_features"]    # (B, 2)
        track = batch["track_features"]  # (B, 40, 19)
        mask  = batch["track_mask"]      # (B, 40)
        label = batch["label"]           # (B,)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Optional, Dict, Iterator
import logging
import math

logger = logging.getLogger(__name__)

# ── Variabili (identiche al notebook ATLAS) ───────────────────────────────────

JET_FEATURES = ["pt_btagJes", "eta_btagJes"]

TRACK_FEATURES = [
    "qOverP", "deta", "dphi", "d0", "z0SinTheta",
    "qOverPUncertainty", "thetaUncertainty", "phiUncertainty",
    "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
    "numberOfPixelHits", "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits", "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits", "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits", "numberOfPixelSplitHits",
    "numberOfSCTSharedHits",
]

LABEL_MAP = {5: 0, 4: 1, 0: 2, 15: 3}   # b, c, light, tau
LABEL_FIELD = "HadronConeExclTruthLabelID"


# ── Stima statistiche di normalizzazione ──────────────────────────────────────

def estimate_stats(
    h5_path: str,
    n_sample: int = 200_000,
    jet_features: List[str] = JET_FEATURES,
    track_features: List[str] = TRACK_FEATURES,
    label_field: str = LABEL_FIELD,
    label_map: Dict[int, int] = LABEL_MAP,
    max_tracks: int = 40,
) -> Dict:
    """
    Stima media e std su n_sample jet.
    Chiamata UNA SOLA VOLTA prima del training; salva i risultati e riusali.
    """
    logger.info(f"Stima statistiche su {n_sample} jet da {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        raw_labels = f["jets"][label_field][:]   # campo dentro il gruppo jets
        valid_idx = np.where(np.isin(raw_labels, list(label_map.keys())))[0]
        n = min(n_sample, len(valid_idx))
        idx = valid_idx[:n]

        jets_raw   = f["jets"][idx]
        tracks_raw = f["tracks"][idx]

    # Jet features
    j_mat = np.stack(
        [jets_raw[f].astype(np.float32) for f in jet_features], axis=1
    )
    j_mean = np.nanmean(j_mat, axis=0)
    j_std  = np.nanstd(j_mat,  axis=0) + 1e-6

    # Track features — solo tracce valide
    valid_mask = tracks_raw["valid"].astype(bool)
    t_mat = np.stack(
        [tracks_raw[f].astype(np.float32) for f in track_features], axis=2
    )  # (N, MAX_TRACKS, F)
    t_flat = t_mat[valid_mask]   # (N_valid_tracks, F)
    t_mean = np.nanmean(t_flat, axis=0)
    t_std  = np.nanstd(t_flat,  axis=0) + 1e-6

    logger.info(f"  jet mean: {j_mean},  jet std: {j_std}")
    logger.info(f"  track mean[:3]: {t_mean[:3]},  track std[:3]: {t_std[:3]}")
    return {
        "jet_mean":   j_mean.astype(np.float32),
        "jet_std":    j_std.astype(np.float32),
        "track_mean": t_mean.astype(np.float32),
        "track_std":  t_std.astype(np.float32),
    }


def save_stats(stats: Dict, path: str):
    np.savez(path, **stats)
    logger.info(f"Stats salvate in {path}")


def load_stats(path: str) -> Dict:
    d = np.load(path)
    return {k: d[k] for k in d}


# ── IterableDataset ottimizzato ───────────────────────────────────────────────

class GN2IterableDataset(IterableDataset):
    """
    IterableDataset che legge il file HDF5 a chunk sequenziali.

    Con num_workers=N:
      - PyTorch assegna automaticamente chunk diversi a worker diversi
        (via worker_info.id e worker_info.num_workers)
      - Ogni worker apre il file indipendentemente (sicuro con h5py read-only)
      - Lo shuffle avviene dentro ogni chunk (in RAM, velocissimo)
    """

    def __init__(
        self,
        h5_path:        str,
        stats:          Dict,
        chunk_size:     int  = 100_000,
        max_tracks:     int  = 40,
        shuffle:        bool = True,
        jet_features:   List[str] = JET_FEATURES,
        track_features: List[str] = TRACK_FEATURES,
        label_field:    str  = LABEL_FIELD,
        label_map:      Dict = LABEL_MAP,
        seed:           int  = 42,
        rank:           int  = 0,
        world_size:     int  = 1,
    ):
        super().__init__()
        self.h5_path        = h5_path
        self.chunk_size     = chunk_size
        self.max_tracks     = max_tracks
        self.shuffle        = shuffle
        self.jet_features   = jet_features
        self.track_features = track_features
        self.label_field    = label_field
        self.label_map      = label_map
        self.seed           = seed
        self.rank           = rank
        self.world_size     = world_size

        # Normalizzazione: converte a float32 numpy arrays
        self.jet_mean  = stats["jet_mean"].astype(np.float32)
        self.jet_std   = stats["jet_std"].astype(np.float32)
        self.trk_mean  = stats["track_mean"].astype(np.float32)
        self.trk_std   = stats["track_std"].astype(np.float32)

        # Pre-calcola gli indici validi (solo label noti)
        logger.info(f"Caricamento indici validi da {h5_path}...")
        with h5py.File(h5_path, "r") as f:
            raw_labels = f["jets"][label_field][:]
        valid_mask = np.isin(raw_labels, list(label_map.keys()))
        self.valid_indices = np.where(valid_mask)[0].astype(np.int64)
        logger.info(f"  {len(self.valid_indices):,} jet validi su {len(raw_labels):,}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_worker_indices(self) -> np.ndarray:
        """
        Divide gli indici validi tra processi DDP e tra worker interni.
        Prima divide per DDP rank, poi per worker_info.id.
        """
        indices = self.valid_indices

        # Divisione tra processi DDP
        if self.world_size > 1:
            total_ddp  = len(indices)
            per_rank   = math.ceil(total_ddp / self.world_size)
            r_start    = self.rank * per_rank
            r_end      = min(r_start + per_rank, total_ddp)
            indices    = indices[r_start:r_end]

        # Divisione tra worker interni di DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return indices

        wid  = worker_info.id
        nw   = worker_info.num_workers
        total = len(indices)
        per_worker = math.ceil(total / nw)
        start = wid * per_worker
        end   = min(start + per_worker, total)
        return indices[start:end]

    def _process_chunk(
        self,
        jets_raw: np.ndarray,
        tracks_raw: np.ndarray,
        labels_raw: np.ndarray,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Versione vettoriale: tutte le operazioni su array (N,...) senza loop
        Python per singolo jet. 5-10x più veloce della versione precedente.
        """
        N = len(jets_raw)

        # ── Labels vettoriale ────────────────────────────────────────────
        lut = np.full(256, -1, dtype=np.int64)
        for raw, cls in self.label_map.items():
            lut[int(raw) % 256] = cls
        labels = lut[labels_raw.astype(np.int64) % 256]  # (N,)

        # ── Jet features (N, F_j) ─────────────────────────────────────────
        j_mat = np.stack(
            [jets_raw[f].astype(np.float32) for f in self.jet_features], axis=1
        )
        j_mat = (j_mat - self.jet_mean) / self.jet_std

        # ── Track features (N, T_file, F_t) ──────────────────────────────
        t_mat = np.stack(
            [tracks_raw[f].astype(np.float32) for f in self.track_features],
            axis=2,
        )
        valid_all = tracks_raw["valid"].astype(bool)   # (N, T_file)

        # Ordina per |d0 significance| decrescente — vettoriale su tutti i jet
        sig = np.abs(tracks_raw["lifetimeSignedD0Significance"].astype(np.float32))
        sort_idx  = np.argsort(-sig, axis=1)           # (N, T_file)
        row_idx   = np.arange(N)[:, None]
        t_mat     = t_mat[row_idx, sort_idx]
        valid_all = valid_all[row_idx, sort_idx]

        # Tronca a max_tracks
        T = self.max_tracks
        t_mat     = t_mat[:, :T, :]
        valid_all = valid_all[:, :T]

        # Padding se T_file < max_tracks
        if t_mat.shape[1] < T:
            pad = T - t_mat.shape[1]
            t_mat     = np.pad(t_mat,     ((0,0),(0,pad),(0,0)))
            valid_all = np.pad(valid_all, ((0,0),(0,pad)))

        # Normalizzazione vettoriale con np.where
        vm_3d = valid_all[:, :, None]
        t_mat = np.where(vm_3d, (t_mat - self.trk_mean) / self.trk_std, 0.0).astype(np.float32)

        # Shuffle interno al chunk
        order = np.random.permutation(N) if self.shuffle else np.arange(N)
        j_mat     = j_mat[order]
        t_mat     = t_mat[order]
        valid_all = valid_all[order]
        labels    = labels[order]

        # Converti in tensori PyTorch una sola volta per chunk
        j_tensor = torch.from_numpy(np.ascontiguousarray(j_mat))
        t_tensor = torch.from_numpy(np.ascontiguousarray(t_mat))
        m_tensor = torch.from_numpy(np.ascontiguousarray(valid_all))
        l_tensor = torch.from_numpy(np.ascontiguousarray(labels))

        for i in range(N):
            yield {
                "jet_features":   j_tensor[i],
                "track_features": t_tensor[i],
                "track_mask":     m_tensor[i],
                "label":          l_tensor[i],
            }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Itera sul file a chunk sequenziali.
        L'ordine dei chunk è shuffled se shuffle=True.
        """
        my_indices = self._get_worker_indices()

        # Opzionalmente shuffla l'ordine dei chunk (non solo il contenuto)
        n_chunks = math.ceil(len(my_indices) / self.chunk_size)
        chunk_order = (
            np.random.permutation(n_chunks) if self.shuffle
            else np.arange(n_chunks)
        )

        with h5py.File(self.h5_path, "r", swmr=True) as f:
            for ci in chunk_order:
                start = ci * self.chunk_size
                end   = min(start + self.chunk_size, len(my_indices))
                chunk_idx = np.sort(my_indices[start:end])
                # np.sort: h5py è più veloce con indici ordinati

                # UNA sola lettura vettoriale per tutto il chunk
                jets_raw   = f["jets"][chunk_idx]
                tracks_raw = f["tracks"][chunk_idx]
                labels_raw = f["jets"][self.label_field][chunk_idx]

                yield from self._process_chunk(jets_raw, tracks_raw, labels_raw)


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_fast_dataloader(
    h5_path:     str,
    stats:       Optional[Dict] = None,
    stats_path:  Optional[str]  = None,
    batch_size:  int  = 2048,
    chunk_size:  int  = 100_000,
    max_tracks:  int  = 40,
    num_workers: int  = 4,
    shuffle:     bool = True,
    pin_memory:  bool = True,
    n_stats_sample: int = 200_000,
) -> DataLoader:
    """
    Costruisce il DataLoader ottimizzato.

    Priorità delle statistiche:
      1. stats       — dict passato direttamente
      2. stats_path  — file .npz pre-salvato (raccomandato!)
      3. stima automatica da n_stats_sample jet

    Esempio uso tipico (prima esecuzione):
        loader = make_fast_dataloader("data.h5", stats_path="stats.npz")
        # Salva le stats automaticamente la prima volta

    Esempio uso tipico (esecuzioni successive):
        stats = load_stats("stats.npz")
        loader = make_fast_dataloader("data.h5", stats=stats)
    """
    # Carica o stima le statistiche
    if stats is None:
        if stats_path is not None:
            try:
                stats = load_stats(stats_path)
                logger.info(f"Stats caricate da {stats_path}")
            except FileNotFoundError:
                logger.info("File stats non trovato, stimo dal dataset...")
                stats = estimate_stats(h5_path, n_sample=n_stats_sample)
                save_stats(stats, stats_path)
        else:
            stats = estimate_stats(h5_path, n_sample=n_stats_sample)

    dataset = GN2IterableDataset(
        h5_path    = h5_path,
        stats      = stats,
        chunk_size = chunk_size,
        max_tracks = max_tracks,
        shuffle    = shuffle,
    )

    loader = DataLoader(
        dataset,
        batch_size        = batch_size,
        num_workers       = num_workers,
        pin_memory        = pin_memory and torch.cuda.is_available(),
        persistent_workers= num_workers > 0,
        prefetch_factor   = 2 if num_workers > 0 else None,
    )
    logger.info(
        f"DataLoader pronto: {len(dataset):,} jet, "
        f"~{len(dataset)//batch_size} batch/epoca, "
        f"{num_workers} worker"
    )
    return loader


# ── Test e benchmark ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", help="Percorso al file .h5")
    parser.add_argument("--stats",       default="stats.npz")
    parser.add_argument("--batch_size",  type=int, default=2048)
    parser.add_argument("--chunk_size",  type=int, default=100_000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_bench",     type=int, default=50,
                        help="Numero di batch per il benchmark")
    args = parser.parse_args()

    loader = make_fast_dataloader(
        h5_path    = args.h5_path,
        stats_path = args.stats,
        batch_size = args.batch_size,
        chunk_size = args.chunk_size,
        num_workers= args.num_workers,
    )

    print(f"\nDataset: {len(loader.dataset):,} jet")
    print(f"Batch size: {args.batch_size}  |  Chunk size: {args.chunk_size}")
    print(f"Worker: {args.num_workers}\n")

    # Warmup (primo chunk lento per apertura file)
    for i, batch in enumerate(loader):
        if i == 0:
            print("Primo batch:")
            for k, v in batch.items():
                print(f"  {k:20s}: {v.shape}  dtype={v.dtype}")
            print()
        if i >= 2:
            break

    # Benchmark throughput
    print(f"Benchmark su {args.n_bench} batch...")
    t0 = time.perf_counter()
    total = 0
    for i, batch in enumerate(loader):
        total += batch["label"].shape[0]
        if i >= args.n_bench - 1:
            break
    elapsed = time.perf_counter() - t0
    print(f"  {total:,} jet in {elapsed:.1f}s")
    print(f"  Throughput: {total/elapsed:,.0f} jet/s")
    print(f"  Tempo/batch: {elapsed/args.n_bench*1000:.1f} ms")

















# """
# dataset.py
# ==========
# Caricamento e preprocessing dei dati GN2 da file HDF5 con struttura:
#   /jets          — variabili jet-level (una riga per jet)
#   /tracks        — variabili track-level (una riga per traccia)
#   /eventwise     — variabili evento
#   /truth_hadrons — truth hadron info

# La struttura /tracks è un array "piatto" di tracce, raggruppate per jet
# tramite un offset/count implicito nell'ordine delle righe: le prime N0
# righe appartengono al jet 0, le successive N1 al jet 1, ecc.
# Il numero di tracce per jet è variabile; si usa la colonna "valid"
# come maschera di qualità.

# Pipeline:
#   1. Leggi /jets e /tracks
#   2. Filtra jet (pT, η, JVT, label valido)
#   3. Filtra tracce (colonna valid == True)
#   4. Per ogni jet: estrai le sue tracce, applica padding a max_tracks,
#      costruisci la maschera booleana
#   5. Concatena jet features alle track features (broadcast)
#   6. Normalizza (fit solo su train)
#   7. Re-sampling 2D (pT, η) per bilanciare le classi
#   8. Restituisce DataLoader PyTorch
# """

# import h5py
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import logging
# import pickle
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional

# logger = logging.getLogger(__name__)

# CLASS_NAMES = ["b-jet", "c-jet", "light-jet", "tau-jet"]


# # ──────────────────────────────────────────────────────────────────────────────
# # 1. Read HDF5
# # ──────────────────────────────────────────────────────────────────────────────

# def load_raw_data(file_path, jet_features, track_features, label_column, batch_size = 500_000):
#     """
#     Load raw HDF5 data

#     Args:
#         file_path       (str):  dataset path
#         jet_features    (str):  features to extract from jet key
#         track_features  (str):  features to extract from track key
#         label_column    (str):  TODO

#     Returns:
#         jet_X       (N_jets, n_jet_feat):           jet input
#         jet_labels  (N_jets,):                      PDG ID
#         jet_pt      (N_jets,):                      pT in MeV (for filters)
#         jet_eta     (N_jets,):                      η (for filters)
#         track_X     (N_tracks_total, n_track_feat): tracks input
#     """
#     p = Path(file_path)
#     if not p.exists():
#         raise FileNotFoundError(f"File not found: {file_path}")

#     logger.info(f"Apertura {file_path}")

#     with h5py.File(file_path, "r") as f:

#         jets = f["jets"]
#         tracks = f["tracks"]

#         # jets
#         n_jets = min(batch_size, jets.shape[0])

#         jets_slice = jets[:n_jets]

#         jet_X = {feat: jets_slice[feat] for feat in jet_features}
#         jet_labels = jets_slice[label_column]
#         isJvtPU = jets_slice["isJvtPU"]

#         #  tracks
#         for i in range(0, n, batch_size):
#             chunk = tracks[i:i+batch_size]

#             mask = chunk["valid"]

#             yield {
#                 feat: chunk[feat][mask]
#                 for feat in track_features
#             }

#         # filtro subito
#         valid_mask = tracks_slice["valid"]

#         track_X = {
#             feat: tracks_slice[feat][valid_mask]
#             for feat in track_features
#         }

#         track_labels = valid_mask  # già booleano

#     logger.info(f"Caricati {len(jet_labels)} jet")
#     logger.info(f"Caricati {len(track_X[track_features[0]])} tracks validi")

#     return jet_X, jet_labels, isJvtPU, track_X, track_labels

#     # f = h5py.File(file_path, "r")

#     # jets_data = f["jets"]
#     # tracks_data = f["tracks"]

#     # jets_slice = slice(0, batch_size)
#     # tracks_slice = slice(0, batch_size)

#     # jets_data = jets_data[jets_slice]
#     # tracks_data = tracks_data[tracks_slice]

#     # print(jets_data.shape)

#     # jets_df = pd.DataFrame(jets_data)
#     # tracks_df = pd.DataFrame(tracks_data)
    

#     # logger.info(f"Number of jets:\t{jets_df.shape[0]}\n"
#     #             f"\tdtype fields: {list(jets_df.dtype.names)[:8]}...")
#     # logger.info(f"Number of tracks:\t{tracks_df.shape[0]}\n"
#     #             f"\tdtype fields: {list(tracks_df.dtype.names)[:8]}...")

#     # # jet features e label
#     # jet_X = read_structured(jets_df, jet_features)
#     # jet_labels = jets_df[label_column][:]

#     # # flag jet selection
#     # isJvtPU = jets_df["isJvtPU"][:]

#     # # track features
#     # track_X = read_structured(tracks_df, track_features)
#     # track_labels = tracks_df["valid"][:]

#     # for t in track_features:
#     #     track_X[t]=track_X[t][track_labels]

#     # logger.info(f"Caricati {len(jet_labels)} jet")

#     # return jet_X, jet_labels, isJvtPU, track_X, track_labels


# def read_structured(data, columns):
#     """
#     Read columns from an HDF5 structured array.

#     Args:
#         data    (Dataset):  all h5 dataset
#         columns (str):      name of columns to read
    
#     Returns:
#         TODO
#     """
#     missing_columns = [c for c in columns if c not in data.dtype.names]
#     if missing_columns:
#         raise ValueError(f"Columns not found: {missing_columns}\nColumns: {list(data.dtype.names)}")
    
#     return np.stack([data[c] for c in columns], axis=1).astype(np.float32)


# # ──────────────────────────────────────────────────────────────────────────────
# # 2. Jet filter and labels
# # ──────────────────────────────────────────────────────────────────────────────

# def filter_jets(jet_X, jet_labels, isJvtPU, label_map, pt_col_idx, eta_col_idx,
#                 pt_min_mev = 20_000., pt_max_mev = 250_000., eta_max = 2.5, filter_pileup = True):      ######################### cos'e filter pileup TODO
#     """
#     Apply event selection by mask.

#     Args:
#         jet_X           (ndarray):          jet input
#         jet_labels      (ndarray):          PDG ID
#         isJvtPU         (ndarray):          TODO
#         label_map       (dict of int: int): TODO
#         pt_col_idx      (int):              index of pT column (for filters)
#         eta_col_idx     (int):              index of η column (for filters)
#         pt_min_mev      (float, optional):  minimum of pt threshold (in MeV)
#         pt_max_mev      (float, optional):  maximum of pt threshold (in MeV)
#         eta_max         (float, optional):  maximum of η threshold (in MeV)
#         filter_pileup   (bool, optional):   TODO

#     Returns:
#         jet_X_filt      (N_jets, n_jet_feat):   jet input filtered
#         y_class         (N_jets,):              TODO
#         jet_pt_filt     (N_jets,):              pT filtered in MeV
#         jet_eta_filt    (N_jets,):              η filtered in MeV
#         jet_mask        (N_jets,):              index of selected jets
#     """
#     N = len(jet_labels)
#     mask = np.ones(N, dtype=bool)       # initialized all true

#     jet_pt  = jet_X[:, pt_col_idx]
#     jet_eta = jet_X[:, eta_col_idx]

#     pt_min = pt_min_mev
#     pt_max = pt_max_mev

#     mask &= (jet_pt > pt_min) & (jet_pt < pt_max)               # mask on pT
#     logger.info(f"\tDopo taglio {pt_min}<pT<{pt_max}: {mask.sum()}/{N}")

#     mask &= (np.abs(jet_eta) < eta_max)                         # mask on η
#     logger.info(f"\tDopo taglio |η|<{eta_max}: {mask.sum()}")

#     if filter_pileup:                                                           ###################################################
#         mask &= (isJvtPU == 0)
#         logger.info(f"\tDopo filtro JVT pile-up: {mask.sum():,}")

#     # Label: tieni solo PDG ID presenti in label_map                            ###################################################
#     valid_pdg = np.array(list(label_map.keys()), dtype=np.int32)
#     mask &= np.isin(jet_labels, valid_pdg)
#     logger.info(f"\tDopo filtro label: {mask.sum():,}")

#     # apply masks
#     jet_indices = np.where(mask)[0]
#     jet_X_filt = jet_X[jet_indices]
#     jet_pt_filt = jet_pt[jet_indices]
#     jet_eta_filt = jet_eta[jet_indices]

#     # Mappa PDG ID → classe
#     y_pdg = jet_labels[jet_indices]
#     y_class = np.vectorize(label_map.get)(y_pdg).astype(np.int64)

#     for pdg, cls in sorted(label_map.items()):
#         n = (y_class == cls).sum()
#         logger.info(f"\t{CLASS_NAMES[cls]} (PDG {pdg}): {n:,}\n({100*n/len(y_class):.1f}%)")

#     return jet_X_filt, y_class, jet_pt_filt, jet_eta_filt, jet_indices


# # ──────────────────────────────────────────────────────────────────────────────
# # 3. Costruzione array tracce per jet con padding
# # ──────────────────────────────────────────────────────────────────────────────

# def build_track_arrays(jet_indices, track_X, track_valid, n_jets_total, max_tracks, n_track_feat):
#     """
#     Costruisce array (N_jets, max_tracks, n_track_feat) con padding e maschera.

#     Le tracce sono memorizzate in un array piatto; le prime k_0 righe
#     appartengono al jet 0, le successive k_1 al jet 1, ecc.
#     La lunghezza di ogni "blocco" è variabile.

#     Strategia di selezione se il jet ha > max_tracks tracce valide:
#       Paper: "Tracks with smaller absolute track impact parameter
#               significance are dropped if there are more than 40 tracks"
#       → ordina per |lifetimeSignedD0Significance| decrescente,
#         tieni le prime max_tracks.

        
#     Args:
#         jet_indices     (ndarray):  indices of jets data
#         track_X         (ndarray):  jet input
#         track_valid     (ndarray):  bool, quality mask ATLAS
#         n_jets_total    (int):      number of total jets
#         max_tracks      (int):      max number of tracks
#         n_tracks_feat   (int):      number of tracks features
        
#     Returns:
#         tracks_padded (N_sel, max_tracks, n_track_feat):    float32    TODO                     ######################################################3
#         track_mask    (N_sel, max_tracks):                  bool, True = posizione valida, False = padding
#     """
#     N_sel = len(jet_indices)

#     # Calcola gli offset: quante tracce ha ogni jet
#     # Assumiamo che track_X abbia le stesse righe ordinate per jet
#     # e che n_jets_total sia il numero totale di jet nel file originale.
#     # Il numero di tracce per jet-i è:
#     #   n_tracks_i = offset[i+1] - offset[i]
#     # Ma il file HDF5 ATLAS tipicamente salva le tracce come array
#     # piatto con lunghezza fissa per jet (zero-padded) oppure con
#     # un dataset separato di offset.
#     #
#     # Strategia robusta: controlliamo il rapporto
#     #   len(track_X) / n_jets_total
#     # Se è un intero, le tracce sono organizzate con slot fissi.
#     # Altrimenti usiamo offsets da un dataset apposito se disponibile.

#     n_tracks_total = len(track_X)

#     if n_tracks_total % n_jets_total == 0:
#         # Slot fissi per jet (comune nei file ATLAS preprocessati)
#         slot_size = n_tracks_total // n_jets_total
#         logger.info(f"Struttura tracce: slot fissi di {slot_size} tracce/jet")

#         tracks_padded = np.zeros(
#             (N_sel, max_tracks, n_track_feat), dtype=np.float32
#         )
#         track_mask = np.zeros((N_sel, max_tracks), dtype=bool)

#         for out_idx, jet_idx in enumerate(jet_indices):
#             start = jet_idx * slot_size
#             end   = start + slot_size

#             # Tracce di questo jet, filtra per valid
#             t_feat  = track_X[start:end]       # (slot_size, n_feat)
#             t_valid = track_valid[start:end]    # (slot_size,)

#             valid_tracks = t_feat[t_valid]      # (n_valid, n_feat)
#             n_valid = len(valid_tracks)

#             if n_valid == 0:
#                 continue

#             # Se abbiamo più tracce di max_tracks:
#             # ordina per |lifetimeSignedD0Significance| decrescente
#             if n_valid > max_tracks:
#                 # Indice della feature lifetimeSignedD0Significance
#                 # (quarta colonna, indice 3 in track_features)
#                 sig_col = 3  # lifetimeSignedD0Significance
#                 significance = np.abs(valid_tracks[:, sig_col])
#                 order = np.argsort(-significance)   # decrescente
#                 valid_tracks = valid_tracks[order[:max_tracks]]
#                 n_valid = max_tracks

#             tracks_padded[out_idx, :n_valid, :] = valid_tracks
#             track_mask[out_idx, :n_valid] = True

#     else:
#         # Numero di tracce variabile per jet — questo caso richiede
#         # un dataset di offsets. Solleviamo un errore informativo.
#         raise ValueError(
#             f"Il numero di tracce ({n_tracks_total}) non è divisibile "
#             f"per il numero di jet ({n_jets_total}). "
#             f"Il file potrebbe usare un dataset di offsets separato "
#             f"(es. 'track_offsets' o 'n_tracks_per_jet'). "
#             f"Aggiorna load_raw_data() per leggerlo."
#         )

#     # Statistiche
#     n_valid_per_jet = track_mask.sum(axis=1)
#     logger.info(f"Tracce valide per jet: "
#                 f"media={n_valid_per_jet.mean():.1f}, "
#                 f"max={n_valid_per_jet.max()}, "
#                 f"min={n_valid_per_jet.min()}")
#     logger.info(f"Jet con 0 tracce valide: "
#                 f"{(n_valid_per_jet == 0).sum():,}")

#     return tracks_padded, track_mask


# # ──────────────────────────────────────────────────────────────────────────────
# # 4. Preprocessing: log-transform del pT
# # ──────────────────────────────────────────────────────────────────────────────

# def log_transform_pt(
#     jet_X:        np.ndarray,
#     jet_features: List[str],
#     tracks_X:     np.ndarray,
#     track_features: List[str],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Applica log-transform alle feature di pT prima della normalizzazione.

#     Il pT ha una distribuzione molto skewed (coda power-law): usare log(pT)
#     migliora la stabilità del training. Stesso approccio usato in GN1/GN2
#     implicitamente tramite il preprocessing del dataset ATLAS.

#     Feature trasformate:
#       - jet: pt_btagJes → log(pt_btagJes)
#       - track: pt → log(pt), ptfrac → log(ptfrac + eps)
#     """
#     eps = 1e-6
#     jet_X_out    = jet_X.copy()
#     tracks_X_out = tracks_X.copy()

#     # Jet pT
#     jet_feat_lower = [f.lower() for f in jet_features]
#     for candidate in ["pt_btagjes", "pt"]:
#         if candidate in jet_feat_lower:
#             idx = jet_feat_lower.index(candidate)
#             jet_X_out[:, idx] = np.log(np.maximum(jet_X[:, idx], eps))
#             logger.info(f"Log-transform applicato a jet feature '{jet_features[idx]}'")
#             break

#     # Track pT e ptfrac
#     track_feat_lower = [f.lower() for f in track_features]
#     for candidate in ["pt"]:
#         if candidate in track_feat_lower:
#             idx = track_feat_lower.index(candidate)
#             tracks_X_out[..., idx] = np.log(
#                 np.maximum(tracks_X[..., idx], eps)
#             )
#             logger.info(f"Log-transform applicato a track feature '{track_features[idx]}'")

#     for candidate in ["ptfrac"]:
#         if candidate in track_feat_lower:
#             idx = track_feat_lower.index(candidate)
#             tracks_X_out[..., idx] = np.log(
#                 np.maximum(tracks_X[..., idx], eps)
#             )
#             logger.info(f"Log-transform applicato a track feature '{track_features[idx]}'")

#     return jet_X_out, tracks_X_out


# # ──────────────────────────────────────────────────────────────────────────────
# # 5. Normalizzazione
# # ──────────────────────────────────────────────────────────────────────────────

# class GN2Scaler:
#     """
#     Normalizzazione separata per jet features e track features.

#     Paper: "All input variables to the algorithm training are normalised
#             to have zero mean and unit variance"

#     Importante: fit SOLO sui dati di training, apply a val/test.
#     Le posizioni di padding (track_mask == False) rimangono a zero
#     e non influenzano il calcolo di media/std (le escludiamo).
#     """

#     def __init__(self):
#         self.jet_scaler   = StandardScaler()
#         self.track_scaler = StandardScaler()
#         self._fitted = False

#     def fit(
#         self,
#         jet_X_train:    np.ndarray,   # (N, n_jet_feat)
#         tracks_X_train: np.ndarray,   # (N, max_tracks, n_track_feat)
#         mask_train:     np.ndarray,   # (N, max_tracks) bool
#     ) -> "GN2Scaler":
#         # Fit jet scaler
#         self.jet_scaler.fit(jet_X_train)

#         # Fit track scaler: considera solo le posizioni non-padding
#         # Reshape a (N*max_tracks, n_track_feat) e filtra con mask
#         N, T, F = tracks_X_train.shape
#         tracks_flat = tracks_X_train.reshape(-1, F)   # (N*T, F)
#         mask_flat   = mask_train.reshape(-1)           # (N*T,)
#         valid_tracks = tracks_flat[mask_flat]           # (N_valid, F)

#         self.track_scaler.fit(valid_tracks)
#         self._fitted = True
#         logger.info(f"GN2Scaler fittato su {valid_tracks.shape[0]:,} tracce valide")
#         return self

#     def transform(
#         self,
#         jet_X:    np.ndarray,
#         tracks_X: np.ndarray,
#         mask:     np.ndarray,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         if not self._fitted:
#             raise RuntimeError("Chiama fit() prima di transform()")

#         # Jet
#         jet_X_norm = self.jet_scaler.transform(jet_X).astype(np.float32)

#         # Track: trasforma tutto, poi rimetti a zero le posizioni di padding
#         N, T, F = tracks_X.shape
#         tracks_flat = tracks_X.reshape(-1, F)
#         tracks_norm = self.track_scaler.transform(tracks_flat).reshape(N, T, F)
#         tracks_norm = tracks_norm.astype(np.float32)

#         # Azzera le posizioni di padding dopo la normalizzazione
#         tracks_norm[~mask] = 0.

#         return jet_X_norm, tracks_norm

#     def fit_transform(
#         self,
#         jet_X:    np.ndarray,
#         tracks_X: np.ndarray,
#         mask:     np.ndarray,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         return self.fit(jet_X, tracks_X, mask).transform(jet_X, tracks_X, mask)

#     def save(self, path: str):
#         with open(path, "wb") as f:
#             pickle.dump((self.jet_scaler, self.track_scaler), f)

#     @classmethod
#     def load(cls, path: str) -> "GN2Scaler":
#         obj = cls()
#         with open(path, "rb") as f:
#             obj.jet_scaler, obj.track_scaler = pickle.load(f)
#         obj._fitted = True
#         return obj


# # ──────────────────────────────────────────────────────────────────────────────
# # 6. Re-sampling 2D (pT, η)
# # ──────────────────────────────────────────────────────────────────────────────

# def resample_to_reference(
#     y_class:      np.ndarray,
#     jet_pt:       np.ndarray,
#     jet_eta:      np.ndarray,
#     reference_class: int = 1,          # c-jet
#     n_pt_bins:    int = 10,
#     n_eta_bins:   int = 4,
#     pt_range:     Tuple[float, float] = (20_000., 250_000.),
#     eta_range:    Tuple[float, float] = (-2.5, 2.5),
#     rng_seed:     int = 42,
# ) -> np.ndarray:
#     """
#     Re-sampling 2D in (pT, η) per bilanciare le classi.

#     Paper: "b-jets, light-jets and τ-jets are re-sampled in pT and η
#             to match the corresponding c-jet distributions, thereby
#             preventing the models from discriminating between jet
#             flavours based on relative kinematic differences"

#     Algoritmo:
#     1. Calcola istogramma 2D (pT, η) della classe di riferimento (c-jet)
#     2. Per ogni altra classe, in ogni bin 2D, campiona (con rimpiazzo se
#        necessario) lo stesso numero di jet del riferimento
#     3. Restituisce gli indici selezionati

#     Returns:
#         selected_indices: array di indici nel dataset originale
#     """
#     rng = np.random.default_rng(rng_seed)

#     pt_bins  = np.linspace(pt_range[0],  pt_range[1],  n_pt_bins + 1)
#     eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)

#     # Assegna ogni jet a un bin 2D
#     pt_bin_idx  = np.digitize(jet_pt,  pt_bins)  - 1
#     eta_bin_idx = np.digitize(jet_eta, eta_bins) - 1

#     # Clip ai bordi (jet fuori range vanno nel bin estremo)
#     pt_bin_idx  = np.clip(pt_bin_idx,  0, n_pt_bins - 1)
#     eta_bin_idx = np.clip(eta_bin_idx, 0, n_eta_bins - 1)

#     # Istogramma di riferimento
#     ref_mask = (y_class == reference_class)
#     ref_counts = np.zeros((n_pt_bins, n_eta_bins), dtype=int)
#     for i in range(n_pt_bins):
#         for j in range(n_eta_bins):
#             bin_mask = ref_mask & (pt_bin_idx == i) & (eta_bin_idx == j)
#             ref_counts[i, j] = bin_mask.sum()

#     # Per ogni classe, campiona per matchare il riferimento
#     all_classes = np.unique(y_class)
#     selected_per_class = []

#     for cls in all_classes:
#         cls_mask = (y_class == cls)
#         cls_indices = []

#         for i in range(n_pt_bins):
#             for j in range(n_eta_bins):
#                 target_n = ref_counts[i, j]
#                 if target_n == 0:
#                     continue

#                 bin_mask = cls_mask & (pt_bin_idx == i) & (eta_bin_idx == j)
#                 available = np.where(bin_mask)[0]

#                 if len(available) == 0:
#                     continue

#                 # Campiona con rimpiazzo se disponibili < target
#                 sampled = rng.choice(
#                     available,
#                     size=target_n,
#                     replace=(len(available) < target_n),
#                 )
#                 cls_indices.append(sampled)

#         if cls_indices:
#             cls_indices = np.concatenate(cls_indices)
#             selected_per_class.append(cls_indices)
#             n_orig = cls_mask.sum()
#             logger.info(
#                 f"  {CLASS_NAMES[cls]}: {n_orig:,} → "
#                 f"{len(cls_indices):,} dopo resampling"
#             )

#     return np.concatenate(selected_per_class)


# # ──────────────────────────────────────────────────────────────────────────────
# # 7. PyTorch Dataset
# # ──────────────────────────────────────────────────────────────────────────────

# class GN2Dataset(Dataset):
#     """
#     PyTorch Dataset.

#     Ogni item: (jet_feat, tracks_feat, track_mask, label)
#       jet_feat   : (n_jet_feat,)               float32
#       tracks_feat: (max_tracks, n_track_feat)  float32
#       track_mask : (max_tracks,)               bool
#       label      : ()                          int64
#     """

#     def __init__(
#         self,
#         jet_X:      np.ndarray,   # (N, n_jet_feat)
#         tracks_X:   np.ndarray,   # (N, max_tracks, n_track_feat)
#         track_mask: np.ndarray,   # (N, max_tracks)
#         y:          np.ndarray,   # (N,)
#     ):
#         assert len(jet_X) == len(y)
#         self.jet_X      = torch.from_numpy(jet_X.astype(np.float32))
#         self.tracks_X   = torch.from_numpy(tracks_X.astype(np.float32))
#         self.track_mask = torch.from_numpy(track_mask.astype(bool))
#         self.y          = torch.from_numpy(y.astype(np.int64))

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return (
#             self.jet_X[idx],
#             self.tracks_X[idx],
#             self.track_mask[idx],
#             self.y[idx],
#         )

#     @property
#     def n_jet_features(self) -> int:
#         return self.jet_X.shape[1]

#     @property
#     def n_track_features(self) -> int:
#         return self.tracks_X.shape[2]

#     @property
#     def max_tracks(self) -> int:
#         return self.tracks_X.shape[1]


# # ──────────────────────────────────────────────────────────────────────────────
# # 8. Factory principale
# # ──────────────────────────────────────────────────────────────────────────────

# def build_dataloaders(cfg: dict):
#     """
#     Pipeline completa: legge HDF5 → filtra → padding → normalizza →
#     resample → split → DataLoader.

#     Returns:
#         train_loader, val_loader, test_loader,
#         scaler (GN2Scaler da salvare per inference),
#         n_jet_features, n_track_features
#     """
#     dcfg = cfg["data"]
#     tcfg = cfg["training"]

#     jet_features   = dcfg["jet_features"]
#     track_features = dcfg["track_features"]
#     max_tracks     = dcfg["max_tracks"]
#     label_map      = {int(k): v for k, v in dcfg["label_map"].items()}

#     # 1. Lettura raw
#     (jet_X, jet_labels, jet_pt, jet_eta,
#      isJvtPU, track_X, track_valid) = load_raw_data(
#         file_path=dcfg["h5_path"],
#         jet_features=jet_features,
#         track_features=track_features,
#         label_column=dcfg["label_column"],
#     )

#     n_jets_total = len(jet_labels)

#     # 2. Filtro jet
#     (jet_X_filt, y_class, jet_pt_filt, jet_eta_filt,
#      jet_indices) = filter_jets(
#         jet_X, jet_labels, jet_pt, jet_eta, isJvtPU,
#         label_map=label_map,
#         pt_min_mev=dcfg.get("pt_min_mev", 20_000.),
#         pt_max_mev=dcfg.get("pt_max_mev", 6_000_000.),
#         eta_max=dcfg.get("eta_max", 2.5),
#         filter_pileup=dcfg.get("filter_pileup", True),
#     )

#     # 3. Costruzione array tracce con padding
#     logger.info("Costruzione array tracce con padding...")
#     tracks_padded, track_mask = build_track_arrays(
#         jet_indices=jet_indices,
#         track_X=track_X,
#         track_valid=track_valid,
#         n_jets_total=n_jets_total,
#         max_tracks=max_tracks,
#         n_track_feat=len(track_features),
#     )

#     # 4. Log-transform pT
#     jet_X_filt, tracks_padded = log_transform_pt(
#         jet_X_filt, jet_features,
#         tracks_padded, track_features,
#     )

#     # 5. Re-sampling
#     rs_cfg = dcfg.get("resampling", {})
#     if rs_cfg.get("enabled", True):
#         logger.info("Re-sampling 2D (pT, η)...")
#         sel_indices = resample_to_reference(
#             y_class=y_class,
#             jet_pt=jet_pt_filt,
#             jet_eta=jet_eta_filt,
#             reference_class=rs_cfg.get("reference_class", 1),
#             n_pt_bins=rs_cfg.get("n_pt_bins", 10),
#             n_eta_bins=rs_cfg.get("n_eta_bins", 4),
#             pt_range=tuple(rs_cfg.get("pt_range_mev", [20_000., 250_000.])),
#             eta_range=tuple(rs_cfg.get("eta_range", [-2.5, 2.5])),
#             rng_seed=dcfg.get("split_seed", 42),
#         )
#         jet_X_filt    = jet_X_filt[sel_indices]
#         tracks_padded = tracks_padded[sel_indices]
#         track_mask    = track_mask[sel_indices]
#         y_class       = y_class[sel_indices]
#         jet_pt_filt   = jet_pt_filt[sel_indices]
#         jet_eta_filt  = jet_eta_filt[sel_indices]
#         logger.info(f"Dopo resampling: {len(y_class):,} jet totali")

#     # 6. Split train/val/test (stratificato)
#     seed = dcfg.get("split_seed", 42)
#     indices = np.arange(len(y_class))

#     # Split in due passi per stratificazione
#     idx_tv, idx_test = train_test_split(
#         indices,
#         test_size=dcfg.get("test_fraction", 0.15),
#         stratify=y_class,
#         random_state=seed,
#     )
#     val_frac_rel = dcfg.get("val_fraction", 0.15) / (
#         dcfg.get("train_fraction", 0.70) + dcfg.get("val_fraction", 0.15)
#     )
#     idx_train, idx_val = train_test_split(
#         idx_tv,
#         test_size=val_frac_rel,
#         stratify=y_class[idx_tv],
#         random_state=seed + 1,
#     )

#     logger.info(f"Split: train={len(idx_train):,}, "
#                 f"val={len(idx_val):,}, test={len(idx_test):,}")

#     def _subset(idx):
#         return (
#             jet_X_filt[idx],
#             tracks_padded[idx],
#             track_mask[idx],
#             y_class[idx],
#         )

#     # 7. Normalizzazione: fit su train, apply a tutti
#     scaler = GN2Scaler()
#     jet_train, trk_train, msk_train, y_train = _subset(idx_train)
#     jet_val,   trk_val,   msk_val,   y_val   = _subset(idx_val)
#     jet_test,  trk_test,  msk_test,  y_test  = _subset(idx_test)

#     jet_train, trk_train = scaler.fit_transform(jet_train, trk_train, msk_train)
#     jet_val,   trk_val   = scaler.transform(jet_val,   trk_val,   msk_val)
#     jet_test,  trk_test  = scaler.transform(jet_test,  trk_test,  msk_test)

#     # 8. Datasets e DataLoaders
#     batch_size = tcfg.get("batch_size", 12000)

#     def _loader(jet, trk, msk, y, shuffle):
#         ds = GN2Dataset(jet, trk, msk, y)
#         return DataLoader(
#             ds,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=4,
#             pin_memory=False,
#             drop_last=shuffle,   # drop_last solo per train (evita batch=1 con BN)
#         )

#     train_loader = _loader(jet_train, trk_train, msk_train, y_train, shuffle=True)
#     val_loader   = _loader(jet_val,   trk_val,   msk_val,   y_val,   shuffle=False)
#     test_loader  = _loader(jet_test,  trk_test,  msk_test,  y_test,  shuffle=False)

#     n_jet_feat   = len(jet_features)
#     n_track_feat = len(track_features)

#     logger.info("DataLoader pronti.")
#     logger.info(f"  n_jet_features={n_jet_feat}, "
#                 f"n_track_features={n_track_feat}, "
#                 f"max_tracks={max_tracks}")

#     return (train_loader, val_loader, test_loader,
#             scaler, n_jet_feat, n_track_feat)


# if __name__ == "__main__":

#     jet_features = ["pt", "eta"]
#     track_features = ["qOverP", "deta", "dphi", "d0", "z0SinTheta",
#                         "qOverPUncertainty", "thetaUncertainty", "phiUncertainty",
#                         "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
#                         "numberOfPixelHits", "numberOfSCTHits", "numberOfInnermostPixelLayerHits", "numberOfNextToInnermostPixelLayerHits", "numberOfInnermostPixelLayerSharedHits", "numberOfInnermostPixelLayerSplitHits", "numberOfPixelSharedHits", "numberOfPixelSplitHits", "numberOfSCTSharedHits"]
    
#     jet_X, jet_labels, isJvtPU, track_X, track_labels = load_raw_data("../dataset/mc-flavtag-ttbar-small.h5",
#                                                                      jet_features, track_features,
#                                                                      "HadronConeExclTruthLabelID",
#                                                                      10_000)
    
#     print(jet_X.shape)
#     print(jet_labels)
#     print(track_X.shape)
#     print(track_labels.shape)