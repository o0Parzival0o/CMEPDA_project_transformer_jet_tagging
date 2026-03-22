"""
GN2-Lite Training — supporto GPU singola e multi-GPU (DDP).

Modalità supportate:
  1. CPU only                  python train.py --no_cuda
  2. GPU singola               python train.py
  3. Multi-GPU stesso nodo     torchrun --nproc_per_node=4 train.py
  4. Multi-nodo                torchrun --nproc_per_node=4 --nnodes=2 ... train.py

DistributedDataParallel (DDP) è lo stesso approccio usato da ATLAS con
PyTorch Lightning. Qui lo implementiamo direttamente per capire come funziona.

Uso rapido:
    # Verifica GPU disponibili
    python train.py --check_gpu

    # Training su GPU singola
    python train.py --h5_path ../dataset/mc-flavtag-ttbar-small.h5 \
                    --stats_path stats.npz --epochs 10

    # Training su 4 GPU (stesso nodo)
    torchrun --nproc_per_node=4 train.py \
             --h5_path ../dataset/mc-flavtag-ttbar-small.h5 \
             --stats_path stats.npz --epochs 10
"""

import os
import time
import logging
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# ── Import locali ─────────────────────────────────────────────────────────────
# Aggiusta il path se i file sono in cartelle diverse
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gn2_dataloader_fast import GN2IterableDataset, make_fast_dataloader, load_stats, estimate_stats, save_stats
from gn2_lite_model import GN2Lite, N_JET_FEAT, N_TRACK_FEAT, N_CLASSES


# ── Setup logging ─────────────────────────────────────────────────────────────

def setup_logging(rank: int = 0):
    """Solo il rank 0 stampa i log in DDP."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger(__name__)


# ── GPU utilities ─────────────────────────────────────────────────────────────

def check_gpu():
    """Stampa informazioni sulle GPU disponibili."""
    print(f"\nPyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"GPU count       : {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            mem_gb = p.total_memory / 1024**3
            print(f"  GPU {i}: {p.name}  ({mem_gb:.1f} GB VRAM)")
    else:
        print("  Nessuna GPU trovata — il training userà la CPU")
    print()


def get_device(rank: int = 0, no_cuda: bool = False) -> torch.device:
    """Restituisce il device corretto per questo processo."""
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{rank}")


# ── DDP setup / teardown ──────────────────────────────────────────────────────

def setup_ddp(rank: int, world_size: int):
    """
    Inizializza il processo group per DDP.
    torchrun imposta automaticamente RANK, LOCAL_RANK, WORLD_SIZE e
    MASTER_ADDR/MASTER_PORT — non serve impostarli a mano.
    """
    dist.init_process_group(
        backend="nccl",   # NCCL = ottimizzato per GPU NVIDIA
        rank=rank,
        world_size=world_size,
    )
    # Ogni processo usa la sua GPU
    torch.cuda.set_device(rank)
    logger.info(f"DDP inizializzato: rank {rank}/{world_size}")


def teardown_ddp():
    dist.destroy_process_group()


def is_main_process() -> bool:
    """True se siamo nel processo principale (rank 0 o no DDP)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ── DataLoader per DDP ────────────────────────────────────────────────────────

def make_ddp_dataloader(
    h5_path: str,
    stats: dict,
    batch_size: int,
    chunk_size: int,
    num_workers: int,
    shuffle: bool,
    rank: int,
    world_size: int,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Con DDP ogni processo legge una porzione diversa del dataset.
    GN2IterableDataset gestisce già questo tramite _get_worker_indices()
    che divide gli indici per worker_info.id — ma per DDP dobbiamo anche
    dividere tra i processi.

    Soluzione: passiamo rank/world_size al dataset che li usa per
    pre-dividere gli indici validi prima ancora dei worker interni.
    """
    dataset = GN2IterableDataset(
        h5_path    = h5_path,
        stats      = stats,
        chunk_size = chunk_size,
        shuffle    = shuffle,
        rank       = rank,        # nuovo parametro
        world_size = world_size,  # nuovo parametro
    )

    return DataLoader(
        dataset,
        batch_size         = batch_size,
        num_workers        = num_workers,
        pin_memory         = pin_memory and torch.cuda.is_available(),
        persistent_workers = num_workers > 0,
        prefetch_factor    = 2 if num_workers > 0 else None,
    )


# ── Loss con class weights ────────────────────────────────────────────────────

def compute_class_weights(
    h5_path: str,
    label_map: dict,
    label_field: str = "HadronConeExclTruthLabelID",
    n_sample: int = 500_000,
) -> torch.Tensor:
    """
    Conta la frequenza di ogni classe su un campione e restituisce
    pesi inversi (classi rare pesano di più).
    GN2 nel paper fa resampling in pT/eta invece di class weights —
    qui usiamo weights per semplicità, l'effetto è simile.
    """
    import h5py
    logger.info("Calcolo class weights...")
    with h5py.File(h5_path, "r") as f:
        labels_raw = f["jets"][label_field][:n_sample]

    counts = np.zeros(len(label_map), dtype=np.float64)
    for raw, cls in label_map.items():
        counts[cls] = (labels_raw == raw).sum()

    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * counts)   # inversamente proporzionale
    weights = weights / weights.sum() * len(counts)   # normalizza alla media 1.0
    logger.info(f"  Class weights: {weights.round(3)}")
    return torch.tensor(weights, dtype=torch.float32)


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    torch.cuda.amp.GradScaler,
    epoch:     int,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_jets = 0
    t0 = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        # Trasferimento CPU → GPU non bloccante (pin_memory=True lo rende veloce)
        jf    = batch["jet_features"].to(device, non_blocking=True)
        tf    = batch["track_features"].to(device, non_blocking=True)
        mask  = batch["track_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # set_to_none risparmia memoria

        # Mixed precision (AMP): calcola in fp16, accumula gradienti in fp32
        # Sui datacentre NVIDIA A100 (come quelli ATLAS) questo raddoppia
        # il throughput senza perdita di accuratezza apprezzabile
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(jf, tf, mask)
            loss   = criterion(logits, label)

        # Gradient scaling per AMP (evita underflow in fp16)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # Metriche (solo rank 0 per non duplicare)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            total_correct += (pred == label).sum().item()
            total_jets    += label.size(0)
            total_loss    += loss.item() * label.size(0)

        if is_main_process() and batch_idx % 100 == 0:
            elapsed = time.perf_counter() - t0
            jets_so_far = (batch_idx + 1) * label.size(0)
            throughput  = jets_so_far / elapsed if elapsed > 0 else 0
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                f"loss={loss.item():.4f}  "
                f"throughput={throughput:,.0f} jet/s"
            )

    elapsed = time.perf_counter() - t0
    return {
        "loss":       total_loss / max(total_jets, 1),
        "accuracy":   total_correct / max(total_jets, 1),
        "throughput": total_jets / elapsed,
        "time_s":     elapsed,
    }


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_jets = 0

    for batch in loader:
        jf    = batch["jet_features"].to(device, non_blocking=True)
        tf    = batch["track_features"].to(device, non_blocking=True)
        mask  = batch["track_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(jf, tf, mask)
            loss   = criterion(logits, label)

        pred = logits.argmax(dim=1)
        total_correct += (pred == label).sum().item()
        total_jets    += label.size(0)
        total_loss    += loss.item() * label.size(0)

    return {
        "loss":     total_loss / max(total_jets, 1),
        "accuracy": total_correct / max(total_jets, 1),
    }


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Salva checkpoint. Con DDP salva solo i pesi del modello (non il wrapper)."""
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "epoch":        epoch,
        "model_state":  raw_model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict() if scheduler else None,
        "metrics":      metrics,
    }, path)
    logger.info(f"Checkpoint salvato: {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and ckpt.get("sched_state"):
        scheduler.load_state_dict(ckpt["sched_state"])
    logger.info(f"Checkpoint caricato da {path}, epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt.get("metrics", {})


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    # ── Detect DDP ────────────────────────────────────────────────────────
    # torchrun imposta LOCAL_RANK automaticamente
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp      = world_size > 1

    setup_logging(local_rank)

    if is_ddp:
        setup_ddp(local_rank, world_size)

    device = get_device(local_rank, args.no_cuda)
    logger.info(f"Device: {device}  |  DDP: {is_ddp}  |  World size: {world_size}")

    # ── Stats di normalizzazione ──────────────────────────────────────────
    from gn2_dataloader_fast import LABEL_MAP
    if Path(args.stats_path).exists():
        stats = load_stats(args.stats_path)
        logger.info(f"Stats caricate da {args.stats_path}")
    else:
        stats = estimate_stats(args.h5_path)
        if is_main_process():
            save_stats(stats, args.stats_path)

    # ── DataLoader ────────────────────────────────────────────────────────
    # In DDP ogni processo legge ~1/world_size del dataset
    # Aggiustamento batch_size: in DDP il batch "globale" è batch_size*world_size
    # Per mantenere la stessa lr schedule del paper, batch_size locale rimane fisso
    train_loader = make_fast_dataloader(
        h5_path     = args.h5_path,
        stats       = stats,
        batch_size  = args.batch_size,
        chunk_size  = args.chunk_size,
        num_workers = args.num_workers,
        shuffle     = True,
    )

    val_loader = make_fast_dataloader(
        h5_path     = args.h5_path,    # idealmente un file di val separato
        stats       = stats,
        batch_size  = args.batch_size * 2,
        chunk_size  = args.chunk_size,
        num_workers = args.num_workers,
        shuffle     = False,
    ) if args.val_h5 is None else make_fast_dataloader(
        h5_path     = args.val_h5,
        stats       = stats,
        batch_size  = args.batch_size * 2,
        chunk_size  = args.chunk_size,
        num_workers = args.num_workers,
        shuffle     = False,
    )

    # ── Modello ───────────────────────────────────────────────────────────
    model = GN2Lite(
        embed_dim = args.embed_dim,
        n_heads   = args.n_heads,
        n_layers  = args.n_layers,
        ffn_dim   = args.ffn_dim,
        dropout   = args.dropout,
    ).to(device)

    if is_ddp:
        # DDP: sincronizza i gradienti tra tutti i processi ad ogni backward
        # find_unused_parameters=False → più veloce se tutti i parametri
        # sono usati (è il nostro caso con GN2Lite)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main_process():
        raw = model.module if is_ddp else model
        logger.info(f"Parametri totali: {raw.count_parameters():,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    class_weights = compute_class_weights(args.h5_path, LABEL_MAP).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # ── Optimizer: AdamW come nel paper GN2 ───────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = 1e-5,
    )

    # ── Scheduler: cosine annealing come nel paper ─────────────────────────
    # Il paper usa cosine annealing con warmup del primo 1% degli step
    total_steps  = args.epochs * math.ceil(len(train_loader.dataset) / args.batch_size)
    warmup_steps = max(1, total_steps // 100)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps          # warmup lineare
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── AMP scaler (solo CUDA) ────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ── Resume da checkpoint ──────────────────────────────────────────────
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if is_main_process():
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, scaler, epoch + 1,
        )

        # Valida solo ogni val_every epoche (è costoso sul dataset intero)
        val_metrics = {}
        if (epoch + 1) % args.val_every == 0:
            val_metrics = validate(model, val_loader, criterion, device)

        if is_main_process():
            logger.info(
                f"Train  loss={train_metrics['loss']:.4f}  "
                f"acc={train_metrics['accuracy']:.3f}  "
                f"throughput={train_metrics['throughput']:,.0f} jet/s"
            )
            if val_metrics:
                logger.info(
                    f"Val    loss={val_metrics['loss']:.4f}  "
                    f"acc={val_metrics['accuracy']:.3f}"
                )

            # Salva checkpoint
            ckpt_path = Path(args.checkpoint_dir) / f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            {**train_metrics, **val_metrics}, ckpt_path)

            # Salva il best model
            v_loss = val_metrics.get("loss", train_metrics["loss"])
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                save_checkpoint(model, optimizer, scheduler, epoch + 1,
                                {**train_metrics, **val_metrics},
                                Path(args.checkpoint_dir) / "best.pt")

    if is_ddp:
        teardown_ddp()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GN2-Lite training")

    # Dati
    p.add_argument("--h5_path",    required=False, default="../dataset/mc-flavtag-ttbar-small.h5")
    p.add_argument("--val_h5",     default=None,   help="File HDF5 di validazione separato (opzionale)")
    p.add_argument("--stats_path", default="stats.npz")

    # Modello
    p.add_argument("--embed_dim", type=int,   default=128)
    p.add_argument("--n_heads",   type=int,   default=4)
    p.add_argument("--n_layers",  type=int,   default=2)
    p.add_argument("--ffn_dim",   type=int,   default=256)
    p.add_argument("--dropout",   type=float, default=0.1)

    # Training
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=2048)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--val_every",  type=int,   default=1)

    # DataLoader
    p.add_argument("--chunk_size",  type=int, default=100_000)
    p.add_argument("--num_workers", type=int, default=4)

    # Misc
    p.add_argument("--no_cuda",        action="store_true")
    p.add_argument("--check_gpu",      action="store_true")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume",         default=None, help="Path a un checkpoint da cui riprendere")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.check_gpu:
        check_gpu()
        import sys; sys.exit(0)

    train(args)