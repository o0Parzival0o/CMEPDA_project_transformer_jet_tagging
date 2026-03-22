"""
main.py
=======
Entry point del progetto GN2.

Uso:
    python main.py                          # usa configs/default.yaml
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --data data/miofile.h5
    python main.py --eval-only --checkpoint outputs/best_model.pt
"""

import argparse
import logging
import sys
import yaml
import torch
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(save_dir: str = "outputs/"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(save_dir) / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GN2 Jet Flavour Tagger")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path al file di configurazione YAML"
    )
    parser.add_argument(
        "--data", default=None,
        help="Sovrascrive data.h5_path nel config"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Sovrascrive output.save_dir nel config"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Salta il training, esegui solo la valutazione"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path a un checkpoint .pt da caricare (per --eval-only o fine-tuning)"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Carica config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override da CLI
    if args.data:
        cfg["data"]["h5_path"] = args.data
    if args.output_dir:
        cfg["output"]["save_dir"] = args.output_dir

    save_dir = cfg.get("output", {}).get("save_dir", "outputs/")
    logger = setup_logging(save_dir)

    logger.info("=" * 60)
    logger.info("Jet Flavour Tagger")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data:   {cfg['data']['h5_path']}")
    logger.info(f"Output: {save_dir}")
    logger.info("=" * 60)

    # Import qui per evitare import circolari col logging
    from src.dataset  import build_dataloaders
    from src.model    import build_model
    from src.train    import train
    from src.evaluate import run_evaluation

    # ── 1. Dati ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("Caricamento dati...")
        (train_loader, val_loader, test_loader,
         scaler, n_jet_feat, n_track_feat) = build_dataloaders(cfg)

        # Salva scaler per inference futura
        scaler.save(Path(save_dir) / "scaler.pkl")
        logger.info(f"Scaler salvato in {save_dir}/scaler.pkl")

    # ── 2. Modello ───────────────────────────────────────────────────────────
    if args.checkpoint and args.eval_only:
        # Carica da checkpoint
        logger.info(f"Caricamento checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Ricostruisci il modello dalla config salvata nel checkpoint
        ckpt_cfg = ckpt.get("cfg", cfg)
        model = build_model(ckpt_cfg)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Checkpoint caricato (epoch {ckpt.get('epoch', '?')}, "
                    f"val_loss={ckpt.get('val_loss', '?'):.4f})")

        # Per eval-only servono i dati test
        logger.info("Caricamento dati (solo test)...")
        (_, _, test_loader,
         scaler, n_jet_feat, n_track_feat) = build_dataloaders(cfg)

    else:
        model = build_model(cfg, n_jet_features=n_jet_feat,
                            n_track_features=n_track_feat)

        if args.checkpoint:
            # Fine-tuning: carica i pesi ma continua il training
            logger.info(f"Caricamento pesi da {args.checkpoint} per fine-tuning")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])

    # ── 3. Training ──────────────────────────────────────────────────────────
    history = None
    if not args.eval_only:
        logger.info("Avvio training...")
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            save_dir=save_dir,
        )

        # Carica il best model per la valutazione
        best_ckpt = Path(save_dir) / "best_model.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            logger.info(f"Best model caricato per valutazione "
                        f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # ── 4. Valutazione ───────────────────────────────────────────────────────
    logger.info("Valutazione sul test set...")
    run_evaluation(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        save_dir=save_dir,
        history=history,
    )

    logger.info("Completato.")


if __name__ == "__main__":
    main()