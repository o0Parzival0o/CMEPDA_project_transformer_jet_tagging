# GN2 Jet Flavour Tagger

Implementazione del tagger GN2 descritto in:
> *"Transforming jet flavour tagging at ATLAS"*, Nature Communications (2026) 17:541

Versione semplificata con solo il **jet flavour classification head**
(senza gli obiettivi ausiliari di track origin e vertex grouping).

---

## Struttura del progetto

```
gn2_project/
├── main.py                  ← entry point
├── configs/
│   └── default.yaml         ← tutti gli iperparametri
├── data/
│   └── jets.h5              ← metti qui il tuo file HDF5
├── src/
│   ├── dataset.py           ← caricamento dati, preprocessing, DataLoader
│   ├── model.py             ← architettura GN2 (transformer)
│   ├── discriminant.py      ← D_b, D_c, operating points
│   ├── train.py             ← training loop
│   └── evaluate.py          ← valutazione, plot
└── outputs/                 ← creata automaticamente
    ├── best_model.pt
    ├── last_model.pt
    ├── scaler.pkl
    ├── run.log
    ├── rejection_curves.png
    ├── discriminant_distribution.png
    ├── training_history.png
    ├── test_probs.npy
    └── test_labels.npy
```

---

## Requisiti

Python 3.9+ con:

```
torch>=2.0.0
numpy>=1.24.0
h5py>=3.8.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## Installazione

### 1. Crea un ambiente virtuale

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# oppure: venv\Scripts\activate   # Windows
```

### 2. Installa le dipendenze

**Con GPU (CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy h5py scikit-learn matplotlib pyyaml
```

**Con GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy h5py scikit-learn matplotlib pyyaml
```

**Solo CPU (più lento ma funziona):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py scikit-learn matplotlib pyyaml
```

### 3. Metti il file HDF5 nella cartella `data/`

```bash
cp /percorso/al/tuo/file.h5 data/jets.h5
```

Il file deve avere questa struttura interna:
```
/jets          — variabili jet-level (structured array)
/tracks        — variabili track-level (structured array)
/eventwise     — variabili evento
/truth_hadrons — info truth hadron
```

Se il tuo file ha un nome diverso per il gruppo principale, aggiorna
`configs/default.yaml`.

---

## Esecuzione

### Training completo

```bash
python main.py
```

Usa automaticamente `configs/default.yaml` e salva tutto in `outputs/`.

### Specificare file dati e output

```bash
python main.py --data data/miofile.h5 --output-dir outputs/esperimento1/
```

### Usare un config personalizzato

```bash
python main.py --config configs/mio_config.yaml
```

### Solo valutazione (su checkpoint esistente)

```bash
python main.py --eval-only --checkpoint outputs/best_model.pt
```

---

## Output prodotti

Dopo il training troverai in `outputs/`:

| File | Descrizione |
|---|---|
| `best_model.pt` | Checkpoint con la miglior validation loss |
| `last_model.pt` | Checkpoint dell'ultima epoca |
| `scaler.pkl` | Normalizzatore fittato su train (serve per inference) |
| `run.log` | Log completo del training |
| `rejection_curves.png` | c-jet e light-jet rejection vs b-jet efficiency |
| `discriminant_distribution.png` | Distribuzione di D_b per classe |
| `training_history.png` | Train/val loss e learning rate per epoca |
| `test_probs.npy` | Probabilità [pb, pc, pu, ptau] sul test set |
| `test_labels.npy` | Label vere sul test set |

---

## Configurazione

Tutti i parametri si trovano in `configs/default.yaml`.
I più importanti da modificare:

```yaml
data:
  h5_path: "data/jets.h5"    # percorso al file HDF5
  max_tracks: 40              # paper GN2: 40

training:
  max_epochs: 100
  batch_size: 12000           # paper GN2: 12000; riduci se OOM
  device: "auto"              # "cuda", "cpu", o "auto"

model:
  transformer_n_layers: 4     # paper GN2: 4
  transformer_n_heads: 8      # paper GN2: 8
```

### Out of Memory (GPU)

Se ricevi errori CUDA OOM, riduci il batch size:
```yaml
training:
  batch_size: 4096   # o anche 1024
```

---

## Feature usate

### Jet-level (njf = 2)
- `pt_btagJes` — pT calibrato (log-trasformato)
- `eta_btagJes` — η calibrato

### Track-level (ntf = 24)
Le feature più discriminanti secondo il paper sono:
- `lifetimeSignedD0Significance` — d₀/σ(d₀) con segno lifetime
- `lifetimeSignedZ0SinThetaSignificance` — z₀sinθ/σ con segno lifetime

Tutte le 24 feature track sono listate in `configs/default.yaml`.

### Target
- `HadronConeExclTruthLabelID`: PDG ID → classe
  - 5 → b-jet (0)
  - 4 → c-jet (1)
  - 0 → light-jet (2)
  - 15 → τ-jet (3)

---

## Architettura (fedele al paper)

```
Per ogni jet (B jet nel batch):

  [jet_pt, jet_eta]               (2 feature)
  [40 tracce × 24 feature]        + maschera booleana
        │
        ▼
  Concatena jet features a ogni traccia → (B, 40, 26)
        │
        ▼
  Track Initialiser MLP: 26 → 256 → 256   (hidden=256)
        │
        ▼
  Transformer Encoder × 4 layer
  (8 heads, embed=256, ffn=512, preLayerNorm, attention masking)
        │
        ▼
  Proiezione: 256 → 128
        │
        ▼
  Attention Pooling → (B, 128)   [jet representation]
        │
        ▼
  Classification Head: 128 → 64 → 32 → 4
        │
        ▼
  Logit → softmax → [pb, pc, pu, pτ]
        │
        ▼
  D_b = log[ pb / (0.2·pc + 0.05·pτ + 0.75·pu) ]
```