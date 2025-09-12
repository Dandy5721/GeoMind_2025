
# GeoMind_2025

Minimal, reproducible code for **GeoMind** and a set of **baseline comparison methods**. The repository is organised as:

### Repository layout

```text
GeoMind_2025/
├─ Comparsion_methods/
│  ├─ Datasets_load.py        # Baseline data loader / split utilities
│  ├─ help_funs.py            # Common helpers used by baselines
│  ├─ help_train.py           # Training/evaluation loops for baselines
│  ├─ main.py                 # Entry point to run all baselines
│  └─ models.py               # Baseline model definitions
├─ GeoMind/
│  ├─ dataset.py              # Dataset class(es) for GeoMind
│  ├─ geo_mind.py             # GeoMind model and building blocks
│  ├─ geomind_train.py        # Entry point for training GeoMind
│  ├─ optimizer.py            # Optimiser / scheduler setup
│  ├─ spd.py                  # SPD utils
│  ├─ train_disease_10fold.py # Convenience script for 10-fold experiments
│  └─ utils.py                # Logging, metrics, seeding, misc utils
└─ conda_requirements.txt     # Environment spec
```



## Environment

Create and activate the environment. The `conda_requirements.txt` may be either an explicit spec or a simple list; use one of the following:

```bash
# A) explicit conda spec
conda create -n geomind --file conda_requirements.txt
conda activate geomind

# B) requirements list
conda create -n geomind python=3.10 -y
conda activate geomind
pip install -r conda_requirements.txt
```

## Data
```text
GeoMind_2025/data/
├── HCPWM/
├── ADNI/
├── OASIS/
├── PPMI/
├── ABIDE/
├── SEED/
├── DEAP/
└── HCI/
```

## Train GeoMind
```text
python GeoMind/geomind_train.py \
  --dataset ADNI \
  --data_root ./data/ADNI \
  --epochs 300 \
  --layer 2 \
  --lr 0.00005 \
  --hidden 360 \
  --gpu cuda:0
```
