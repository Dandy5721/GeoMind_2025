# GeoMind

Code for **GeoMind: Dancing with Geometric Brain Rhythm on Symmetric Positive Definite Manifolds**
(IEEE Transactions on Medical Imaging, 2026, [doi:10.1109/TMI.2026.3745506](https://doi.org/10.1109/TMI.2026.3745506)).

GeoMind is a geometric deep state-space model (SSM). It tracks how functional connectivity (FC) evolves
over time as a trajectory on the Riemannian manifold of symmetric positive definite (SPD) matrices.
The state transition and the observation both run *on* the SPD manifold, so every intermediate state is a
valid connectivity matrix. The implementation is pure PyTorch and needs no custom CUDA kernels.

## Method overview

A windowed FC sequence `X = {X_t}` (each `X_t` an `N x N` SPD matrix) is treated as a discrete-time
dynamical system whose latent state also lives on `Sym_N^+`:

- **Conjugation map** `Psi_g(X) = g X g^T`, `g in O(N)`: an SPD-preserving, Stein-isometric "translation" (Eq. 2).
- **Geodesic barycenter** `X ⊞_α Y`: a weighted Stein barycenter that replaces Euclidean `+` and `x` (Eq. 3).
- **Geometric SSM** (Eqs. 5-7):
  - transition `Σ_t = Psi_{A_Δ}(Σ_{t-1}) ⊞_α Psi_{B_Δ}(X_t)`, with `Σ_0 = I_N`
  - readout `Ξ_t = Psi_C(Σ_t) ⊞_β Psi_D(X_t)`
  - `A_Δ, B_Δ` come from a zero-order-hold discretization with a learnable step `Δ`.
- **SPD convolution** with an SPD kernel `H = Z^T Z + εI` (Eq. 4), used as a feature front-end.
- **Geometric-adaptive attention (GaA)**, an interpretable gate `Ω(X) in [0,1]^{N x N}` applied as `Ω(X) ∘ X` (Eq. 11).
- **Classifier**: the readouts are averaged over time, mapped to the tangent space with the matrix
  logarithm, and passed to a linear (or one-hidden-layer MLP) head (Eq. 8).

Every operation is defined independently of `N`, so the same model runs unchanged on fMRI
(regions of an atlas) and EEG (electrodes).

## Repository layout

```
GeoMind_2025/
├── geomind/
│   ├── spd.py               # SPD algebra: logm/expm with stable backward, Stein distance,
│   │                        #   conjugation map (Eq. 2), Stein barycenter (Eq. 3)
│   ├── conv.py              # SPD convolution (Eq. 4) and multi-channel SPD conv front-end
│   ├── attention.py         # Geometric-adaptive attention, GaA (Eq. 11)
│   ├── ssm.py               # Geometric state-space recurrence (Eqs. 5-7)
│   ├── model.py             # End-to-end GeoMind classifier
│   ├── mamba_model.py       # Mamba baseline (optional, needs mamba-ssm)
│   └── transformer_model.py # Transformer baseline
├── data/
│   └── datasets.py          # Sliding-window FC pipeline and loaders for SEED, DEAP, HCI, ADNI, ABIDE
├── experiments/
│   ├── baselines_seed.py    # SPDNet (CPD-Net) and GANN baselines on SEED / DEAP / HCI
│   └── baselines/           # Source of the CPD-Net and GANN baselines
├── train.py                 # k-fold cross-validation training and evaluation
├── demo.py                  # Smoke test, no data required
└── requirements.txt
```

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
# optional, only for the Mamba baseline (--model mamba / geomind2):
pip install mamba-ssm
```

## Quick start

Check the manifold operations and a forward/backward pass (no data needed):

```bash
python demo.py
```

Run cross-validation on the bundled synthetic SPD dataset:

```bash
python train.py --dataset synthetic --folds 5 --epochs 20 --device cpu
```

## Data

The datasets are not redistributed here; download them from their providers. By default `train.py`
looks under `./datasets/`, and every path can be overridden with `--data_dir` (and `--label_file`).

```
datasets/
├── SEED/Preprocessed_EEG/   # official SEED files (*.mat) + label.mat
├── DEAP/trials/             # s01_trials.mat ... s32_trials.mat (see below)
├── HCI/                     # Part_<id>.mat, one file per participant (see below)
├── ADNI/timeseries/         # sub_<id>_aal.txt BOLD time series (T x N)
├── ADNI/labels.csv          # columns: subject_id, DX
└── ABIDE/                   # sub-<control|patient><id>/<folder>_AAL116_features_timeseries.mat
```

- **SEED**: the official `Preprocessed_EEG` folder. Each session file holds `<name>_eeg1` ... `<name>_eeg15`
  arrays (62 channels x T, 200 Hz), and `label.mat` holds the 15 trial labels in `{-1, 0, 1}`.
- **DEAP**: one file per subject holding `trial1` ... `trial40` (each 40 channels x 8064 samples, 128 Hz) and
  `labels` (40 x 4: valence, arousal, dominance, liking). This is the official
  `data_preprocessed_matlab` release split per trial:

  ```python
  import scipy.io as sio
  d = sio.loadmat("s01.mat")  # official DEAP file: data (40, 40, 8064), labels (40, 4)
  out = {f"trial{k + 1}": d["data"][k] for k in range(40)}
  out["labels"] = d["labels"]
  sio.savemat("datasets/DEAP/trials/s01_trials.mat", out)
  ```

- **MAHNOB-HCI**: one file per participant holding `Trial<k>` arrays (all 47 recorded channels x T, 256 Hz)
  and the self-assessment arrays `feltVlnc` and `feltArsl`, where entry `k` is the rating of `Trial<k>`
  (entry 0 is a placeholder).

Ratings above 5 are the high class for DEAP and HCI. All recorded channels are used by default;
`--eeg_channels 32` keeps only the EEG electrodes.

## Reproducing the EEG results (Table 1)

```bash
C="--model geomind --folds 10 --segments 1 --head mlp --readout mean --lr 1e-3 --stein_iters 5 \
   --grad_clip 1.0 --eval_every 3 --epochs 100 --dropout 0.2 --conv_layers 2"
SEED="--dataset SEED --kernel_size 7 --window_size 20 --window_len 400 --eeg_step 2400 \
      --eeg_sampling_rate 200 --batch_size 32 --eeg_band broadband+gamma"
DEAP="--dataset DEAP --kernel_size 3 --window_size 12 --window_len 1200 --eeg_step 0 \
      --eeg_sampling_rate 128 --batch_size 64 --eeg_band broadband+gamma"
HCI="--dataset HCI --kernel_size 17 --window_size 20 --window_len 400 --eeg_step 0 \
     --eeg_sampling_rate 256 --batch_size 32 --eeg_band beta --class_weights"

python train.py $SEED $C                       # SEED
python train.py $DEAP $C                       # DEAP valence
python train.py $DEAP $C --deap_target 1       # DEAP arousal
python train.py $HCI  $C                       # HCI valence
python train.py $HCI  $C --hci_target arousal  # HCI arousal
```

The evaluation protocol is as follows:
- **Splits**: 10-fold `StratifiedGroupKFold`, grouped by trial, so windows from one trial never fall in
  both the training and the test fold.
- **Reported epoch**: each fold is trained for 100 epochs, the held-out fold is evaluated every 3 epochs,
  and the best evaluation is reported. This is the default (`--best_epoch`). Pass `--no_best_epoch` to
  report the final epoch instead, or `--val_fraction 0.2` to select the epoch on an inner validation split.
- **DEAP and HCI scores**: the per-fold average of the valence and arousal runs; the ± is the std over folds.
- **Variation between runs**: GPU kernels are not deterministic, so a rerun typically moves the 10-fold
  mean by about ±0.5 points.

`train.py` prints per-fold accuracy, precision, recall, F1 and macro-F1, then the mean ± std.

## Baselines

```bash
# SPDNet (CPD-Net) on the same windowed input, e.g. DEAP valence:
python experiments/baselines_seed.py --method cpdnet --dataset DEAP --window_size 12 --window_len 1200 \
    --eeg_step 0 --folds 10 --epochs 100 --batch_size 64
# GANN:      --method gann
# Mamba / Transformer backbones on the same input:  python train.py ... --model mamba | transformer
```

## Citation

```bibtex
@article{dan2026geomind,
  title     = {GeoMind: Dancing with Geometric Brain Rhythm on Symmetric Positive Definite Manifolds},
  author    = {Dan, Tingting and Huang, Xinwei and Cao, Yuwei and Ding, Jiaqi and Yang, Yang and Wu, Guorong},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2026},
  pages     = {1--1},
  doi       = {10.1109/TMI.2026.3745506},
  publisher = {IEEE},
  note      = {Early Access}
}
```
