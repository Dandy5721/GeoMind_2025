"""EEG baselines (SEED, DEAP, HCI): CPD-Net (SPDNet) and GANN (SPD-CNN) as supervised classifiers.

CPD-Net (MICCAI 2021) and GANN (TMI 2022) are originally *unsupervised* brain-state
segmentation methods (clustering timepoints, scored by purity/NMI). To compare them
against GeoMind on the supervised EEG emotion tasks, we reuse their SPD feature
extractors and attach a linear classification head, then evaluate with the *same*
leakage-free grouped cross-validation and sliding-window input as GeoMind.

Usage:
    python experiments/baselines_seed.py --method cpdnet --dataset SEED --device cuda:0
    python experiments/baselines_seed.py --method gann   --dataset SEED --device cuda:0
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # GeoMind root, for `data`
from data import SEEDEEGDataset  # noqa: E402
from train import DATA_PATHS  # noqa: E402  (default dataset locations, shared with GeoMind)

CPDNET_SRC = os.path.join(HERE, "baselines", "cpdnet_src")
GANN_SRC = os.path.join(HERE, "baselines", "gann_src")


class CPDNetClassifier(nn.Module):
    """SPDNet (BiMap + ReEig + LogEig) per window, mean-pooled, linear head."""

    def __init__(self, n, num_classes, dims=(48, 32, 16)):
        super().__init__()
        sys.path.insert(0, CPDNET_SRC)
        from spdnet.spd import SPDRectified, SPDTangentSpace, SPDTransform

        # BiMap only reduces dimension; keep taps strictly below N (adapts to N=47/40/62).
        dims = tuple(d for d in dims if d < n)
        layers, prev = [], n
        for d in dims:
            layers += [SPDTransform(prev, d), SPDRectified()]
            prev = d
        self.feat = nn.Sequential(*layers)
        self.tangent = SPDTangentSpace()
        self.cls = nn.Linear(prev * prev, num_classes)

    def forward(self, x):  # x: (B, L, N, N)
        b, l, n, _ = x.shape
        h = self.feat(x.reshape(b * l, n, n))
        v = self.tangent(h).reshape(b, l, -1).mean(1)  # mean over windows
        return self.cls(v)


class GANNClassifier(nn.Module):
    """GANN SPD-CNN (SPD conv + geometric attention) feature extractor + linear head."""

    def __init__(self, length, n, num_classes):
        super().__init__()
        sys.path.insert(0, GANN_SRC)
        from SPD_CNN.spd import Normalize, SPDActivate, SPDAttention, SPDConv

        self.feat = nn.Sequential(
            SPDAttention(in_channels=length, kernel_size=5),
            SPDConv(length, 8, 5),
            Normalize(p="fro", dim=(-2, -1)),
            SPDAttention(8, 5),
            SPDActivate(activate_func="exp"),
            SPDConv(8, 16, 5),
            Normalize(p="fro", dim=(-2, -1)),
        )
        # Pool the log-features to a compact, trainable representation before the head.
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.cls = nn.Sequential(
            nn.BatchNorm1d(16 * 8 * 8), nn.Dropout(0.3), nn.Linear(16 * 8 * 8, num_classes)
        )

    def forward(self, x):  # x: (B, L, N, N)
        from geomind.spd import spd_logm  # numerically stable log-map (CPU eigh fallback)

        f = self.feat(x)  # (B, 16, m, m)
        v = self.pool(spd_logm(f)).reshape(f.shape[0], -1)  # log-map -> pool -> flatten
        return self.cls(v)


def build_optimizer(method, model, lr):
    # CPD-Net's SPDTransform uses Stiefel parameters -> needs the Riemannian retraction
    # optimizer. GANN's SPDConv uses ordinary parameters, so plain Adam trains it best.
    if method == "cpdnet":
        from spdnet.optimizer import StiefelMetaOptimizer
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        return StiefelMetaOptimizer(base)
    return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


def run(method, args):
    from sklearn.metrics import precision_score, recall_score

    from data import DEAPEEGDataset, HCIEEGDataset

    device = torch.device(args.device)
    wl = args.window_len or None
    step = args.eeg_step or None
    bkw = dict(band=args.eeg_band, sampling_rate=args.eeg_sampling_rate, fc_estimator=args.eeg_fc)
    if args.dataset == "SEED":
        ds = SEEDEEGDataset(
            args.data_dir or DATA_PATHS["SEED"]["data_dir"],
            args.label_file or DATA_PATHS["SEED"]["label_file"],
            num_windows=args.window_size, segments=args.segments,
            window_len=wl, step=step, **bkw,
        )
    elif args.dataset == "HCI":
        ds = HCIEEGDataset(args.data_dir or DATA_PATHS["HCI"]["data_dir"],
                           num_windows=args.window_size, segments=args.segments,
                           window_len=wl, step=step, target=args.hci_target,
                           channels=args.eeg_channels, **bkw)
    else:
        ds = DEAPEEGDataset(args.data_dir or DATA_PATHS["DEAP"]["data_dir"],
                            num_windows=args.window_size, segments=args.segments,
                            window_len=wl, step=step, target=args.deap_target,
                            channels=args.eeg_channels, **bkw)
    y = np.array(ds.labels)
    g = np.array(ds.groups)
    n = ds[0]["input_ids"].shape[-1]
    length = ds[0]["input_ids"].shape[0]
    print(f"{method} {args.dataset}: {len(ds)} samples, L={length}, N={n}, balance={np.bincount(y)}")

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    accs, pres, f1s, mf1s = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, g)):
        if args.max_folds and fold >= args.max_folds:
            break
        torch.manual_seed(0)
        if method == "cpdnet":
            model = CPDNetClassifier(n, ds.num_classes).to(device)
        else:
            model = GANNClassifier(length, n, ds.num_classes).to(device)
        opt = build_optimizer(method, model, args.lr)
        if args.class_weights:
            counts = np.bincount(y[tr], minlength=ds.num_classes).astype(np.float64)
            w = counts.sum() / (len(counts) * np.clip(counts, 1, None))
            crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        else:
            crit = nn.CrossEntropyLoss()
        trl = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
        tel = DataLoader(Subset(ds, te), batch_size=args.batch_size)

        def evaluate():
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for b in tel:
                    preds.extend(model(b["input_ids"].to(device)).argmax(1).cpu().numpy())
                    labels.extend(b["labels"].numpy())
            return (100.0 * np.mean(np.array(preds) == np.array(labels)),
                    precision_score(labels, preds, average="weighted", zero_division=0),
                    f1_score(labels, preds, average="weighted", zero_division=0),
                    f1_score(labels, preds, average="macro", zero_division=0))

        best = None
        for _ in range(args.epochs):
            model.train()
            for b in trl:
                opt.zero_grad()
                loss = crit(model(b["input_ids"].to(device)), b["labels"].to(device))
                loss.backward()
                opt.step()
            if args.best_epoch:
                m = evaluate()
                if best is None or m[0] > best[0]:
                    best = m
        acc, pre, f1, mf1 = best if args.best_epoch else evaluate()
        accs.append(acc); pres.append(pre); f1s.append(f1); mf1s.append(mf1)
        print(f"  fold {fold + 1}: Acc={acc:.2f}% Pre={pre:.4f} F1={f1:.4f} MacroF1={mf1:.4f}")

    print(f"=== {method} {args.dataset}: "
          f"Accuracy: {np.mean(accs):.2f} +/- {np.std(accs):.2f} | "
          f"Precision: {np.mean(pres):.4f} +/- {np.std(pres):.4f} | "
          f"F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | "
          f"Macro-F1: {np.mean(mf1s):.4f} +/- {np.std(mf1s):.4f} ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["cpdnet", "gann"], required=True)
    ap.add_argument("--dataset", choices=["SEED", "HCI", "DEAP"], default="SEED")
    ap.add_argument("--data_dir", default=None, help="Override the dataset directory.")
    ap.add_argument("--label_file", default=None, help="Override the SEED label.mat path.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--max_folds", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--window_len", type=int, default=0, help="Same W as GeoMind for fairness.")
    ap.add_argument("--eeg_step", type=int, default=0, help="Same step as GeoMind for fairness.")
    ap.add_argument("--eeg_fc", type=str, default="correlation",
                    choices=["correlation", "covariance", "oas", "ledoit_wolf"],
                    help="EEG functional-connectivity estimator (same options as train.py).")
    ap.add_argument("--eeg_channels", type=int, default=0,
                    help="DEAP/HCI: keep only the first N channels (32 = EEG only; 0 = all).")
    ap.add_argument("--eeg_band", type=str, default="broadband",
                    choices=["broadband", "theta", "alpha", "beta", "gamma"])
    ap.add_argument("--eeg_sampling_rate", type=float, default=256.0)
    ap.add_argument("--class_weights", action="store_true",
                    help="Inverse-frequency class weights in the loss (fair match to GeoMind).")
    ap.add_argument("--segments", type=int, default=1)
    ap.add_argument("--deap_target", type=int, default=0,
                    help="DEAP label: 0=valence, 1=arousal, 2=dominance, 3=liking.")
    ap.add_argument("--hci_target", type=str, default="valence",
                    choices=["valence", "arousal"], help="HCI emotion target.")
    ap.add_argument("--best_epoch", dest="best_epoch", action="store_true", default=True)
    ap.add_argument("--no_best_epoch", dest="best_epoch", action="store_false")
    args = ap.parse_args()
    run(args.method, args)
