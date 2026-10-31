"""k-fold cross-validation training for GeoMind.

Reproduces the evaluation protocol of the paper: stratified k-fold cross-validation
reporting accuracy, precision, recall and F1 (Sec. V-C). Works on the bundled synthetic
dataset out of the box, or on your own SPD-sequence dataset.

Examples
--------
Smoke test on synthetic data::

    python train.py --dataset synthetic --epochs 5 --folds 3

Real data: implement a ``torch.utils.data.Dataset`` returning
``{"input_ids": (L, N, N), "labels": int}`` and plug it into :func:`build_dataset`.
"""

import argparse
import copy
import math
import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import (
    ABIDEFMRIDataset,
    ADNIFMRIDataset,
    DEAPEEGDataset,
    HCIEEGDataset,
    SEEDEEGDataset,
    SyntheticSPDDataset,
    explode_windows,
)
from geomind import GeoMind, GeoMind2, TransformerClassifier

# Source of the SPDNet baseline, whose Riemannian meta-optimizer GeoMind also uses.
CPDNET_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "experiments", "baselines", "cpdnet_src")

# Default on-disk locations for the real cohorts, under ./datasets (see README "Data");
# override with --data_dir/--label_file.
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
DATA_PATHS = {
    "ADNI": {
        "data_dir": os.path.join(DATA_ROOT, "ADNI", "timeseries"),
        "label_file": os.path.join(DATA_ROOT, "ADNI", "labels.csv"),
    },
    "ABIDE": {
        "data_dir": os.path.join(DATA_ROOT, "ABIDE"),
    },
    "SEED": {
        "data_dir": os.path.join(DATA_ROOT, "SEED", "Preprocessed_EEG"),
        "label_file": os.path.join(DATA_ROOT, "SEED", "Preprocessed_EEG", "label.mat"),
    },
    "DEAP": {"data_dir": os.path.join(DATA_ROOT, "DEAP", "trials")},
    "HCI": {"data_dir": os.path.join(DATA_ROOT, "HCI")},
}


def build_dataset(name: str, args):
    """Return a dataset of SPD sequences and its ``(dim, time_points, num_classes)``."""
    if name == "synthetic":
        ds = SyntheticSPDDataset(
            num_samples=args.num_samples,
            num_classes=args.num_classes,
            dim=args.dim,
            time_points=args.time_points,
        )
        return ds, args.dim, args.time_points, args.num_classes

    if name == "ADNI":
        data_dir = args.data_dir or DATA_PATHS["ADNI"]["data_dir"]
        label_file = args.label_file or DATA_PATHS["ADNI"]["label_file"]
        ds = ADNIFMRIDataset(
            data_dir=data_dir,
            label_file=label_file,
            window_size=args.window_size,
            step=args.window_step,
            binary=True,
        )
        # Infer dim and sequence length from the first sample.
        sample = ds[0]["input_ids"]
        length, dim = sample.shape[0], sample.shape[-1]
        print(f"ADNI: {len(ds)} subjects, sequence (L={length}, N={dim}), "
              f"class balance {np.bincount(get_labels(ds))}")
        return ds, dim, length, ds.num_classes

    if name == "ABIDE":
        ds = ABIDEFMRIDataset(
            root_dir=args.data_dir or DATA_PATHS["ABIDE"]["data_dir"],
            window_size=args.window_size,
            step=args.window_step,
        )
        sample = ds[0]["input_ids"]
        length, dim = sample.shape[0], sample.shape[-1]
        print(f"ABIDE: {len(ds)} subjects, sequence (L={length}, N={dim}), "
              f"class balance {np.bincount(get_labels(ds))}")
        return ds, dim, length, ds.num_classes

    if name == "SEED":
        ds = SEEDEEGDataset(
            mat_dir=args.data_dir or DATA_PATHS["SEED"]["data_dir"],
            label_file=args.label_file or DATA_PATHS["SEED"]["label_file"],
            num_windows=args.window_size,
            segments=args.segments,
            window_len=args.window_len or None,
            step=args.eeg_step or None,
            fc_estimator=args.eeg_fc,
            band=args.eeg_band,
            sampling_rate=args.eeg_sampling_rate,
        )
        if args.window_level:
            explode_windows(ds)
        sample = ds[0]["input_ids"]  # (L, N, N), or (K, L, N, N) for multi-band input
        length, dim = sample.shape[-3], sample.shape[-1]
        print(f"SEED: {len(ds)} samples, sequence (L={length}, N={dim}), "
              f"class balance {np.bincount(get_labels(ds))}")
        return ds, dim, length, ds.num_classes

    if name in ("DEAP", "HCI"):
        if name == "DEAP":
            ds = DEAPEEGDataset(args.data_dir or DATA_PATHS["DEAP"]["data_dir"],
                                num_windows=args.window_size, segments=args.segments,
                                window_len=args.window_len or None, step=args.eeg_step or None,
                                target=args.deap_target, fc_estimator=args.eeg_fc,
                                band=args.eeg_band, sampling_rate=args.eeg_sampling_rate,
                                channels=args.eeg_channels)
        else:
            ds = HCIEEGDataset(args.data_dir or DATA_PATHS["HCI"]["data_dir"],
                               num_windows=args.window_size, segments=args.segments,
                               window_len=args.window_len or None, step=args.eeg_step or None,
                               fc_estimator=args.eeg_fc, target=args.hci_target,
                               group_level=args.hci_group_level, band=args.eeg_band,
                               sampling_rate=args.eeg_sampling_rate, channels=args.eeg_channels)
        if args.window_level:
            explode_windows(ds)
        sample = ds[0]["input_ids"]  # (L, N, N), or (K, L, N, N) for multi-band input
        length, dim = sample.shape[-3], sample.shape[-1]
        print(f"{name}: {len(ds)} samples, sequence (L={length}, N={dim}), "
              f"class balance {np.bincount(get_labels(ds))}")
        return ds, dim, length, ds.num_classes

    raise ValueError(
        f"Unknown dataset {name!r}. Add your own SPD-sequence Dataset here "
        "(see data/datasets.py:SlidingWindowFCDataset)."
    )


def get_labels(dataset) -> np.ndarray:
    return np.array([int(dataset[i]["labels"]) for i in range(len(dataset))])


class MemberEnsemble(torch.nn.Module):
    """Average the class probabilities of independently trained members.

    ``inputs[k]`` is the band index fed to member ``k`` for a multi-band ``(B, K, L, N, N)``
    input, or None when every member sees the same ``(B, L, N, N)`` input. Each member gets
    its own loss in :func:`run_epoch`, so members train independently.
    """

    def __init__(self, members, inputs):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.inputs = inputs

    def forward(self, x):
        logits = torch.stack([m(x if i is None else x[:, i])[0]
                              for m, i in zip(self.members, self.inputs)])  # (K, B, C)
        combined = torch.logsumexp(torch.log_softmax(logits, dim=-1), dim=0) - math.log(len(logits))
        return combined, logits


def split_train_validation(train_idx, labels, groups, val_fraction, seed):
    """Split an outer training fold without crossing groups."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("--val_fraction must be in (0, 0.5), or 0 to disable")
    inner_splits = max(2, round(1.0 / val_fraction))
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    inner_groups = groups[train_idx] if groups is not None else np.asarray(train_idx)
    local_train, local_val = next(inner.split(
        np.zeros(len(train_idx)), labels[train_idx], inner_groups))
    return np.asarray(train_idx)[local_train], np.asarray(train_idx)[local_val]


def riemannian_optimizer(base):
    """Wrap ``base`` in the Riemannian meta-optimizer of the SPDNet baseline.

    Around each step of ``base`` it projects the gradients of Stiefel/SPD parameters onto the
    tangent space and retracts the result onto the manifold, and it zeroes NaN gradients.
    """
    sys.path.insert(0, CPDNET_SRC)
    try:
        from spdnet.optimizer import StiefelMetaOptimizer
    finally:
        sys.path.remove(CPDNET_SRC)
    return StiefelMetaOptimizer(base)


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=0.0, step_hook=None):
    """Run one epoch; train if ``optimizer`` is given, else evaluate. Returns metrics.

    ``step_hook`` (e.g. an EMA update) is called after every optimizer step.
    """
    train = optimizer is not None
    model.train(train)
    total, correct, total_loss = 0, 0, 0.0
    labels_all, preds_all = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits, aux = model(x)
            if isinstance(model, MemberEnsemble):
                loss = torch.stack([criterion(member_logits, y) for member_logits in aux]).mean()
            else:
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    # Clip each ensemble member on its own so members stay independent.
                    for part in (model.members if isinstance(model, MemberEnsemble) else [model]):
                        torch.nn.utils.clip_grad_norm_(part.parameters(), grad_clip)
                optimizer.step()
                if step_hook is not None:
                    step_hook()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            labels_all.extend(y.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

    acc = 100.0 * correct / total
    pre = precision_score(labels_all, preds_all, average="weighted", zero_division=0)
    rec = recall_score(labels_all, preds_all, average="weighted", zero_division=0)
    f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(labels_all, preds_all)
    return {"loss": total_loss / len(loader), "acc": acc, "pre": pre, "rec": rec,
            "f1": f1, "macro_f1": macro_f1, "balanced_acc": balanced_acc}


def main():
    parser = argparse.ArgumentParser(description="Train GeoMind with k-fold CV.")
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_attention", action="store_true", help="Ablate the GaA module.")
    parser.add_argument("--seed", type=int, default=0)
    # Real-data options.
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset data dir.")
    parser.add_argument("--label_file", type=str, default=None, help="Override dataset label file.")
    parser.add_argument("--window_size", type=int, default=15, help="Sliding-window length W.")
    parser.add_argument("--window_step", type=int, default=1, help="Sliding-window stride.")
    parser.add_argument("--window_len", type=int, default=0, help="EEG sliding-window length W in timepoints (0 = auto).")
    parser.add_argument("--eeg_step", type=int, default=0, help="EEG sliding-window stride in timepoints (0 = even spacing).")
    parser.add_argument("--eeg_fc", type=str, default="correlation",
                        choices=["correlation", "covariance", "oas", "ledoit_wolf"],
                        help="EEG functional-connectivity estimator.")
    parser.add_argument("--eeg_channels", type=int, default=0,
                        help="DEAP/HCI: keep only the first N channels (32 = the EEG channels; 0 = "
                             "all, which adds DEAP's 8 and HCI's 15 peripheral/auxiliary ones).")
    parser.add_argument("--eeg_band", type=str, default="broadband",
                        help="Optional fourth-order zero-phase EEG band-pass before FC: one of "
                             "broadband/theta/alpha/beta/gamma, or a '+'-joined list "
                             "(e.g. broadband+gamma) for a multi-band ensemble.")
    parser.add_argument("--eeg_sampling_rate", type=float, default=256.0,
                        help="EEG sampling rate in Hz used by band-pass filtering.")
    # Training levers (all default to the original behavior).
    parser.add_argument("--class_weights", action="store_true",
                        help="Use inverse-frequency class weights in the loss.")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Max grad norm (0 = off).")
    parser.add_argument("--readout", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--dt_init", type=float, default=0.1, help="Initial ZOH step of the SSM.")
    parser.add_argument("--head", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--stein_iters", type=int, default=8, help="Stein barycenter iterations.")
    parser.add_argument("--conv_channels", type=int, default=0,
                        help="Multi-channel SPD conv front-end channels (0 = off).")
    parser.add_argument("--conv_layers", type=int, default=2, help="SPD conv front-end layers.")
    parser.add_argument("--max_folds", type=int, default=0, help="Only run the first N folds (0 = all).")
    parser.add_argument("--segments", type=int, default=1,
                        help="Split each EEG trial into N leakage-free segments (more samples).")
    parser.add_argument("--window_level", action="store_true",
                        help="Window-level samples with a random split (not leakage-free).")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (L2).")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing in the cross-entropy loss.")
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"],
                        help="Per-epoch learning-rate schedule.")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear learning-rate warmup epochs.")
    parser.add_argument("--ema", type=float, default=0.0,
                        help="Evaluate an exponential moving average of the weights with this "
                             "per-step decay (0 = off).")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Independently initialized GeoMind members per band, trained "
                             "jointly and averaged in probability space.")
    parser.add_argument("--split_seed", type=int, default=-1,
                        help="Seed of the CV split (-1 = use --seed). Fixing it lets --seed vary "
                             "the initialization while keeping the folds paired.")
    parser.add_argument("--best_epoch", dest="best_epoch", action="store_true", default=True,
                        help="Report the best test epoch per fold (default).")
    parser.add_argument("--no_best_epoch", dest="best_epoch", action="store_false",
                        help="Report the final epoch instead of the best.")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate the test set every N epochs for best-epoch selection.")
    parser.add_argument("--no_group", action="store_true",
                        help="Ignore trial groups: plain random StratifiedKFold (per-window protocol).")
    parser.add_argument("--deap_target", type=int, default=0,
                        help="DEAP label: 0=valence, 1=arousal, 2=dominance, 3=liking.")
    parser.add_argument("--hci_target", type=str, default="valence",
                        choices=["valence", "arousal"], help="HCI emotion target.")
    parser.add_argument("--hci_group_level", type=str, default="trial",
                        choices=["trial", "subject"],
                        help="HCI CV grouping: trial keeps each trial in one fold; subject tests cross-subject generalization.")
    parser.add_argument("--val_fraction", type=float, default=0.0,
                        help="Fraction of each outer training fold reserved for epoch selection (0 = no inner validation split).")
    parser.add_argument("--selection_metric", type=str, default="macro_f1",
                        choices=["acc", "balanced_acc", "macro_f1"],
                        help="Validation metric used to select the checkpoint.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in the classifier head.")
    # Backbone selection: original geometric SSM vs Mamba-backbone GeoMind2.
    parser.add_argument("--model", type=str, default="geomind",
                        choices=["geomind", "geomind2", "mamba", "transformer"])
    parser.add_argument("--d_model", type=int, default=128, help="GeoMind2 Mamba width.")
    parser.add_argument("--n_layers", type=int, default=2, help="GeoMind2 number of Mamba blocks.")
    parser.add_argument("--d_state", type=int, default=16, help="GeoMind2 Mamba state dim.")
    # Synthetic-data options.
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--time_points", type=int, default=12)
    args = parser.parse_args()
    bands = args.eeg_band.split("+")
    for band in bands:
        if band not in ("broadband", "theta", "alpha", "beta", "gamma"):
            parser.error(f"unknown --eeg_band component {band!r}")
    if (len(bands) > 1 or args.ensemble > 1) and args.model != "geomind":
        parser.error("multi-band input and --ensemble are only implemented for --model geomind")
    split_seed = args.seed if args.split_seed < 0 else args.split_seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    dataset, dim, time_points, num_classes = build_dataset(args.dataset, args)
    labels = get_labels(dataset)

    # Leakage-free grouped CV when the dataset defines groups (e.g. EEG trial segments):
    # all segments of one trial stay within the same fold. --no_group forces a plain random
    # StratifiedKFold over samples (per-window protocol; segments of a trial can span folds).
    groups = None if args.no_group else getattr(dataset, "groups", None)
    if groups is not None:
        groups = np.asarray(groups)
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=split_seed)
        split_iter = splitter.split(np.zeros(len(labels)), labels, groups)
        print(f"Grouped CV over {len(np.unique(groups))} trial groups "
              f"({len(labels)} segment samples).")
    else:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=split_seed)
        split_iter = skf.split(np.zeros(len(labels)), labels)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(split_iter):
        if args.max_folds and fold >= args.max_folds:
            break
        print(f"\n=== Fold {fold + 1}/{args.folds} ===")
        fit_idx = train_idx
        val_loader = None
        if args.val_fraction > 0:
            fit_idx, val_idx = split_train_validation(
                train_idx, labels, groups, args.val_fraction, split_seed + fold)
            val_loader = DataLoader(
                Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
            print(f"Inner split: {len(fit_idx)} train / {len(val_idx)} validation; "
                  f"checkpoint metric={args.selection_metric}.")
        train_loader = DataLoader(Subset(dataset, fit_idx), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

        if args.model == "transformer":
            model = TransformerClassifier(
                dim=dim, num_classes=num_classes, time_points=time_points,
                d_model=args.d_model, n_layers=args.n_layers, readout=args.readout,
                dropout=args.dropout,
            ).to(device)
        elif args.model in ("geomind2", "mamba"):
            geometric = args.model == "geomind2"
            model = GeoMind2(
                dim=dim,
                num_classes=num_classes,
                time_points=time_points,
                d_model=args.d_model,
                n_layers=args.n_layers,
                d_state=args.d_state,
                # vanilla Mamba: raw FC features, no geometric attention.
                use_attention=(not args.no_attention) and geometric,
                readout=args.readout,
                dropout=args.dropout,
                geometric=geometric,
            ).to(device)
        else:
            def make_geomind():
                return GeoMind(
                    dim=dim,
                    num_classes=num_classes,
                    time_points=time_points,
                    kernel_size=args.kernel_size,
                    use_attention=not args.no_attention,
                    dt_init=args.dt_init,
                    readout=args.readout,
                    head=args.head,
                    stein_iters=args.stein_iters,
                    dropout=args.dropout,
                    conv_channels=args.conv_channels,
                    conv_layers=args.conv_layers,
                )

            if len(bands) == 1 and args.ensemble == 1:
                model = make_geomind().to(device)
            else:
                inputs = [None if len(bands) == 1 else b
                          for b in range(len(bands)) for _ in range(args.ensemble)]
                model = MemberEnsemble([make_geomind() for _ in inputs], inputs).to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {n_params:,}")

        # Optional inverse-frequency class weights to counter label imbalance.
        weight = None
        if args.class_weights:
            train_labels = labels[fit_idx]
            counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
            inv = counts.sum() / (num_classes * np.maximum(counts, 1.0))
            weight = torch.tensor(inv, dtype=torch.float32, device=device)

        adam = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = adam
        if args.model == "geomind":
            # Riemannian (M-Adam) updates, as for SPDNet. GeoMind's orthogonal gains and SPD
            # kernel taps are generated from unconstrained matrices, so the step equals Adam's.
            optimizer = riemannian_optimizer(adam)
        criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

        scheduler = None
        if args.lr_schedule != "constant" or args.warmup_epochs > 0:
            def lr_factor(epoch):
                if epoch < args.warmup_epochs:
                    return (epoch + 1) / args.warmup_epochs
                if args.lr_schedule == "cosine":
                    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(adam, lr_factor)

        # Optional EMA of the weights; when enabled it replaces the raw model for evaluation.
        eval_model, ema_update = model, None
        if args.ema > 0:
            eval_model = copy.deepcopy(model)
            for p in eval_model.parameters():
                p.requires_grad_(False)
            ema_tensors = list(eval_model.state_dict().values())
            raw_tensors = list(model.state_dict().values())

            def ema_update():
                with torch.no_grad():
                    for e, r in zip(ema_tensors, raw_tensors):
                        if e.dtype.is_floating_point:
                            e.mul_(args.ema).add_(r, alpha=1.0 - args.ema)
                        else:
                            e.copy_(r)

        best = None
        best_state = None
        for epoch in tqdm(range(args.epochs), desc=f"Fold {fold + 1}"):
            run_epoch(model, train_loader, criterion, device, optimizer, grad_clip=args.grad_clip,
                      step_hook=ema_update)
            if scheduler is not None:
                scheduler.step()
            do_eval = args.best_epoch and (
                (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1)
            if do_eval:
                # With an inner validation split the outer test fold remains untouched until
                # training is complete. val_fraction=0 retains the historical test-selected
                # checkpoint solely for backward-compatible reproduction.
                selection_loader = val_loader if val_loader is not None else test_loader
                m = run_epoch(eval_model, selection_loader, criterion, device)
                metric = args.selection_metric if val_loader is not None else "acc"
                if best is None or m[metric] > best[metric]:
                    best = m
                    if val_loader is not None:
                        best_state = copy.deepcopy(eval_model.state_dict())
        if best_state is not None:
            eval_model.load_state_dict(best_state)
        metrics = (run_epoch(eval_model, test_loader, criterion, device)
                   if val_loader is not None or not args.best_epoch else best)
        tag = ("validation-selected" if val_loader is not None else
               ("best-epoch" if args.best_epoch else "final"))
        print(
            f"Fold {fold + 1} test ({tag}): Acc={metrics['acc']:.2f}% "
            f"Pre={metrics['pre']:.4f} Rec={metrics['rec']:.4f} "
            f"F1={metrics['f1']:.4f} MacroF1={metrics['macro_f1']:.4f}"
        )
        fold_metrics.append(metrics)

    print("\n=== Cross-validation summary ===")
    for key, name in [("acc", "Accuracy"), ("balanced_acc", "Balanced accuracy"),
                      ("pre", "Precision"), ("rec", "Recall"),
                      ("f1", "F1"), ("macro_f1", "Macro-F1")]:
        vals = np.array([m[key] for m in fold_metrics])
        print(f"{name}: {vals.mean():.4f} +/- {vals.std():.4f}")


if __name__ == "__main__":
    main()
