"""Datasets that turn raw neural time series into sequences of SPD connectivity matrices.

Following Sec. V-B of the paper, a single sliding-window pipeline serves both modalities:
for each scan/trial with ``N`` channels and ``T`` time points, a windowed correlation
(fMRI) or covariance (EEG) matrix is computed at every window, yielding a sequence
``X = {X_t} in R^{T x N x N}`` of SPD matrices. A tiny diagonal shrinkage guarantees
positive definiteness regardless of the window-to-N ratio.

Each item is returned as ``{"input_ids": (L, N, N) float tensor, "labels": int tensor}``,
matching the interface expected by :mod:`train`.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "SlidingWindowFCDataset",
    "SyntheticSPDDataset",
    "ADNIFMRIDataset",
    "ABIDEFMRIDataset",
    "SEEDEEGDataset",
    "DEAPEEGDataset",
    "HCIEEGDataset",
]


def explode_windows(ds) -> None:
    """Convert a sequence dataset into window-level samples (in place).

    Each ``(L, N, N)`` sequence is split into ``L`` independent ``(1, N, N)`` samples that
    share the sequence's label, and grouping is dropped. WARNING: with a random (non-grouped)
    train/test split this leaks information, because windows from the same trial are highly
    correlated and may land in both splits, so the resulting accuracy is not leakage-free.
    """
    new_samples, new_labels = [], []
    for seq, label in zip(ds.samples, ds.labels):
        for window in seq:
            new_samples.append(window[None])  # (1, N, N)
            new_labels.append(label)
    ds.samples, ds.labels = new_samples, new_labels
    ds.groups = None  # force a random split -> the (leaky) window-level protocol


def segment_signal(signal: np.ndarray, segments: int) -> list:
    """Split an ``(N, T)`` signal into ``segments`` non-overlapping time chunks.

    Used to turn one trial into several independent sequence samples (more training data)
    while keeping every chunk traceable to its source trial for leakage-free grouped CV.
    """
    if segments <= 1:
        return [signal]
    size = signal.shape[1] // segments
    return [signal[:, i * size : (i + 1) * size] for i in range(segments)]


def eeg_window_sequence(
    signal: np.ndarray,
    num_windows: int,
    window_len: int | None = None,
    step: int | None = None,
    fc_estimator: str = "correlation",
    band: str = "broadband",
    sampling_rate: float = 256.0,
) -> np.ndarray:
    """Turn an ``(N, T)`` EEG trial into a fixed-length sequence of SPD covariance matrices.

    The three sliding-window controls are independent (Sec. V-B):

    - ``window_len`` (``W``): timepoints per covariance. Larger ``W`` -> smoother,
      better-conditioned covariance.
    - ``step``: stride between consecutive windows. If given, windows start at
      ``0, step, 2*step, ...`` (the classic sliding window); window starts are clamped so
      every window fits inside the trial.
    - ``num_windows`` (``L``): number of windows / SSM states. Always exactly ``L`` windows
      are returned so trials of different length batch together.

    If ``step`` is None the ``L`` windows are spread evenly across the trial; if
    ``window_len`` is None it defaults to ``T // (L + 1)`` (kept larger than ``N``).
    A small diagonal shrinkage guarantees positive definiteness. Returns ``(L, N, N)``.
    A ``"+"``-joined ``band`` (e.g. ``"broadband+gamma"``) returns one sequence per band,
    stacked to ``(K, L, N, N)``.
    """
    if "+" in band:
        return np.stack([
            eeg_window_sequence(signal, num_windows, window_len, step, fc_estimator, b,
                                sampling_rate)
            for b in band.split("+")
        ])
    bands = {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    if band != "broadband":
        if band not in bands:
            raise ValueError(f"Unknown EEG band: {band!r}")
        from scipy.signal import butter, sosfiltfilt
        low, high = bands[band]
        if high >= sampling_rate / 2:
            raise ValueError(f"Band {band} upper edge must be below Nyquist frequency")
        sos = butter(4, (low, high), btype="bandpass", fs=sampling_rate, output="sos")
        signal = sosfiltfilt(sos, np.nan_to_num(signal), axis=-1)

    if fc_estimator != "correlation":
        # Covariance-type estimators keep each channel's scale, so channels recorded in very
        # different units (e.g. the peripheral signals in DEAP and HCI) dominate the matrix and
        # destabilize training. Standardize every channel over the whole trial first; unlike
        # correlation, the changes in variance from window to window are kept.
        signal = np.nan_to_num(signal)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        signal = np.divide(signal - mean, std, out=np.zeros_like(signal), where=std > 1e-12)

    n, t = signal.shape
    w = window_len if window_len is not None else max(n + 1, t // (num_windows + 1))
    w = min(max(w, n + 1), t)  # at least N+1 (full rank), at most T
    if step is not None and step > 0:
        starts = np.minimum(np.arange(num_windows) * step, max(0, t - w))
    else:
        starts = np.linspace(0, max(0, t - w), num_windows).astype(int)
    if fc_estimator not in ("correlation", "covariance", "oas", "ledoit_wolf"):
        raise ValueError(f"Unknown EEG FC estimator: {fc_estimator!r}")
    mats = []
    for s in starts:
        window = np.nan_to_num(signal[:, int(s) : int(s) + w])
        if fc_estimator == "correlation":
            # Pearson correlation as a Gram matrix. Unlike np.corrcoef followed by
            # nan_to_num, this remains positive semidefinite when a filtered EEG channel
            # has (near-)zero variance. Such channels map to a zero row/column.
            centered = window - window.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(centered, axis=1, keepdims=True)
            standardized = np.divide(
                centered, norm, out=np.zeros_like(centered), where=norm > 1e-12)
            c = standardized @ standardized.T
        elif fc_estimator == "covariance":
            c = np.cov(window)
        else:
            # sklearn estimators expect observations x features.
            if fc_estimator == "oas":
                from sklearn.covariance import OAS
                c = OAS(assume_centered=False).fit(window.T).covariance_
            else:
                from sklearn.covariance import LedoitWolf
                c = LedoitWolf(assume_centered=False).fit(window.T).covariance_
        c = np.nan_to_num(c)
        if fc_estimator != "correlation":
            # EEG amplitude units and trial-wise gains vary widely. Without scale control,
            # covariance values can overflow the quadratic SPD convolution kernel. Normalize
            # the trace to N (mean eigenvalue 1), retaining relative channel variances and
            # connectivity while putting every trial on the same numerical scale.
            scale = np.trace(c) / n
            if np.isfinite(scale) and scale > 1e-12:
                c = c / scale
        mats.append(0.5 * (c + c.T) + 1e-4 * np.eye(n))
    return np.stack(mats).astype(np.float32)


def sliding_window_fc(
    signal: np.ndarray,
    window_size: int,
    step: int = 1,
    mode: str = "correlation",
    shrinkage: float = 1e-4,
) -> np.ndarray:
    """Build a sequence of SPD connectivity matrices from an ``(N, T)`` signal.

    Args:
        signal: array of shape ``(N, T)`` (channels/regions by time).
        window_size: number of time points per window ``W``.
        step: stride between consecutive windows.
        mode: ``"correlation"`` (fMRI) or ``"covariance"`` (EEG).
        shrinkage: diagonal ridge added for positive definiteness (Ledoit--Wolf style).

    Returns:
        Array of shape ``(L, N, N)`` of SPD matrices.
    """
    n, t = signal.shape
    mats = []
    for start in range(0, t - window_size + 1, step):
        window = signal[:, start : start + window_size]
        if mode == "correlation":
            m = np.corrcoef(window)
        else:
            m = np.cov(window)
        m = np.nan_to_num(m)
        m = m + shrinkage * np.eye(n)
        mats.append(m)
    return np.stack(mats).astype(np.float32)


class SlidingWindowFCDataset(Dataset):
    """Wrap a list of raw ``(N, T)`` signals + labels into SPD-sequence samples.

    This is a modality-agnostic adapter: load your fMRI/EEG signals however you like,
    pass them in as a list of ``(N, T)`` arrays with matching integer labels, and the
    dataset yields the SPD sequences GeoMind consumes.
    """

    def __init__(
        self,
        signals: list[np.ndarray],
        labels: list[int],
        window_size: int = 15,
        step: int = 1,
        mode: str = "correlation",
        shrinkage: float = 1e-4,
    ):
        if len(signals) != len(labels):
            raise ValueError("signals and labels must have the same length")
        self.samples = [
            sliding_window_fc(s, window_size, step, mode, shrinkage) for s in signals
        ]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.from_numpy(self.samples[idx]).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.int64),
        }


class ADNIFMRIDataset(Dataset):
    """Alzheimer's Disease Neuroimaging Initiative (ADNI) resting-state fMRI cohort.

    Each subject is an ``AAL``-parcellated BOLD time series (``T x N``). Following the
    paper, diagnostic labels are grouped into a ``CN-like`` class (CN, SMC, EMCI) and an
    ``AD-like`` class (LMCI, AD). The BOLD signal is truncated to ``time_points`` frames
    and turned into a sequence of windowed correlation (SPD) matrices (Sec. V-B).

    Args:
        data_dir: directory of ``sub_<id>_aal.txt`` BOLD time-series files.
        label_file: CSV with ``subject_id`` and ``DX`` columns.
        window_size: sliding-window length ``W`` (default 15, as in the paper).
        step: stride between consecutive windows (use >1 to shorten the sequence).
        time_points: number of leading BOLD frames to keep.
        binary: if True use the CN-like / AD-like 2-class grouping, else 4 classes.
    """

    _CN_LIKE = {"CN", "SMC", "EMCI"}
    _AD_LIKE = {"LMCI", "AD"}
    _FOUR_CLASS = {"CN": 0, "SMC": 0, "EMCI": 1, "LMCI": 2, "AD": 3}

    def __init__(
        self,
        data_dir: str,
        label_file: str,
        window_size: int = 15,
        step: int = 1,
        time_points: int = 140,
        binary: bool = True,
    ):
        import pandas as pd

        self.window_size = window_size
        self.step = step
        self.time_points = time_points
        labels_df = pd.read_csv(label_file)

        self.files, self.labels = [], []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.startswith("sub_"):
                continue
            subject_id = fname.split("_")[1]
            row = labels_df[labels_df["subject_id"] == subject_id]
            if row.empty:
                continue
            dx = row.iloc[0]["DX"]
            if binary:
                if dx in self._CN_LIKE:
                    label = 0
                elif dx in self._AD_LIKE:
                    label = 1
                else:
                    continue
            else:
                if dx not in self._FOUR_CLASS:
                    continue
                label = self._FOUR_CLASS[dx]
            self.files.append(os.path.join(data_dir, fname))
            self.labels.append(label)

        self.num_classes = 2 if binary else 4

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        signal = np.loadtxt(self.files[idx])[: self.time_points]  # (T, N)
        seq = sliding_window_fc(
            signal.T, self.window_size, self.step, mode="correlation"
        )  # (L, N, N)
        return {
            "input_ids": torch.from_numpy(np.nan_to_num(seq)).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.int64),
        }


class ABIDEFMRIDataset(Dataset):
    """Autism Brain Imaging Data Exchange (ABIDE) resting-state fMRI cohort.

    Each subject folder holds an ``AAL116`` BOLD time series; the binary label is encoded
    in the folder name (``control`` -> 0, ``patient`` -> 1). The BOLD signal is truncated
    to ``time_points`` frames and converted to a sequence of windowed correlation (SPD)
    matrices (Sec. V-B).

    Args:
        root_dir: directory containing ``sub-<control|patient><id>/`` folders.
        atlas: parcellation tag used to pick the time-series file (default ``AAL116``).
        window_size: sliding-window length ``W``.
        step: stride between consecutive windows.
        time_points: number of leading BOLD frames to keep (sequences are truncated).
    """

    def __init__(
        self,
        root_dir: str,
        atlas: str = "AAL116",
        window_size: int = 15,
        step: int = 1,
        time_points: int = 137,
    ):
        from scipy.io import loadmat

        self._loadmat = loadmat
        self.window_size = window_size
        self.step = step
        self.time_points = time_points
        # ABIDE scans have variable length; fix the number of windows per subject so every
        # sequence has the same length L and can be batched (adaptive per-subject striding).
        self.num_windows = (time_points - window_size) // step + 1

        self.files, self.labels = [], []
        for sub in sorted(os.listdir(root_dir)):
            sub_dir = os.path.join(root_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            ts_file = os.path.join(sub_dir, f"{sub}_{atlas}_features_timeseries.mat")
            if not os.path.exists(ts_file):
                continue
            if "control" in sub:
                label = 0
            elif "patient" in sub:
                label = 1
            else:
                continue
            self.files.append(ts_file)
            self.labels.append(label)

        self.num_classes = 2

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        signal = self._loadmat(self.files[idx])["data"][: self.time_points].T  # (N, T)
        n, t = signal.shape
        w = min(self.window_size, t)
        # Fixed number of windows via evenly spaced (adaptive) start positions.
        starts = np.linspace(0, max(0, t - w), self.num_windows).astype(int)
        mats = []
        for s in starts:
            c = np.corrcoef(signal[:, s : s + w])
            mats.append(np.nan_to_num(c) + 1e-4 * np.eye(n))
        seq = np.stack(mats).astype(np.float32)
        return {
            "input_ids": torch.from_numpy(seq).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.int64),
        }


class SEEDEEGDataset(Dataset):
    """SEED EEG emotion-recognition dataset (3 classes: negative / neutral / positive).

    Each ``.mat`` session file holds up to 15 trials keyed ``*_eeg1`` .. ``*_eeg15``, each
    an ``(N_channels, T)`` EEG recording. Labels (one per trial index) are loaded from a
    shared ``label.mat`` and shifted from ``{-1, 0, 1}`` to ``{0, 1, 2}``. Following the
    paper, each trial is turned into a sequence of windowed covariance (SPD) matrices.

    Args:
        mat_dir: directory of SEED session ``.mat`` files.
        label_file: ``label.mat`` with a ``label`` array of length 15.
        num_windows: number of windows per trial (controls the adaptive window length).
    """

    def __init__(self, mat_dir: str, label_file: str, num_windows: int = 10, segments: int = 1,
                 window_len: int | None = None, step: int | None = None,
                 fc_estimator: str = "correlation", band: str = "broadband",
                 sampling_rate: float = 256.0):
        from scipy.io import loadmat

        self.num_windows = num_windows
        base_labels = loadmat(label_file)["label"].reshape(-1) + 1  # {-1,0,1} -> {0,1,2}

        self.samples, self.labels, self.groups = [], [], []
        group_id = 0
        for fname in sorted(os.listdir(mat_dir)):
            if not fname.endswith(".mat"):
                continue
            mat = loadmat(os.path.join(mat_dir, fname))
            for i in range(1, 16):
                keys = [k for k in mat if k.endswith(f"eeg{i}")]
                if not keys:
                    continue
                for seg in segment_signal(mat[keys[0]], segments):
                    self.samples.append(eeg_window_sequence(
                        seg, num_windows, window_len, step, fc_estimator, band, sampling_rate))
                    self.labels.append(int(base_labels[i - 1]))
                    self.groups.append(group_id)
                group_id += 1
        self.num_classes = 3

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.from_numpy(self.samples[idx]).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.int64),
        }


class DEAPEEGDataset(Dataset):
    """DEAP EEG emotion dataset (binary valence: low <= 5 < high).

    Each subject file holds 40 trials keyed ``trial1`` .. ``trial40``, every trial an
    ``(N_channels, T)`` recording, with a ``labels`` array whose first column is the
    valence rating. Each trial becomes a fixed-length sequence of windowed covariance
    (SPD) matrices.

    Args:
        mat_dir: directory of ``s*_trials.mat`` subject files.
        num_windows: number of windows per trial (sequence length L).
        target: label column, 0 = valence (default), 1 = arousal.
    """

    def __init__(self, mat_dir: str, num_windows: int = 10, target: int = 0, segments: int = 1,
                 window_len: int | None = None, step: int | None = None,
                 fc_estimator: str = "correlation", band: str = "broadband",
                 sampling_rate: float = 256.0, channels: int = 0):
        from scipy.io import loadmat

        self.num_windows = num_windows
        self.samples, self.labels, self.groups = [], [], []
        group_id = 0
        for fname in sorted(os.listdir(mat_dir)):
            if not fname.endswith(".mat"):
                continue
            mat = loadmat(os.path.join(mat_dir, fname))
            if "labels" not in mat:
                continue
            ratings = mat["labels"][:, target]
            for i in range(1, 41):
                key = f"trial{i}"
                if key not in mat:
                    continue
                label = 0 if ratings[i - 1] <= 5 else 1
                signal = mat[key][:channels] if channels else mat[key]
                for seg in segment_signal(signal, segments):
                    self.samples.append(eeg_window_sequence(
                        seg, num_windows, window_len, step, fc_estimator, band, sampling_rate))
                    self.labels.append(label)
                    self.groups.append(group_id)
                group_id += 1
        self.num_classes = 2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.from_numpy(self.samples[idx]).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.int64),
        }


class HCIEEGDataset(Dataset):
    """MAHNOB-HCI EEG emotion dataset (binary valence: low <= 5 < high).

    Each participant file holds ``Trial<k>`` keys (``(N_channels, T)`` recordings) and a
    ``feltVlnc`` array of per-trial valence ratings (its first entry is a placeholder, so
    ``Trial<k>`` maps to ``feltVlnc[k]``). Each trial becomes a fixed-length sequence of
    windowed covariance (SPD) matrices.

    Args:
        mat_dir: directory of ``Part_*.mat`` participant files.
        num_windows: number of windows per trial (sequence length L).
    """

    def __init__(self, mat_dir: str, num_windows: int = 10, segments: int = 1,
                 window_len: int | None = None, step: int | None = None,
                 fc_estimator: str = "correlation", target: str = "valence",
                 group_level: str = "trial", band: str = "broadband",
                 sampling_rate: float = 256.0, channels: int = 0):
        from scipy.io import loadmat

        if target not in ("valence", "arousal"):
            raise ValueError("HCI target must be 'valence' or 'arousal'")
        if group_level not in ("trial", "subject"):
            raise ValueError("HCI group_level must be 'trial' or 'subject'")
        self.num_windows = num_windows
        self.samples, self.labels, self.groups = [], [], []
        trial_group_id = 0
        subject_id = 0
        for fname in sorted(os.listdir(mat_dir)):
            if not fname.endswith(".mat"):
                continue
            mat = loadmat(os.path.join(mat_dir, fname))
            rating_key = "feltVlnc" if target == "valence" else "feltArsl"
            if rating_key not in mat:
                continue
            ratings = [str(v).strip() for v in mat[rating_key].reshape(-1)]
            for key in mat:
                if not key.startswith("Trial"):
                    continue
                k = int(key[5:])
                try:
                    rating = int(ratings[k])
                except (ValueError, IndexError):
                    continue
                label = 0 if rating <= 5 else 1
                signal = mat[key][:channels] if channels else mat[key]
                for seg in segment_signal(signal, segments):
                    self.samples.append(eeg_window_sequence(
                        seg, num_windows, window_len, step, fc_estimator, band, sampling_rate))
                    self.labels.append(label)
                    self.groups.append(subject_id if group_level == "subject" else trial_group_id)
                trial_group_id += 1
            subject_id += 1
        self.num_classes = 2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.from_numpy(self.samples[idx]).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.int64),
        }


class SyntheticSPDDataset(Dataset):
    """Synthetic SPD-sequence dataset for smoke tests and demos.

    Each class is associated with a random SPD "template"; samples are noisy SPD
    trajectories drawn around their class template, so a working model should separate
    them well above chance. No external data files are required.
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_classes: int = 2,
        dim: int = 20,
        time_points: int = 12,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.time_points = time_points

        templates = []
        for _ in range(num_classes):
            a = rng.standard_normal((dim, dim))
            templates.append(a @ a.T + dim * np.eye(dim))

        self.samples, self.labels = [], []
        for i in range(num_samples):
            cls = i % num_classes
            seq = []
            for _ in range(time_points):
                noise = rng.standard_normal((dim, dim)) * 0.3
                noise = noise @ noise.T
                m = templates[cls] + noise + 1e-3 * np.eye(dim)
                seq.append(0.5 * (m + m.T))
            self.samples.append(np.stack(seq).astype(np.float32))
            self.labels.append(cls)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.from_numpy(self.samples[idx]).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.int64),
        }
