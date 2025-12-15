import numpy as np
import torch
from typing import List, Tuple

from src.dataloaders.utils.signal import whitesignal


def _draw_non_overlapping_windows(
    l_seq: int, l_window_min: int, l_window_max: int, n_windows: int
) -> List[Tuple[int, int]]:
    """Sample non-overlapping windows (start, length) within the sequence."""
    windows: List[Tuple[int, int]] = []
    max_attempts = 1000 * n_windows  # generous budget because l_seq is large
    attempts = 0
    while len(windows) < n_windows and attempts < max_attempts:
        window_len = int(np.random.randint(l_window_min, l_window_max + 1))
        start = int(np.random.randint(0, l_seq - window_len + 1))
        if not any(start < s + length and s < start + window_len for s, length in windows):
            windows.append((start, window_len))
        attempts += 1
    if len(windows) < n_windows:
        raise RuntimeError(f"Could not place {n_windows} non-overlapping windows in {l_seq} steps")
    # Number windows left-to-right for consistent labels
    return sorted(windows, key=lambda pair: pair[0])


def _generate_local_window_sample(
    l_seq: int,
    l_window_min: int,
    l_window_max: int,
    dt: float,
    freq: float,
    n_windows_min: int,
    n_windows_max: int,
    query_length: int,
):
    """Create a single local copying sample with multiple marked windows and replay queries."""
    if l_window_min <= 0 or l_window_min > l_window_max:
        raise ValueError("Require 0 < l_window_min <= l_window_max")
    if l_window_max > l_seq:
        raise ValueError("Window length must not exceed l_seq")
    if n_windows_min <= 0 or n_windows_min > n_windows_max:
        raise ValueError("Require 0 < n_windows_min <= n_windows_max")
    if n_windows_max * l_window_min > l_seq:
        raise ValueError("Total minimum window length exceeds sequence length")
    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if n_windows_max > query_length:
        raise ValueError("query_length must be at least as large as n_windows_max to hold queries")

    signal = torch.as_tensor(whitesignal(l_seq * dt, dt, freq), dtype=torch.float)

    n_windows = int(np.random.randint(n_windows_min, n_windows_max + 1))
    windows = _draw_non_overlapping_windows(l_seq, l_window_min, l_window_max, n_windows)

    # Targets are padded to the maximum possible number of queries.
    targets = torch.zeros((query_length, l_window_max), dtype=signal.dtype)

    total_len = l_seq + query_length
    markers = torch.zeros(total_len, dtype=signal.dtype)

    # Markers: +id over each window interior; queries remain negative markers in tail.
    for idx, (start, length) in enumerate(windows, start=1):
        markers[start : start + length] = float(idx)

    # Shuffle the replay order and encode queries with negative markers.
    query_order = np.random.permutation(n_windows)
    for query_idx, window_idx in enumerate(query_order):
        start, length = windows[int(window_idx)]
        markers[l_seq + query_idx] = -(float(window_idx) + 1.0)
        targets[query_idx, :length] = signal[start : start + length]

    padded_signal = torch.cat([signal, torch.zeros(query_length, dtype=signal.dtype)], dim=0)
    x = torch.stack([padded_signal, markers], dim=-1)
    return x, targets


class LocalWindowCopyingTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: int,
        l_seq: int,
        l_window_min: int,
        l_window_max: int,
        dt: float,
        freq: float,
        n_windows_min: int,
        n_windows_max: int,
        query_length: int,
    ):
        super().__init__()
        self.samples = samples
        self.l_seq = l_seq
        self.l_window_min = l_window_min
        self.l_window_max = l_window_max
        self.dt = dt
        self.freq = freq
        self.n_windows_min = n_windows_min
        self.n_windows_max = n_windows_max
        self.query_length = query_length

    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        return _generate_local_window_sample(
            self.l_seq,
            self.l_window_min,
            self.l_window_max,
            self.dt,
            self.freq,
            self.n_windows_min,
            self.n_windows_max,
            self.query_length,
        )

    def __len__(self):
        return self.samples


class LocalWindowCopyingEvalDataset(torch.utils.data.TensorDataset):
    def __init__(
        self,
        samples: int,
        l_seq: int,
        l_window_min: int,
        l_window_max: int,
        dt: float,
        freq: float,
        n_windows_min: int,
        n_windows_max: int,
        query_length: int,
    ):
        xs, ys = zip(
            *[
                _generate_local_window_sample(
                    l_seq,
                    l_window_min,
                    l_window_max,
                    dt,
                    freq,
                    n_windows_min,
                    n_windows_max,
                    query_length,
                )
                for _ in range(samples)
            ]
        )
        X = torch.stack(xs, dim=0)
        Y = torch.stack(ys, dim=0)
        super().__init__(X, Y)
