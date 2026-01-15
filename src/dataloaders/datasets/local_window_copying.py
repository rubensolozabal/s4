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


def _draw_fixed_windows(
    l_seq: int, l_window_max: int, n_windows: int
) -> List[Tuple[int, int]]:
    """Place equal-length windows evenly across the sequence, left to right."""
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if n_windows * l_window_max > l_seq:
        raise ValueError("Total fixed window length exceeds sequence length")
    gap = (l_seq - n_windows * l_window_max) // (n_windows + 1)
    starts: List[int] = []
    pos = gap
    for _ in range(n_windows):
        starts.append(pos)
        pos += l_window_max + gap
    windows = [(int(s), int(l_window_max)) for s in starts]
    return windows


def _generate_local_window_sample(
    l_seq: int,
    l_window_min: int,
    l_window_max: int,
    dt: float,
    freq: float,
    n_windows_min: int,
    n_windows_max: int,
    query_length: int,
    target_mode: str,
    window_op: str,
    fixed_windows: bool = False,
    ordered_queries: bool = False,
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

    target_mode = target_mode.lower()
    if target_mode not in {"reconstruct", "aggregate"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    if target_mode == "reconstruct" and n_windows_max > query_length:
        raise ValueError("query_length must be at least as large as n_windows_max to hold queries")

    signal = torch.as_tensor(whitesignal(l_seq * dt, dt, freq), dtype=torch.float)

    n_windows = int(np.random.randint(n_windows_min, n_windows_max + 1))
    if fixed_windows:
        if l_window_min != l_window_max:
            raise ValueError("fixed_windows requires l_window_min == l_window_max")
        windows = _draw_fixed_windows(l_seq, l_window_max, n_windows)
    else:
        windows = _draw_non_overlapping_windows(l_seq, l_window_min, l_window_max, n_windows)

    # Targets are padded to the maximum possible number of queries.
    targets = torch.zeros((query_length, l_window_max), dtype=signal.dtype)

    total_len = l_seq + query_length
    markers = torch.zeros(total_len, dtype=signal.dtype)

    # Markers: +id over each window interior; queries remain negative markers in tail.
    for idx, (start, length) in enumerate(windows, start=1):
        markers[start : start + length] = float(idx)

    if target_mode == "aggregate":
        window_op = window_op.lower()
        if window_op == "add":
            aggregate = torch.zeros(l_window_max, dtype=signal.dtype)
        elif window_op == "multiply":
            aggregate = torch.ones(l_window_max, dtype=signal.dtype)
        else:
            raise ValueError(f"Unsupported window_op: {window_op}")

        for start, length in windows:
            segment = signal[start : start + length]
            if window_op == "add":
                padded = torch.zeros(l_window_max, dtype=signal.dtype)
                padded[:length] = segment
                aggregate = aggregate + padded
            else:
                padded = torch.ones(l_window_max, dtype=signal.dtype)
                padded[:length] = segment
                aggregate = aggregate * padded

        markers[l_seq:] = -1.0
        targets[:] = aggregate.unsqueeze(0)
    else:
        # Shuffle the replay order and encode queries with negative markers.
        query_order = np.arange(n_windows) if ordered_queries else np.random.permutation(n_windows)
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
        target_mode: str,
        window_op: str,
        fixed_windows: bool = False,
        ordered_queries: bool = False,
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
        self.target_mode = target_mode
        self.window_op = window_op
        self.fixed_windows = fixed_windows
        self.ordered_queries = ordered_queries

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
            self.target_mode,
            self.window_op,
            self.fixed_windows,
            self.ordered_queries,
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
        target_mode: str,
        window_op: str,
        fixed_windows: bool = False,
        ordered_queries: bool = False,
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
                    target_mode,
                    window_op,
                    fixed_windows,
                    ordered_queries,
                )
                for _ in range(samples)
            ]
        )
        X = torch.stack(xs, dim=0)
        Y = torch.stack(ys, dim=0)
        super().__init__(X, Y)
