import numpy as np
import torch

from src.dataloaders.utils.signal import whitesignal


def _generate_local_window_sample(
    l_seq: int,
    l_window_min: int,
    l_window_max: int,
    dt: float,
    freq: float,
    query_length: int,
):
    """Create a single local copying sample."""
    if l_window_min <= 0 or l_window_min > l_window_max:
        raise ValueError("Require 0 < l_window_min <= l_window_max")
    if l_window_max > l_seq:
        raise ValueError("Window length must not exceed l_seq")
    if query_length <= 0:
        raise ValueError("query_length must be positive")

    signal = torch.as_tensor(whitesignal(l_seq * dt, dt, freq), dtype=torch.float)

    window_len = int(np.random.randint(l_window_min, l_window_max + 1))
    start = int(np.random.randint(0, l_seq - window_len + 1))

    target = torch.zeros(l_window_max, dtype=signal.dtype)
    target[:window_len] = signal[start : start + window_len]

    # Second channel carries markers: +1 over the segment, -1 for the final query step
    total_len = l_seq + query_length
    markers = torch.zeros(total_len, dtype=signal.dtype)
    markers[start : start + window_len] = 1.0
    markers[-query_length:] = -1.0

    padded_signal = torch.cat([signal, torch.zeros(query_length, dtype=signal.dtype)], dim=0)
    x = torch.stack([padded_signal, markers], dim=-1)
    return x, target


class LocalWindowCopyingTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: int,
        l_seq: int,
        l_window_min: int,
        l_window_max: int,
        dt: float,
        freq: float,
        query_length: int,
    ):
        super().__init__()
        self.samples = samples
        self.l_seq = l_seq
        self.l_window_min = l_window_min
        self.l_window_max = l_window_max
        self.dt = dt
        self.freq = freq
        self.query_length = query_length

    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        return _generate_local_window_sample(
            self.l_seq,
            self.l_window_min,
            self.l_window_max,
            self.dt,
            self.freq,
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
        query_length: int,
    ):
        xs, ys = zip(
            *[
                _generate_local_window_sample(
                    l_seq, l_window_min, l_window_max, dt, freq, query_length
                )
                for _ in range(samples)
            ]
        )
        X = torch.stack(xs, dim=0)
        Y = torch.stack(ys, dim=0)
        super().__init__(X, Y)
