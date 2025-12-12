#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S4_ROOT = PROJECT_ROOT / "S4"
S4_SRC = S4_ROOT / "src"
for extra_path in (S4_ROOT, S4_SRC):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from dataloaders.datasets.local_window_copying import (  # type: ignore  # pylint: disable=import-error
    _generate_local_window_sample,
)

DEFAULT_EXPERIMENT = S4_ROOT / "configs" / "experiment" / "synthetic" / "s4-local-window-copying.yaml"
DEFAULT_DATASET = S4_ROOT / "configs" / "dataset" / "local_window_copying.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def load_dataset_config(experiment_path: Path, dataset_path: Path) -> dict[str, Any]:
    dataset_cfg = load_yaml(dataset_path).copy()
    dataset_cfg.pop("_name_", None)
    experiment_cfg = load_yaml(experiment_path)
    dataset_cfg.update(experiment_cfg.get("dataset", {}))
    return dataset_cfg


def sample_sequence(config: dict[str, Any], seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a single local-window-copying example."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    x, y = _generate_local_window_sample(
        l_seq=int(config["l_seq"]),
        l_window_min=int(config["l_window_min"]),
        l_window_max=int(config["l_window_max"]),
        dt=float(config["dt"]),
        freq=float(config["freq"]),
        query_length=int(config["query_length"]),
    )
    return x, y


def extract_window(markers: np.ndarray) -> tuple[int, int]:
    window_idx = np.flatnonzero(markers > 0)
    if window_idx.size == 0:
        raise ValueError("No positive markers found for window region.")
    start = int(window_idx.min())
    end = int(window_idx.max()) + 1
    return start, end


def print_summary(inputs: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> None:
    signal = inputs[:, 0]
    markers = inputs[:, 1]
    total_length = signal.numel()
    query_length = int(config["query_length"])
    window_start, window_end = extract_window(markers.numpy())
    window_len = window_end - window_start

    print(f"Sequence length: {total_length} (signal={config['l_seq']} + query={query_length})")
    print(f"Window start: {window_start}, length: {window_len} (max {config['l_window_max']})")
    print(f"Query region: [{total_length - query_length}:{total_length})")

    source_segment = signal[window_start:window_end]
    target_segment = targets[:window_len]
    exact_match = torch.allclose(source_segment, target_segment)
    print(f"Sanity check: source window matches target -> {exact_match}")


def plot_example(inputs: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> plt.Figure:
    signal = inputs[:, 0].cpu().numpy()
    markers = inputs[:, 1].cpu().numpy()
    query_length = int(config["query_length"])
    time = np.arange(signal.shape[0])

    window_start, window_end = extract_window(markers)
    window_len = window_end - window_start
    query_start = signal.shape[0] - query_length

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True, sharex=False)

    # Input signal with markers
    axes[0].plot(time, signal, color="#1f78b4", linewidth=1.2, label="Signal")
    axes[0].step(time, markers, where="mid", color="#d95f02", alpha=0.7, label="Markers (+window, -query)")
    axes[0].axvspan(window_start, window_end - 1, color="#cce5ff", alpha=0.6, label="Window region")
    axes[0].axvspan(query_start - 0.5, signal.shape[0] - 0.5, color="#f5c6cb", alpha=0.35, label="Query steps")
    axes[0].set_title("Local window copying input")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Value / marker")
    axes[0].legend()

    # Target vs source window
    target_np = targets.cpu().numpy()
    axes[1].plot(np.arange(target_np.shape[0]), target_np, marker="o", color="#1b9e77", label="Target (padded)")
    axes[1].plot(
        np.arange(window_len),
        signal[window_start:window_end],
        marker="x",
        linestyle="--",
        color="#7570b3",
        label="Source window",
    )
    axes[1].set_xlim(-0.5, max(target_np.shape[0], window_len) + 1)
    axes[1].set_title("Window reconstruction target")
    axes[1].set_xlabel("Position in window")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a random local-window-copying example.")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="Hydra experiment config to read dataset overrides from.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Base dataset config referenced by the experiment.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the generated figure.")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = (args.experiment if args.experiment.is_absolute() else (PROJECT_ROOT / args.experiment)).resolve()
    dataset_path = (args.dataset if args.dataset.is_absolute() else (PROJECT_ROOT / args.dataset)).resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_path}")

    dataset_config = load_dataset_config(experiment_path, dataset_path)
    inputs, targets = sample_sequence(dataset_config, seed=args.seed)
    print_summary(inputs, targets, dataset_config)

    if not args.no_plot or args.save:
        fig = plot_example(inputs, targets, dataset_config)
        if args.save:
            fig.savefig(args.save, dpi=200)
            print(f"\nSaved figure to {args.save}")
        if not args.no_plot:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
