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
from omegaconf import OmegaConf

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


def load_dataset_config(experiment_path: Path, dataset_path: Path) -> dict[str, Any]:
    # Minimal resolver registry for standalone usage outside Hydra.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))  # noqa: S307

    dataset_cfg = OmegaConf.load(dataset_path)
    dataset_cfg.pop("_name_", None)
    experiment_cfg = OmegaConf.load(experiment_path)
    merged = OmegaConf.merge(dataset_cfg, experiment_cfg.get("dataset", {}))
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


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
        n_windows_min=int(config["n_windows_min"]),
        n_windows_max=int(config["n_windows_max"]),
        query_length=int(config["query_length"]),
    )
    return x, y


def extract_windows(markers: np.ndarray, query_length: int) -> list[tuple[int, int, int]]:
    """Return (window_id, start, end) tuples for all positive marker regions."""
    sequence_markers = markers if query_length == 0 else markers[:-query_length]
    window_ids = sorted({int(val) for val in np.unique(sequence_markers) if val > 0})
    windows: list[tuple[int, int, int]] = []
    for window_id in window_ids:
        window_idx = np.flatnonzero(sequence_markers == window_id)
        if window_idx.size == 0:
            continue
        start = int(window_idx.min())
        end = int(window_idx.max()) + 1
        windows.append((window_id, start, end))
    if not windows:
        raise ValueError("No positive markers found for window regions.")
    return windows


def extract_queries(markers: np.ndarray, query_length: int) -> list[tuple[int, int]]:
    """Return (offset, window_id) for each replay query encoded by negative markers."""
    query_region = markers[-query_length:]
    queries: list[tuple[int, int]] = []
    for offset, marker in enumerate(query_region):
        if marker < 0:
            queries.append((offset, int(abs(marker))))
    return queries


def print_summary(inputs: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> None:
    signal = inputs[:, 0]
    markers = inputs[:, 1]
    total_length = signal.numel()
    query_length = int(config["query_length"])
    markers_np = markers.numpy()
    windows = extract_windows(markers_np, query_length)
    queries = extract_queries(markers_np, query_length)
    window_desc = ", ".join(f"{wid}:[{start},{end})" for wid, start, end in windows)
    query_order = ", ".join(f"{q_idx + 1}→w{wid}" for q_idx, (_, wid) in enumerate(queries))

    print(f"Sequence length: {total_length} (signal={config['l_seq']} + query={query_length})")
    print(f"Windows ({len(windows)}): {window_desc}")
    print(f"Query region: [{total_length - query_length}:{total_length})")
    print(f"Replay order ({len(queries)} queries): {query_order or 'none'}")

    if queries:
        first_query_idx, window_id = queries[0]
        _, start, end = next(item for item in windows if item[0] == window_id)
        window_len = end - start
        source_segment = signal[start:end]
        target_segment = targets[first_query_idx, :window_len]
        exact_match = torch.allclose(source_segment, target_segment)
        print(f"Sanity check (first query -> window {window_id}): match={exact_match}")


def plot_example(inputs: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> plt.Figure:
    signal = inputs[:, 0].cpu().numpy()
    markers = inputs[:, 1].cpu().numpy()
    query_length = int(config["query_length"])
    time = np.arange(signal.shape[0])

    windows = extract_windows(markers, query_length)
    window_lookup = {wid: (start, end) for wid, start, end in windows}
    queries = extract_queries(markers, query_length)
    query_start = signal.shape[0] - query_length

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True, sharex=False)

    # Input signal with markers
    axes[0].plot(time, signal, color="#1f78b4", linewidth=1.2, label="Signal")
    axes[0].step(time, markers, where="mid", color="#d95f02", alpha=0.7, label="Markers (+window id, -query id)")
    for i, (window_id, start, end) in enumerate(windows):
        axes[0].axvspan(
            start,
            end - 1,
            color="#cce5ff",
            alpha=0.5,
            label="Window regions" if i == 0 else None,
        )
    axes[0].axvspan(query_start - 0.5, signal.shape[0] - 0.5, color="#f5c6cb", alpha=0.35, label="Query steps")
    for i, (offset, window_id) in enumerate(queries):
        axes[0].axvline(query_start + offset, color="#e31a1c", linestyle="--", alpha=0.8, label="Replay prompts" if i == 0 else None)
        axes[0].text(
            query_start + offset + 0.1,
            max(1.0, np.max(np.abs(markers))) * 0.8,
            f"w{window_id}",
            fontsize=8,
            color="#e31a1c",
        )
    axes[0].set_title("Local window copying input")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Value / marker")
    axes[0].legend()

    # Target vs source window
    colors = plt.cm.tab10.colors
    for idx, (offset, window_id) in enumerate(queries):
        start, end = window_lookup[window_id]
        window_len = end - start
        color = colors[idx % len(colors)]
        axes[1].plot(
            np.arange(window_len),
            targets[idx, :window_len].cpu().numpy(),
            marker="o",
            color=color,
            label=f"Target query {idx + 1} → window {window_id}",
        )
        axes[1].plot(
            np.arange(window_len),
            signal[start:end],
            marker="x",
            linestyle="--",
            color=color,
            alpha=0.8,
            label=f"Source window {window_id}",
        )
    max_len = targets.shape[-1]
    axes[1].set_xlim(-0.5, max_len + 1)
    axes[1].set_title("Replay targets vs source windows")
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
