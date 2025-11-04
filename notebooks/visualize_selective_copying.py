#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S4_ROOT = PROJECT_ROOT / "S4"
S4_SRC = S4_ROOT / "src"
for extra_path in (S4_ROOT, S4_SRC):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from dataloaders.datasets.copying import torch_copying_data  # type: ignore  # pylint: disable=import-error

DEFAULT_EXPERIMENT = S4_ROOT / "configs" / "experiment" / "synthetic" / "s4-selective-copying.yaml"
DEFAULT_DATASET = S4_ROOT / "configs" / "dataset" / "selective_copying.yaml"


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
    dataset_cfg.setdefault("variable", False)
    dataset_cfg.setdefault("variable_length", False)
    dataset_cfg.setdefault("one_hot", False)
    dataset_cfg.setdefault("reverse", False)
    return dataset_cfg


def sample_sequence(config: dict[str, Any], seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a single selective-copying example."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    l_noise = int(config["l_noise"])
    l_mem = int(config["l_memorize"])
    n_tokens = int(config["n_tokens"])
    inputs, targets = torch_copying_data(
        l_noise,
        l_mem,
        n_tokens,
        variable=bool(config.get("variable", False)),
        variable_length=bool(config.get("variable_length", False)),
        batch_shape=(1,),
        one_hot=bool(config.get("one_hot", False)),
        reverse=bool(config.get("reverse", False)),
    )
    return inputs.squeeze(0), targets.squeeze(0)


def pretty_join(values: Sequence[int]) -> str:
    return " ".join(f"{val:2d}" for val in values)


def print_summary(inputs: torch.Tensor, targets: torch.Tensor, config: dict[str, Any], context: int) -> None:
    marker_token = int(config["n_tokens"]) - 1
    memorize_mask = (inputs > 0) & (inputs != marker_token)
    memorize_positions = torch.nonzero(memorize_mask, as_tuple=False).squeeze(1)
    memorize_tokens = inputs[memorize_positions]

    total_length = inputs.numel()
    l_noise = int(config["l_noise"])
    l_mem = int(config["l_memorize"])
    print(f"Sequence length: {total_length} (noise={l_noise} + memorize={l_mem} + markers={l_mem})")
    print("\nMemorized token positions (0-based):")
    for pos, token in zip(memorize_positions.tolist(), memorize_tokens.tolist(), strict=False):
        print(f"  {pos:5d} -> {token:2d}")

    if context > 0:
        print(f"\nContext ±{context} around each memorized token:")
        for pos in memorize_positions.tolist():
            start = max(pos - context, 0)
            end = min(pos + context + 1, inputs.numel())
            window = inputs[start:end].tolist()
            pointer = " " * (3 * (pos - start)) + "^"
            print(f"  [{start:4d}:{end - 1:4d}] {pretty_join(window)}")
            print(f"               {pointer}")

    print("\nExpected output sequence:")
    print(f"  {pretty_join(targets.tolist())}")
    copied_back = [inputs[pos].item() for pos in memorize_positions.tolist()]
    print(f"\nSanity check: copied tokens == targets -> {copied_back == targets.tolist()}")


def plot_example(inputs: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> plt.Figure:
    marker_token = int(config["n_tokens"]) - 1
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    memorize_mask = (inputs_np > 0) & (inputs_np != marker_token)
    marker_mask = inputs_np == marker_token
    categories = np.zeros_like(inputs_np)
    categories[memorize_mask] = 1
    categories[marker_mask] = 2
    memorize_positions = np.flatnonzero(memorize_mask)

    cmap = ListedColormap(["#d9d9d9", "#1f78b4", "#e31a1c"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True, sharex=False)

    axes[0].imshow(categories[np.newaxis, :], aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[0].set_yticks([])
    axes[0].set_title("Input token categories (0=noise, 1=memorize, 2=marker)")
    if memorize_positions.size <= 24:
        axes[0].set_xticks(memorize_positions)
        axes[0].set_xticklabels([str(pos) for pos in memorize_positions], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Category")
    legend_handles = [
        Patch(facecolor=cmap(0), label="Noise"),
        Patch(facecolor=cmap(1), label="Memorize token"),
        Patch(facecolor=cmap(2), label="Marker"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right")

    axes[1].plot(range(len(targets_np)), targets_np, marker="o", color="#1f78b4", label="Target token id")
    axes[1].set_title("Target sequence order (model output after markers)")
    axes[1].set_xlabel("Output position")
    axes[1].set_ylabel("Token id")
    axes[1].set_xticks(range(len(targets_np)))
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a random selective-copying example.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT, help="Hydra experiment config to read dataset overrides from.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Base dataset config referenced by the experiment.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the generated figure.")
    parser.add_argument("--context", type=int, default=4, help="How many tokens of context to print around each memorized token.")
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
    print_summary(inputs, targets, dataset_config, context=max(args.context, 0))

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
