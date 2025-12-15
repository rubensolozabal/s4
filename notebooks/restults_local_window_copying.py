import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

# Ensure we can import the Lightning module and src package
FILE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(FILE_DIR))

from train import SequenceLightningModule  # noqa: E402


def _load_config(run_dir: Path):
    # Ensure Hydra's `${eval:...}` interpolations work when loading outside Hydra
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find Hydra config at {cfg_path}")
    return OmegaConf.load(cfg_path)


def _default_ckpt(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoints" / "val" / "loss.ckpt",
        run_dir / "checkpoints" / "last.ckpt",
    ]
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    raise FileNotFoundError(f"No checkpoint found in {run_dir}/checkpoints")


def _extract_window(markers: torch.Tensor) -> Tuple[int, int]:
    """Deprecated: kept for backward compatibility; returns first window bounds if present."""
    windows = _extract_windows(markers, query_length=0)
    return (windows[0][1], windows[0][2]) if windows else (0, 0)


def _extract_windows(markers: torch.Tensor, query_length: int) -> List[Tuple[int, int, int]]:
    """Return (window_id, start, end) for all positive marker regions."""
    sequence_markers = markers if query_length == 0 else markers[:-query_length]
    window_ids = sorted({int(v.item()) for v in sequence_markers.unique() if v.item() > 0})
    windows: List[Tuple[int, int, int]] = []
    for wid in window_ids:
        idx = (sequence_markers == float(wid)).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        start = int(idx.min().item())
        end = int(idx.max().item()) + 1
        windows.append((wid, start, end))
    return windows


def _extract_queries(markers: torch.Tensor, query_length: int) -> List[Tuple[int, int]]:
    """Return (offset, window_id) for each replay query encoded by negative markers."""
    query_start = markers.numel() - query_length
    query_region = markers[query_start:]
    queries: List[Tuple[int, int]] = []
    for offset, marker in enumerate(query_region):
        if marker.item() < 0:
            queries.append((offset, int(abs(marker.item()))))
    return queries


def _plot_example(
    idx: int,
    x: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    save_dir: Path,
    show: bool,
):
    signal = x[:, 0]
    markers = x[:, 1]
    query_length = target.shape[0]
    windows = _extract_windows(markers, query_length)
    window_lookup = {wid: (start, end) for wid, start, end in windows}
    queries = _extract_queries(markers, query_length)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    time = torch.arange(signal.numel())
    axes[0].plot(time, signal, label="signal")
    axes[0].plot(time, markers, label="markers", alpha=0.7)
    for i, (wid, start, end) in enumerate(windows):
        axes[0].axvspan(
            start,
            end - 1,
            color="#cce5ff",
            alpha=0.5,
            label="windows" if i == 0 else None,
        )
    query_start = signal.numel() - query_length
    axes[0].axvspan(
        query_start - 0.5,
        signal.numel() - 0.5,
        color="#f5c6cb",
        alpha=0.35,
        label="query steps",
    )
    for i, (offset, wid) in enumerate(queries):
        axes[0].axvline(query_start + offset, color="#e31a1c", linestyle="--", alpha=0.8, label="replay" if i == 0 else None)
        axes[0].text(
            query_start + offset + 0.1,
            max(1.0, markers.abs().max().item()) * 0.8,
            f"w{wid}",
            fontsize=8,
            color="#e31a1c",
        )
    axes[0].set_title(f"Input sequence (example {idx})")
    axes[0].legend()

    colors = plt.cm.tab10.colors
    per_query_mae = []
    for q_idx, (offset, wid) in enumerate(queries):
        if wid not in window_lookup:
            continue
        start, end = window_lookup[wid]
        window_len = end - start
        targ_row = target[q_idx, :window_len]
        pred_row = pred[q_idx, :window_len]
        mae = torch.nn.functional.l1_loss(pred_row, targ_row).item()
        per_query_mae.append(mae)
        color = colors[q_idx % len(colors)]
        axes[1].plot(
            range(window_len),
            targ_row,
            label=f"target q{q_idx+1}→w{wid}",
            linewidth=2,
            alpha=0.85,
            color=color,
        )
        axes[1].plot(
            range(window_len),
            pred_row,
            label=f"pred q{q_idx+1}",
            linestyle="--",
            linewidth=2,
            color=color,
        )

    avg_mae = sum(per_query_mae) / len(per_query_mae) if per_query_mae else float("nan")
    axes[1].set_title(f"Window replays (avg MAE={avg_mae:.4f}, queries={len(per_query_mae)})")
    axes[1].set_xlabel("Position in window")
    axes[1].legend()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"val_example_{idx}.png"
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def _to_dict(cfg_node) -> Dict:
    return OmegaConf.to_container(cfg_node, resolve=True) if cfg_node is not None else {}


def gather_examples(
    model: SequenceLightningModule,
    loader: Iterable,
    device: torch.device,
    num_examples: int,
):
    """Collect predictions for a handful of validation samples."""
    collected = []
    model.eval()
    model._initialize_state()  # reset any persistent state

    with torch.no_grad():
        for batch in loader:
            inputs, targets, extra = batch
            extra = extra if isinstance(extra, dict) else {}
            for b in range(inputs.size(0)):
                if len(collected) >= num_examples:
                    return collected
                x = inputs[b : b + 1].to(device)
                y = targets[b : b + 1].to(device)
                batch_extra = {k: v[b : b + 1] for k, v in extra.items()}
                preds, _, _ = model((x, y, batch_extra))
                collected.append(
                    (
                        inputs[b].cpu(),  # (L, 2)
                        targets[b].cpu(),  # (query_length, l_window_max)
                        preds.squeeze(0).detach().cpu(),  # (query_length, l_window_max)
                    )
                )
            if len(collected) >= num_examples:
                break
    return collected


def main():
    parser = argparse.ArgumentParser(
        description="Visualize local-window-copying predictions from a trained checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        # default=Path("S4/outputs/2025-12-12/14-08-51-883639"), # original local-window-copying run
        # default=Path("S4/outputs/2025-12-12/18-39-41-894048"), # new local-window-copying run multuple windows 1 to 2
        default=Path("S4/outputs/2025-12-15/09-36-16-852462"), # new local-window-copying start-end markers
        help="Path to the Hydra output directory for the run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to val/loss.ckpt or last.ckpt inside the run.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of random validation examples to visualize.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for picking validation samples."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cpu or cuda). Defaults to auto.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to <run-dir>/visualizations.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.run_dir
    config = _load_config(run_dir)
    ckpt_path = args.checkpoint or _default_ckpt(run_dir)

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading checkpoint from {ckpt_path} on {device}")
    model = SequenceLightningModule.load_from_checkpoint(
        ckpt_path, config=config, map_location=device
    )
    model.to(device)

    loader_kwargs = _to_dict(config.loader)
    val_loader = model.dataset.val_dataloader(**loader_kwargs)

    save_dir = args.save_dir or (run_dir / "visualizations")
    examples = gather_examples(
        model=model,
        loader=val_loader,
        device=device,
        num_examples=args.num_examples,
    )

    for idx, (x, target, pred) in enumerate(examples, start=1):
        _plot_example(idx, x, target, pred, save_dir=save_dir, show=args.show)


if __name__ == "__main__":
    main()
