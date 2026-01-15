#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Ensure we can import the Lightning module and src package
FILE_DIR = Path(__file__).resolve().parent.parent
if str(FILE_DIR) not in sys.path:
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


def _to_dict(cfg_node) -> Dict:
    return OmegaConf.to_container(cfg_node, resolve=True) if cfg_node is not None else {}


def _load_model(
    ckpt_path: Path,
    device: torch.device,
    config,
    use_ckpt_config: bool,
    strict: bool,
):
    """
    Try loading with explicit config first; if shapes mismatch, retry with the checkpoint's saved hyperparameters.
    """
    kwargs = {"map_location": device, "strict": strict}
    try:
        if use_ckpt_config:
            return SequenceLightningModule.load_from_checkpoint(ckpt_path, **kwargs)
        return SequenceLightningModule.load_from_checkpoint(ckpt_path, config=config, **kwargs)
    except RuntimeError as exc:
        if use_ckpt_config or "size mismatch" not in str(exc):
            raise
        print(
            "Detected size mismatch when constructing model from provided config; "
            "retrying with checkpoint hyperparameters."
        )
        return SequenceLightningModule.load_from_checkpoint(ckpt_path, **kwargs)


def _iterate_to_batch(loader: Iterable, batch_index: int):
    for idx, batch in enumerate(loader):
        if idx == batch_index:
            return batch
    raise IndexError(f"batch_index={batch_index} exceeds validation loader length")


def _select_example(batch, device: torch.device, example_index: int):
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Expected batch to be a tuple (inputs, targets[, extra]).")

    inputs, targets = batch[0], batch[1]
    extra = batch[2] if len(batch) > 2 else {}
    extra = extra if isinstance(extra, dict) else {}

    if inputs.size(0) <= example_index or targets.size(0) <= example_index:
        raise IndexError(f"example_index={example_index} exceeds batch size={inputs.size(0)}")

    inputs = inputs[example_index : example_index + 1].to(device)
    targets = targets[example_index : example_index + 1].to(device)
    extra_device: Dict = {}
    for key, value in extra.items():
        if torch.is_tensor(value):
            if value.size(0) > example_index:
                extra_device[key] = value[example_index : example_index + 1].to(device)
            else:
                extra_device[key] = value.to(device)
        else:
            extra_device[key] = value

    return inputs, targets, extra_device


def _prepare_state_inputs(backbone, batch_size: int, device: torch.device):
    # Not used in input attribution version.
    return [], [], []


def compute_input_gradients(
    model: SequenceLightningModule,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    extra: Dict,
) -> tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
    model.eval()
    with torch.enable_grad():
        inputs = inputs.clone().detach().requires_grad_(True)
        enc_x, enc_kwargs = model.encoder(inputs, **extra)
        enc_out, _ = model.model(enc_x, state=model.model.default_state(inputs.size(0), device=inputs.device), **enc_kwargs)
        y_pred, *_ = model.decoder(enc_out, **enc_kwargs)

        num_heads = y_pred.shape[1]
        head_losses: List[torch.Tensor] = []
        grads_per_head: List[Optional[torch.Tensor]] = []
        for h in range(num_heads):
            loss_h = F.mse_loss(
                y_pred[:, h : h + 1, :],
                targets[:, h : h + 1, :],
            )
            head_losses.append(loss_h.detach())

            grad_input = torch.autograd.grad(
                loss_h,
                inputs,
                retain_graph=h < num_heads - 1,
                allow_unused=True,
            )[0]
            grads_per_head.append(grad_input)

        stacked_losses = torch.stack(head_losses)
        return stacked_losses, y_pred.detach(), grads_per_head


def _plot_single_head(head_idx: int, grad: Optional[torch.Tensor], output_path: Path) -> None:
    if grad is None:
        print(f"No gradients to plot for head {head_idx}.")
        return

    # grad: (1, L, C)
    g = grad.detach().abs().squeeze(0).cpu()
    if g.dim() != 2:
        print(f"Unexpected grad shape for head {head_idx}: {g.shape}")
        return

    l_seq, n_channels = g.shape
    channel_names = ["signal", "marker"] if n_channels == 2 else [str(i) for i in range(n_channels)]

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )

    im = ax0.imshow(g.T, aspect="auto", cmap="magma")
    ax0.set_title(f"|d output / d input| (head {head_idx})")
    ax0.set_ylabel("Input channel")
    ax0.set_xlabel("Time step")
    ax0.set_yticks(range(n_channels))
    ax0.set_yticklabels(channel_names)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    channel_means = g.mean(dim=0)
    ax1.bar(range(n_channels), channel_means, color="#1f77b4")
    ax1.set_xticks(range(n_channels))
    ax1.set_xticklabels(channel_names)
    ax1.set_ylabel("Mean |grad|")
    ax1.set_xlabel("Input channel")
    ax1.set_title("Channel summary")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved gradient attribution figure to {output_path}")


def plot_gradients_per_head(grads_per_head: List[Optional[torch.Tensor]], output_path: Path) -> None:
    base = Path(output_path)
    base = base.with_suffix(".pdf")
    for head_idx, gradients in enumerate(grads_per_head):
        head_path = base.with_name(f"{base.stem}_head{head_idx}{base.suffix}")
        _plot_single_head(head_idx, gradients, head_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute gradients of model outputs with respect to SSM hidden states."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("S4/outputs/2026-01-09/17-04-17-094343"),  # s4f-local-window-copying-v3 default 
        help="Path to the Hydra output directory for the run (same default as probing_local_window_copying.py).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to val/loss.ckpt or last.ckpt inside the run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cpu, cuda, or auto).",
    )
    parser.add_argument(
        "--use-ckpt-config",
        action="store_true",
        help="Load hyperparameters from the checkpoint instead of the run directory config.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require an exact state_dict match when loading the checkpoint.",
    )
    parser.add_argument(
        "--loader-batch-size",
        type=int,
        default=1,
        help="Override validation dataloader batch size to reduce memory (defaults to config).",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=1,
        help="Which validation batch to inspect.",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Which element within the selected batch to use.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the matplotlib figure(s). Defaults to RUN_DIR/grad_attribution_head*.pdf",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    config = _load_config(run_dir)
    ckpt_path = args.checkpoint or _default_ckpt(run_dir)

    if args.device is None or args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Loading checkpoint from {ckpt_path} on {device}")
    model = _load_model(
        ckpt_path=ckpt_path,
        device=device,
        config=config,
        use_ckpt_config=args.use_ckpt_config,
        strict=args.strict_load,
    )
    model.to(device)

    loader_kwargs = _to_dict(config.loader)
    if args.loader_batch_size is not None:
        loader_kwargs["batch_size"] = args.loader_batch_size

    val_loader = model.dataset.val_dataloader(**loader_kwargs)
    if isinstance(val_loader, dict):
        val_loader = next(iter(val_loader.values()))
    elif isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

    batch = _iterate_to_batch(val_loader, args.batch_index)
    inputs, targets, extra = _select_example(batch, device, args.example_index)
    head_losses, predictions, grads_per_head = compute_input_gradients(model, inputs, targets, extra)

    print(f"Selected batch={args.batch_index}, example={args.example_index}")
    print(f"Prediction shape: {tuple(predictions.shape)} | Target shape: {tuple(targets.shape)}")
    for idx, loss_val in enumerate(head_losses):
        non_empty = grads_per_head[idx] is not None
        print(f"Head {idx}: MSE={loss_val.item():.6f} | grad available={non_empty}")

    default_output = args.output
    if default_output is None:
        default_output = Path(run_dir) / "grad_attribution_input.pdf"
    plot_gradients_per_head(grads_per_head, default_output)


if __name__ == "__main__":
    main()
