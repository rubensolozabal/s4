#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import random
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
    raw_states = backbone.default_state(batch_size, device=device)
    state_inputs: List[Optional[torch.Tensor]] = []
    grad_targets: List[torch.Tensor] = []
    grad_indices: List[int] = []
    for idx, state in enumerate(raw_states):
        if state is None:
            state_inputs.append(None)
            continue
        state_req = state.detach().clone().requires_grad_(True)
        state_inputs.append(state_req)
        grad_targets.append(state_req)
        grad_indices.append(idx)
    return state_inputs, grad_targets, grad_indices


def compute_state_gradients(
    model: SequenceLightningModule,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    extra: Dict,
) -> tuple[torch.Tensor, torch.Tensor, List[List[Optional[torch.Tensor]]]]:
    model.eval()
    with torch.enable_grad():
        enc_x, enc_kwargs = model.encoder(inputs, **extra)
        state_inputs, grad_targets, grad_indices = _prepare_state_inputs(
            model.model, inputs.size(0), inputs.device
        )
        enc_out, _ = model.model(enc_x, state=state_inputs, **enc_kwargs)
        y_pred, *_ = model.decoder(enc_out, state=state_inputs, **enc_kwargs)

        num_heads = y_pred.shape[1]
        head_losses: List[torch.Tensor] = []
        grads_per_head: List[List[Optional[torch.Tensor]]] = []
        for h in range(num_heads):
            loss_h = F.mse_loss(
                y_pred[:, h : h + 1, :],
                targets[:, h : h + 1, :],
            )
            head_losses.append(loss_h.detach())

            grads: List[Optional[torch.Tensor]] = [None] * len(state_inputs)
            if grad_targets:
                raw_grads = torch.autograd.grad(
                    loss_h,
                    grad_targets,
                    retain_graph=h < num_heads - 1,
                    allow_unused=True,
                )
                for idx, g in zip(grad_indices, raw_grads):
                    grads[idx] = g
            grads_per_head.append(grads)

        stacked_losses = torch.stack(head_losses)
        return stacked_losses, y_pred.detach(), grads_per_head


def _flatten_state_tensors(states: Iterable) -> List[torch.Tensor]:
    """Extract tensors from nested state structures, preserving order."""
    flat: List[torch.Tensor] = []

    def _collect(state_obj):
        if state_obj is None:
            return
        if torch.is_tensor(state_obj):
            flat.append(state_obj)
        elif isinstance(state_obj, (list, tuple)):
            for item in state_obj:
                _collect(item)

    _collect(states)
    return flat


def _compute_linear_jacobian(Abar: torch.Tensor, Bbar: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Build analytic Jacobian J[t] = Abar^{T-1-t} Bbar for t in [0, T-1].
    Returns a tensor of shape (seq_len, d_state, d_in).
    """
    d_state, d_in = Bbar.shape
    A_pows = [torch.eye(d_state, device=Abar.device, dtype=Abar.dtype)]
    for _ in range(seq_len - 1):
        A_pows.append(Abar @ A_pows[-1])

    jac = []
    for t in range(seq_len):
        k = seq_len - 1 - t
        jac_t = A_pows[k] @ Bbar  # (d_state, d_in)
        jac.append(jac_t)
    return torch.stack(jac, dim=0)  # (seq_len, d_state, d_in)


def _extract_analytic_params_from_model(model: SequenceLightningModule) -> List[Dict]:
    """Gather discretized (Abar, Bbar) from each SSM kernel in the backbone."""
    params: List[Dict] = []
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        return params

    for layer_idx, block in enumerate(backbone.layers):
        # Search recursively within the block for a module exposing get_discrete_state_matrices
        kernels = [
            m for m in block.modules() if hasattr(m, "get_discrete_state_matrices")
        ]
        for kernel in kernels:
            try:
                dA, dB = kernel.get_discrete_state_matrices()
            except Exception:
                continue
            if dA is None or dB is None:
                continue
            for h in range(dA.shape[0]):
                params.append(
                    {
                        "layer": layer_idx,
                        "ssm": h,
                        "Abar": dA[h],
                        "Bbar": dB[h],
                    }
                )
    return params


def _prepare_state_slices(
    states,
    max_ssm_per_layer: int = 5,
    sample_seed: Optional[int] = 0,
) -> List[Dict]:
    """
    Build a list of (layer, ssm_idx, hidden_idx, tensor) entries for selected SSMs.
    Assumes SSM states have shape (..., channels, hidden_dim) or (..., hidden_dim).
    """
    rng = torch.Generator()
    if sample_seed is not None:
        rng.manual_seed(sample_seed)

    entries: List[Dict] = []
    state_list = states if isinstance(states, (list, tuple)) else [states]
    for layer_idx, state in enumerate(state_list):
        layer_tensors: List[torch.Tensor] = []

        def _collect(obj):
            if obj is None:
                return
            if torch.is_tensor(obj):
                layer_tensors.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)

        _collect(state)

        for tensor in layer_tensors:
            if tensor.dim() < 2:
                continue
            if tensor.dim() == 2:
                channels = 1
                hidden_dim = tensor.shape[-1]
            else:
                channels = tensor.shape[-2]
                hidden_dim = tensor.shape[-1]

            ssm_indices = torch.arange(channels)
            if channels > max_ssm_per_layer:
                perm = torch.randperm(channels, generator=rng)
                ssm_indices = perm[:max_ssm_per_layer]

            for ssm_idx in ssm_indices.tolist():
                for h_idx in range(hidden_dim):
                    if tensor.dim() == 2:
                        slice_tensor = tensor[..., h_idx]
                    else:
                        slice_tensor = tensor.select(-2, ssm_idx).select(-1, h_idx)
                    entries.append(
                        {
                            "layer": layer_idx,
                            "ssm": ssm_idx,
                            "hidden": h_idx,
                            "tensor": slice_tensor,
                        }
                    )

    return entries


def compute_state_input_gradients(
    model: SequenceLightningModule,
    inputs: torch.Tensor,
    extra: Dict,
    max_ssm_per_layer: int = 5,
    sample_seed: Optional[int] = 0,
    analytic_params: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Compute gradients of selected SSM hidden states with respect to the input signal.
    Returns a list of dicts: {"layer", "ssm", "grads"} where grads is (hidden_dim, seq_len).
    """
    model.eval()
    with torch.enable_grad():
        inp = inputs.detach().clone().requires_grad_(True)
        enc_x, enc_kwargs = model.encoder(inp, **extra)
        init_state = model.model.default_state(inp.size(0), device=inp.device)
        enc_out, next_states = model.model(enc_x, state=init_state, **enc_kwargs)
        y_pred, *_ = model.decoder(enc_out, state=init_state, **enc_kwargs)

        # Analytic path using dc_T/dx_t = Abar^{T-1-t} Bbar
        params = analytic_params
        if params is None:
            params = _extract_analytic_params_from_model(model)
        results: List[Dict] = []
        if params:
            seq_len = inputs.shape[1]
            for entry in params:
                Abar = entry["Abar"].to(inputs.device)
                Bbar = entry["Bbar"].to(inputs.device)
                if Bbar.dim() == 1:
                    Bbar = Bbar.unsqueeze(-1)
                jac = _compute_linear_jacobian(Abar, Bbar, seq_len=seq_len)  # (T, d_state, d_in)
                grad_map = jac.abs().sum(dim=-1).transpose(0, 1).detach().cpu()  # (d_state, T)
                results.append(
                    {
                        "layer": int(entry.get("layer", 0)),
                        "ssm": int(entry.get("ssm", 0)),
                        "grads": grad_map,
                    }
                )
            if results:
                return results

        slices = _prepare_state_slices(next_states, max_ssm_per_layer=max_ssm_per_layer, sample_seed=sample_seed)

        layer_results: Dict[int, Dict[int, torch.Tensor]] = {}
        seq_len = inputs.shape[1]

        for idx, entry in enumerate(slices):
            state_tensor = entry["tensor"]
            if not (torch.is_tensor(state_tensor) and state_tensor.requires_grad and state_tensor.grad_fn is not None):
                continue

            grad = torch.autograd.grad(
                state_tensor.sum(),
                inp,
                retain_graph=idx < len(slices) - 1,
                allow_unused=True,
            )[0]
            if grad is None:
                continue

            grad_map = grad.abs().sum(dim=-1).squeeze(0)  # (seq_len,)
            if grad_map.ndim > 1:
                grad_map = grad_map.flatten()
            if grad_map.numel() != seq_len:
                # Best effort align to input length
                grad_padded = torch.zeros(seq_len, device=grad_map.device)
                n = min(seq_len, grad_map.numel())
                grad_padded[:n] = grad_map[:n]
                grad_map = grad_padded

            layer = entry["layer"]
            ssm = entry["ssm"]
            hidden = entry["hidden"]
            layer_results.setdefault(layer, {})
            mat = layer_results[layer].get(ssm)
            if mat is None or mat.numel() == 0:
                mat = torch.zeros(hidden + 1, seq_len, device=grad_map.device)
            elif hidden >= mat.shape[0]:
                pad_rows = hidden - mat.shape[0] + 1
                mat = torch.cat([mat, torch.zeros(pad_rows, seq_len, device=grad_map.device)], dim=0)
            mat[hidden] = grad_map
            layer_results[layer][ssm] = mat

    results: List[Dict] = []
    for layer_idx in sorted(layer_results.keys()):
        for ssm_idx in sorted(layer_results[layer_idx].keys()):
            results.append(
                {
                    "layer": layer_idx,
                    "ssm": ssm_idx,
                    "grads": layer_results[layer_idx][ssm_idx].detach().cpu(),
                }
            )

    if not results:
        # Fallback: gradient of model output w.r.t. input to avoid empty results
        fallback_grad = torch.autograd.grad(y_pred.sum(), inp, allow_unused=True)[0]
        if fallback_grad is None:
            fallback_grad = torch.zeros_like(inp)
        grad_map = fallback_grad.abs().sum(dim=-1).squeeze(0)
        if grad_map.dim() == 1:
            grad_map = grad_map.unsqueeze(0)
        results.append(
            {
                "layer": 0,
                "ssm": 0,
                "grads": grad_map.detach().cpu(),
            }
        )

    return results


def _summarize_input_grads(grads: List[Dict]) -> None:
    for entry in grads:
        mean_mag = float(entry["grads"].abs().mean().item())
        shape = tuple(entry["grads"].shape)
        print(
            f"Layer {entry['layer'] + 1} SSM {entry['ssm']}: grad matrix shape {shape} | mean|grad|={mean_mag:.4e}"
        )


def _plot_single_head(head_idx: int, gradients: List[Optional[torch.Tensor]], output_path: Path) -> None:
    usable = [g for g in gradients if g is not None]
    if not usable:
        print(f"No gradients to plot for head {head_idx}.")
        return

    # Reduce batch/time dims, keep per-SSM channel and hidden dimension.
    layer_maps: List[Optional[torch.Tensor]] = []
    layer_norms: List[float] = []
    max_hidden = 0
    max_channels = 0
    global_max = 0.0
    for g in gradients:
        if g is None:
            layer_maps.append(None)
            layer_norms.append(0.0)
            continue

        mag = g.detach().abs()
        if mag.dim() >= 2:
            mag = mag.mean(dim=0)  # average over batch
        if mag.dim() == 1:
            mag = mag.unsqueeze(0)
        elif mag.dim() > 2:
            mag = mag.reshape(-1, mag.shape[-1])  # merge extra dims into "channels"

        layer_maps.append(mag.cpu())
        layer_norms.append(float(mag.mean().item()))
        max_channels = max(max_channels, mag.shape[0])
        max_hidden = max(max_hidden, mag.shape[-1])
        global_max = max(global_max, float(mag.max().item()))

    n_layers = len(layer_maps)
    fig = plt.figure(
        figsize=(14, max(6, 1.8 * n_layers)),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(n_layers, 2, width_ratios=[4, 1], wspace=0.35)

    for idx, layer_map in enumerate(layer_maps):
        ax = fig.add_subplot(gs[idx, 0])
        if layer_map is None:
            ax.text(0.5, 0.5, "No grad", ha="center", va="center")
            ax.set_axis_off()
            continue

        padded = torch.zeros(max_channels, max_hidden, dtype=torch.float)
        c, h = layer_map.shape
        padded[:c, :h] = layer_map
        im = ax.imshow(
            padded.numpy(),
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=global_max if global_max > 0 else None,
        )

        # log scale for better visibility
        im.set_norm(matplotlib.colors.LogNorm(vmin=1e-6, vmax=global_max if global_max > 0 else None))
        ax.set_ylabel(f"Layer {idx + 1} (SSMs)")
        ax.set_xticks([])
        ax.set_yticks(range(c))
        ax.set_yticklabels([str(i + 1) for i in range(c)])
        if idx == 0:
            ax.set_title(f"|d output / d state| per SSM/hidden (head {head_idx})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_bar = fig.add_subplot(gs[:, 1])
    ax_bar.barh(range(n_layers), layer_norms, color="#1f77b4")
    ax_bar.set_xlabel("Mean grad magnitude")
    ax_bar.set_ylabel("Layer")
    ax_bar.set_yticks(range(n_layers))
    ax_bar.set_yticklabels([str(i + 1) for i in range(n_layers)])
    ax_bar.set_title("Layer summary")


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved gradient attribution figure to {output_path}")


def plot_gradients_per_head(grads_per_head: List[List[Optional[torch.Tensor]]], output_path: Path) -> None:
    base = Path(output_path)
    base = base.with_suffix(".pdf")
    for head_idx, gradients in enumerate(grads_per_head):
        head_path = base.with_name(f"{base.stem}_head{head_idx}{base.suffix}")
        _plot_single_head(head_idx, gradients, head_path)


def plot_input_gradients_per_ssm(samples: List[Dict], output_path: Path, seed: Optional[int] = 0) -> None:
    if not samples:
        print("No input gradients to plot.")
        return

    # Focus on a random SSM from the first layer
    first_layer_samples = [s for s in samples if s["layer"] == 0]
    candidates = first_layer_samples if first_layer_samples else samples
    if not candidates:
        print("No input gradients to plot.")
        return

    rng = random.Random(seed)
    sample = rng.choice(candidates)
    grads = sample["grads"]  # (hidden_dim, seq_len)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    vmax = float(grads.max().item()) if grads.numel() > 0 else None
    im = ax.imshow(
        grads,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vmax if vmax is not None and vmax > 0 else None,
        extent=[0, grads.shape[1], 0, grads.shape[0]],
    )

    # color scale log for better visibility
    im.set_norm(matplotlib.colors.LogNorm(vmin=1e-6, vmax=vmax if vmax is not None and vmax > 0 else None))
    ax.set_title(f"Layer 1 SSM {sample['ssm']} input gradients")
    ax.set_xlabel("Input position")
    ax.set_ylabel("Hidden dimension")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved input-gradient figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute gradients of model outputs with respect to SSM hidden states."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        # default=Path("S4/outputs/2026-01-09/17-04-17-094343"),  # s4f-local-window-copying-v3 default 
        default=Path("S4/outputs/2026-01-13/14-23-06-106225"),  # s4f-local-window-copying-v3-fixed-gabors_multiWin8-8_layer2_frozenL1
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
    parser.add_argument(
        "--save-input-grads",
        type=Path,
        default=None,
        help="Optional path to save gradients of each SSM state w.r.t. the input signal (torch.save). Defaults to RUN_DIR/state_input_grads.pt",
    )
    parser.add_argument(
        "--skip-input-grads",
        action="store_true",
        help="Disable computing gradients of states with respect to the input to save time.",
    )
    parser.add_argument(
        "--max-ssm-per-layer",
        type=int,
        default=5,
        help="Randomly sample up to this many SSMs per layer for input-gradient visualization.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for sampling SSMs per layer when computing input gradients.",
    )
    parser.add_argument(
        "--analytic-params",
        type=Path,
        default=None,
        help="Path to a torch file containing a list of dicts with keys {layer, ssm, Abar, Bbar} to use analytic dc_T/dx_t = Abar^{T-1-t} Bbar instead of autograd.",
    )
    parser.add_argument(
        "--plot-ssm-seed",
        type=int,
        default=3,
        help="Seed for selecting which SSM to plot in the input-gradient heatmap.",
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
    head_losses, predictions, grads_per_head = compute_state_gradients(model, inputs, targets, extra)

    print(f"Selected batch={args.batch_index}, example={args.example_index}")
    print(f"Prediction shape: {tuple(predictions.shape)} | Target shape: {tuple(targets.shape)}")
    for idx, loss_val in enumerate(head_losses):
        non_empty = sum(g is not None for g in grads_per_head[idx])
        print(
            f"Head {idx}: MSE={loss_val.item():.6f} | non-empty grads {non_empty}/{len(grads_per_head[idx])}"
        )

    if not args.skip_input_grads:
        analytic_params = None
        if args.analytic_params is not None:
            analytic_params = torch.load(args.analytic_params)

        state_input_grads = compute_state_input_gradients(
            model,
            inputs,
            extra,
            max_ssm_per_layer=args.max_ssm_per_layer,
            sample_seed=args.sample_seed,
            analytic_params=analytic_params,
        )
        _summarize_input_grads(state_input_grads)
        save_path = args.save_input_grads
        if save_path is None:
            save_path = Path(run_dir) / "state_input_grads.pt"
        torch.save(state_input_grads, save_path)
        print(f"Saved state-input gradients to {save_path}")
        input_grad_fig = Path(run_dir) / "input_grad_attribution.pdf" if args.output is None else args.output.with_name(args.output.stem + "_input.pdf")
        plot_input_gradients_per_ssm(state_input_grads, input_grad_fig, seed=args.plot_ssm_seed)

    default_output = args.output
    if default_output is None:
        default_output = Path(run_dir) / "grad_attribution.pdf"
    plot_gradients_per_head(grads_per_head, default_output)


if __name__ == "__main__":
    main()
