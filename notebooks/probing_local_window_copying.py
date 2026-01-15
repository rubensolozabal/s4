import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
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


def _to_dict(cfg_node) -> Dict:
    return OmegaConf.to_container(cfg_node, resolve=True) if cfg_node is not None else {}


def _flatten_state_vector(state, batch_size: int) -> torch.Tensor:
    """Flatten raw per-layer states (list of tensors) to shape (B, D)."""
    if isinstance(state, dict):
        state = state.get("state", state.get("states"))
    if not isinstance(state, (list, tuple)):
        return torch.zeros(batch_size, 0)

    pieces: List[torch.Tensor] = []
    for layer_state in state:
        if layer_state is None or not torch.is_tensor(layer_state):
            continue
        piece = layer_state.real if torch.is_complex(layer_state) else layer_state

        if piece.dim() == 1:
            piece = piece.unsqueeze(0)
        piece = piece.reshape(piece.shape[0], -1)
        if piece.shape[0] == 1 and batch_size > 1:
            piece = piece.expand(batch_size, -1)
        pieces.append(piece)

    if not pieces:
        return torch.zeros(batch_size, 0)

    return torch.cat(pieces, dim=1)


def _infer_hidden_dim(samples: List[Dict]) -> Optional[int]:
    """Infer hidden dimension N from collected layer_states."""
    for sample in samples:
        for layer_state in sample.get("layer_states", []):
            if layer_state is None or not torch.is_tensor(layer_state):
                continue
            return int(layer_state.shape[-1])
    return None


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Compute MSE over valid positions defined by window lengths."""
    max_len = target.size(1)
    lengths = lengths.to(pred.device)
    mask = torch.arange(max_len, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp(min=1)
    return diff.sum() / denom


def collect_states_and_windows(
    model: SequenceLightningModule,
    loader: Iterable,
    device: torch.device,
    query_length: int,
    l_window_max: int,
    max_examples: Optional[int] = None,
):
    """Run the model on the validation set and store inputs, per-layer states, and flattened c_T."""
    collected = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, targets, extra = batch
            extra = extra if isinstance(extra, dict) else {}
            inputs = inputs.to(device)
            targets = targets.to(device)
            extra_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in extra.items()}

            enc_x, enc_kwargs = model.encoder(inputs, **extra_device)
            init_state = model.model.default_state(inputs.shape[0], device=device)
            # Just backbone
            enc_out, states = model.model(enc_x, state=init_state, **enc_kwargs)
            # Rest of model (e.g., decoder)
            y_pred, *_ = model.decoder(enc_out, state=states, **enc_kwargs)
            
            # States [layers][batch, d_model, hidden_state]
            flat_states = _flatten_state_vector(states, batch_size=inputs.size(0)).detach().cpu()
            layer_states = [s.detach().cpu() if s is not None else None for s in states]
            inputs_cpu = inputs.detach().cpu()
            pred_cpu = y_pred.detach().cpu()
            target_cpu = targets.detach().cpu()
            sample_mse = ((pred_cpu - target_cpu) ** 2).mean(dim=(1, 2))

            for b in range(inputs_cpu.size(0)):
                signal = inputs_cpu[b, :, 0]
                markers = inputs_cpu[b, :, 1]
                windows = _extract_windows(markers, query_length=query_length)
                window_entries = []
                for wid, start, end in windows:
                    segment = signal[start:end]
                    length = end - start
                    padded = torch.zeros(l_window_max, dtype=signal.dtype)
                    padded[:length] = segment
                    window_entries.append(
                        {
                            "id": wid,
                            "start": start,
                            "end": end,
                            "length": length,
                            "target": padded,
                        }
                    )

                collected.append(
                    {
                        "input": inputs_cpu[b],
                        "state_vector": flat_states[b],
                        "layer_states": [s[b] if s is not None else None for s in layer_states],
                        "windows": window_entries,
                        "y_pred": pred_cpu[b],
                        "target": target_cpu[b],
                        "mse": float(sample_mse[b].item()),
                    }
                )
                if max_examples is not None and len(collected) >= max_examples:
                    return collected

    return collected


def build_probe_datasets(
    samples: List[Dict],
    max_windows: int,
    l_window_max: int,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], int]:
    """Collect feature/target tensors for each window index."""
    datasets: Dict[int, Dict[str, torch.Tensor]] = {}
    state_dim = samples[0]["state_vector"].numel() if samples else 0

    for window_idx in range(1, max_windows + 1):
        features, targets, lengths = [], [], []
        for sample in samples:
            if len(sample["windows"]) >= window_idx:
                window_info = sample["windows"][window_idx - 1]  # windows are sorted by id
                features.append(sample["state_vector"])
                targets.append(window_info["target"])
                lengths.append(window_info["length"])

        if features:
            datasets[window_idx] = {
                "features": torch.stack(features),
                "targets": torch.stack(targets),
                "lengths": torch.tensor(lengths, dtype=torch.long),
            }

    return datasets, state_dim


def train_linear_probes(
    window_datasets: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[int, Dict[str, Union[torch.Tensor, float, int]]]:
    """Fit a linear probe P_i for each window index using least squares."""
    probes: Dict[int, Dict[str, Union[torch.Tensor, float, int]]] = {}
    for window_idx, data in window_datasets.items():
        X = data["features"]
        Y = data["targets"]
        lengths = data["lengths"]

        X_aug = torch.cat([X, torch.ones(X.size(0), 1)], dim=1)  # Add bias term
        solution = torch.linalg.lstsq(X_aug, Y).solution  # (D+1, L)
        weight = solution[:-1]
        bias = solution[-1]

        full_pred = X @ weight + bias
        mse = _masked_mse(full_pred, Y, lengths)

        probes[window_idx] = {
            "weight": weight,
            "bias": bias,
            "train_mse": mse.item(),
            "n": X.size(0),
        }
    return probes


def evaluate_masked_probes(
    window_datasets: Dict[int, Dict[str, torch.Tensor]],
    probes: Dict[int, Dict[str, Union[torch.Tensor, float, int]]],
    state_dim: int,
    window_size: int,
    hidden_dim: int,
) -> Dict[int, List[Dict[str, Union[float, int]]]]:
    """
    Mask the flattened state by contiguous slices along the hidden dimension
    (size `window_size`) for every d_model and every layer. Assumes the flat
    vector layout is [channels*d_models*hidden_dim], contiguous over hidden_dim.
    """
    results: Dict[int, List[Dict[str, Union[float, int]]]] = {}

    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if state_dim % hidden_dim != 0:
        raise ValueError(f"state_dim={state_dim} not divisible by hidden_dim={hidden_dim}")

    n_channels = state_dim // hidden_dim  # includes all layers * d_model
    window_starts = list(range(0, hidden_dim, window_size))

    for window_idx, data in window_datasets.items():
        if window_idx not in probes:
            continue

        print(f"Evaluating masked probe for window {window_idx}...")
        probe = probes[window_idx]
        X = data["features"]
        Y = data["targets"]
        lengths = data["lengths"]
        weight = probe["weight"]
        bias = probe["bias"]

        window_results: List[Dict[str, Union[float, int]]] = []
        for start in window_starts:
            end = min(start + window_size, hidden_dim)
            mask = torch.zeros(state_dim)
            for c in range(n_channels):
                base = c * hidden_dim
                mask[base + start : base + end] = 1.0
            masked_X = X * mask
            pred = masked_X @ weight + bias
            mse = _masked_mse(pred, Y, lengths).item()
            window_results.append({"start": start, "end": end, "mse": mse})
        results[window_idx] = window_results
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Probe SSM hidden state c_T for local-window-copying."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        # default=Path("S4/outputs/2026-01-09/17-04-17-094343"),  # s4f-local-window-copying-v3 default 
        default=Path("S4/outputs/2026-01-13/14-23-06-106225"),  # s4f-local-window-copying-v3-fixed-gabors_multiWin8-8_layer2_frozenL1

        help="Path to the Hydra output directory for the run.",
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
        help="Device to run on (cpu, cuda, or auto). Defaults to cpu to avoid GPU OOM.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Optional cap on number of validation examples to process.",
    )
    parser.add_argument(
        "--mask-size",
        type=int,
        default=20,
        help="Number of coefficients to keep in each mask window.",
    )
    parser.add_argument(
        "--loader-batch-size",
        type=int,
        default=1,
        help="Override validation dataloader batch size to reduce memory (defaults to config).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    config = _load_config(run_dir)
    ckpt_path = args.checkpoint or _default_ckpt(run_dir)
    dataset_cfg = _to_dict(config.dataset)
    query_length = int(dataset_cfg.get("query_length", 0))
    l_window_max = int(dataset_cfg.get("l_window_max", 0))
    max_windows = int(dataset_cfg.get("n_windows_max", 0))

    if args.device is None or args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Loading checkpoint from {ckpt_path} on {device}")
    model = SequenceLightningModule.load_from_checkpoint(
        ckpt_path, config=config, map_location=device
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

    samples = collect_states_and_windows(
        model=model,
        loader=val_loader,
        device=device,
        query_length=query_length,
        l_window_max=l_window_max,
        max_examples=args.max_examples,
    )
    if not samples:
        raise RuntimeError("No validation samples were collected.")

    if max_windows <= 0:
        max_windows = max(len(s["windows"]) for s in samples)

    avg_mse = sum(s["mse"] for s in samples) / len(samples)
    print(f"Decoder MSE over collected samples: {avg_mse:.6f}")

    window_datasets, state_dim = build_probe_datasets(
        samples=samples, max_windows=max_windows, l_window_max=l_window_max
    )
    probes = train_linear_probes(window_datasets)

    hidden_dim = _infer_hidden_dim(samples)
    if hidden_dim is None:
        raise RuntimeError("Could not infer hidden_dim from collected layer_states.")

    mask_results = evaluate_masked_probes(
        window_datasets,
        probes,
        state_dim=state_dim,
        window_size=args.mask_size,
        hidden_dim=hidden_dim,
    )

    print(f"Collected {len(samples)} validation samples. State dim={state_dim}.")
    for window_idx in sorted(probes.keys()):
        probe_info = probes[window_idx]
        print(
            f"Window {window_idx}: n={probe_info['n']} "
            f"| full-state MSE={probe_info['train_mse']:.6f}"
        )
        for mask_info in mask_results.get(window_idx, []):
            print(
                f"  mask[{mask_info['start']:4d}:{mask_info['end']:4d}) -> "
                f"MSE={mask_info['mse']:.6f}"
            )


if __name__ == "__main__":
    main()
