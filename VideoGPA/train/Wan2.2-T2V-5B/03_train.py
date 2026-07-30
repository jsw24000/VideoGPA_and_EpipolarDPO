"""
Wan2.2 TI2V-5B text-only VideoGPA DPO LoRA smoke trainer.

This sibling keeps the official WAN VideoGPA DPO ingredients while removing the
TI2V image-conditioned first-frame path. It intentionally uses a direct PyTorch
loop for smoke execution because the local runnable environment does not include
the optional Lightning/W&B orchestration packages used by the official script.
The DPO loss, reference-policy comparison, flow-matching target, LoRA modules,
and optimizer/scheduler choices remain aligned with the official WAN script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import get_cosine_schedule_with_warmup

CURRENT_DIR = Path(__file__).resolve().parent
VIDEOGPA_ROOT = CURRENT_DIR.parents[1]
PROJECT_ROOT = VIDEOGPA_ROOT.parent
TRAIN_DIR = VIDEOGPA_ROOT / "train"
WAN_PATH = VIDEOGPA_ROOT / "Wan2.2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for path in [TRAIN_DIR, VIDEOGPA_ROOT, WAN_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vgm_common.config import resolve_experiment_config, write_resolved_config  # noqa: E402
from dataset import DPODataset, collate_fn  # noqa: E402
from loss import create_loss_strategy  # noqa: E402
from wan.modules.model import WanModel  # noqa: E402


DEFAULT_CONFIG = {
    "metric_name": "consistency_score",
    "metric_mode": "min",
    "min_gap": 0.05,
    "metric_threshold": 0.8,
    "motion_threshold": 0.001,
    "learning_rate": 5e-6,
    "beta": 1.0,
    "max_steps": 5,
    "warmup_steps": 500,
    "batch_size": 1,
    "accumulate_grad_batches": 1,
    "gradient_clip_val": 1.0,
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "lora_rank": 64,
    "lora_alpha": 128.0,
    "lora_dropout": 0.0,
    "lora_target_modules": ["q", "k", "v", "o"],
    "vae_stride": (4, 16, 16),
    "patch_size": (1, 2, 2),
    "enable_gradient_checkpointing": True,
    "seed": 2026,
    "device": 0,
    "save_steps": 5,
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_path(project_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def iter_dirs_limited(root: Path, max_depth: int = 5):
    root = root.resolve()
    if not root.exists():
        return
    for parent, dirs, _files in os.walk(root):
        parent_path = Path(parent)
        try:
            depth = len(parent_path.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            dirs[:] = []
            continue
        yield parent_path
        if depth == max_depth:
            dirs[:] = []


def find_unique_wan_model(models_root: Path) -> Path:
    candidates = sorted(
        {
            p.resolve()
            for p in iter_dirs_limited(models_root, max_depth=5)
            if p.is_dir()
            and p.name == "Wan2.2-TI2V-5B"
            and (p / "Wan2.2_VAE.pth").is_file()
            and (p / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
            and (p / "diffusion_pytorch_model.safetensors.index.json").is_file()
        }
    )
    if not candidates:
        raise FileNotFoundError(f"No WAN2.2-TI2V-5B model found under {models_root}")
    if len(candidates) > 1:
        raise RuntimeError("Multiple WAN candidates found:\n" + "\n".join(str(p) for p in candidates))
    return candidates[0]


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    if not args.config:
        raise ValueError("--config is required so paths can be resolved through the active VGM profile")
    cfg = resolve_experiment_config(args.config, args.run_dir, model_path_override=args.model_path)
    cfg.setdefault("training", {})
    cfg["project"]["task"] = "t2v"
    train_cfg = DEFAULT_CONFIG.copy()
    yaml_train = cfg.get("training", {})
    train_cfg.update(
        {
            "metric_name": cfg.get("scoring", {}).get("metric_name", train_cfg["metric_name"]),
            "metric_mode": cfg.get("scoring", {}).get("metric_mode", train_cfg["metric_mode"]),
            "min_gap": cfg.get("scoring", {}).get("min_score_gap", train_cfg["min_gap"]),
            "metric_threshold": cfg.get("scoring", {}).get("winner_score_threshold", train_cfg["metric_threshold"]),
            "motion_threshold": cfg.get("scoring", {}).get("motion_threshold", train_cfg["motion_threshold"]),
        }
    )
    train_cfg.update(
        {
            "learning_rate": yaml_train.get("learning_rate", train_cfg["learning_rate"]),
            "beta": yaml_train.get("dpo_beta", yaml_train.get("beta", train_cfg["beta"])),
            "max_steps": yaml_train.get("max_train_steps", yaml_train.get("max_steps", train_cfg["max_steps"])),
            "warmup_steps": yaml_train.get("warmup_steps", train_cfg["warmup_steps"]),
            "batch_size": yaml_train.get("batch_size_per_gpu", yaml_train.get("batch_size", train_cfg["batch_size"])),
            "accumulate_grad_batches": yaml_train.get(
                "gradient_accumulation_steps", train_cfg["accumulate_grad_batches"]
            ),
            "gradient_clip_val": yaml_train.get("gradient_clip_val", train_cfg["gradient_clip_val"]),
            "lora_rank": yaml_train.get("lora_rank", train_cfg["lora_rank"]),
            "lora_alpha": yaml_train.get("lora_alpha", train_cfg["lora_alpha"]),
            "lora_dropout": yaml_train.get("lora_dropout", train_cfg["lora_dropout"]),
            "lora_target_modules": yaml_train.get("lora_target_modules", train_cfg["lora_target_modules"]),
            "num_train_timesteps": yaml_train.get("num_train_timesteps", train_cfg["num_train_timesteps"]),
            "shift": yaml_train.get("shift", train_cfg["shift"]),
            "seed": yaml_train.get("seed", train_cfg["seed"]),
            "device": yaml_train.get("device", train_cfg["device"]),
            "save_steps": yaml_train.get("save_steps", train_cfg["save_steps"]),
            "enable_gradient_checkpointing": yaml_train.get(
                "enable_gradient_checkpointing", train_cfg["enable_gradient_checkpointing"]
            ),
        }
    )
    if args.max_train_steps:
        train_cfg["max_steps"] = args.max_train_steps
    if args.device is not None:
        train_cfg["device"] = args.device
    cfg["training_resolved"] = train_cfg
    return cfg


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sigma_from_timestep(timestep: torch.Tensor, num_train_timesteps: int = 1000, shift: float = 5.0) -> torch.Tensor:
    sigma = timestep.float() / num_train_timesteps
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def flow_matching_add_noise(z_0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    while sigma.dim() < z_0.dim():
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * z_0 + sigma * noise


def flow_matching_get_velocity(z_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - z_0


def compute_seq_len(z: torch.Tensor, patch_size: tuple[int, int, int]) -> int:
    _, _, f, h, w = z.shape
    return f * (h // patch_size[1]) * (w // patch_size[2])


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    base = model.base_model.model if hasattr(model, "base_model") else model
    for block in base.blocks:
        original_forward = block.forward

        def make_forward(fn):
            def ckpt_forward(*args, **kwargs):
                return torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)

            return ckpt_forward

        block.forward = make_forward(original_forward)


def trainable_stats(model: torch.nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    names = []
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            names.append(name)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
        "trainable_parameter_names": names,
    }


def clone_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def trainable_delta(before: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name in before:
            total += (param.detach().cpu().float() - before[name].float()).abs().sum().item()
    return total


def frozen_param_changed(model: torch.nn.Module, sample_before: dict[str, torch.Tensor]) -> bool:
    for name, before in sample_before.items():
        now = dict(model.named_parameters())[name].detach().cpu()
        if not torch.equal(before, now):
            return True
    return False


def sample_frozen(model: torch.nn.Module, limit: int = 16) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            out[name] = param.detach().cpu().clone()
            if len(out) >= limit:
                break
    return out


def ensure_encoded_conditions_no_image(run_dir: Path, metadata_path: Path) -> None:
    payload = read_json(metadata_path)
    pairs = payload.get("pairs", [])
    if not pairs:
        raise ValueError("encoded manifest contains no pairs")
    for pair in pairs:
        cond = torch_load(run_dir / pair["condition_path"])
        if "image_latent" in cond:
            raise ValueError(f"T2V condition contains image_latent: {pair['condition_path']}")
        emb = cond.get("encoder_hidden_states")
        if not isinstance(emb, torch.Tensor) or emb.numel() == 0:
            raise ValueError(f"Invalid text embedding in {pair['condition_path']}")


def build_dataloader(run_dir: Path, metadata_path: Path, cfg: dict[str, Any]):
    dataset = DPODataset(
        base_path=str(run_dir),
        metadata_path=str(metadata_path),
        metric_name=cfg["metric_name"],
        metric_mode=cfg["metric_mode"],
        min_gap=0.0,
        metric_threshold=None,
        motion_threshold=0.0,
    )
    if len(dataset) < 2:
        raise RuntimeError(f"Need at least 2 preference pairs for DPO smoke, got {len(dataset)}")
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=generator,
    )


def shared_step(
    transformer: torch.nn.Module,
    ref_transformer: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[Any, dict[str, float]]:
    x_win = batch["x_win"].to(device=device, dtype=torch.bfloat16)
    x_lose = batch["x_lose"].to(device=device, dtype=torch.bfloat16)
    prompt_emb = batch["prompt_emb"].to(device=device, dtype=torch.bfloat16)
    if "image_latent" in batch or "image_emb" in batch:
        raise ValueError("T2V training batch unexpectedly contains image condition")

    batch_size = x_win.shape[0]
    patch_size = tuple(cfg["patch_size"])
    seq_len = compute_seq_len(x_win, patch_size)
    timesteps = torch.randint(1, int(cfg["num_train_timesteps"]), (batch_size,), device=device)
    sigma = get_sigma_from_timestep(timesteps, int(cfg["num_train_timesteps"]), float(cfg["shift"]))
    noise = torch.randn_like(x_win)

    x_win_noisy = flow_matching_add_noise(x_win, noise, sigma)
    x_lose_noisy = flow_matching_add_noise(x_lose, noise, sigma)
    context_list = [prompt_emb[b] for b in range(batch_size)]
    x_win_input = [x_win_noisy[b] for b in range(batch_size)]
    x_lose_input = [x_lose_noisy[b] for b in range(batch_size)]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        v_win_ref = torch.stack(ref_transformer(x_win_input, t=timesteps, context=context_list, seq_len=seq_len))
        v_lose_ref = torch.stack(ref_transformer(x_lose_input, t=timesteps, context=context_list, seq_len=seq_len))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        v_win_pred = torch.stack(transformer(x_win_input, t=timesteps, context=context_list, seq_len=seq_len))
        v_lose_pred = torch.stack(transformer(x_lose_input, t=timesteps, context=context_list, seq_len=seq_len))

    v_win_target = flow_matching_get_velocity(x_win.float(), noise.float())
    v_lose_target = flow_matching_get_velocity(x_lose.float(), noise.float())
    loss_out = loss_fn(
        v_win_pred.float(),
        v_lose_pred.float(),
        v_win_ref.float(),
        v_lose_ref.float(),
        v_win_target,
        v_lose_target,
    )
    with torch.no_grad():
        win_policy_err = (v_win_pred.float() - v_win_target).pow(2).mean().item()
        lose_policy_err = (v_lose_pred.float() - v_lose_target).pow(2).mean().item()
        win_ref_err = (v_win_ref.float() - v_win_target).pow(2).mean().item()
        lose_ref_err = (v_lose_ref.float() - v_lose_target).pow(2).mean().item()
    debug = {
        "winner_policy_error": win_policy_err,
        "loser_policy_error": lose_policy_err,
        "winner_reference_error": win_ref_err,
        "loser_reference_error": lose_ref_err,
        "winner_loser_latent_shape": list(x_win.shape),
        "timestep_shape": list(timesteps.shape),
        "noise_shape": list(noise.shape),
        "text_embedding_shape": list(prompt_emb.shape),
        "seq_len": seq_len,
    }
    return loss_out, debug


def save_checkpoint(
    checkpoint_dir: Path,
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
    resolved_config: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    write_json(checkpoint_dir / "trainer_state.json", state)
    write_yaml(checkpoint_dir / "config_resolved.yaml", resolved_config)


def train(args: argparse.Namespace) -> None:
    cfg_all = resolve_config(args)
    cfg = cfg_all["training_resolved"]
    run_dir = Path(cfg_all["paths"]["run_dir"]).resolve()
    model_path = Path(cfg_all["paths"]["wan_model_path"]).resolve()
    metadata_path = Path(args.metadata_path or run_dir / "manifests/encoded_pairs.json").expanduser().resolve()
    output_dir = Path(args.output_dir or run_dir).expanduser().resolve()
    checkpoint_root = output_dir / "checkpoints"

    print("Resolved paths:")
    print(f"  run_dir={run_dir}")
    print(f"  model_path={model_path}")
    print(f"  metadata_path={metadata_path}")
    print(f"  checkpoint_root={checkpoint_root}")
    print("Training mode:")
    print("  task=t2v")
    print("  image-conditioned branch=false")
    print(f"  dpo_beta={cfg['beta']}")
    print(f"  learning_rate={cfg['learning_rate']}")
    print(f"  max_steps={cfg['max_steps']}")
    print(f"  effective_global_batch_size={cfg['batch_size'] * cfg['accumulate_grad_batches']}")

    if "test_t2v" in str(metadata_path) or "test_i2v" in str(metadata_path):
        raise ValueError(f"Refusing test metadata: {metadata_path}")
    ensure_encoded_conditions_no_image(run_dir, metadata_path)

    set_seed(int(cfg["seed"]))
    device = torch.device(f"cuda:{int(cfg['device'])}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    dataloader = build_dataloader(run_dir, metadata_path, cfg)
    print(f"DPO dataset pairs: {len(dataloader.dataset)}")

    print(f"Loading policy WanModel from {model_path}")
    transformer = WanModel.from_pretrained(str(model_path))
    transformer.to(device=device, dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=float(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_target_modules"]),
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.to(device)
    if cfg.get("enable_gradient_checkpointing"):
        enable_gradient_checkpointing(transformer)
    stats = trainable_stats(transformer)
    print("LoRA target modules:", cfg["lora_target_modules"])
    print("Trainable parameter count:", stats["trainable_params"])
    print("Trainable parameter ratio:", stats["trainable_ratio"])
    print("Trainable parameter names:")
    for name in stats["trainable_parameter_names"]:
        print(f"  {name}")
    if not stats["trainable_parameter_names"] or not all("lora_" in name for name in stats["trainable_parameter_names"]):
        raise RuntimeError("Expected only LoRA parameters to be trainable")

    print(f"Loading frozen reference WanModel from {model_path}")
    ref_transformer = WanModel.from_pretrained(str(model_path))
    ref_transformer.to(device=device, dtype=torch.bfloat16)
    ref_transformer.requires_grad_(False)
    ref_transformer.eval()
    transformer.train()

    loss_fn = create_loss_strategy(strategy="dpo", beta=float(cfg["beta"]))
    optimizer = torch.optim.AdamW((p for p in transformer.parameters() if p.requires_grad), lr=float(cfg["learning_rate"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["warmup_steps"]),
        num_training_steps=int(cfg["max_steps"]),
    )

    lora_before = clone_trainable(transformer)
    frozen_before = sample_frozen(transformer)
    ref_before = sample_frozen(ref_transformer)
    metrics = []
    grad_nonzero = False
    step = 0
    data_iter = iter(dataloader)
    while step < int(cfg["max_steps"]):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_out, debug = shared_step(transformer, ref_transformer, loss_fn, batch, cfg, device)
        if not torch.isfinite(loss_out.loss).item():
            raise RuntimeError(f"Non-finite DPO loss at step {step + 1}")
        loss_out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in transformer.parameters() if p.requires_grad],
            max_norm=float(cfg["gradient_clip_val"]),
        )
        grad_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
        grad_nonzero = grad_nonzero or grad_value > 0
        optimizer.step()
        scheduler.step()
        step += 1
        allocated = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        reserved = torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
        row = {
            "step": step,
            "total_loss": float(loss_out.loss.detach().cpu().item()),
            "dpo_loss": float(loss_out.loss.detach().cpu().item()),
            "winner_policy_error": debug["winner_policy_error"],
            "loser_policy_error": debug["loser_policy_error"],
            "winner_reference_error": debug["winner_reference_error"],
            "loser_reference_error": debug["loser_reference_error"],
            "implicit_reward_margin": float(loss_out.reward_margin.detach().cpu().item()),
            "grad_norm": grad_value,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gpu_allocated_gb": allocated,
            "gpu_reserved_gb": reserved,
            "step_time_sec": time.time() - step_start,
            "debug_shapes": debug,
        }
        metrics.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if step % int(cfg["save_steps"]) == 0 or step == int(cfg["max_steps"]):
            save_checkpoint(
                checkpoint_root / f"step_{step:06d}",
                transformer,
                optimizer,
                scheduler,
                {
                    "step": step,
                    "time": dt.datetime.now().isoformat(),
                    "config": cfg,
                    "metrics": metrics,
                    "trainable_stats": stats,
                },
                cfg_all,
            )

    lora_delta = trainable_delta(lora_before, transformer)
    base_changed = frozen_param_changed(transformer, frozen_before)
    ref_changed = frozen_param_changed(ref_transformer, ref_before)
    final_ckpt = checkpoint_root / f"step_{step:06d}"
    if lora_delta <= 0:
        raise RuntimeError("LoRA parameters did not change")
    if not grad_nonzero:
        raise RuntimeError("No non-zero gradient observed")
    if base_changed:
        raise RuntimeError("A sampled frozen base parameter changed")
    if ref_changed:
        raise RuntimeError("A sampled reference parameter changed")
    if not (final_ckpt / "adapter_config.json").exists():
        raise RuntimeError(f"Missing adapter_config.json in {final_ckpt}")

    del ref_transformer
    del transformer
    torch.cuda.empty_cache()
    print(f"Reloading checkpoint adapter from {final_ckpt}")
    reload_base = WanModel.from_pretrained(str(model_path))
    reloaded = PeftModel.from_pretrained(reload_base, str(final_ckpt), adapter_name="default")
    del reloaded, reload_base
    torch.cuda.empty_cache()

    summary = {
        "status": "PASS",
        "task": "t2v",
        "image_conditioned_branch": False,
        "steps": step,
        "checkpoint_path": str(final_ckpt),
        "checkpoint_reloaded": True,
        "lora_delta_l1": lora_delta,
        "grad_nonzero": grad_nonzero,
        "base_parameters_changed": base_changed,
        "reference_parameters_changed": ref_changed,
        "trainable_stats": stats,
        "metrics": metrics,
    }
    write_json(run_dir / "reports/training_summary.json", summary)
    write_resolved_config(run_dir, cfg_all)
    print(f"Training smoke PASS. Checkpoint: {final_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2 T2V VideoGPA DPO LoRA smoke trainer")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_dir", "--run-dir", dest="run_dir", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
