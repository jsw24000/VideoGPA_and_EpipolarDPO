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
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
try:
    from peft import set_peft_model_state_dict
except ImportError:  # pragma: no cover - depends on remote PEFT version
    set_peft_model_state_dict = None
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data.distributed import DistributedSampler
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
from resume_utils import (  # noqa: E402
    ResumeError,
    compute_resume_data_cursor,
    discover_latest_checkpoint,
    ensure_checkpoint_save_target_safe,
    parse_checkpoint_step,
    read_json as read_resume_json,
    read_yaml as read_resume_yaml,
    validate_checkpoint_manifest,
    validate_resume_config,
    validate_resume_metadata,
)
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


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def cleanup_distributed(state: dict[str, int | bool]) -> None:
    if not state["distributed"]:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(state: dict[str, int | bool]) -> None:
    if not state["distributed"]:
        return
    import torch.distributed as dist

    dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


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


def build_dataloader(run_dir: Path, metadata_path: Path, cfg: dict[str, Any], dist_state: dict[str, int | bool]):
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
    sampler = None
    shuffle = True
    if dist_state["distributed"]:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(dist_state["world_size"]),
            rank=int(dist_state["rank"]),
            shuffle=True,
            seed=int(cfg["seed"]),
            drop_last=False,
        )
        shuffle = False
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
        generator=generator,
    ), sampler


def resolve_resume_checkpoint(args: argparse.Namespace, checkpoint_root: Path) -> Path | None:
    if args.resume_from_checkpoint:
        return Path(args.resume_from_checkpoint).expanduser().resolve()
    if args.resume:
        return discover_latest_checkpoint(checkpoint_root)
    if args.validate_resume_only:
        raise ValueError("--validate_resume_only requires --resume or --resume_from_checkpoint")
    return None


def _load_adapter_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - remote dependency
            raise RuntimeError("safetensors is required to load adapter_model.safetensors") from exc
        state = load_file(str(path), device="cpu")
    else:
        state = torch_load(path)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Adapter state is empty or invalid: {path}")
    non_tensors = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise RuntimeError(f"Adapter state contains non-tensor entries: {non_tensors[:8]}")
    return state


def _adapter_key_candidates(key: str) -> list[str]:
    candidates = [key]
    replacements = {
        ".lora_A.": ".lora_A.default.",
        ".lora_B.": ".lora_B.default.",
        ".lora_embedding_A.": ".lora_embedding_A.default.",
        ".lora_embedding_B.": ".lora_embedding_B.default.",
    }
    for old, new in replacements.items():
        if old in key:
            candidates.append(key.replace(old, new))
    if not key.startswith("base_model.model."):
        candidates.extend([f"base_model.model.{candidate}" for candidate in list(candidates)])
    return candidates


def _map_adapter_keys_to_model(
    adapter_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, str]:
    model_state = model.state_dict()
    model_lora_keys = sorted(key for key in model_state if "lora_" in key)
    mapped: dict[str, str] = {}
    missing = []
    shape_mismatches = []
    for checkpoint_key, checkpoint_value in adapter_state.items():
        model_key = next((candidate for candidate in _adapter_key_candidates(checkpoint_key) if candidate in model_state), None)
        if model_key is None:
            missing.append(checkpoint_key)
            continue
        if tuple(model_state[model_key].shape) != tuple(checkpoint_value.shape):
            shape_mismatches.append(
                f"{checkpoint_key}: checkpoint={tuple(checkpoint_value.shape)} model={tuple(model_state[model_key].shape)}"
            )
            continue
        mapped[checkpoint_key] = model_key
    missing_model_lora = sorted(set(model_lora_keys) - set(mapped.values()))
    if missing or shape_mismatches or missing_model_lora:
        details = []
        if missing:
            details.append(f"unmatched checkpoint LoRA keys: {missing[:8]}")
        if shape_mismatches:
            details.append(f"shape mismatches: {shape_mismatches[:8]}")
        if missing_model_lora:
            details.append(f"model LoRA keys absent from checkpoint: {missing_model_lora[:8]}")
        raise RuntimeError("; ".join(details))
    return mapped


def load_lora_adapter_strict(model: torch.nn.Module, adapter_path: Path) -> dict[str, Any]:
    if set_peft_model_state_dict is None:
        raise RuntimeError("This PEFT version does not expose set_peft_model_state_dict; refusing unsafe resume")
    adapter_state = _load_adapter_tensor_file(adapter_path)
    mapped_keys = _map_adapter_keys_to_model(adapter_state, model)
    result = set_peft_model_state_dict(model, adapter_state, adapter_name="default")
    missing_lora = [key for key in getattr(result, "missing_keys", []) if "lora_" in key]
    unexpected_lora = [key for key in getattr(result, "unexpected_keys", []) if "lora_" in key]
    if missing_lora or unexpected_lora:
        raise RuntimeError(f"PEFT adapter load mismatch: missing={missing_lora[:8]} unexpected={unexpected_lora[:8]}")

    model_state = model.state_dict()
    max_abs_diff = 0.0
    worst_key = None
    for checkpoint_key, model_key in mapped_keys.items():
        expected = adapter_state[checkpoint_key].to(dtype=model_state[model_key].dtype)
        actual = model_state[model_key].detach().cpu()
        diff = (actual.float() - expected.float()).abs().max().item()
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst_key = checkpoint_key
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Adapter verification failed: max_abs_diff={max_abs_diff} at {worst_key}")
    return {
        "adapter_path": str(adapter_path),
        "tensor_count": len(adapter_state),
        "parameter_count": int(sum(tensor.numel() for tensor in adapter_state.values())),
        "max_abs_diff": max_abs_diff,
    }


def load_optimizer_state_strict(optimizer: torch.optim.Optimizer, path: Path) -> dict[str, Any]:
    state = torch_load(path)
    if not isinstance(state, dict) or "state" not in state or "param_groups" not in state:
        raise RuntimeError(f"Invalid optimizer checkpoint: {path}")
    current = optimizer.state_dict()
    if len(current["param_groups"]) != len(state["param_groups"]):
        raise RuntimeError(
            f"Optimizer param group count mismatch: checkpoint={len(state['param_groups'])} "
            f"current={len(current['param_groups'])}"
        )
    for idx, (current_group, checkpoint_group) in enumerate(zip(current["param_groups"], state["param_groups"])):
        if len(current_group["params"]) != len(checkpoint_group["params"]):
            raise RuntimeError(
                f"Optimizer param count mismatch in group {idx}: "
                f"checkpoint={len(checkpoint_group['params'])} current={len(current_group['params'])}"
            )
    optimizer.load_state_dict(state)
    return {
        "optimizer_path": str(path),
        "param_groups": len(state["param_groups"]),
        "state_entries": len(state["state"]),
    }


def load_scheduler_state_strict(scheduler: Any, path: Path, resume_step: int) -> dict[str, Any]:
    state = torch_load(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid scheduler checkpoint: {path}")
    scheduler.load_state_dict(state)
    if int(getattr(scheduler, "last_epoch", -1)) != int(resume_step):
        raise RuntimeError(f"Scheduler last_epoch mismatch after load: {scheduler.last_epoch} != {resume_step}")
    return {
        "scheduler_path": str(path),
        "last_epoch": int(scheduler.last_epoch),
        "last_lr": [float(value) for value in scheduler.get_last_lr()],
    }


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state_if_present(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"restored": False, "reason": "rng_state.pt not present in checkpoint"}
    state = torch_load_full(path)
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "torch_cpu_rng_state" in state:
        torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
    return {"restored": True, "path": str(path)}


def make_data_iterator_at_cursor(
    dataloader: torch.utils.data.DataLoader,
    sampler: DistributedSampler | None,
    cursor: dict[str, int],
) -> tuple[Any, int]:
    data_epoch = int(cursor["data_epoch"])
    data_offset = int(cursor["data_offset"])
    if sampler is not None:
        sampler.set_epoch(data_epoch)
    data_iter = iter(dataloader)
    for _ in range(data_offset):
        try:
            next(data_iter)
        except StopIteration as exc:
            raise RuntimeError(f"Could not advance dataloader to resume offset {data_offset}") from exc
    return data_iter, data_epoch


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
    ensure_checkpoint_save_target_safe(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    torch.save(capture_rng_state(), checkpoint_dir / "rng_state.pt")
    write_json(checkpoint_dir / "trainer_state.json", state)
    write_yaml(checkpoint_dir / "config_resolved.yaml", resolved_config)


def train(args: argparse.Namespace) -> None:
    dist_state = init_distributed()
    try:
        cfg_all = resolve_config(args)
        cfg = cfg_all["training_resolved"]
        if dist_state["distributed"]:
            cfg["device"] = int(dist_state["local_rank"])
        run_dir = Path(cfg_all["paths"]["run_dir"]).resolve()
        model_path = Path(cfg_all["paths"]["wan_model_path"]).resolve()
        metadata_path = Path(args.metadata_path or run_dir / "manifests/encoded_pairs.json").expanduser().resolve()
        output_dir = Path(args.output_dir or run_dir).expanduser().resolve()
        checkpoint_root = output_dir / "checkpoints"
        is_main = bool(dist_state["is_main"])
        resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_root)
        resume_files = None
        resume_step = 0
        resume_trainer_state: dict[str, Any] | None = None
        resume_scheduler_state: dict[str, Any] | None = None
        resume_config_report: dict[str, Any] | None = None
        resume_report: dict[str, Any] = {"enabled": resume_checkpoint is not None}

        def log(*items: object) -> None:
            if is_main:
                print(*items)

        log("Resolved paths:")
        log(f"  run_dir={run_dir}")
        log(f"  model_path={model_path}")
        log(f"  metadata_path={metadata_path}")
        log(f"  checkpoint_root={checkpoint_root}")
        log("Training mode:")
        log("  task=t2v")
        log("  image-conditioned branch=false")
        log(f"  distributed={dist_state['distributed']}")
        log(f"  world_size={dist_state['world_size']}")
        log(f"  dpo_beta={cfg['beta']}")
        log(f"  learning_rate={cfg['learning_rate']}")
        log(f"  max_steps={cfg['max_steps']}")
        effective_batch = int(cfg["batch_size"]) * int(cfg["accumulate_grad_batches"]) * int(dist_state["world_size"])
        log(f"  effective_global_batch_size={effective_batch}")
        if resume_checkpoint is not None:
            log(f"  resume_checkpoint={resume_checkpoint}")
            resume_files = validate_checkpoint_manifest(resume_checkpoint)
            checkpoint_config = read_resume_yaml(resume_files["config_resolved"])
            assert isinstance(checkpoint_config, dict)
            resume_config_report = validate_resume_config(cfg_all, checkpoint_config)
            resume_trainer_state = read_resume_json(resume_files["trainer_state"])
            resume_scheduler_state = torch_load(resume_files["scheduler"])
            assert isinstance(resume_trainer_state, dict)
            assert isinstance(resume_scheduler_state, dict)
            resume_step = validate_resume_metadata(resume_checkpoint, resume_trainer_state, resume_scheduler_state)
            if resume_step >= int(cfg["max_steps"]):
                raise ResumeError(f"Checkpoint step {resume_step} is already >= max_steps {cfg['max_steps']}")
            resume_report.update(
                {
                    "checkpoint": str(resume_checkpoint),
                    "resume_step": resume_step,
                    "next_step": resume_step + 1,
                    "config": resume_config_report,
                }
            )

        if "test_t2v" in str(metadata_path) or "test_i2v" in str(metadata_path):
            raise ValueError(f"Refusing test metadata: {metadata_path}")
        ensure_encoded_conditions_no_image(run_dir, metadata_path)

        set_seed(int(cfg["seed"]) + int(dist_state["rank"]))
        device = torch.device(f"cuda:{int(cfg['device'])}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)

        dataloader, sampler = build_dataloader(run_dir, metadata_path, cfg, dist_state)
        log(f"DPO dataset pairs: {len(dataloader.dataset)}")

        log(f"Loading policy WanModel from {model_path}")
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
        if resume_files is not None:
            adapter_report = load_lora_adapter_strict(transformer, resume_files["adapter_model"])
            resume_report["adapter"] = adapter_report
        stats = trainable_stats(transformer)
        log("LoRA target modules:", cfg["lora_target_modules"])
        log("Trainable parameter count:", stats["trainable_params"])
        log("Trainable parameter ratio:", stats["trainable_ratio"])
        log("Trainable parameter names:")
        for name in stats["trainable_parameter_names"]:
            log(f"  {name}")
        if not stats["trainable_parameter_names"] or not all("lora_" in name for name in stats["trainable_parameter_names"]):
            raise RuntimeError("Expected only LoRA parameters to be trainable")

        lora_before = clone_trainable(transformer)
        frozen_before = sample_frozen(transformer)

        if dist_state["distributed"]:
            find_unused = os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "0") == "1"
            transformer = DistributedDataParallel(
                transformer,
                device_ids=[int(dist_state["local_rank"])] if device.type == "cuda" else None,
                output_device=int(dist_state["local_rank"]) if device.type == "cuda" else None,
                find_unused_parameters=find_unused,
            )

        log(f"Loading frozen reference WanModel from {model_path}")
        ref_transformer = WanModel.from_pretrained(str(model_path))
        ref_transformer.to(device=device, dtype=torch.bfloat16)
        ref_transformer.requires_grad_(False)
        ref_transformer.eval()
        if any("lora_" in name for name, _param in ref_transformer.named_parameters()):
            raise RuntimeError("Reference model unexpectedly contains LoRA parameters")
        transformer.train()

        loss_fn = create_loss_strategy(strategy="dpo", beta=float(cfg["beta"]))
        optimizer = torch.optim.AdamW((p for p in transformer.parameters() if p.requires_grad), lr=float(cfg["learning_rate"]))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg["warmup_steps"]),
            num_training_steps=int(cfg["max_steps"]),
        )
        if resume_files is not None:
            optimizer_report = load_optimizer_state_strict(optimizer, resume_files["optimizer"])
            scheduler_report = load_scheduler_state_strict(scheduler, resume_files["scheduler"], resume_step)
            resume_report["optimizer"] = optimizer_report
            resume_report["scheduler"] = scheduler_report
            resume_report["reference_model"] = {"has_lora_parameters": False, "requires_grad": False}

        ref_before = sample_frozen(ref_transformer)
        metrics = []
        grad_nonzero = False
        accum_steps = max(1, int(cfg["accumulate_grad_batches"]))
        step = resume_step
        if resume_files is not None:
            cursor = compute_resume_data_cursor(step, accum_steps, len(dataloader))
            data_iter, data_epoch = make_data_iterator_at_cursor(dataloader, sampler, cursor)
            rng_report = restore_rng_state_if_present(resume_files["rng_state"])
            resume_report["data_cursor"] = cursor
            resume_report["rng"] = rng_report
            resume_report["first_new_update_step"] = step + 1
            resume_report["next_checkpoint_step"] = ((step // int(cfg["save_steps"])) + 1) * int(cfg["save_steps"])
            resume_report["current_lr_before_update"] = float(scheduler.get_last_lr()[0])
            if is_main:
                print("Resume validation report:")
                print(json.dumps(resume_report, indent=2, ensure_ascii=False))
            distributed_barrier(dist_state)
            if args.validate_resume_only:
                if is_main:
                    print("Validate-resume-only PASS. No backward, optimizer step, scheduler step, checkpoint save, or summary write was performed.")
                distributed_barrier(dist_state)
                return
        else:
            data_epoch = 0
            if sampler is not None:
                sampler.set_epoch(data_epoch)
            data_iter = iter(dataloader)

        def next_batch() -> dict[str, Any]:
            nonlocal data_iter, data_epoch
            try:
                return next(data_iter)
            except StopIteration:
                data_epoch += 1
                if sampler is not None:
                    sampler.set_epoch(data_epoch)
                data_iter = iter(dataloader)
                return next(data_iter)

        while step < int(cfg["max_steps"]):
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            debug: dict[str, Any] = {}
            step_loss = 0.0
            reward_margin = 0.0
            for micro_idx in range(accum_steps):
                batch = next_batch()
                sync_context = (
                    transformer.no_sync()
                    if dist_state["distributed"] and micro_idx < accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    loss_out, debug = shared_step(transformer, ref_transformer, loss_fn, batch, cfg, device)
                    if not torch.isfinite(loss_out.loss).item():
                        raise RuntimeError(f"Non-finite DPO loss at step {step + 1}, microbatch {micro_idx + 1}")
                    (loss_out.loss / accum_steps).backward()
                step_loss += float(loss_out.loss.detach().cpu().item()) / accum_steps
                reward_margin += float(loss_out.reward_margin.detach().cpu().item()) / accum_steps

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
                "rank": int(dist_state["rank"]),
                "world_size": int(dist_state["world_size"]),
                "total_loss": step_loss,
                "dpo_loss": step_loss,
                "winner_policy_error": debug["winner_policy_error"],
                "loser_policy_error": debug["loser_policy_error"],
                "winner_reference_error": debug["winner_reference_error"],
                "loser_reference_error": debug["loser_reference_error"],
                "implicit_reward_margin": reward_margin,
                "grad_norm": grad_value,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gpu_allocated_gb": allocated,
                "gpu_reserved_gb": reserved,
                "step_time_sec": time.time() - step_start,
                "debug_shapes": debug,
            }
            if is_main:
                metrics.append(row)
                print(json.dumps(row, ensure_ascii=False))
            should_save = step % int(cfg["save_steps"]) == 0 or step == int(cfg["max_steps"])
            if should_save:
                checkpoint_path = checkpoint_root / f"step_{step:06d}"
                ensure_checkpoint_save_target_safe(checkpoint_path)
                distributed_barrier(dist_state)
            if is_main and should_save:
                save_checkpoint(
                    checkpoint_path,
                    unwrap_model(transformer),
                    optimizer,
                    scheduler,
                    {
                        "step": step,
                        "time": dt.datetime.now().isoformat(),
                        "config": cfg,
                        "metrics": metrics,
                        "trainable_stats": stats,
                        "distributed": dist_state,
                        "effective_global_batch_size": effective_batch,
                    },
                    cfg_all,
                )
            distributed_barrier(dist_state)

        if dist_state["distributed"]:
            import torch.distributed as dist

            flag = torch.tensor([1 if grad_nonzero else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            grad_nonzero = bool(flag.item())

        final_ckpt = checkpoint_root / f"step_{step:06d}"
        if is_main:
            policy_model = unwrap_model(transformer)
            lora_delta = trainable_delta(lora_before, policy_model)
            base_changed = frozen_param_changed(policy_model, frozen_before)
            ref_changed = frozen_param_changed(ref_transformer, ref_before)
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

        distributed_barrier(dist_state)
        del ref_transformer
        del transformer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if is_main:
            print(f"Reloading checkpoint adapter from {final_ckpt}")
            reload_base = WanModel.from_pretrained(str(model_path))
            reloaded = PeftModel.from_pretrained(reload_base, str(final_ckpt), adapter_name="default")
            del reloaded, reload_base
            if device.type == "cuda":
                torch.cuda.empty_cache()

            summary = {
                "status": "PASS",
                "task": "t2v",
                "image_conditioned_branch": False,
                "distributed": dist_state,
                "effective_global_batch_size": effective_batch,
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
            print(f"Training PASS. Checkpoint: {final_ckpt}")
        distributed_barrier(dist_state)
    finally:
        cleanup_distributed(dist_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2 T2V VideoGPA DPO LoRA smoke trainer")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_dir", "--run-dir", dest="run_dir", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest complete checkpoint under output_dir/checkpoints")
    parser.add_argument("--resume_from_checkpoint", "--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--validate_resume_only",
        "--validate-resume-only",
        action="store_true",
        help="Load model/optimizer/scheduler/data cursor for resume validation, then exit without training or writing outputs",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
