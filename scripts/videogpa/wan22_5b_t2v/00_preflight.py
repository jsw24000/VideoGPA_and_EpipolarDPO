from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import iter_dirs_limited, read_json, read_yaml, resolve_config, sha256_file, write_json, write_yaml
from vgm_common.config import write_resolved_config
from vgm_common.paths import get_manifest_root, get_model_root, get_output_root


REQUIRED_REPO_PATHS = [
    "VideoGPA",
    "VideoGPA/train/Wan2.2-TI2V-5B",
    "VideoGPA/train/CogVideoX-5B",
    "VideoGPA/train/CogVideoX-I2V-5B",
    "VideoGPA/train/01_preference_pair.py",
    "VideoGPA/train/dataset.py",
    "VideoGPA/train/loss.py",
    "VideoGPA/generate",
    "VideoGPA/Wan2.2",
]

REQUIRED_MANIFEST_PATHS = [
    "videogpa_protocol",
    "videogpa_protocol/train_t2v.json",
]

OPTIONAL_MANIFEST_PATHS = [
    "videogpa_protocol/train_i2v.json",
    "videogpa_protocol/test_t2v.json",
    "videogpa_protocol/test_i2v.json",
    "master_all.jsonl",
    "caption_index.jsonl",
]


def run_text(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def import_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "python_executable": sys.executable,
    }
    for name in [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "decord",
        "lpips",
        "easydict",
        "yaml",
    ]:
        try:
            mod = importlib.import_module(name)
            versions[name] = str(getattr(mod, "__version__", "ok"))
        except Exception as exc:
            versions[name] = f"IMPORT_ERROR {type(exc).__name__}: {exc}"
    try:
        import torch

        versions["torch_cuda_runtime"] = str(torch.version.cuda)
        versions["torch_cuda_available"] = str(torch.cuda.is_available())
        versions["torch_cuda_device_count"] = str(torch.cuda.device_count())
        versions["torch_cuda_bf16_supported"] = str(
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None
        )
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            versions[f"gpu_{idx}"] = f"{props.name}, {props.total_memory} bytes"
    except Exception as exc:
        versions["torch_cuda_probe"] = f"ERROR {type(exc).__name__}: {exc}"
    return versions


def audit_runtime_tools(project_root: Path) -> dict[str, str]:
    output_root = get_output_root()
    disk = shutil.disk_usage(project_root)
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    writable = False
    writable_error = ""
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".videogpa_preflight_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        writable = True
    except Exception as exc:
        writable_error = f"{type(exc).__name__}: {exc}"

    return {
        "ffmpeg_path": ffmpeg_path or "",
        "ffmpeg_version": run_text(["ffmpeg", "-version"], project_root).splitlines()[0] if ffmpeg_path else "missing",
        "ffprobe_path": ffprobe_path or "",
        "nvidia_smi_query": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            project_root,
        )
        if shutil.which("nvidia-smi")
        else "missing",
        "disk_total_gb": f"{disk.total / (1024**3):.2f}",
        "disk_free_gb": f"{disk.free / (1024**3):.2f}",
        "outputs_writable": str(writable),
        "outputs_writable_error": writable_error,
    }


def audit_models(project_root: Path, resolved: dict) -> dict:
    wan = Path(resolved["paths"]["wan_model_path"])
    vggt = Path(resolved["paths"]["vggt_model_path"])
    official_lora = project_root / "VideoGPA/checkpoints/VideoGPA-Wan2.2TI2V-lora"
    wan_files = {
        "config_json": wan / "config.json",
        "configuration_json": wan / "configuration.json",
        "vae": wan / "Wan2.2_VAE.pth",
        "t5": wan / "models_t5_umt5-xxl-enc-bf16.pth",
        "dit_index": wan / "diffusion_pytorch_model.safetensors.index.json",
        "t5_tokenizer": wan / "google/umt5-xxl",
    }
    dit_shards = sorted(wan.glob("diffusion_pytorch_model-*.safetensors"))
    vggt_files = {
        "config_json": vggt / "config.json",
        "model_safetensors": vggt / "model.safetensors",
        "model_pt": vggt / "model.pt",
    }
    return {
        "wan_root": str(wan),
        "wan_required": {key: path.exists() for key, path in wan_files.items()},
        "wan_dit_shard_count": len(dit_shards),
        "wan_complete": all(path.exists() for path in wan_files.values()) and bool(dit_shards),
        "vggt_root": str(vggt),
        "vggt_required": {key: path.exists() for key, path in vggt_files.items()},
        "vggt_complete": vggt_files["config_json"].exists()
        and (vggt_files["model_safetensors"].exists() or vggt_files["model_pt"].exists()),
        "official_videogpa_wan_ti2v_lora": str(official_lora) if official_lora.exists() else None,
        "official_videogpa_wan_ti2v_lora_complete": (official_lora / "adapter_config.json").exists()
        and (official_lora / "adapter_model.safetensors").exists(),
    }


def audit_model_candidates(project_root: Path) -> dict:
    models_root = get_model_root()
    wan_candidates = [
        str(p.resolve())
        for p in iter_dirs_limited(models_root, max_depth=5)
        if p.name == "Wan2.2-TI2V-5B"
        and (p / "Wan2.2_VAE.pth").is_file()
        and (p / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
        and (p / "diffusion_pytorch_model.safetensors.index.json").is_file()
    ]
    vggt_candidates = [
        str(p.resolve())
        for p in iter_dirs_limited(models_root, max_depth=5)
        if p.name == "VGGT-1B"
        and (p / "config.json").is_file()
        and ((p / "model.safetensors").is_file() or (p / "model.pt").is_file())
    ]
    return {"wan_candidates": sorted(wan_candidates), "vggt_candidates": sorted(vggt_candidates)}


def audit_wan_defaults(project_root: Path) -> dict:
    textimage_path = project_root / "VideoGPA/Wan2.2/wan/textimage2video.py"
    config_path = project_root / "VideoGPA/Wan2.2/wan/configs/wan_ti2v_5B.py"
    text = textimage_path.read_text(encoding="utf-8")
    cfg_text = config_path.read_text(encoding="utf-8")

    def signature_default(fn_name: str, arg_name: str) -> str:
        match = re.search(rf"def {fn_name}\([\s\S]*?\):", text)
        if not match:
            return "unknown"
        sig = match.group(0)
        if arg_name == "size":
            size_arg = re.search(r"size=\(([^)]*)\)", sig)
            if size_arg:
                return f"({size_arg.group(1).strip()})"
        arg = re.search(rf"{arg_name}=([^,\n)]+)", sig)
        return arg.group(1).strip() if arg else "unknown"

    def config_value(name: str) -> str:
        match = re.search(rf"ti2v_5B\.{name}\s*=\s*([^\n]+)", cfg_text)
        return match.group(1).strip() if match else "unknown"

    return {
        "generate_dispatch": "WanTI2V.generate calls i2v when img is not None, otherwise t2v",
        "generate_default_size": signature_default("generate", "size"),
        "generate_default_frame_num": signature_default("generate", "frame_num"),
        "generate_default_sampling_steps": signature_default("generate", "sampling_steps"),
        "generate_default_shift": signature_default("generate", "shift"),
        "generate_default_guide_scale": signature_default("generate", "guide_scale"),
        "t2v_default_frame_num": signature_default("t2v", "frame_num"),
        "i2v_default_sampling_steps": signature_default("i2v", "sampling_steps"),
        "config_frame_num": config_value("frame_num"),
        "config_sample_steps": config_value("sample_steps"),
        "config_sample_shift": config_value("sample_shift"),
        "config_sample_guide_scale": config_value("sample_guide_scale"),
        "config_sample_fps": config_value("sample_fps"),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_optional_jsonl(path: Path) -> list[dict]:
    return load_jsonl(path) if path.exists() else []


def audit_manifest(project_root: Path, train_manifest: Path) -> dict:
    data = read_json(train_manifest)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict manifest for T2V train, got {type(data).__name__}")
    manifest_root = get_manifest_root()
    master_path = manifest_root / "master_all.jsonl"
    caption_path = manifest_root / "caption_index.jsonl"
    master_rows = load_optional_jsonl(master_path)
    caption_rows = load_optional_jsonl(caption_path)
    master_by_uid = {row.get("scene_uid"): row for row in master_rows}
    caption_by_uid = {row.get("scene_uid"): row for row in caption_rows}
    source_counts: dict[str, int] = {}
    empty_prompts = []
    image_keys = []
    traced_master = 0
    traced_caption = 0
    prompt_master = 0
    prompt_caption = 0
    scene_ids = []
    for scene_uid, item in data.items():
        bucket = scene_uid.split("/", 1)[0] if "/" in scene_uid else "NO_PREFIX"
        source_counts[bucket] = source_counts.get(bucket, 0) + 1
        scene_ids.append(scene_uid.split("/", 1)[-1])
        prompt = item.get("text_prompt", "") if isinstance(item, dict) else ""
        if not prompt.strip():
            empty_prompts.append(scene_uid)
        if isinstance(item, dict):
            image_keys.extend(
                (scene_uid, key) for key in item if "image" in key.lower() or "frame" in key.lower()
            )
        master = master_by_uid.get(scene_uid)
        caption = caption_by_uid.get(scene_uid)
        if master:
            traced_master += 1
            if master.get("t2v_train_text_prompt") == prompt:
                prompt_master += 1
        if caption:
            traced_caption += 1
            if caption.get("vlm_caption") == prompt:
                prompt_caption += 1
    duplicate_count = len(scene_ids) - len(set(scene_ids))
    return {
        "sha256": sha256_file(train_manifest),
        "top_type": type(data).__name__,
        "total_samples": len(data),
        "master_manifest_present": master_path.exists(),
        "caption_index_present": caption_path.exists(),
        "scene_id_field": "dict key scene_uid, formatted source_subset/scene_id",
        "prompt_field": "text_prompt",
        "source_counts": source_counts,
        "empty_prompt_count": len(empty_prompts),
        "duplicate_scene_id_count": duplicate_count,
        "image_related_key_examples": image_keys[:5],
        "trace_master_count": traced_master,
        "trace_caption_count": traced_caption,
        "prompt_exact_master_count": prompt_master,
        "prompt_exact_caption_count": prompt_caption,
        "first5_sanitized": [
            {
                "scene_uid": scene_uid,
                "keys": sorted(item.keys()) if isinstance(item, dict) else [],
                "prompt_chars": len(item.get("text_prompt", "")) if isinstance(item, dict) else 0,
                "prompt_preview": (item.get("text_prompt", "")[:96] + "...")
                if isinstance(item, dict)
                else "",
            }
            for scene_uid, item in list(data.items())[:5]
        ],
    }


def audit_manifest_samples(project_root: Path) -> dict:
    manifest_root = get_manifest_root()
    train_i2v_path = manifest_root / "videogpa_protocol/train_i2v.json"
    train_i2v = read_json(train_i2v_path) if train_i2v_path.exists() else None
    master_rows = load_optional_jsonl(manifest_root / "master_all.jsonl")
    caption_rows = load_optional_jsonl(manifest_root / "caption_index.jsonl")
    train_i2v_image_keys = 0
    train_i2v_abs_image_examples = []
    if isinstance(train_i2v, dict):
        for uid, item in train_i2v.items():
            if not isinstance(item, dict):
                continue
            image_value = item.get("image_prompt") or item.get("image_path")
            if image_value:
                train_i2v_image_keys += 1
                if str(image_value).startswith("/"):
                    train_i2v_abs_image_examples.append({"scene_uid": uid, "image_path": image_value})
            if len(train_i2v_abs_image_examples) >= 3:
                break
    return {
        "train_i2v_top_type": type(train_i2v).__name__,
        "train_i2v_total_samples": len(train_i2v) if isinstance(train_i2v, dict) else None,
        "train_i2v_present": train_i2v_path.exists(),
        "train_i2v_image_key_count_preview": train_i2v_image_keys,
        "train_i2v_absolute_image_examples": train_i2v_abs_image_examples,
        "master_rows": len(master_rows),
        "caption_rows": len(caption_rows),
        "master_first_row_keys": sorted(master_rows[0].keys()) if master_rows else [],
        "caption_first_row_keys": sorted(caption_rows[0].keys()) if caption_rows else [],
    }


def write_report(
    path: Path,
    resolved: dict,
    env: dict,
    runtime: dict,
    model_audit: dict,
    candidate_audit: dict,
    wan_defaults: dict,
    manifest: dict,
    manifest_samples: dict,
    static: dict,
) -> None:
    lines = [
        "# VideoGPA WAN2.2 5B T2V Smoke Preflight",
        "",
        f"Status: {static['status']}",
        "",
        "## Resolved Paths",
        "",
        f"- project_root: `{resolved['project']['project_root']}`",
        f"- videogpa_root: `{resolved['paths']['videogpa_root']}`",
        f"- wan_model_path: `{resolved['paths']['wan_model_path']}`",
        f"- vggt_model_path: `{resolved['paths']['vggt_model_path']}`",
        f"- train_manifest: `{resolved['paths']['train_manifest']}`",
        f"- output_root: `{resolved['paths']['output_root']}`",
        f"- run_dir: `{resolved['paths'].get('run_dir', '')}`",
        "",
        "## Git",
        "",
        f"- branch: `{static['branch']}`",
        f"- commit: `{static['commit']}`",
        f"- git_status_short: `{static['git_status'] or 'clean'}`",
        f"- VideoGPA git mode: `{static['videogpa_git_mode']}`",
        "",
        "## Environment",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in env.items())
    lines.extend(
        [
            "",
            "## Runtime Tools",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in runtime.items())
    lines.extend(
        [
            "",
            "## Model Discovery",
            "",
            f"- WAN candidates under model root: `{candidate_audit['wan_candidates']}`",
            f"- VGGT candidates under model root: `{candidate_audit['vggt_candidates']}`",
            f"- WAN required files: `{model_audit['wan_required']}`",
            f"- WAN DiT shard count: `{model_audit['wan_dit_shard_count']}`",
            f"- WAN complete: `{model_audit['wan_complete']}`",
            f"- VGGT required files: `{model_audit['vggt_required']}`",
            f"- VGGT complete: `{model_audit['vggt_complete']}`",
            f"- official VideoGPA WAN TI2V LoRA: `{model_audit['official_videogpa_wan_ti2v_lora']}`",
            f"- official VideoGPA WAN TI2V LoRA complete: `{model_audit['official_videogpa_wan_ti2v_lora_complete']}`",
            "",
            "## Local WAN Defaults",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in wan_defaults.items())
    lines.extend(
        [
            "",
            "## Manifest",
            "",
            f"- top_type: `{manifest['top_type']}`",
            f"- total_samples: `{manifest['total_samples']}`",
            f"- sha256: `{manifest['sha256']}`",
            f"- master_manifest_present: `{manifest['master_manifest_present']}`",
            f"- caption_index_present: `{manifest['caption_index_present']}`",
            f"- scene_id_field: `{manifest['scene_id_field']}`",
            f"- prompt_field: `{manifest['prompt_field']}`",
            "- split_field: `master_all.jsonl when present; otherwise inferred from train_t2v manifest`",
            f"- source_counts: `{manifest['source_counts']}`",
            f"- empty_prompt_count: `{manifest['empty_prompt_count']}`",
            f"- duplicate_scene_id_count: `{manifest['duplicate_scene_id_count']}`",
            f"- image_related_key_examples: `{manifest['image_related_key_examples']}`",
            f"- trace_master_count: `{manifest['trace_master_count']}`",
            f"- trace_caption_count: `{manifest['trace_caption_count']}`",
            f"- prompt_exact_master_count: `{manifest['prompt_exact_master_count']}`",
            f"- prompt_exact_caption_count: `{manifest['prompt_exact_caption_count']}`",
            f"- train_i2v_present: `{manifest_samples['train_i2v_present']}`",
            f"- train_i2v_top_type: `{manifest_samples['train_i2v_top_type']}`",
            f"- train_i2v_total_samples: `{manifest_samples['train_i2v_total_samples']}`",
            f"- train_i2v_absolute_image_examples: `{manifest_samples['train_i2v_absolute_image_examples']}`",
            f"- master_rows: `{manifest_samples['master_rows']}`",
            f"- caption_rows: `{manifest_samples['caption_rows']}`",
            "",
            "First 5 train_t2v samples, sanitized:",
            "",
            "```json",
            json.dumps(manifest["first5_sanitized"], indent=2, ensure_ascii=False),
            "```",
            "",
            "Important: this manifest is all train split but includes 8K/9K/10K/11K. Smoke subset creation filters `8K/` only and never reads test manifests.",
            "",
            "## Code Diff Decisions",
            "",
            "- Directly reusable: VideoGPA `loss.py`, `dataset.py`, VGGT `VideoProcessor`/`Consistency_Score`, WAN T5, WAN VAE, `WanModel`, flow-matching target, LoRA target modules, optimizer family, and reference-policy DPO setup.",
            "- Image-bound in TI2V: `image_path`/`image_prompt` loading, PIL first-frame loading, VAE image-latent encoding, `image_latent` condition files, clean first latent replacement, and TI2V timestep mask construction.",
            "- T2V minimum change set: reject image fields, call `WanTI2V.generate(..., img=None)`, save text-only conditions, encode only winner/loser video latents, and train all temporal latent positions with the same T2V noise/timestep rule.",
            "- WAN text-only behavior is determined by `WanTI2V.generate(..., img=None)`, which calls the local WAN `t2v` branch.",
            "- CogVideoX T2V/I2V differences were used only to confirm which image-condition fields disappear in pure T2V.",
            "- WAN-specific training behavior is determined by local `WanModel.forward()`: a 1D timestep tensor is expanded over the whole sequence when no image mask tensor is supplied.",
            "",
            "## Readiness",
            "",
            f"- ready_for_smoke: `{static['ready']}`",
            f"- caveats: `{'; '.join(static['caveats']) if static['caveats'] else 'none'}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight for WAN2.2 5B T2V VideoGPA smoke")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    resolved = resolve_config(config_path, run_dir)
    project_root = Path(resolved["project"]["project_root"])
    run_dir = Path(resolved["paths"].get("run_dir") or Path(resolved["paths"]["output_root"]) / "preflight_only")

    manifest_root = get_manifest_root()
    missing = [p for p in REQUIRED_REPO_PATHS if not (project_root / p).exists()]
    missing.extend(f"{manifest_root}/{p}" for p in REQUIRED_MANIFEST_PATHS if not (manifest_root / p).exists())
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(missing))

    env = import_versions()
    runtime = audit_runtime_tools(project_root)
    model_audit = audit_models(project_root, resolved)
    candidate_audit = audit_model_candidates(project_root)
    wan_defaults = audit_wan_defaults(project_root)
    manifest = audit_manifest(project_root, Path(resolved["paths"]["train_manifest"]))
    manifest_samples = audit_manifest_samples(project_root)
    branch = run_text(["git", "branch", "--show-current"], project_root)
    commit = run_text(["git", "rev-parse", "HEAD"], project_root)
    git_status = run_text(["git", "status", "--short"], project_root)
    videogpa_git_mode = "independent-git" if (project_root / "VideoGPA/.git").exists() else "ordinary-dir"

    caveats = []
    ready = True
    for dep in ["torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "decord", "lpips"]:
        if "IMPORT_ERROR" in env.get(dep, ""):
            ready = False
            caveats.append(f"{dep} import failed")
    if shutil.which("ffmpeg") is None:
        ready = False
        caveats.append("ffmpeg missing")
    if runtime["outputs_writable"] != "True":
        ready = False
        caveats.append("outputs not writable")
    if not model_audit["wan_complete"]:
        ready = False
        caveats.append("WAN model incomplete")
    if not model_audit["vggt_complete"]:
        ready = False
        caveats.append("VGGT model incomplete")
    if len(candidate_audit["wan_candidates"]) != 1:
        ready = False
        caveats.append("WAN model auto-discovery is ambiguous or missing")
    if len(candidate_audit["vggt_candidates"]) != 1:
        ready = False
        caveats.append("VGGT model auto-discovery is ambiguous or missing")
    if manifest["empty_prompt_count"]:
        ready = False
        caveats.append("empty prompts in train_t2v")
    if manifest["image_related_key_examples"]:
        ready = False
        caveats.append("unexpected image keys in train_t2v")
    expected_train_prompts = resolved.get("formal_requirements", {}).get("expected_train_prompts")
    if expected_train_prompts is not None and manifest["total_samples"] != int(expected_train_prompts):
        ready = False
        caveats.append(
            f"train_t2v sample count {manifest['total_samples']} != expected {int(expected_train_prompts)}"
        )
    if manifest["master_manifest_present"] and manifest["trace_master_count"] != manifest["total_samples"]:
        ready = False
        caveats.append(
            f"master_all trace count {manifest['trace_master_count']} != train_t2v count {manifest['total_samples']}"
        )

    static = {
        "status": "PASS" if ready else "FAIL",
        "ready": ready,
        "caveats": caveats,
        "branch": branch,
        "commit": commit,
        "git_status": git_status,
        "videogpa_git_mode": videogpa_git_mode,
    }

    config_dir = run_dir / "config"
    preflight_dir = run_dir / "preflight"
    reports_dir = run_dir / "reports"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_dir, resolved)
    if config_path.suffix == ".json":
        write_json(config_dir / "source_config.json", read_json(config_path))
    else:
        write_yaml(config_dir / "source_config.yaml", read_yaml(config_path))
    (config_dir / "environment.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in {**env, **runtime}.items()) + "\n",
        encoding="utf-8",
    )
    write_json(
        run_dir / "preflight" / "preflight_data.json",
        {
            "env": env,
            "runtime": runtime,
            "models": model_audit,
            "model_candidates": candidate_audit,
            "wan_defaults": wan_defaults,
            "manifest": manifest,
            "manifest_samples": manifest_samples,
            "static": static,
        },
    )
    write_report(
        preflight_dir / "preflight_report.md",
        resolved,
        env,
        runtime,
        model_audit,
        candidate_audit,
        wan_defaults,
        manifest,
        manifest_samples,
        static,
    )
    write_report(
        get_output_root() / "videogpa/_preflight/latest/preflight_report.md",
        resolved,
        env,
        runtime,
        model_audit,
        candidate_audit,
        wan_defaults,
        manifest,
        manifest_samples,
        static,
    )
    official_diff = reports_dir / "official_diff.md"
    official_diff.parent.mkdir(parents=True, exist_ok=True)
    official_diff.write_text(
        "\n".join(
            [
                "# Official Diff",
                "",
                "## A. Identical To Official WAN2.2-TI2V VideoGPA",
                "",
                "- Same Wan2.2-TI2V-5B base checkpoint.",
                "- Same WAN T5 text encoder and VAE latent format.",
                "- Same flow-matching target, DPO loss, frozen reference model, LoRA target modules, rank, alpha, optimizer, and scheduler family.",
                "- Same VGGT consistency score and motion filtering semantics.",
                "",
                "## B. T2V-Required Changes",
                "",
                "- No first-frame manifest field is read.",
                "- No PIL image is opened for conditioning.",
                "- No `image_latent` is saved in condition files.",
                "- No clean first latent replacement is applied.",
                "- No TI2V timestep mask is constructed; WAN expands the sampled 1D timestep over the whole T2V sequence.",
                "",
                "## C. Smoke-Only Shrinks",
                "",
                "- Subset is limited to 4 train 8K prompts by default.",
                "- If strict official thresholds produce fewer than 2 pairs, `run_smoke.sh` extends only the train 8K subset to 8 prompts before allowing debug fallback pairs.",
                "- Candidate count is 3 seeds per prompt.",
                "- Training defaults to 5 optimizer steps and batch size 1.",
                "- Debug fallback preference pairs are marked `DEBUG_ONLY_NOT_COMPARABLE` and are not valid for formal experiments.",
                "",
                "## D. Remaining Uncertainty",
                "",
                "- Full 81-frame generation runtime depends on current GPU availability.",
                "- Local WAN `WanTI2V.generate()` defaults to 81 frames while the local `ti2v_5B` config and direct `t2v()` signature default to 121; the smoke YAML intentionally keeps 81 frames as requested.",
                "- The local official Lightning/W&B orchestration packages are absent, so smoke training uses a direct PyTorch loop while preserving the official DPO math and WAN model calls.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
