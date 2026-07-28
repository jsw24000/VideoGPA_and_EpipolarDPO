#!/usr/bin/env python3
import importlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "wan22_smoke"
MODEL_DIR = PROJECT_ROOT / "models" / "wan" / "Wan2.2-TI2V-5B"
WAN_SRC_DIR = PROJECT_ROOT / "third_party" / "Wan2.2"
VIDEOGPA_DIR = PROJECT_ROOT / "VideoGPA"


def package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def git_commit(path: Path) -> str:
    if not (path / ".git").exists():
        return "not a git repository"
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out.strip()
    except subprocess.CalledProcessError as exc:
        return f"git error: {exc.output.strip()}"


def check_model_tree() -> list[str]:
    missing: list[str] = []
    required_files = [
        "config.json",
        "configuration.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2_VAE.pth",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
    ]
    for rel in required_files:
        if not (MODEL_DIR / rel).is_file():
            missing.append(rel)

    shards = sorted(MODEL_DIR.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        missing.append("diffusion_pytorch_model-*.safetensors")
    return missing


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    lines: list[str] = []

    if str(WAN_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(WAN_SRC_DIR))

    lines.append(f"Python: {sys.version.split()[0]} ({sys.executable})")
    lines.append(f"Platform: {platform.platform()}")

    import_names = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "peft": "peft",
        "accelerate": "accelerate",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface-hub",
        "flash_attn": "flash-attn",
        "numpy": "numpy",
    }
    imported = {}
    for module_name, dist_name in import_names.items():
        try:
            imported[module_name] = importlib.import_module(module_name)
            lines.append(f"{module_name}: {package_version(dist_name)}")
        except Exception as exc:
            errors.append(f"failed to import {module_name}: {exc}")
            lines.append(f"{module_name}: IMPORT FAILED ({exc})")

    torch = imported.get("torch")
    if torch is None:
        errors.append("torch import failed, cannot check CUDA")
    else:
        lines.append(f"CUDA used by torch: {torch.version.cuda}")
        cuda_available = torch.cuda.is_available()
        lines.append(f"torch.cuda.is_available(): {cuda_available}")
        if not cuda_available:
            errors.append("torch.cuda.is_available() is False")
        else:
            lines.append(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            lines.append(f"torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                total_gib = props.total_memory / 1024**3
                lines.append(
                    "GPU {idx}: {name}, capability {major}.{minor}, "
                    "{mem:.2f} GiB".format(
                        idx=idx,
                        name=torch.cuda.get_device_name(idx),
                        major=props.major,
                        minor=props.minor,
                        mem=total_gib,
                    )
                )

    try:
        importlib.import_module("wan.configs")
        lines.append("wan.configs: import ok")
    except Exception as exc:
        errors.append(f"failed to import WAN2.2 source from {WAN_SRC_DIR}: {exc}")
        lines.append(f"wan.configs: IMPORT FAILED ({exc})")

    missing = check_model_tree()
    if missing:
        errors.append("model tree is incomplete: " + ", ".join(missing))
    else:
        shards = sorted(MODEL_DIR.glob("diffusion_pytorch_model-*.safetensors"))
        lines.append(f"model tree: ok ({len(shards)} diffusion safetensor shards)")

    lines.append(f"WAN2.2 source path: {WAN_SRC_DIR}")
    lines.append(f"WAN2.2 git commit: {git_commit(WAN_SRC_DIR)}")
    lines.append(f"VideoGPA path: {VIDEOGPA_DIR}")
    lines.append(f"VideoGPA git commit: {git_commit(VIDEOGPA_DIR)}")
    lines.append(f"VideoGPA Wan2.2 link: {(VIDEOGPA_DIR / 'Wan2.2').resolve() if (VIDEOGPA_DIR / 'Wan2.2').exists() else 'missing'}")
    lines.append(f"Model path: {MODEL_DIR}")

    status = "PASS" if not errors else "FAIL"
    lines.append(f"Status: {status}")
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {err}" for err in errors)

    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "environment_versions.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
