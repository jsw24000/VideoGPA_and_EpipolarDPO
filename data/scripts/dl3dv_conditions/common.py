from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator


TRAIN_SUBSETS = ("8K", "9K", "10K", "11K")
TEST_SUBSETS = ("1K",)
ALL_SUBSETS = TEST_SUBSETS + TRAIN_SUBSETS
HF_DATASET = "DL3DV/DL3DV-ALL-960P"
DEFAULT_SEED = 2026
DATA_DIRNAME = "3DVGM_data"
CAPTION_KEY_RE = re.compile(r"^(?P<subset>1K|8K|9K|10K|11K)/(?P<scene_id>[^/]+)/(?P<image_dir>images_8)$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class PipelineError(RuntimeError):
    """Raised for expected pipeline failures with user-actionable messages."""


@dataclass(frozen=True)
class StorageLayout:
    project_root: Path
    project_data: Path
    scratch_root: Path
    asset_root: Path
    dl3dv_raw_960p: Path
    first_frames: Path
    download_cache: Path
    staging: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "project_data": str(self.project_data),
            "scratch_root": str(self.scratch_root),
            "asset_root": str(self.asset_root),
            "dl3dv_raw_960p": str(self.dl3dv_raw_960p),
            "first_frames": str(self.first_frames),
            "download_cache": str(self.download_cache),
            "staging": str(self.staging),
        }


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "VideoGPA").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise PipelineError(f"Could not find project root from {current}")


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    return usage.free


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _can_write(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def normalize_scratch_arg(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.name == DATA_DIRNAME:
        return path
    return path / DATA_DIRNAME


def resolve_storage_layout(
    project_root: Path | None = None,
    scratch_root_arg: str | None = None,
    min_free_gb: float = 1.0,
    create: bool = True,
) -> StorageLayout:
    project_root = find_project_root(project_root)
    project_data = (project_root / "data").resolve()
    min_free = int(min_free_gb * 1024**3)

    candidates: list[tuple[str, Path]] = []
    if scratch_root_arg:
        candidates.append(("argument", normalize_scratch_arg(scratch_root_arg)))
    elif os.environ.get("DL3DV_SCRATCH_ROOT"):
        candidates.append(("DL3DV_SCRATCH_ROOT", normalize_scratch_arg(os.environ["DL3DV_SCRATCH_ROOT"])))
    else:
        for base in ("/data1", "/disk1", "/mnt/data1", "/mnt/disk1"):
            base_path = Path(base)
            if base_path.exists():
                candidates.append((base, base_path.resolve() / DATA_DIRNAME))

    errors: list[str] = []
    for source, asset_root in candidates:
        try:
            if create:
                asset_root.mkdir(parents=True, exist_ok=True)
            if not asset_root.exists():
                errors.append(f"{source}: {asset_root} does not exist")
                continue
            if not _can_write(asset_root):
                errors.append(f"{source}: {asset_root} is not writable")
                continue
            if free_bytes(asset_root) < min_free:
                errors.append(
                    f"{source}: {asset_root} has {human_bytes(free_bytes(asset_root))} free, "
                    f"requires at least {human_bytes(min_free)}"
                )
                continue
            if is_relative_to(asset_root, project_root):
                raise PipelineError(f"External data root must not be inside project root: {asset_root}")
            layout = StorageLayout(
                project_root=project_root,
                project_data=project_data,
                scratch_root=asset_root,
                asset_root=asset_root,
                dl3dv_raw_960p=asset_root / "dl3dv_raw_960p",
                first_frames=asset_root / "first_frames",
                download_cache=asset_root / "download_cache",
                staging=asset_root / "staging",
            )
            if is_relative_to(layout.dl3dv_raw_960p, project_data):
                raise PipelineError(f"Raw DL3DV path resolves inside project data: {layout.dl3dv_raw_960p}")
            if create:
                ensure_dirs(
                    [
                        layout.dl3dv_raw_960p,
                        layout.first_frames / "train" / "8K",
                        layout.first_frames / "train" / "9K",
                        layout.first_frames / "train" / "10K",
                        layout.first_frames / "train" / "11K",
                        layout.first_frames / "test" / "1K",
                        layout.download_cache,
                        layout.staging,
                    ]
                )
            return layout
        except PipelineError:
            raise
        except OSError as exc:
            errors.append(f"{source}: {asset_root}: {exc}")

    if not candidates:
        errors.append("no DL3DV_SCRATCH_ROOT and no writable candidate among /data1, /disk1, /mnt/data1, /mnt/disk1")
    raise PipelineError("Could not resolve writable external scratch root. " + "; ".join(errors))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:
        path.write_text(_simple_yaml(data), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise PipelineError(f"YAML root is not a mapping: {path}")
        return loaded
    except ModuleNotFoundError as exc:
        raise PipelineError(f"PyYAML is required to read {path}") from exc


def _simple_yaml(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_simple_yaml(value, indent + 2).rstrip())
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                lines.append(f"{prefix}  - {item}")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {str(value).lower()}")
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise PipelineError(f"JSONL item is not an object at {path}:{line_no}")
            records.append(item)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
            count += 1
    return count


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
        handle.write("\n")


def parse_caption_key(key: str) -> dict[str, str]:
    match = CAPTION_KEY_RE.match(key)
    if not match:
        raise ValueError(f"Invalid DL3DV caption key: {key}")
    return match.groupdict()


def split_group_for_subset(subset: str) -> str:
    if subset in TRAIN_SUBSETS:
        return "train"
    if subset in TEST_SUBSETS:
        return "test"
    raise ValueError(f"Unknown source subset: {subset}")


def scene_uid(source_subset: str, scene_id: str) -> str:
    return f"{source_subset}/{scene_id}"


def caption_source_file(source_subset: str) -> str:
    return f"VideoGPA/dl3dv_video_captions/captions_{source_subset}.json"


def stable_scene_seed(global_seed: int, uid: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


@lru_cache(maxsize=1)
def load_official_i2v_module(project_root_text: str | None = None):
    project_root = find_project_root(Path(project_root_text) if project_root_text else None)
    script_path = project_root / "VideoGPA" / "data_prep" / "generate_i2v_prompts.py"
    if not script_path.exists():
        raise PipelineError(f"Missing official I2V prompt script: {script_path}")
    spec = importlib.util.spec_from_file_location("videogpa_generate_i2v_prompts", script_path)
    if spec is None or spec.loader is None:
        raise PipelineError(f"Could not import official I2V prompt script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("videogpa_generate_i2v_prompts", module)
    spec.loader.exec_module(module)
    for name in ("PREFIX_PROMPT", "TRANSLATIONS", "ROTATIONS", "COMPLEX_PATHS", "generate_multi_stage_motion"):
        if not hasattr(module, name):
            raise PipelineError(f"Official I2V prompt script is missing {name}")
    return module


@contextlib.contextmanager
def _patched_official_rng(module: Any, rng: random.Random) -> Iterator[None]:
    original_random = module.random
    module.random = rng
    try:
        yield
    finally:
        module.random = original_random


def generate_official_motion_from_seed(scene_seed: int, project_root: Path | None = None) -> str:
    module = load_official_i2v_module(str(find_project_root(project_root)))
    rng = random.Random(scene_seed)
    with _patched_official_rng(module, rng):
        return module.generate_multi_stage_motion()


def generate_official_i2v_prompt(scene_uid_text: str, global_seed: int = DEFAULT_SEED, project_root: Path | None = None) -> dict[str, Any]:
    seed = stable_scene_seed(global_seed, scene_uid_text)
    motion = generate_official_motion_from_seed(seed, project_root)
    module = load_official_i2v_module(str(find_project_root(project_root)))
    text_prompt = module.PREFIX_PROMPT + f" Camera motion: {motion}."
    return {
        "scripted_camera_seed": seed,
        "scripted_camera_motion": motion,
        "i2v_train_text_prompt": text_prompt,
    }


def official_motion_vocab(project_root: Path | None = None) -> set[str]:
    module = load_official_i2v_module(str(find_project_root(project_root)))
    return set(module.TRANSLATIONS) | set(module.ROTATIONS) | set(module.COMPLEX_PATHS)


def validate_official_motion_structure(motion: str, project_root: Path | None = None) -> tuple[bool, str]:
    if ", followed by " in motion:
        pieces = motion.split(", then ", 1)
        if len(pieces) != 2:
            return False, "missing ', then ' connective"
        second = pieces[1].split(", followed by ", 1)
        if len(second) != 2:
            return False, "malformed ', followed by ' connective"
        parsed = [pieces[0], second[0], second[1]]
    elif ", then " in motion:
        parsed = motion.split(", then ")
    else:
        return False, "missing official connective"
    if len(parsed) not in (2, 3):
        return False, f"expected 2 or 3 pieces, got {len(parsed)}"
    vocab = official_motion_vocab(project_root)
    unknown = [piece for piece in parsed if piece not in vocab]
    if unknown:
        return False, f"unknown motion piece(s): {unknown}"
    return True, ""


def natural_sort_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        return img.size


def copy_stream_to_file(stream: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    with tmp.open("wb") as handle:
        shutil.copyfileobj(stream, handle)
    tmp.replace(target)


def first_frame_relpath(split_group: str, subset: str, scene_id: str, extension: str) -> str:
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"first_frames/{split_group}/{subset}/{scene_id}/first_frame{ext.lower()}"


def resolve_asset_relpath(asset_root: Path, relpath: str) -> Path:
    path = Path(relpath)
    if path.is_absolute():
        raise PipelineError(f"Canonical asset path must be relative: {relpath}")
    return (asset_root / path).resolve()


def storage_from_local_config(project_root: Path | None = None) -> StorageLayout:
    project_root = find_project_root(project_root)
    config_path = project_root / "data" / "configs" / "storage.local.yaml"
    if not config_path.exists():
        raise PipelineError(f"Missing storage config: {config_path}. Run resolve_storage.py first.")
    data = load_yaml(config_path)
    asset_root_value = data.get("asset_root") or data.get("scratch_root")
    if not asset_root_value:
        raise PipelineError(f"storage.local.yaml does not contain asset_root: {config_path}")
    asset_root = Path(asset_root_value).expanduser().resolve()
    return StorageLayout(
        project_root=project_root,
        project_data=project_root / "data",
        scratch_root=asset_root,
        asset_root=asset_root,
        dl3dv_raw_960p=asset_root / "dl3dv_raw_960p",
        first_frames=asset_root / "first_frames",
        download_cache=asset_root / "download_cache",
        staging=asset_root / "staging",
    )


def load_storage_or_resolve(project_root: Path | None = None, scratch_root: str | None = None, min_free_gb: float = 1.0) -> StorageLayout:
    if scratch_root:
        return resolve_storage_layout(project_root, scratch_root, min_free_gb=min_free_gb, create=True)
    try:
        return storage_from_local_config(project_root)
    except PipelineError:
        return resolve_storage_layout(project_root, None, min_free_gb=min_free_gb, create=True)


def filter_records_by_splits(records: Iterable[dict[str, Any]], splits: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    normalized = set(splits or [])
    selected: list[dict[str, Any]] = []
    for record in records:
        subset = record.get("source_subset")
        split = record.get("split") or record.get("split_group")
        uid = record.get("scene_uid") or (scene_uid(subset, record["scene_id"]) if subset and record.get("scene_id") else "")
        if normalized and subset not in normalized and split not in normalized and uid not in normalized:
            continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def require_relative_path(relpath: str | None, field: str) -> str | None:
    if relpath is None:
        return None
    if Path(relpath).is_absolute():
        raise PipelineError(f"{field} must be relative, got {relpath}")
    return Path(relpath).as_posix()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
