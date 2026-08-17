from __future__ import annotations

import json
import shlex
import subprocess
import sys
import types
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
I2V_CONFIG = "configs/videogpa/wan22_5b_i2v_formal.yaml"
I2V_CONFIG_ABS = REPO_ROOT / I2V_CONFIG


def run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-lc", command], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_i2v_generator_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object))
    monkeypatch.setitem(sys.modules, "wan", types.ModuleType("wan"))
    monkeypatch.setitem(
        sys.modules,
        "wan.configs",
        types.SimpleNamespace(SIZE_CONFIGS={"1280*704": (1280, 704)}, WAN_CONFIGS={"ti2v-5B": object()}),
    )
    monkeypatch.setitem(sys.modules, "wan.textimage2video", types.SimpleNamespace(WanTI2V=object))
    path = REPO_ROOT / "VideoGPA" / "generate" / "Wan2.2-I2V-5B.py"
    spec = importlib.util.spec_from_file_location("wan22_i2v_generator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_i2v_formal_output_roots_are_t2v_siblings() -> None:
    local = run_bash(
        "source scripts/env/activate_profile.sh local >/dev/null && "
        f"python -m vgm_common.config --config {I2V_CONFIG} --print output_root"
    )
    assert local.returncode == 0, local.stdout + local.stderr
    assert local.stdout.strip() == str(REPO_ROOT / "outputs" / "videogpa" / "wan22_5b_i2v" / "formal")

    cluster = run_bash(
        "source scripts/env/activate_profile.sh cluster_zk >/dev/null && "
        f"python -m vgm_common.config --config {shlex.quote(str(I2V_CONFIG_ABS))} --print output_root"
    )
    assert cluster.returncode == 0, cluster.stdout + cluster.stderr
    expected_cluster = run_bash(
        "source scripts/env/activate_profile.sh cluster_zk >/dev/null && "
        "printf '%s' \"${VGM_OUTPUT_ROOT}/videogpa/wan22_5b_i2v/formal\""
    )
    assert expected_cluster.returncode == 0, expected_cluster.stdout + expected_cluster.stderr
    assert cluster.stdout.strip() == expected_cluster.stdout.strip()


def test_i2v_subset_resolves_profile_first_frames(tmp_path: Path) -> None:
    run_dir = tmp_path / "i2v_run"
    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        "python scripts/videogpa/wan22_5b_i2v/01_make_train_subset.py "
        f"--config {I2V_CONFIG} --run-dir {shlex.quote(str(run_dir))} --subset-size 2"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = json.loads((run_dir / "manifests" / "input_subset.json").read_text(encoding="utf-8"))
    assert payload["task"] == "i2v"
    assert payload["image_conditioned"] is True
    assert payload["selected_count"] == 2
    assert payload["required_buckets"] == ["8k", "9k", "10k", "11k"]
    assert payload["first_frames_root"] == str(REPO_ROOT / "data" / "first_frames")

    for sample in payload["samples"]:
        image_path = Path(sample["image_path"])
        assert sample["task"] == "i2v"
        assert sample["image_conditioned"] is True
        assert "Camera motion:" in sample["text_prompt"]
        assert sample["camera_motion"]
        assert image_path.is_file()
        assert image_path.is_relative_to(REPO_ROOT / "data" / "first_frames")


def test_i2v_generator_resolves_relpath_protocol_images(monkeypatch, tmp_path: Path) -> None:
    module = load_i2v_generator_module(monkeypatch)
    monkeypatch.setenv("VGM_PROFILE", "local")
    monkeypatch.setenv("VGM_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("VGM_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("VGM_DL3DV_ROOT", str(REPO_ROOT / "data"))
    monkeypatch.setenv("VGM_MODEL_ROOT", str(REPO_ROOT / "models"))
    monkeypatch.setenv("VGM_OUTPUT_ROOT", str(REPO_ROOT / "outputs"))

    scene_uid = "1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
    rel_image = f"first_frames/test/1K/{scene_uid.split('/', 1)[1]}/first_frame.png"
    manifest = tmp_path / "test_i2v_relpaths.json"
    manifest.write_text(json.dumps({scene_uid: {"text_prompt": "A test prompt.", "image_prompt": rel_image}}), encoding="utf-8")

    samples = module.load_samples(manifest)
    assert samples[0]["source_split"] == "test"
    assert samples[0]["source_bucket"] == "1k"
    assert samples[0]["scene_id"] == scene_uid.split("/", 1)[1]

    cfg = {
        "project": {"project_root": str(REPO_ROOT)},
        "paths": {"first_frames_root": str(REPO_ROOT / "data" / "first_frames")},
    }
    resolved = module.resolve_image_path(samples[0], cfg, tmp_path / "run")
    assert resolved == REPO_ROOT / "data" / rel_image


def test_i2v_generator_repairs_stale_absolute_first_frame_paths(monkeypatch, tmp_path: Path) -> None:
    module = load_i2v_generator_module(monkeypatch)
    monkeypatch.setenv("VGM_PROFILE", "local")
    monkeypatch.setenv("VGM_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("VGM_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("VGM_DL3DV_ROOT", str(REPO_ROOT / "data"))
    monkeypatch.setenv("VGM_MODEL_ROOT", str(REPO_ROOT / "models"))
    monkeypatch.setenv("VGM_OUTPUT_ROOT", str(REPO_ROOT / "outputs"))

    scene_id = "001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
    stale = f"/old/local/machine/data/first_frames/test/1K/{scene_id}/first_frame.png"
    sample = {
        "scene_uid": f"1K/{scene_id}",
        "scene_id": scene_id,
        "source_split": "test",
        "source_bucket": "1K",
        "text_prompt": "A test prompt.",
        "image_prompt": stale,
    }
    cfg = {
        "project": {"project_root": str(REPO_ROOT)},
        "paths": {"first_frames_root": str(REPO_ROOT / "data" / "first_frames")},
    }
    resolved = module.resolve_image_path(sample, cfg, tmp_path / "run")
    assert resolved == REPO_ROOT / "data" / "first_frames" / "test" / "1K" / scene_id / "first_frame.png"
