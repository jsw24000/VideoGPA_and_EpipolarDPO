from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-lc", command], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def iter_strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)
    elif isinstance(value, str):
        yield value


def load_t2v_generator_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object))
    monkeypatch.setitem(sys.modules, "wan", types.ModuleType("wan"))
    monkeypatch.setitem(
        sys.modules,
        "wan.configs",
        types.SimpleNamespace(SIZE_CONFIGS={"1280*704": (1280, 704)}, WAN_CONFIGS={"ti2v-5B": object()}),
    )
    monkeypatch.setitem(sys.modules, "wan.textimage2video", types.SimpleNamespace(WanTI2V=object))
    path = REPO_ROOT / "VideoGPA" / "generate" / "Wan2.2-T2V-5B.py"
    spec = importlib.util.spec_from_file_location("wan22_t2v_generator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dl3dv1k_eval_manifest_is_canonical_and_portable(tmp_path: Path) -> None:
    out = tmp_path / "eval_1k_seed456.json"
    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        f"python scripts/videogpa/wan22_5b_eval/make_eval_manifest.py --output {out} --seed 456"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["protocol"] == "wan22_5b_dl3dv1k_eval_v1"
    assert payload["num_samples"] == 1000
    assert payload["prompt"] == "CogVLM2 natural caption"
    assert payload["seed"] == 456
    assert payload["path_policy"]["contains_absolute_paths"] is False
    assert payload["path_policy"]["first_frames_are_not_copied"] is True
    assert payload["generation_settings"]["frame_num"] == 81
    assert payload["generation_settings"]["size"] == "1280*704"
    assert payload["evaluator_settings"]["geometry_backbone"] == "DA3-Large"

    strings = list(iter_strings(payload))
    assert not [text for text in strings if text.startswith("/")]

    first = payload["samples"][0]
    assert first["source_split"] == "test"
    assert first["source_bucket"] == "1K"
    assert first["group_id"].startswith("1K__")
    assert first["first_frame_relpath"].startswith("first_frames/test/1K/")
    assert (REPO_ROOT / "data" / first["first_frame_relpath"]).is_file()

    sha_path = out.with_suffix(out.suffix + ".sha256")
    assert sha_path.is_file()


def test_task_manifests_keep_t2v_pure_and_i2v_conditioned(tmp_path: Path) -> None:
    canonical = tmp_path / "eval_1k_seed456.json"
    t2v_out = tmp_path / "t2v_eval_1k_seed456.json"
    i2v_out = tmp_path / "i2v_eval_1k_seed456.json"
    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        f"python scripts/videogpa/wan22_5b_eval/make_eval_manifest.py --output {canonical} --seed 456 --limit 3 && "
        f"python scripts/videogpa/wan22_5b_eval/make_task_manifest.py --input {canonical} --output {t2v_out} --task t2v && "
        f"python scripts/videogpa/wan22_5b_eval/make_task_manifest.py --input {canonical} --output {i2v_out} --task i2v"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    t2v_payload = json.loads(t2v_out.read_text(encoding="utf-8"))
    i2v_payload = json.loads(i2v_out.read_text(encoding="utf-8"))
    assert t2v_payload["task"] == "t2v"
    assert i2v_payload["task"] == "i2v"
    assert t2v_payload["samples"][0]["text_prompt"] == i2v_payload["samples"][0]["text_prompt"]
    assert t2v_payload["samples"][0]["seed"] == i2v_payload["samples"][0]["seed"] == 456
    assert t2v_payload["samples"][0]["image_conditioned"] is False
    assert i2v_payload["samples"][0]["image_conditioned"] is True
    assert "first_frame_relpath" not in t2v_payload["samples"][0]
    assert i2v_payload["samples"][0]["first_frame_relpath"].startswith("first_frames/test/1K/")
    assert not [text for text in iter_strings(t2v_payload) if text.startswith("/")]
    assert not [text for text in iter_strings(i2v_payload) if text.startswith("/")]


def test_eval_100_subset_is_seeded_and_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        f"python scripts/videogpa/wan22_5b_eval/make_eval_manifest.py --output {first} --seed 456 --limit 100 && "
        f"python scripts/videogpa/wan22_5b_eval/make_eval_manifest.py --output {second} --seed 456 --limit 100"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    first_scene_ids = [sample["scene_uid"] for sample in first_payload["samples"]]
    second_scene_ids = [sample["scene_uid"] for sample in second_payload["samples"]]
    assert first_payload["num_samples"] == 100
    assert first_payload["selection"] == {
        "strategy": "seeded_without_replacement",
        "seed": 456,
        "requested_limit": 100,
        "source_size": 1000,
    }
    assert first_scene_ids == second_scene_ids
    assert len(first_scene_ids) == len(set(first_scene_ids)) == 100


def test_eval_runner_manifest_only_uses_flat_100_sample_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "wan22_5b_t2v_formal_001"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(
        "project:\n  method: videogpa\n  model_scale: 5b\n  task: t2v\n",
        encoding="utf-8",
    )
    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        f"PYTHON_BIN=python bash scripts/videogpa/wan22_5b_eval/run_eval.sh --run-dir {run_dir} --manifest-only"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    eval_dir = run_dir / "evaluation" / "dl3dv1k_seed456"
    canonical = json.loads((eval_dir / "manifests/eval_1k_seed456.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((eval_dir / "manifests/task_eval_1k_seed456.json").read_text(encoding="utf-8"))
    assert canonical["num_samples"] == task_manifest["num_samples"] == 100
    assert task_manifest["task"] == "t2v"
    assert not (eval_dir / "manifests/t2v_eval_1k_seed456.json").exists()

    runner = (REPO_ROOT / "scripts/videogpa/wan22_5b_eval/run_eval.sh").read_text(encoding="utf-8")
    assert 'local out_dir="${GEN_DIR}/${variant}"' in runner
    assert 'local out_dir="${SCORE_DIR}/da3"' in runner
    assert 'local base_dir="${GEN_DIR}/${task}/${variant}"' not in runner
    assert 'assert_variant_video_count "${variant}" "${out_dir}"' in runner
    assert '[[ -f "${adapter_dir}/adapter_model.safetensors" || -f "${adapter_dir}/adapter_model.bin" ]]' in runner
    assert "Only one fine-tuned variant is allowed per RUN_DIR" in runner


def test_eval_runner_discovers_inference_complete_legacy_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "wan22_5b_t2v_formal_001"
    checkpoint = run_dir / "checkpoints/step_010000"
    checkpoint.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        "project:\n  method: videogpa\n  model_scale: 5b\n  task: t2v\n",
        encoding="utf-8",
    )
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").touch()

    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        f"PYTHON_BIN=python bash scripts/videogpa/wan22_5b_eval/run_eval.sh --run-dir {run_dir} "
        "--skip-baseline --skip-generation --skip-score"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    environment = (
        run_dir / "evaluation/dl3dv1k_seed456/config/environment.txt"
    ).read_text(encoding="utf-8")
    assert f"EVAL_VARIANT=videogpa_step_010000={checkpoint}:0.2" in environment


def test_t2v_generator_marks_1k_dict_manifest_as_test(monkeypatch, tmp_path: Path) -> None:
    module = load_t2v_generator_module(monkeypatch)
    scene_uid = "1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
    manifest = tmp_path / "test_t2v_direct.json"
    manifest.write_text(json.dumps({scene_uid: {"text_prompt": "A natural caption."}}), encoding="utf-8")

    samples = module.load_samples(manifest)
    assert samples[0]["task"] == "t2v"
    assert samples[0]["source_split"] == "test"
    assert samples[0]["source_bucket"] == "1k"
    assert samples[0]["scene_id"] == scene_uid.split("/", 1)[1]
