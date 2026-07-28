from __future__ import annotations

from pathlib import Path

from dl3dv_conditions.common import (
    generate_official_i2v_prompt,
    load_official_i2v_module,
    validate_official_motion_structure,
)


def _motion_pieces(motion: str) -> list[str]:
    if ", followed by " in motion:
        first, tail = motion.split(", then ", 1)
        second, third = tail.split(", followed by ", 1)
        return [first, second, third]
    return motion.split(", then ")


def test_i2v_prompt_uses_official_components() -> None:
    project_root = Path(__file__).resolve().parents[2]
    module = load_official_i2v_module(str(project_root))
    prompt = generate_official_i2v_prompt("8K/example-scene", 2026, project_root)
    assert prompt["i2v_train_text_prompt"].startswith(module.PREFIX_PROMPT)
    assert " Camera motion: " in prompt["i2v_train_text_prompt"]
    assert prompt["i2v_train_text_prompt"].endswith(".")
    ok, reason = validate_official_motion_structure(prompt["scripted_camera_motion"], project_root)
    assert ok, reason
    pieces = _motion_pieces(prompt["scripted_camera_motion"])
    assert len(pieces) in (2, 3)
