from __future__ import annotations

from pathlib import Path

from dl3dv_conditions.common import generate_official_i2v_prompt, stable_scene_seed


def test_scene_local_prompt_is_deterministic() -> None:
    project_root = Path(__file__).resolve().parents[2]
    first = generate_official_i2v_prompt("1K/abc", 2026, project_root)
    second = generate_official_i2v_prompt("1K/abc", 2026, project_root)
    assert first == second
    assert stable_scene_seed(2026, "1K/abc") == stable_scene_seed(2026, "1K/abc")
    assert stable_scene_seed(2026, "1K/abc") != stable_scene_seed(2027, "1K/abc")
