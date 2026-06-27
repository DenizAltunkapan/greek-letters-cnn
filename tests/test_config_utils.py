from pathlib import Path

import pytest

from config_utils import load_config, resolve_ml_path


def test_load_config_reads_yaml_mapping(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("paths:\n  model_path: models/best_model.pth\n", encoding="utf-8")

    config, resolved_path = load_config(str(config_path))

    assert config == {"paths": {"model_path": "models/best_model.pth"}}
    assert resolved_path == str(config_path.resolve())


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(str(config_path))


def test_resolve_ml_path_keeps_absolute_paths(tmp_path):
    assert resolve_ml_path(str(tmp_path)) == str(tmp_path)


def test_resolve_ml_path_resolves_relative_paths_under_ml_dir():
    resolved = Path(resolve_ml_path("models/best_model.pth"))

    assert resolved.name == "best_model.pth"
    assert resolved.parent.name == "models"
    assert resolved.parent.parent.name == "ml"
