import os

import yaml


def ml_root():
    return os.path.dirname(os.path.abspath(__file__))


def default_config_path():
    return os.path.join(ml_root(), "config", "config.yml")


def resolve_ml_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(ml_root(), path_value)


def load_config(config_path=None):
    resolved_path = os.path.abspath(config_path or default_config_path())
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must be a YAML mapping.")

    return config, resolved_path
