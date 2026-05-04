"""Cargador de configuraciones YAML con soporte de herencia y merge."""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Carga un archivo YAML y lo devuelve como dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge de dos configs. El override tiene precedencia.
    Las claves de tipo dict se fusionan recursivamente.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_config(
    model: str,
    strategy: str,
    configs_dir: str = "configs",
) -> Dict[str, Any]:
    """
    Carga y combina la config completa para un experimento dado.

    Orden de merge: base_config → model config → strategy config
    Cualquier clave en configs más específicas sobreescribe la base.

    Args:
        model: 'resnet50' | 'densenet121' | 'vgg16'
        strategy: 'straightforward' | 'finetune' | 'warmup'
        configs_dir: Ruta al directorio configs/

    Returns:
        Dict con la configuración final del experimento.

    Example:
        config = load_experiment_config("resnet50", "warmup")
        lr = config["strategy"]["phases"][0]["optimizer"]["learning_rate"]
    """
    configs_path = Path(configs_dir)

    base = load_config(configs_path / "base_config.yaml")
    model_cfg = load_config(configs_path / "models" / f"{model}.yaml")
    strategy_cfg = load_config(configs_path / "strategies" / f"{strategy}.yaml")

    merged = merge_configs(base, model_cfg)
    merged = merge_configs(merged, strategy_cfg)
    return merged
