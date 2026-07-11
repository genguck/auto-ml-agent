import os
import yaml
from typing import Any, Dict, Optional

_config: Optional[Dict[str, Any]] = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    global _config
    if _config is None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def get_config_value(key: str, default: Any = None) -> Any:
    config = load_config()
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    if isinstance(value, str) and value.endswith("_env"):
        env_var_name = value.replace("_env", "")
        return os.getenv(env_var_name, default)
    return value


def update_config(key: str, value: Any) -> None:
    config = load_config()
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def get_env_var(name: str, required: bool = False) -> Optional[str]:
    value = os.getenv(name)
    if required and value is None:
        raise EnvironmentError(f"Environment variable {name} is required but not set")
    return value
