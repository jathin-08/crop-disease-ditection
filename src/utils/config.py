"""
Tomato Disease AI - Configuration Manager
Centralizes YAML configuration parsing and project path management.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Determine Project Root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """Loads master project YAML configuration."""
    cfg_path = config_file or CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_project_root() -> Path:
    """Returns absolute Path to project root."""
    return PROJECT_ROOT


def resolve_model_path(custom_name_or_path: Optional[str] = None) -> Path:
    """
    Resolves the active YOLO26 model checkpoint.
    Order of precedence:
      1. Provided explicit path (if exists)
      2. models/trained/best.pt
      3. models/pretrained/yolo26n.pt
      4. models/best.pt
      5. Fallback pretrained filename
    """
    cfg = load_config()
    
    if custom_name_or_path:
        p = Path(custom_name_or_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists() and p.is_file() and p.stat().st_size > 1024:
            return p

    trained_best = PROJECT_ROOT / cfg.get("model", {}).get("best_weights", "models/trained/best.pt")
    if trained_best.exists() and trained_best.is_file() and trained_best.stat().st_size > 1024:
        return trained_best

    pretrained_best = PROJECT_ROOT / cfg.get("model", {}).get("pretrained_dir", "models/pretrained") / cfg.get("model", {}).get("name", "yolo26n.pt")
    if pretrained_best.exists() and pretrained_best.is_file() and pretrained_best.stat().st_size > 1024:
        return pretrained_best

    # Root models directory check
    legacy_best = PROJECT_ROOT / "models" / "best.pt"
    if legacy_best.exists() and legacy_best.is_file() and legacy_best.stat().st_size > 1024:
        return legacy_best

    legacy_yolo26 = PROJECT_ROOT / "models" / "yolo26n.pt"
    if legacy_yolo26.exists() and legacy_yolo26.is_file() and legacy_yolo26.stat().st_size > 1024:
        return legacy_yolo26

    # Return default weight name for Ultralytics auto-download if needed
    return Path(cfg.get("model", {}).get("name", "yolo26n.pt"))


def get_classes() -> Dict[int, str]:
    """Returns mapping of class_id -> class_name from config."""
    cfg = load_config()
    raw_classes = cfg.get("classes", {})
    return {int(k): str(v) for k, v in raw_classes.items()}
