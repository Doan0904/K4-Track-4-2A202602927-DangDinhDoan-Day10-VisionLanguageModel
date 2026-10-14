"""Resolve official experiment YAML files and a controlled E4 data-size ablation."""
import copy
import json
from pathlib import Path

import yaml


def load_config(path, **overrides):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg.get("name") not in ("E0", "E1", "E2", "E3", "E4", "E5"):
        raise ValueError("Invalid experiment name")
    if "model_id" not in cfg:
        raise ValueError("Use an E0–E3 config or a resolved E4 config")
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    defaults = {"coord": "norm1000", "train_fraction": 1.0, "seed": 42,
                "max_new_tokens": 256, "max_length": 2048, "warmup_ratio": 0.05,
                "weight_decay": 0.0, "save_steps": 100, "revision": "main"}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    if cfg["quantization"] != "nf4-4bit" or cfg["coord"] not in ("norm1000", "pixel"):
        raise ValueError("Unsupported quantization/coordinate configuration")
    if not 0 < cfg["train_fraction"] <= 1:
        raise ValueError("train_fraction must be in (0,1]")
    for k in ("batch_size", "grad_accum", "max_steps", "max_new_tokens", "max_length", "save_steps"):
        if k in cfg and (not isinstance(cfg[k], int) or cfg[k] < 1):
            raise ValueError(f"{k} must be a positive integer")
    return cfg


def create_e4(base_config, out_dir="work/configs"):
    base = load_config(base_config)
    if base["name"] not in ("E1", "E2", "E3"):
        raise ValueError("E4 baseline must be E1, E2 or E3 selected on validation")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for value, suffix in ((0.1, "010"), (1.0, "100")):
        cfg = copy.deepcopy(base)
        cfg.update(name=f"E4", description="Data-size ablation: only train_fraction changes",
                   train_fraction=value, base_experiment=base["name"],
                   ablation={"variable": "data_size", "value": value})
        path = out / f"e4_data_{suffix}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(path)
    return paths


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
