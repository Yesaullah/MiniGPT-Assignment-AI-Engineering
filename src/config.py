"""Typed configuration: YAML files merged in order, then `section.key=value` overrides.

Usage: python -m src.config --config configs/tiny_gpt.yaml [--config overlay.yaml] [--set model.n_layers=6]
prints the fully resolved configuration.
"""
from __future__ import annotations

import argparse
import ast
import math
from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 0  # filled in from the tokenizer
    context_length: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int | None = None  # None -> 4 * d_model
    dropout: float = 0.1
    causal: bool = True  # False only for the target-leakage demonstration


@dataclass(frozen=True)
class DataConfig:
    path: str = "data/input.txt"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    max_steps: int = 5000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 250
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 250
    seed: int = 0
    device: str = "auto"
    out_dir: str = "outputs/runs/baseline"
    sample_prompt: str = "ROMEO:\n"
    sample_tokens: int = 200


@dataclass(frozen=True)
class GenerateConfig:
    temperature: float = 0.8
    top_k: int | None = None
    max_new_tokens: int = 200


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig, "generate": GenerateConfig}


def _merge(base: dict, new: dict, where: str = "") -> dict:
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def _parse_value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {"true": True, "false": False, "null": None, "none": None}.get(text.lower(), text)


def from_dict(raw: dict) -> Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        values = dict(raw.get(name) or {})
        allowed = {f.name for f in fields(cls)}
        bad = set(values) - allowed
        if bad:
            raise ValueError(f"Unknown keys in '{name}': {sorted(bad)}")
        sections[name] = cls(**values)
    cfg = Config(**sections)
    return validate(resolve(cfg))


def resolve(cfg: Config) -> Config:
    m = cfg.model
    if m.ffn_dim is None:
        m = ModelConfig(**{**asdict(m), "ffn_dim": 4 * m.d_model})
    return Config(model=m, data=cfg.data, train=cfg.train, generate=cfg.generate)


def validate(cfg: Config) -> Config:
    m, t = cfg.model, cfg.train
    if m.d_model % m.n_heads != 0:
        raise ValueError(f"d_model ({m.d_model}) must be divisible by n_heads ({m.n_heads})")
    if not 0.0 <= m.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    for name in ("lr", "weight_decay", "grad_clip"):
        if not math.isfinite(getattr(t, name)) or getattr(t, name) < 0:
            raise ValueError(f"train.{name} must be finite and >= 0")
    if t.warmup_steps >= t.max_steps:
        raise ValueError("warmup_steps must be smaller than max_steps")
    if cfg.generate.temperature < 0:
        raise ValueError("temperature must be >= 0")
    return cfg


def load_config(paths: list[str] | str, overrides: list[str] | None = None) -> Config:
    if isinstance(paths, str):
        paths = [paths]
    raw: dict = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw = _merge(raw, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        if not name:
            raise ValueError(f"Override must look like section.key=value, got '{item}'")
        raw.setdefault(section, {})[name] = _parse_value(value)
    return from_dict(raw)


def with_vocab(cfg: Config, vocab_size: int) -> Config:
    model = ModelConfig(**{**asdict(cfg.model), "vocab_size": vocab_size})
    return Config(model=model, data=cfg.data, train=cfg.train, generate=cfg.generate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a fully resolved config.")
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config or ["configs/tiny_gpt.yaml"], args.overrides)
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


if __name__ == "__main__":
    main()
