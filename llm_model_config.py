"""Resolve inference models from explicit overrides, environment, or TOML."""

import os
from pathlib import Path
import tomllib


MODEL_CONFIG_PATH = Path(__file__).with_name("llm_models.toml")
MODEL_ENV_VARS = {
    "openai": "OPENAI_MODEL",
    "gemini": "GEMINI_MODEL",
    "claude": "ANTHROPIC_MODEL",
}
EMBEDDING_MODEL_ENV_VAR = "EMBEDDING_MODEL"


def resolve_model(provider: str, model: str | None = None) -> str:
    """Use an explicit model, then a provider env var, then the config default."""
    env_vars = {**MODEL_ENV_VARS, "embedding": EMBEDDING_MODEL_ENV_VAR}
    if provider not in env_vars:
        raise ValueError("Unknown provider: choose one of openai|gemini|claude|embedding")

    source = "model override"
    value = model
    if value is None:
        source = env_vars[provider]
        value = os.environ.get(source)
    if value is None:
        source = f"{MODEL_CONFIG_PATH} [models].{provider}"
        try:
            with MODEL_CONFIG_PATH.open("rb") as config_file:
                value = tomllib.load(config_file).get("models", {}).get(provider)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Cannot read model configuration: {exc}") from exc

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {source} to a non-empty model ID")
    return value.strip()
