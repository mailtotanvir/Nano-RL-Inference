"""ASGI entrypoint for the provider container."""

from pathlib import Path

from mini_stack.provider.app import create_app
from mini_stack.schemas.config import load_model_registry

_registry_path = Path(__file__).parents[3] / "configs" / "models" / "registry.yaml"
app = create_app(registry=load_model_registry(_registry_path).models)
