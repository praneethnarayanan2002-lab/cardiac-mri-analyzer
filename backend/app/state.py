"""What every request handler needs: settings, the loaded models and the case store."""

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from backend.app.config import Settings
from backend.app.services.models import ModelBundle
from backend.app.services.store import CaseStore


@dataclass
class AppState:
    settings: Settings
    models: ModelBundle | None  # None if the weights are missing: samples still work
    store: CaseStore
    models_error: str | None = None
    metrics_cache: dict[str, Any] | None = None


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.cardiac
    return state
