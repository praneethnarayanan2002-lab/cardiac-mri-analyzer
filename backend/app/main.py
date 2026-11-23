"""The FastAPI app. Run with `make api` (uvicorn on port 8000).

Models load once at startup. If the trained weights are missing, the app still starts: sample cases
and test-set metrics work, uploads answer 503 with the reason."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings, get_settings
from backend.app.routes import cases, info
from backend.app.services.models import ModelBundle, ModelsMissingError, load_models
from backend.app.services.store import CaseStore
from backend.app.state import AppState

log = logging.getLogger("cardiac")


def create_app(settings: Settings | None = None, models: ModelBundle | None = None) -> FastAPI:
    """`models` lets tests pass small stand-in models instead of loading trained weights."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(
            settings=settings,
            models=models,
            store=CaseStore(settings.samples_dir, settings.case_ttl_s),
        )
        if state.models is None:
            try:
                state.models = load_models(settings)
            except ModelsMissingError as error:
                state.models_error = str(error)
                log.warning("uploads disabled: %s", error)
        app.state.cardiac = state
        yield

    app = FastAPI(title="Cardiac MRI Analyzer", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(info.router)
    app.include_router(cases.router)
    return app


app = create_app()
