"""Health, sample list and test-set metrics."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from backend.app.schemas import Health, SampleCase
from backend.app.services.metrics import metrics_summary
from backend.app.state import AppState, get_state

router = APIRouter(prefix="/api")


@router.get("/health", response_model=Health)
def health(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    models = state.models
    versions = (
        {"error": state.models_error or "not loaded"}
        if models is None
        else {
            "unet": models.unet.run,
            "unet_pretrained": models.unet_pretrained.run,
            "classifier": "xgboost",
            "code_version": models.code_version,
            "device": str(models.device),
        }
    )
    return {"status": "ok", "models": versions, "samples": len(state.store.sample_ids())}


@router.get("/samples", response_model=list[SampleCase])
def samples(state: Annotated[AppState, Depends(get_state)]) -> list[dict[str, Any]]:
    out = []
    for case_id in state.store.sample_ids():
        case = state.store.get(case_id)
        assert case is not None
        s: dict[str, Any] = case.summary
        out.append(
            {
                "id": case_id,
                "pid": s["pid"],
                "true_group": s["true_group"],
                "predicted": s["diagnosis"]["predicted"],
                "flagged": s["uncertainty"]["flagged"],
            }
        )
    return out


@router.get("/metrics/summary")
def metrics(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    if state.metrics_cache is None:
        try:
            state.metrics_cache = metrics_summary(state.settings.results_dir)
        except FileNotFoundError as error:
            raise HTTPException(503, f"results missing: {error.filename}") from None
    return state.metrics_cache
