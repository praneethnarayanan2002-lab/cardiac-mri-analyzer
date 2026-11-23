"""Upload, case summary, slice and cine PNGs, PDF report."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from backend.app.report import build_report
from backend.app.schemas import CaseSummary, Layer, Phase
from backend.app.services.analysis import analyse
from backend.app.services.images import png_bytes, window_volume
from backend.app.services.store import Case
from backend.app.services.uploads import UploadError, check_pair, read_volume
from backend.app.state import AppState, get_state

router = APIRouter(prefix="/api")
PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}


def _case(state: AppState, case_id: str) -> Case:
    case = state.store.get(case_id)
    if case is None:
        raise HTTPException(404, f"no case {case_id!r}; uploaded cases expire after an hour")
    return case


async def _read(upload: UploadFile, limit_mb: int) -> bytes:
    data = await upload.read(limit_mb * 1024 * 1024 + 1)
    if len(data) > limit_mb * 1024 * 1024:
        raise HTTPException(413, f"{upload.filename}: larger than {limit_mb} MB")
    if not data:
        raise HTTPException(422, f"{upload.filename}: empty file")
    return data


@router.post("/analyze")
async def analyze(
    state: Annotated[AppState, Depends(get_state)],
    ed: Annotated[UploadFile, File(description="end-diastole frame, .nii or .nii.gz")],
    es: Annotated[UploadFile, File(description="end-systole frame, .nii or .nii.gz")],
    height_cm: Annotated[float, Form(gt=50, lt=250)],
    weight_kg: Annotated[float, Form(gt=10, lt=300)],
) -> dict[str, str]:
    """Analyse an uploaded ED + ES pair. Returns the new case id."""
    if state.models is None:
        raise HTTPException(503, f"models are not loaded: {state.models_error}")
    limit = state.settings.max_upload_mb
    try:
        ed_volume = read_volume(await _read(ed, limit), ed.filename or "ED file")
        es_volume = read_volume(await _read(es, limit), es.filename or "ES file")
        check_pair(ed_volume, es_volume)
    except UploadError as error:
        raise HTTPException(422, str(error)) from None
    images = {"ed": ed_volume[0], "es": es_volume[0]}
    # Inference takes seconds; run it off the event loop so other requests keep being served.
    result = await run_in_threadpool(
        analyse, state.models, images, ed_volume[1], height_cm, weight_kg
    )
    summary = {
        "source": "upload",
        "layers": ["image", "unet", "unet_pretrained", "doubt"],
        **result.summary,
    }
    case = state.store.add_upload(
        summary, {p: window_volume(v) for p, v in images.items()}, result.layers
    )
    return {"case_id": case.case_id}


@router.get("/cases/{case_id}", response_model=CaseSummary)
def get_case(case_id: str, state: Annotated[AppState, Depends(get_state)]) -> dict[str, object]:
    return _case(state, case_id).summary


@router.get("/cases/{case_id}/slice/{phase}/{k}")
def get_slice(
    case_id: str,
    phase: Phase,
    k: int,
    state: Annotated[AppState, Depends(get_state)],
    layer: Annotated[Layer, Query()] = "image",
) -> Response:
    """One slice as PNG: greyscale image, a label map (pixel values 0-3) or a doubt map (0-255)."""
    case = _case(state, case_id)
    volume = case.images[phase] if layer == "image" else case.layers[phase].get(layer)
    if volume is None:
        raise HTTPException(404, f"layer {layer!r} is not available for this case")
    if not 0 <= k < volume.shape[2]:
        raise HTTPException(404, f"slice {k} out of range 0-{volume.shape[2] - 1}")
    return Response(png_bytes(volume[:, :, k]), media_type="image/png", headers=PNG_HEADERS)


@router.get("/cases/{case_id}/cine/{k}")
def get_cine(
    case_id: str,
    k: int,
    state: Annotated[AppState, Depends(get_state)],
    frame: Annotated[int, Query(ge=0)] = 0,
) -> Response:
    """Slice k of one cine frame (samples only)."""
    case = _case(state, case_id)
    if case.cine is None:
        raise HTTPException(404, "no cine for this case")
    if not (0 <= k < case.cine.shape[2] and frame < case.cine.shape[3]):
        raise HTTPException(404, "slice or frame out of range")
    return Response(
        png_bytes(case.cine[:, :, k, frame]), media_type="image/png", headers=PNG_HEADERS
    )


@router.get("/cases/{case_id}/report.pdf")
def get_report(case_id: str, state: Annotated[AppState, Depends(get_state)]) -> Response:
    case = _case(state, case_id)
    name = case.summary.get("pid") or case_id
    return Response(
        build_report(case),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="cardiac-report-{name}.pdf"'},
    )
