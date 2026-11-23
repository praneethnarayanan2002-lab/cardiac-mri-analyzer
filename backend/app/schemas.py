"""Response shapes. The frontend's `data/types.ts` mirrors these by hand; keep them in step."""

from typing import Literal

from pydantic import BaseModel

Phase = Literal["ed", "es"]
Layer = Literal["image", "unet", "unet_pretrained", "medsam", "expert", "doubt"]
Status = Literal["normal", "borderline", "abnormal", "unknown"]


class Health(BaseModel):
    status: Literal["ok"]
    models: dict[str, str]
    samples: int


class NormalRange(BaseModel):
    low: float
    high: float
    mean: float
    sd: float
    n: int


class Measurement(BaseModel):
    key: str
    label: str
    unit: str
    value: float | None
    range: NormalRange | None
    status: Status


class Diagnosis(BaseModel):
    predicted: str
    probabilities: dict[str, float]


class Reason(BaseModel):
    feature: str
    label: str
    value: float
    unit: str
    shap: float
    text: str


class Uncertainty(BaseModel):
    score: float
    per_phase: dict[str, float]
    threshold: float
    flagged: bool


class CaseSummary(BaseModel):
    id: str
    source: Literal["sample", "upload"]
    pid: str | None = None
    true_group: str | None = None
    spacing_mm: list[float]
    shape: list[int]
    n_frames: int = 0
    ed_frame: int | None = None
    es_frame: int | None = None
    layers: list[Layer]
    height_cm: float
    weight_kg: float
    bsa_m2: float
    measurements: list[Measurement]
    diagnosis: Diagnosis
    why: list[Reason]
    uncertainty: Uncertainty
    models: dict[str, str]


class SampleCase(BaseModel):
    id: str
    pid: str
    true_group: str
    predicted: str
    flagged: bool
