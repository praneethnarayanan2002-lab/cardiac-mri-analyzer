"""API tests with small stand-in models: an untrained scratch U-Net and a 5-tree XGBoost trained on
the committed expert features. They check the plumbing, not the quality of the answers."""

import io
import json
from collections.abc import Iterator
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import shap
import torch
from fastapi.testclient import TestClient
from PIL import Image
from xgboost import XGBClassifier

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.images import window_volume
from backend.app.services.models import ModelBundle, UNet
from backend.app.services.store import Case, save_sample
from cardiac.classify import encode, feature_matrix
from cardiac.config import load_config
from cardiac.models.unet import build_scratch_unet

RESULTS = Path("results")


def _bundle() -> ModelBundle:
    torch.manual_seed(0)
    config = load_config(Path("configs/unet_scratch.yaml"))
    unet = UNet(model=build_scratch_unet(0.0).eval(), config=config, run="test-untrained")
    train = pd.read_csv(RESULTS / "train" / "clinical_expert.csv")
    classifier = XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1)
    classifier.fit(feature_matrix(train), encode(train.group))
    ranges = json.loads((RESULTS / "normal_ranges.json").read_text())["ranges"]
    return ModelBundle(
        unet=unet,
        unet_pretrained=unet,
        classifier=classifier,
        explainer=shap.TreeExplainer(classifier),
        normal_ranges=ranges,
        doubt_threshold=0.1,
        device=torch.device("cpu"),
        code_version="test",
    )


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return _bundle()


def _sample(folder: Path) -> None:
    rng = np.random.default_rng(0)
    image = rng.random((64, 64, 3)).astype(np.float32)
    label = np.zeros((64, 64, 3), dtype=np.uint8)
    label[20:40, 20:40] = 3
    summary = {
        "id": "patient999",
        "source": "sample",
        "pid": "patient999",
        "true_group": "NOR",
        "spacing_mm": [1.0, 1.0, 10.0],
        "shape": [64, 64, 3],
        "n_frames": 2,
        "ed_frame": 0,
        "es_frame": 1,
        "layers": ["image", "expert"],
        "height_cm": 170.0,
        "weight_kg": 70.0,
        "bsa_m2": 1.82,
        "measurements": [],
        "diagnosis": {"predicted": "NOR", "probabilities": {"NOR": 1.0}},
        "why": [],
        "uncertainty": {
            "score": 0.05,
            "per_phase": {"ed": 0.05, "es": 0.05},
            "threshold": 0.1,
            "flagged": False,
        },
        "models": {"measurements_from": "x", "classifier": "x", "code_version": "test"},
    }
    layers = {p: {"expert": label, "unet_pretrained": label} for p in ("ed", "es")}
    case = Case(
        case_id="patient999",
        summary=summary,
        images={p: window_volume(image) for p in ("ed", "es")},
        layers=layers,
        cine=window_volume(rng.random((64, 64, 3, 2)).astype(np.float32)),
    )
    save_sample(case, folder)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory, bundle: ModelBundle) -> Iterator[TestClient]:
    samples = tmp_path_factory.mktemp("samples")
    _sample(samples)
    settings = Settings(samples_dir=samples, results_dir=RESULTS)
    with TestClient(create_app(settings, bundle)) as test_client:
        yield test_client


def _nifti(shape: tuple[int, ...], spacing: tuple[float, ...] = (1.5, 1.5, 10.0)) -> bytes:
    rng = np.random.default_rng(1)
    img = nib.Nifti1Image(rng.random(shape).astype(np.float32), np.diag([*spacing, 1.0]))
    img.header.set_zooms(spacing)
    return img.to_bytes()


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["samples"] == 1
    assert body["models"]["unet"] == "test-untrained"


def test_samples_and_case(client: TestClient) -> None:
    samples = client.get("/api/samples").json()
    assert samples == [
        {
            "id": "patient999",
            "pid": "patient999",
            "true_group": "NOR",
            "predicted": "NOR",
            "flagged": False,
        }
    ]
    case = client.get("/api/cases/patient999").json()
    assert case["source"] == "sample" and case["n_frames"] == 2


def test_slice_png_orientation_and_values(client: TestClient) -> None:
    response = client.get("/api/cases/patient999/slice/ed/1", params={"layer": "expert"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    png = np.asarray(Image.open(io.BytesIO(response.content)))
    assert png.shape == (64, 64)
    assert set(np.unique(png).tolist()) == {0, 3}  # label values, not colours


def test_cine_and_report(client: TestClient) -> None:
    assert client.get("/api/cases/patient999/cine/0", params={"frame": 1}).status_code == 200
    assert client.get("/api/cases/patient999/cine/0", params={"frame": 5}).status_code == 404
    report = client.get("/api/cases/patient999/report.pdf")
    assert report.status_code == 200 and report.content.startswith(b"%PDF")


def test_unknown_case_and_layer(client: TestClient) -> None:
    assert client.get("/api/cases/nope").status_code == 404
    assert client.get("/api/cases/patient999/slice/ed/0?layer=medsam").status_code == 404
    assert client.get("/api/cases/patient999/slice/ed/9").status_code == 404


def test_analyze_synthetic_upload(client: TestClient) -> None:
    files = {
        "ed": ("ed.nii.gz", _nifti((80, 80, 2)), "application/gzip"),
        "es": ("es.nii.gz", _nifti((80, 80, 2)), "application/gzip"),
    }
    response = client.post("/api/analyze", files=files, data={"height_cm": 175, "weight_kg": 70})
    assert response.status_code == 200, response.text
    case_id = response.json()["case_id"]
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["source"] == "upload"
    assert case["shape"] == [80, 80, 2]
    assert set(case["diagnosis"]["probabilities"]) == {"NOR", "MINF", "DCM", "HCM", "RV"}
    assert {m["key"] for m in case["measurements"]} >= {"lv_ef_pct", "myo_mass_g"}
    for layer in ("image", "unet", "unet_pretrained", "doubt"):
        png = client.get(f"/api/cases/{case_id}/slice/es/1", params={"layer": layer})
        assert png.status_code == 200
        assert np.asarray(Image.open(io.BytesIO(png.content))).shape == (80, 80)


def test_analyze_rejects_bad_files(client: TestClient) -> None:
    good = _nifti((80, 80, 2))
    text = client.post(
        "/api/analyze",
        files={"ed": ("ed.nii.gz", b"not a scan", "text/plain"), "es": ("es.nii.gz", good)},
        data={"height_cm": 175, "weight_kg": 70},
    )
    assert text.status_code == 422 and "ed.nii.gz" in text.json()["detail"]
    mismatch = client.post(
        "/api/analyze",
        files={"ed": ("ed.nii.gz", good), "es": ("es.nii.gz", _nifti((96, 96, 2)))},
        data={"height_cm": 175, "weight_kg": 70},
    )
    assert mismatch.status_code == 422 and "differ" in mismatch.json()["detail"]
    wrong_type = client.post(
        "/api/analyze",
        files={"ed": ("ed.png", good), "es": ("es.nii.gz", good)},
        data={"height_cm": 175, "weight_kg": 70},
    )
    assert wrong_type.status_code == 422
    no_height = client.post(
        "/api/analyze", files={"ed": ("ed.nii.gz", good), "es": ("es.nii.gz", good)}
    )
    assert no_height.status_code == 422


def test_metrics_summary(client: TestClient) -> None:
    body = client.get("/api/metrics/summary").json()
    assert set(body) == {"segmentation", "clinical", "diagnosis", "uncertainty"}
    assert body["segmentation"]["dice"]["unet_resnet34"]["LV"]["mean"] > 0.5


def test_uploads_disabled_without_models(tmp_path: Path) -> None:
    settings = Settings(samples_dir=tmp_path, results_dir=RESULTS, runs_dir=tmp_path / "none")
    with TestClient(create_app(settings)) as test_client:
        assert "error" in test_client.get("/api/health").json()["models"]
        good = _nifti((80, 80, 2))
        response = test_client.post(
            "/api/analyze",
            files={"ed": ("ed.nii.gz", good), "es": ("es.nii.gz", good)},
            data={"height_cm": 175, "weight_kg": 70},
        )
        assert response.status_code == 503
