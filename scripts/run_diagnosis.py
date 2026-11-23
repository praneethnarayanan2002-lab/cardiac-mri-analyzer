"""Train the diagnosis models on expert features and test them on expert and predicted features.

Usage:
    uv run python scripts/run_diagnosis.py

Reads results/{train,val}/clinical_expert.csv (the 100 training patients) for training and
cross-validation, and results/test/clinical_<source>.csv for testing. Writes results/diagnosis.json,
results/test/diagnosis_predictions.csv, results/test/diagnosis_shap.csv (per-patient SHAP values for
expert and pretrained-seed-0 features), figures/diagnosis_confusion.png, figures/shap_global.png and
the fitted model to runs/classifier/xgboost.json (for the app; not in git).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from cardiac.classify import (
    FEATURES,
    encode,
    feature_matrix,
    fit_logistic,
    fit_xgboost,
    load_classifier_config,
    rule_based,
)
from cardiac.data.io import ACDC_GROUPS
from cardiac.plots import save
from cardiac.repro import git_commit

TEST_SOURCES = {
    "expert": ["expert"],
    "unet_scratch": [f"unet_scratch_seed{k}" for k in range(3)],
    "unet_resnet34": [f"unet_resnet34_seed{k}" for k in range(3)],
    "medsam_jitter0": ["medsam_jitter0"],
}
SOURCE_LABELS = {
    "expert": "expert masks",
    "unet_scratch": "U-Net (scratch)",
    "unet_resnet34": "U-Net (ResNet-34)",
    "medsam_jitter0": "MedSAM (oracle boxes)",
}


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels = list(range(len(ACDC_GROUPS)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1": {g: float(f) for g, f in zip(ACDC_GROUPS, per_class, strict=True)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def mean_std(blocks: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(b[key]) for b in blocks]  # type: ignore[arg-type]
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(np.mean(values)), "std": std}


SHAP_SOURCES = ("expert", "unet_resnet34_seed0")


def shap_table(
    model: object, table: pd.DataFrame, x: np.ndarray, stem: str
) -> list[dict[str, object]]:
    """One row per patient and feature: the feature's value and its SHAP value towards each group.
    A row with feature "(base)" holds the explainer's base score per group, so for each patient and
    group, base + the sum of the feature rows = the model's raw score (checked here)."""
    values, base = shap_values(model, x)
    rows: list[dict[str, object]] = []
    for i, pid in enumerate(table.pid):
        per_feature = [("(base)", np.nan, base)]
        per_feature += [(f, float(x[i, j]), values[i, j]) for j, f in enumerate(FEATURES)]
        for feature, value, contributions in per_feature:
            row: dict[str, object] = {"pid": pid, "features_from": stem, "feature": feature}
            row["value"] = value
            row.update({f"shap_{g}": float(contributions[k]) for k, g in enumerate(ACDC_GROUPS)})
            rows.append(row)
    margins = np.asarray(model.predict(x, output_margin=True))  # type: ignore[attr-defined]
    assert np.allclose(base + values.sum(axis=1), margins, atol=1e-3), "SHAP does not add up"
    return rows


def agreement(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """How often the XGBoost diagnosis from a segmenter's features equals the one from the expert's
    features for the same patient, whether or not either is right. Mean over runs."""
    expert = predictions[predictions.features_from == "expert"].set_index("pid").xgboost
    out = {}
    for source, stems in TEST_SOURCES.items():
        if source == "expert":
            continue
        rates = []
        for stem in stems:
            part = predictions[predictions.features_from == stem].set_index("pid").xgboost
            if len(part):
                rates.append(float((part == expert.reindex(part.index)).mean()))
        if rates:
            std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
            out[source] = {"mean": float(np.mean(rates)), "std": std}
    return out


def plot_confusions(matrices: dict[str, list[list[int]]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.4 * len(matrices), 3.6))
    for ax, (title, matrix) in zip(np.atleast_1d(axes), matrices.items(), strict=True):
        m = np.asarray(matrix)
        ax.imshow(m, cmap="Blues", vmin=0, vmax=10)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                colour = "white" if m[i, j] > 6 else "#3d3d3a"
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=9, color=colour)
        ax.set_xticks(range(len(ACDC_GROUPS)), ACDC_GROUPS, fontsize=8)
        ax.set_yticks(range(len(ACDC_GROUPS)), ACDC_GROUPS, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("XGBoost trained on expert features; test set, 10 patients per group", fontsize=10)
    fig.tight_layout()
    save(fig, out)


def plot_shap(mean_abs: pd.DataFrame, out: Path) -> None:
    """Mean |SHAP| per feature, stacked by class: how much each feature moves predictions."""
    order = mean_abs.sum(axis=1).sort_values().index
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots
    fig, ax = plt.subplots(figsize=(7.5, 6))
    left = np.zeros(len(order))
    for group, colour in zip(ACDC_GROUPS, colours, strict=True):
        values = mean_abs.loc[order, group].to_numpy()
        ax.barh(range(len(order)), values, left=left, color=colour, label=group,
                edgecolor="white", linewidth=1)  # fmt: skip
        left += values
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("mean |SHAP value| on the training patients (log-odds)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(title="towards / away from", fontsize=8, title_fontsize=8, frameon=False,
              loc="lower right")  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Which measurements drive the XGBoost diagnosis", fontsize=10)
    save(fig, out)


def shap_values(model: object, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(patients, features, classes) SHAP values in log-odds units, and the base value per class.

    The base is read after computing the values on the same explainer: shap updates
    expected_value during that call, and only the updated one makes base + values add up to the
    model's raw score."""
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x))
    if values.shape[0] == len(ACDC_GROUPS) and values.ndim == 3:  # older shap: classes first
        values = np.moveaxis(values, 0, 2)
    return values, np.asarray(explainer.expected_value, dtype=float)


def main() -> None:
    commit = git_commit()
    config = load_classifier_config()
    train = pd.concat(
        [pd.read_csv(f"results/{s}/clinical_expert.csv") for s in ("train", "val")],
        ignore_index=True,
    )
    x_train, y_train = feature_matrix(train), encode(train.group)

    xgb = fit_xgboost(x_train, y_train, config)
    logreg = fit_logistic(x_train, y_train, config)
    rules_train = rule_based(train)
    print(f"CV accuracy: XGBoost {xgb.best_score_:.3f} {xgb.best_params_}")
    print(f"CV accuracy: logistic {logreg.best_score_:.3f} {logreg.best_params_}")
    print(f"rules on training patients: {accuracy_score(y_train, rules_train):.3f}")

    results: dict[str, object] = {
        "eval_git_commit": commit,
        "train": {
            "patients": len(train),
            "features": list(FEATURES),
            "xgboost": {"cv_accuracy": float(xgb.best_score_), "best_params": xgb.best_params_},
            "logistic": {
                "cv_accuracy": float(logreg.best_score_),
                "best_params": logreg.best_params_,
            },
            "rules_accuracy_on_train": float(accuracy_score(y_train, rules_train)),
            "cv": f"stratified {config.cv_folds}-fold, seed {config.seed}",
        },
    }

    test_results: dict[str, object] = {}
    predictions = []
    shap_rows: list[dict[str, object]] = []
    confusions: dict[str, list[list[int]]] = {}
    for source, stems in TEST_SOURCES.items():
        per_model: dict[str, list[dict[str, object]]] = {"xgboost": [], "logistic": [], "rules": []}
        for stem in stems:
            path = Path(f"results/test/clinical_{stem}.csv")
            if not path.exists():
                print(f"skipping {stem}: {path} not found")
                continue
            table = pd.read_csv(path)
            x, y = feature_matrix(table), encode(table.group)
            preds = {
                "xgboost": xgb.predict(x),
                "logistic": logreg.predict(np.nan_to_num(x, nan=np.nanmedian(x_train, axis=0))),
                "rules": rule_based(table),
            }
            for name, pred in preds.items():
                per_model[name].append(scores(y, pred))
            probs = xgb.predict_proba(x)
            if stem in SHAP_SOURCES:
                shap_rows += shap_table(xgb.best_estimator_, table, x, stem)
            for i, row in enumerate(table.itertuples()):
                predictions.append(
                    {
                        "pid": row.pid,
                        "group": row.group,
                        "features_from": stem,
                        "xgboost": ACDC_GROUPS[preds["xgboost"][i]],
                        "logistic": ACDC_GROUPS[preds["logistic"][i]],
                        "rules": ACDC_GROUPS[preds["rules"][i]],
                        **{f"p_{g}": float(probs[i, k]) for k, g in enumerate(ACDC_GROUPS)},
                    }
                )
            if stem in ("expert", "unet_resnet34_seed0", "medsam_jitter0"):
                note = " seed 0" if "seed" in stem else ""
                confusions[f"features from {SOURCE_LABELS[source]}{note}"] = per_model["xgboost"][
                    -1
                ]["confusion"]  # type: ignore[assignment]
        test_results[source] = {
            name: {
                "n_runs": len(blocks),
                "accuracy": mean_std(blocks, "accuracy"),
                "macro_f1": mean_std(blocks, "macro_f1"),
                "per_run": blocks,
            }
            for name, blocks in per_model.items()
            if blocks
        }
    results["test"] = test_results
    results["agreement_with_expert_features"] = agreement(pd.DataFrame(predictions))

    values, _ = shap_values(xgb.best_estimator_, x_train)
    mean_abs = pd.DataFrame(np.abs(values).mean(axis=0), index=list(FEATURES), columns=ACDC_GROUPS)
    results["shap_mean_abs_train"] = {
        f: {g: float(mean_abs.loc[f, g]) for g in ACDC_GROUPS} for f in FEATURES
    }

    Path("results/diagnosis.json").write_text(json.dumps(results, indent=2) + "\n")
    pd.DataFrame(predictions).to_csv(
        "results/test/diagnosis_predictions.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(shap_rows).to_csv(
        "results/test/diagnosis_shap.csv", index=False, float_format="%.4f"
    )
    plot_confusions(confusions, Path("figures/diagnosis_confusion.png"))
    plot_shap(mean_abs, Path("figures/shap_global.png"))
    model_dir = Path("runs/classifier")
    model_dir.mkdir(parents=True, exist_ok=True)
    xgb.best_estimator_.save_model(model_dir / "xgboost.json")
    print("wrote results/diagnosis.json, results/test/diagnosis_predictions.csv, figures, model")
    for source, block in test_results.items():
        assert isinstance(block, dict)
        line = "  ".join(
            f"{n} {b['accuracy']['mean']:.3f}±{b['accuracy']['std']:.3f}" for n, b in block.items()
        )
        print(f"  test, features from {source:<15} accuracy: {line}")


if __name__ == "__main__":
    main()
