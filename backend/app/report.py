"""One-page PDF report for a case, drawn with ReportLab from the same summary the app shows."""

import io
from typing import Any

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.services.images import to_display
from backend.app.services.store import Case
from cardiac.plots import STRUCTURE_COLOURS

TEAL = colors.HexColor("#0E7C86")
MUTED = colors.HexColor("#5B6B7C")
BORDER = colors.HexColor("#E3E8EE")
STATUS_COLOURS = {
    "normal": colors.HexColor("#15803D"),
    "borderline": colors.HexColor("#B45309"),
    "abnormal": colors.HexColor("#B91C1C"),
    "unknown": MUTED,
}


def overlay_png(image: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> io.BytesIO:
    """One uint8 (X, Y) image plane + its label plane -> RGB PNG with coloured structures."""
    grey = to_display(image).astype(np.float32)
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    lab = to_display(label)
    for value, hex_colour in STRUCTURE_COLOURS.items():
        colour = np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
        mask = lab == value
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * colour
    buffer = io.BytesIO()
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize(
        (288, 288), Image.Resampling.NEAREST
    ).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def middle_slice(label: np.ndarray) -> int:
    lv = (label == 3).sum(axis=(0, 1))
    return int(np.argmax(lv)) if lv.any() else label.shape[2] // 2


def build_report(case: Case) -> bytes:
    s: dict[str, Any] = case.summary
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8.5, leading=11)
    small = ParagraphStyle("small", parent=body, fontSize=7, textColor=MUTED, leading=9)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, textColor=TEAL, spaceAfter=3)
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=16, alignment=0, spaceAfter=2)

    out = io.BytesIO()
    doc = SimpleDocTemplate(
        out,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=12 * mm,
    )
    who = s.get("pid") or f"upload {s['id']}"
    story: list[Any] = [
        Paragraph("Cardiac MRI analysis", title),
        Paragraph(
            f"Case: {who}. Height {s['height_cm']:.0f} cm, weight {s['weight_kg']:.0f} kg, "
            f"body surface area {s['bsa_m2']:.2f} m². Research demo, not a medical device: "
            "not validated for clinical use.",
            small,
        ),
        Spacer(1, 4 * mm),
    ]

    pics = []
    for phase, name in (("ed", "End-diastole"), ("es", "End-systole")):
        label = case.layers[phase]["unet_pretrained"]
        k = middle_slice(label)
        pics.append(
            [
                RLImage(overlay_png(case.images[phase][:, :, k], label[:, :, k]), 62 * mm, 62 * mm),
                Paragraph(f"{name}, slice {k + 1} of {label.shape[2]}", small),
            ]
        )
    story.append(
        Table([[pics[0][0], pics[1][0]], [pics[0][1], pics[1][1]]], colWidths=[90 * mm, 90 * mm])
    )
    story.append(Paragraph("Red: LV cavity. Green: myocardium. Blue: RV cavity.", small))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Measurements", h2))
    rows = [["", "value", "normal range*", ""]]
    status_styles = []
    for i, m in enumerate(s["measurements"], start=1):
        value = "-" if m["value"] is None else f"{m['value']:.1f} {m['unit']}"
        rng = m["range"]
        rng_text = "-" if rng is None else f"{rng['low']:.1f} to {rng['high']:.1f}"
        rows.append([m["label"], value, rng_text, m["status"]])
        status_styles.append(("TEXTCOLOR", (3, i), (3, i), STATUS_COLOURS[m["status"]]))
    table = Table(rows, colWidths=[70 * mm, 35 * mm, 40 * mm, 25 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, BORDER),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                *status_styles,
            ]
        )
    )
    story += [
        table,
        Paragraph(
            "*Mean ± 2 SD of the 20 normal (NOR) training patients, measured from expert outlines "
            "with the same code. Not a clinical reference range.",
            small,
        ),
    ]

    diagnosis = s["diagnosis"]
    probs = ", ".join(
        f"{g} {p:.0%}" for g, p in sorted(diagnosis["probabilities"].items(), key=lambda kv: -kv[1])
    )
    story += [
        Paragraph("Suggested group", h2),
        Paragraph(f"<b>{diagnosis['predicted']}</b> ({probs})", body),
    ]
    for reason in s["why"]:
        story.append(
            Paragraph(f"• {reason['text']} ({reason['value']:.1f} {reason['unit']})", body)
        )

    unc = s["uncertainty"]
    verdict = (
        "<b>Flagged for review</b>: the model's doubt is in the top 20% of validation patients."
        if unc["flagged"]
        else "Not flagged: the model's doubt is below the review threshold."
    )
    story += [
        Paragraph("Confidence", h2),
        Paragraph(
            f"Doubt score {unc['score']:.3f} (threshold {unc['threshold']:.3f}). {verdict}", body
        ),
    ]
    story += [
        Spacer(1, 4 * mm),
        Paragraph(
            f"Segmentation: {s['models']['measurements_from']}. "
            f"Diagnosis: {s['models']['classifier']}. "
            f"Code version {s['models']['code_version'][:12]}.",
            small,
        ),
    ]
    doc.build(story)
    return out.getvalue()
