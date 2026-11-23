import { useState } from "react";

import { BlandAltman, ConfusionMatrix, DiceBars, JitterCurve, RiskCoverage } from "../components/charts/charts";
import { MODEL_NAMES } from "../components/charts/models";
import { Card, ErrorState, Segmented, Skeleton } from "../components/ui/controls";
import { dataSource } from "../data/dataSource";
import type { ModelKey } from "../data/types";
import { useLoad } from "../lib/useLoad";

const MEASUREMENTS: { key: string; label: string }[] = [
  { key: "lv_ef_pct", label: "LV EF" },
  { key: "lv_edv_ml", label: "LV EDV" },
  { key: "lv_esv_ml", label: "LV ESV" },
  { key: "rv_ef_pct", label: "RV EF" },
  { key: "rv_edv_ml", label: "RV EDV" },
  { key: "rv_esv_ml", label: "RV ESV" },
  { key: "myo_mass_g", label: "Mass" },
  { key: "wall_thickness_max_mm", label: "Max wall" },
];

/** Test-set results, every number read from results/ through the data source. */
export function ComparisonPage() {
  const { data, error, retry } = useLoad(() => dataSource.getMetricsSummary(), "metrics");
  const [model, setModel] = useState<ModelKey>("unet_resnet34");
  const [measurement, setMeasurement] = useState("lv_ef_pct");

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-10">
        <ErrorState message={error} onRetry={retry} />
      </div>
    );
  }
  if (!data) {
    return (
      <div className="mx-auto grid max-w-6xl gap-4 px-4 py-10 md:grid-cols-2">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-72" />
        ))}
      </div>
    );
  }

  const n = data.segmentation.n_patients;
  const xgb = data.diagnosis.test;
  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <header className="rise mb-6 max-w-3xl">
        <h1 className="text-2xl font-semibold text-text">Model comparison</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Everything here is measured on {n} held-out test patients, each model scored once after all
          training was finished. "±" is the spread over 3 training seeds, not over patients.
        </p>
      </header>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="How well each model outlines the heart" delay={0}>
          <DiceBars data={data.segmentation} />
          <p className="mt-2 text-xs text-muted">
            Dice: overlap with the expert, 1 = perfect. Pretraining didn't measurably help; MedSAM is
            given boxes cut from the expert's own outline and still trails, badly on the muscle.
          </p>
        </Card>
        <Card title="Do the clinical numbers agree with the expert's?" delay={40}>
          <div className="mb-3 flex flex-wrap gap-2">
            <Segmented<ModelKey>
              label="Model"
              value={model}
              onChange={setModel}
              options={(Object.keys(MODEL_NAMES) as ModelKey[]).map((m) => ({ value: m, label: MODEL_NAMES[m] }))}
            />
            <select
              aria-label="Measurement"
              value={measurement}
              onChange={(e) => setMeasurement(e.target.value)}
              className="rounded-lg border border-border bg-surface px-2 py-1 text-xs text-text"
            >
              {MEASUREMENTS.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
          <BlandAltman data={data.clinical} model={model} measurement={measurement} />
        </Card>
        <Card title="Diagnosis from measurements (XGBoost)" delay={80}>
          <div className="grid gap-4 sm:grid-cols-2">
            {xgb.expert?.xgboost && <ConfusionMatrix title="from expert outlines" matrix={xgb.expert.xgboost.confusion} />}
            {xgb.unet_resnet34?.xgboost && (
              <ConfusionMatrix title="from pretrained U-Net outlines (seed 0)" matrix={xgb.unet_resnet34.xgboost.confusion} />
            )}
          </div>
          <p className="mt-3 text-xs text-muted">
            Trained on expert measurements of 100 training patients. Most mistakes swap MINF and DCM:
            both are weak, enlarged left ventricles, and what separates them (wall motion) isn't measured.
          </p>
        </Card>
        <Card title="Does the model know when it's wrong?" delay={120}>
          <RiskCoverage data={data.uncertainty} />
          <p className="mt-2 text-xs text-muted">
            Patients sorted by doubt score. Keeping only the more confident ones lowers the error;
            the closer the curve to "perfect", the better the score ranks patients. Spearman
            correlation of doubt and error: {data.uncertainty.spearman_rho.mean.toFixed(2)} ±{" "}
            {data.uncertainty.spearman_rho.std.toFixed(2)} over 3 seeds.
          </p>
        </Card>
        <Card title="MedSAM with sloppier boxes" delay={160} className="lg:col-span-2">
          <div className="lg:max-w-2xl">
            <JitterCurve data={data.segmentation} />
          </div>
          <p className="mt-2 text-xs text-muted">
            Each box edge moved randomly by up to this many pixels. Up to 5 px barely matters; by
            20 px, RV and LV drop sharply.
          </p>
        </Card>
      </div>
    </div>
  );
}
