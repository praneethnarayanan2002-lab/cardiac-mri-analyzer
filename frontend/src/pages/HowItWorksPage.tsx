import { Card } from "../components/ui/controls";

const REPO = "https://github.com/praneethnarayanan2002-lab/cardiac-mri-analyzer";

const STEPS = [
  {
    title: "1. Outline",
    body: "A U-Net labels every pixel of every short-axis slice as background, right ventricle, heart muscle or left ventricle. It learned from 80 patients outlined by experts.",
    doc: "docs/ml_basics.md#7-the-u-net",
  },
  {
    title: "2. Measure",
    body: "Counting voxels and multiplying by their size in millimetres gives volumes at end-diastole and end-systole, the ejection fraction, muscle mass and wall thickness.",
    doc: "docs/ml_basics.md#14-from-masks-to-millimetres",
  },
  {
    title: "3. Suggest a group",
    body: "A gradient-boosted tree model reads the 18 measurements and gives a probability for each of five groups. SHAP values show which measurements pushed it which way.",
    doc: "docs/ml_basics.md#15-decision-trees-and-gradient-boosting",
  },
  {
    title: "4. Flag doubt",
    body: "The scan is shown to the U-Net ten times, slightly rotated and mirrored. Where the answers disagree, the model is unsure; a high average sends the case for review.",
    doc: "docs/ml_basics.md#17-uncertainty",
  },
];

export function HowItWorksPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <header className="rise mb-6 max-w-3xl">
        <h1 className="text-2xl font-semibold text-text">How it works</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Four steps, each a separate piece of the research code, each evaluated on 50 test patients
          the models never saw. The documentation explains every step from first principles.
        </p>
      </header>
      <Pipeline />
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {STEPS.map((s, i) => (
          <Card key={s.title} title={s.title} delay={i * 40}>
            <p className="text-sm leading-relaxed text-muted">{s.body}</p>
            <a className="mt-3 inline-block text-xs font-medium text-primary hover:underline" href={`${REPO}/blob/main/${s.doc}`}>
              Read more
            </a>
          </Card>
        ))}
      </div>
      <Card title="What this is not" className="mt-6">
        <ul className="list-disc space-y-1 pl-5 text-sm text-muted">
          <li>Not a medical device and not validated for clinical use.</li>
          <li>Trained and tested on one public dataset from one hospital (ACDC, 150 patients).</li>
          <li>A 2D model on thick slices: the top slice near the valves is where most errors are.</li>
          <li>MedSAM is shown with boxes taken from the expert's answer: a best case, not real use.</li>
        </ul>
        <p className="mt-3 text-sm">
          <a className="font-medium text-primary hover:underline" href={REPO}>
            Code, results and documentation on GitHub
          </a>
        </p>
      </Card>
    </div>
  );
}

function Pipeline() {
  const boxes = ["Cine MRI (ED + ES)", "U-Net outlines", "Measurements", "Diagnosis + reasons", "Doubt flag"];
  return (
    <ol className="rise flex flex-col gap-2 sm:flex-row sm:items-stretch" aria-label="Pipeline">
      {boxes.map((b, i) => (
        <li key={b} className="flex items-center gap-2 sm:flex-1">
          <span className="flex-1 rounded-lg border border-border bg-surface px-3 py-3 text-center text-sm font-medium text-text shadow-[var(--shadow-card)]">
            {b}
          </span>
          {i < boxes.length - 1 && (
            <span aria-hidden="true" className="text-muted">
              →
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}
