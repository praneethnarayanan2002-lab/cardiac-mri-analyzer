import { useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";

import { dataSource } from "../data/dataSource";
import { Button, Icon } from "./ui/controls";

type Slot = "ed" | "es";

/** Two files (ED and ES frames) plus height and weight, which body-surface-area indexing needs. */
export function UploadForm() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<Record<Slot, File | null>>({ ed: null, es: null });
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState<Slot | null>(null);

  const ready = files.ed && files.es && Number(height) > 50 && Number(weight) > 10;

  async function submit() {
    if (!files.ed || !files.es) return;
    setBusy(true);
    setError(null);
    try {
      const id = await dataSource.analyze({
        ed: files.ed,
        es: files.es,
        heightCm: Number(height),
        weightKg: Number(weight),
      });
      navigate(`/case/${id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function drop(slot: Slot, event: DragEvent) {
    event.preventDefault();
    setOver(null);
    const file = event.dataTransfer.files[0];
    if (file) setFiles((f) => ({ ...f, [slot]: file }));
  }

  return (
    <form
      className="grid gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        {(["ed", "es"] as const).map((slot) => (
          <label
            key={slot}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(slot);
            }}
            onDragLeave={() => setOver(null)}
            onDrop={(e) => drop(slot, e)}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-[var(--radius-card)] border-2 border-dashed p-5 text-center transition-colors duration-150 ${over === slot ? "border-primary bg-primary-soft" : "border-border bg-surface-2 hover:border-primary/60"}`}
          >
            <Icon name="upload" className="h-5 w-5 text-primary" />
            <span className="text-sm font-medium text-text">
              {slot === "ed" ? "End-diastole frame" : "End-systole frame"}
            </span>
            <span className="text-xs text-muted">
              {files[slot]?.name ?? "Drop a .nii or .nii.gz file, or click"}
            </span>
            <input
              type="file"
              accept=".nii,.gz"
              className="sr-only"
              onChange={(e) => setFiles((f) => ({ ...f, [slot]: e.target.files?.[0] ?? null }))}
            />
          </label>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:max-w-sm">
        <label className="grid gap-1 text-xs font-medium text-muted">
          Height (cm)
          <input
            inputMode="decimal"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
            className="rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text"
            placeholder="175"
          />
        </label>
        <label className="grid gap-1 text-xs font-medium text-muted">
          Weight (kg)
          <input
            inputMode="decimal"
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
            className="rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text"
            placeholder="70"
          />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" variant="primary" disabled={!ready || busy}>
          {busy ? "Analysing… (a few seconds)" : "Analyse"}
        </Button>
        <span className="text-xs text-muted">
          Short-axis cine frames, like ACDC's. Nothing is stored; results are forgotten after an hour.
        </span>
      </div>
      {error && (
        <p role="alert" className="rounded-lg border border-alert/40 bg-alert/10 p-3 text-sm text-text">
          {error}
        </p>
      )}
    </form>
  );
}
