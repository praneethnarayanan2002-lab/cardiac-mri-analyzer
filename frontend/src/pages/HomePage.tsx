import { Link } from "react-router-dom";

import { Card, ErrorState, Icon, Skeleton } from "../components/ui/controls";
import { UploadForm } from "../components/UploadForm";
import { dataSource } from "../data/dataSource";
import { GROUP_NAMES, GROUP_ORDER } from "../lib/format";
import { useLoad } from "../lib/useLoad";

export function HomePage() {
  const { data, error, retry } = useLoad(() => dataSource.listSamples(), "samples");
  const samples = data
    ? [...data].sort((a, b) => GROUP_ORDER.indexOf(a.true_group) - GROUP_ORDER.indexOf(b.true_group))
    : null;

  return (
    <div className="mx-auto grid max-w-6xl gap-10 px-4 py-10 sm:px-6">
      <section className="rise grid items-center gap-8 md:grid-cols-[1.4fr_1fr]">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-text sm:text-4xl">
            Outline the heart, measure it, and explain what the numbers suggest.
          </h1>
          <p className="mt-4 max-w-xl text-base leading-relaxed text-muted">
            A cardiac MRI is segmented by a U-Net, turned into the volumes and ejection fractions
            cardiologists use, sorted into one of five groups by an explainable model, and flagged
            when the model is unsure. Built on the public ACDC dataset as a research project.
          </p>
        </div>
        <HeartOutline />
      </section>

      <section aria-labelledby="samples">
        <h2 id="samples" className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted">
          Try a sample patient
        </h2>
        {error ? (
          <ErrorState message={error} onRetry={retry} />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {samples === null
              ? GROUP_ORDER.map((g) => <Skeleton key={g} className="h-32" />)
              : samples.map((s, i) => (
                  <Link
                    key={s.id}
                    to={`/case/${s.id}`}
                    className="rise group flex flex-col justify-between rounded-[var(--radius-card)] border border-border bg-surface p-4 shadow-[var(--shadow-card)] transition-colors duration-150 hover:border-primary"
                    style={{ animationDelay: `${i * 40}ms` }}
                  >
                    <div>
                      <span className="text-xs font-semibold text-primary">{s.true_group}</span>
                      <p className="mt-1 text-sm font-medium text-text">{GROUP_NAMES[s.true_group]}</p>
                    </div>
                    <div className="mt-4 flex items-center justify-between text-xs text-muted">
                      <span className="tabular">{s.pid}</span>
                      <Icon name="arrow" className="h-4 w-4 transition-transform duration-150 group-hover:translate-x-0.5" />
                    </div>
                  </Link>
                ))}
          </div>
        )}
        <p className="mt-2 text-xs text-muted">
          One test patient per group, picked by a fixed rule (the median segmentation score in its
          group), not chosen for looking good.
        </p>
      </section>

      <Card title="Analyse your own scan">
        {dataSource.canUpload ? (
          <UploadForm />
        ) : (
          <p className="text-sm text-muted">
            This public demo shows precomputed results only. To upload your own scan, run the app
            locally:{" "}
            <a
              className="font-medium text-primary hover:underline"
              href="https://github.com/praneethnarayanan2002-lab/cardiac-mri-analyzer#quick-start"
            >
              see the README
            </a>
            .
          </p>
        )}
      </Card>
    </div>
  );
}

/** The only decoration: a heart outline with a slow beat (still when reduced motion is on). */
function HeartOutline() {
  return (
    <svg viewBox="0 0 200 170" className="mx-auto hidden w-56 text-primary md:block" aria-hidden="true">
      <g className="beat" fill="none" stroke="currentColor" strokeLinejoin="round">
        <path
          d="M100 160C52 128 18 98 18 60a40 40 0 0 1 82-22 40 40 0 0 1 82 22c0 38-34 68-82 100z"
          strokeWidth="3"
        />
        <path d="M44 86h34l10-22 14 42 10-20h44" strokeWidth="3" strokeLinecap="round" opacity="0.7" />
      </g>
    </svg>
  );
}
