import type { ButtonHTMLAttributes, ReactNode } from "react";

/** Small hand-built controls on the design tokens (docs/decisions.md D39). */

export function Button({
  variant = "secondary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" }) {
  const styles = {
    primary: "bg-primary text-white hover:opacity-90",
    secondary: "border border-border bg-surface text-text hover:bg-surface-2",
    ghost: "text-muted hover:bg-surface-2 hover:text-text",
  }[variant];
  return (
    <button
      type="button"
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${className}`}
      {...props}
    />
  );
}

export function Card({
  title,
  children,
  className = "",
  delay = 0,
  action,
}: {
  title?: string;
  children: ReactNode;
  className?: string;
  delay?: number;
  action?: ReactNode;
}) {
  return (
    <section
      className={`rise rounded-[var(--radius-card)] border border-border bg-surface p-4 shadow-[var(--shadow-card)] ${className}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {(title || action) && (
        <header className="mb-3 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold tracking-wide text-text">{title}</h2>}
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

/** A segmented choice (ED/ES, model picker). Dark variant for use inside the viewer bar. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  dark = false,
}: {
  options: { value: T; label: string; disabled?: boolean; title?: string }[];
  value: T;
  onChange: (value: T) => void;
  label: string;
  dark?: boolean;
}) {
  const base = dark ? "border-viewer-border bg-viewer-bg" : "border-border bg-surface-2";
  return (
    <div role="radiogroup" aria-label={label} className={`inline-flex rounded-lg border p-0.5 ${base}`}>
      {options.map((o) => {
        const active = o.value === value;
        const tone = active
          ? "bg-primary text-white"
          : dark
            ? "text-viewer-muted hover:text-viewer-text"
            : "text-muted hover:text-text";
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={o.disabled}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors duration-150 disabled:opacity-40 ${tone}`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  dark = false,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  dark?: boolean;
}) {
  return (
    <label
      className={`inline-flex cursor-pointer items-center gap-2 text-xs font-medium ${dark ? "text-viewer-text" : "text-text"}`}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 rounded-full transition-colors duration-150 ${checked ? "bg-primary" : dark ? "bg-viewer-border" : "bg-border"}`}
      >
        <span
          className={`absolute left-0 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-150 ${checked ? "translate-x-4" : "translate-x-0.5"}`}
        />
      </button>
      {label}
    </label>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-[var(--radius-card)] border border-border bg-surface p-6 text-sm">
      <p className="font-medium text-text">Something went wrong.</p>
      <p className="mt-1 text-muted">{message}</p>
      {onRetry && (
        <Button className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

/** Icons drawn inline so the app ships no icon font and makes no external requests. */
export function Icon({ name, className = "h-4 w-4" }: { name: IconName; className?: string }) {
  const paths: Record<IconName, ReactNode> = {
    play: <path d="M7 5v14l11-7z" fill="currentColor" stroke="none" />,
    pause: (
      <>
        <rect x="6" y="5" width="4" height="14" fill="currentColor" stroke="none" />
        <rect x="14" y="5" width="4" height="14" fill="currentColor" stroke="none" />
      </>
    ),
    check: <path d="M5 12l5 5L20 7" />,
    alert: (
      <>
        <path d="M12 3l10 18H2z" />
        <path d="M12 10v5M12 18h.01" />
      </>
    ),
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6M12 7.5h.01" />
      </>
    ),
    download: <path d="M12 4v11m0 0l-4-4m4 4l4-4M5 20h14" />,
    upload: <path d="M12 20V9m0 0l-4 4m4-4l4 4M5 4h14" />,
    arrow: <path d="M5 12h14m-5-5l5 5-5 5" />,
    moon: <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />,
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4l1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4l1.4-1.4" />
      </>
    ),
    columns: <path d="M4 5h7v14H4zM13 5h7v14h-7z" />,
  };
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

export type IconName =
  | "play"
  | "pause"
  | "check"
  | "alert"
  | "info"
  | "download"
  | "upload"
  | "arrow"
  | "moon"
  | "sun"
  | "columns";
