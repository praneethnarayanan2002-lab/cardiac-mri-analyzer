import { useEffect, useState } from "react";
import { createHashRouter, NavLink, Outlet, RouterProvider, useLocation } from "react-router-dom";

import { Icon } from "../components/ui/controls";
import { dataSource } from "../data/dataSource";
import { ComparisonPage } from "../pages/ComparisonPage";
import { HomePage } from "../pages/HomePage";
import { HowItWorksPage } from "../pages/HowItWorksPage";
import { WorkspacePage } from "../pages/WorkspacePage";

// Hash routing, so deep links work on GitHub Pages, which has no server-side routing.
const router = createHashRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <HomePage /> },
      { path: "/case/:id", element: <WorkspacePage /> },
      { path: "/compare", element: <ComparisonPage /> },
      { path: "/how", element: <HowItWorksPage /> },
    ],
  },
]);

export function App() {
  return <RouterProvider router={router} />;
}

type Theme = "light" | "dark";

function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = localStorage.getItem("theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      // storage blocked: fall through to the system preference
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // not essential
    }
  }, [theme]);
  return [theme, setTheme];
}

function Layout() {
  const [theme, setTheme] = useTheme();
  const location = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);
  const link = ({ isActive }: { isActive: boolean }) =>
    `whitespace-nowrap rounded-md px-2 py-1.5 text-sm font-medium transition-colors duration-150 sm:px-2.5 ${isActive ? "bg-primary-soft text-primary" : "text-muted hover:text-text"}`;
  return (
    <div className="flex min-h-screen flex-col">
      {!dataSource.canUpload && (
        <div className="bg-primary px-4 py-1.5 text-center text-xs text-white">
          Precomputed results for 5 test patients. Research demo, not for clinical use.
        </div>
      )}
      <header className="sticky top-0 z-10 border-b border-border bg-bg/90 backdrop-blur">
        <nav className="mx-auto flex max-w-7xl items-center gap-1 px-4 py-2.5 sm:px-6" aria-label="Main">
          <NavLink to="/" className="mr-2 flex shrink-0 items-center gap-2 text-sm font-semibold text-text">
            <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="h-6 w-6 shrink-0" />
            <span className="hidden sm:inline">Cardiac MRI Analyzer</span>
          </NavLink>
          <NavLink to="/" end className={link}>
            Home
          </NavLink>
          <NavLink to="/compare" className={link}>
            Model comparison
          </NavLink>
          <NavLink to="/how" className={link}>
            How it works
          </NavLink>
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="ml-auto rounded-md p-2 text-muted hover:text-text"
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            <Icon name={theme === "dark" ? "sun" : "moon"} />
          </button>
        </nav>
      </header>
      <main key={location.pathname} className="rise flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-border px-4 py-5 text-center text-xs text-muted">
        Research demo, not a medical device; not for clinical use. Images: ACDC dataset (Bernard et
        al., IEEE TMI 2018), CC BY-NC-SA 4.0.
      </footer>
    </div>
  );
}
