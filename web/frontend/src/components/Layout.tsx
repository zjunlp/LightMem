import type { ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Search, PenLine, Settings2 } from "lucide-react";
import { api } from "@/lib/api";
import { setLang, useT } from "@/lib/i18n";
import { cn, formatNumber } from "@/lib/utils";
import { Dot } from "@/components/ui";

const lightmemMark = new URL("../assets/lightmem-mark.png", import.meta.url)
  .href;

const NAV = [
  { to: "/", labelKey: "nav.ask", end: true, icon: Search },
  { to: "/ingest", labelKey: "nav.ingest", end: false, icon: PenLine },
  { to: "/settings", labelKey: "nav.settings", end: false, icon: Settings2 },
];

function LanguageSwitch() {
  const { t, lang } = useT();
  return (
    <div
      className="flex items-center gap-1"
      role="group"
      aria-label={t("lang.toggle")}
    >
      {(["zh", "en"] as const).map((code) => (
        <button
          key={code}
          onClick={() => setLang(code)}
          className={cn(
            "rounded px-1.5 py-0.5 text-micro transition-colors",
            lang === code
              ? "bg-sunken font-medium text-ink"
              : "text-ink4 hover:text-ink2",
          )}
          aria-pressed={lang === code}
        >
          {code === "zh" ? "中文" : "EN"}
        </button>
      ))}
    </div>
  );
}

export default function Layout() {
  const location = useLocation();
  const { t } = useT();

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 5000,
  });
  const instance = useQuery({
    queryKey: ["instance"],
    queryFn: api.instance,
    refetchInterval: 8000,
  });

  const ready = instance.data?.ready ?? false;
  const busy = health.data?.busy ?? false;

  return (
    <div className="min-h-full w-full lg:flex">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-[17rem] flex-col border-r border-rule bg-sunken px-4 py-5 text-ink md:flex">
        <div>
          <div className="flex items-center justify-between px-2">
            <span className="brand-mark">
              <img
                src={lightmemMark}
                alt=""
                aria-hidden="true"
                className="h-5 w-6 shrink-0 object-contain"
              />
              LightMem
            </span>
          </div>
        </div>

        <nav className="mt-12 space-y-1" aria-label="Primary">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  "rail-link group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                  isActive
                    ? "bg-white text-ink shadow-sm"
                    : "text-ink2 hover:bg-white/70 hover:text-ink",
                )
              }
            >
              <item.icon className="h-4 w-4 text-ink4" strokeWidth={2} />
              <span className="flex-1">{t(item.labelKey)}</span>
              {item.to === "/settings" && !ready && <Dot tone="warn" />}
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto space-y-5">
          <div className="rounded-xl border border-rule bg-white p-3.5 shadow-sm">
            <div className="flex items-center gap-2 text-tiny text-ink2">
              <Dot tone={ready ? "good" : "muted"} pulse={busy} />
              <span>
                {ready
                  ? t("shell.entries", {
                      n: formatNumber(instance.data?.points ?? 0),
                    })
                  : t("shell.notReady")}
              </span>
            </div>
          </div>
          <div className="flex items-center justify-between border-t border-rule pt-4">
            <LanguageSwitch />
          </div>
        </div>
      </aside>

      <div className="flex min-h-full flex-1 flex-col md:ml-[17rem]">
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-rule px-6 py-5 sm:px-10 md:hidden">
          <span className="brand-mark">
            <img
              src={lightmemMark}
              alt=""
              aria-hidden="true"
              className="h-5 w-6 shrink-0 object-contain"
            />
            LightMem
          </span>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2 text-tiny text-ink3">
              <Dot tone={ready ? "good" : "muted"} pulse={busy} />
              {ready
                ? t("shell.entries", {
                    n: formatNumber(instance.data?.points ?? 0),
                  })
                : t("shell.notReady")}
            </span>
            <LanguageSwitch />
          </div>
          <nav
            className="order-3 flex w-full items-center gap-5 overflow-x-auto pt-1"
            aria-label="Primary"
          >
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cn(
                    "whitespace-nowrap text-sm transition-colors",
                    isActive ? "font-medium text-ink" : "text-ink3",
                  )
                }
              >
                {t(item.labelKey)}
              </NavLink>
            ))}
          </nav>
        </header>

        <main
          key={location.pathname}
          className="mx-auto w-full max-w-[76rem] flex-1 animate-fade-up px-6 py-12 sm:px-10 sm:py-16 xl:px-14"
        >
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-12 flex items-start justify-between gap-10">
      <div className="min-w-0">
        <h1 className="text-h1 font-semibold tracking-[-0.03em] text-ink sm:text-[2rem]">
          {title}
        </h1>
        {description && (
          <p className="mt-2 max-w-readable text-lead text-ink2">
            {description}
          </p>
        )}
      </div>
      {actions && (
        <div className="flex shrink-0 items-center gap-2 pt-1">{actions}</div>
      )}
    </header>
  );
}

export function Step({
  n,
  title,
  hint,
  children,
}: {
  n: number;
  title: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="step-section border-t border-rule pt-8">
      <div className="mb-5 flex items-baseline gap-3">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-rule text-micro text-ink3">
          {n}
        </span>
        <div>
          <h2 className="text-h3 font-semibold text-ink">{title}</h2>
          {hint ? (
            <p className="mt-1 max-w-prose text-sm text-ink2">{hint}</p>
          ) : null}
        </div>
      </div>
      <div className="pl-8">{children}</div>
    </section>
  );
}
