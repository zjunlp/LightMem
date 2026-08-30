import * as React from "react";
import { Check, Copy, Loader2 } from "lucide-react";
import { useT } from "@/lib/i18n";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";

const buttonVariants: Record<ButtonVariant, string> = {
  primary:
    "bg-ink text-page shadow-[0_5px_14px_-6px_rgba(11,11,11,0.5)] hover:-translate-y-px hover:bg-ink/90 disabled:bg-ink4",
  secondary:
    "border border-rule bg-raised text-ink shadow-sm hover:-translate-y-px hover:border-accent/40 hover:bg-accent-wash",
  ghost: "text-ink2 hover:bg-sunken hover:text-ink",
  danger: "border border-bad/30 text-badInk hover:bg-bad/5",
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: "h-7 px-2.5 text-tiny gap-1.5",
  md: "h-9 px-3.5 text-sm gap-2",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export function Button({
  className,
  variant = "secondary",
  size = "md",
  loading = false,
  disabled,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-lg font-medium transition-[color,background-color,border-color,transform,box-shadow]",
        "disabled:cursor-not-allowed disabled:opacity-55",
        buttonVariants[variant],
        buttonSizes[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
}

export function Section({
  title,
  description,
  actions,
  children,
  className,
  divided = true,
}: {
  title?: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  divided?: boolean;
}) {
  return (
    <section className={cn(divided && "border-t border-rule pt-8", className)}>
      {(title || actions) && (
        <header className="mb-5 flex items-start justify-between gap-6">
          <div className="min-w-0">
            {title && (
              <h2 className="text-h3 font-semibold text-ink">{title}</h2>
            )}
            {description && (
              <p className="mt-1 max-w-prose text-sm text-ink2">
                {description}
              </p>
            )}
          </div>
          {actions && (
            <div className="flex shrink-0 items-center gap-2">{actions}</div>
          )}
        </header>
      )}
      {children}
    </section>
  );
}

export function Surface({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("surface", className)} {...props} />;
}

export function SurfaceHead({
  title,
  meta,
  actions,
  className,
}: {
  title?: React.ReactNode;
  meta?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-4 border-b border-ruleSoft px-4 py-2.5",
        className,
      )}
    >
      <div className="flex min-w-0 items-baseline gap-3">
        {title && (
          <span className="truncate text-sm font-medium text-ink">{title}</span>
        )}
        {meta && <span className="truncate text-tiny text-ink3">{meta}</span>}
      </div>
      {actions && (
        <div className="flex shrink-0 items-center gap-1.5">{actions}</div>
      )}
    </div>
  );
}

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input ref={ref} className={cn("field", className)} {...props} />
));
Input.displayName = "Input";

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn("field resize-y leading-relaxed", className)}
    {...props}
  />
));
Textarea.displayName = "Textarea";

export function Select({
  className,
  options,
  flat = false,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement> & {
  options: (string | { value: string; label: string })[];
  flat?: boolean;
}) {
  return (
    <select
      className={cn(
        flat
          ? "cursor-pointer appearance-none border-0 bg-transparent p-0 focus:outline-none focus:ring-0"
          : "field cursor-pointer appearance-none bg-[right_0.6rem_center] bg-no-repeat pr-8",
        className,
      )}
      style={
        flat
          ? undefined
          : {
              backgroundImage:
                "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='%23898781' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>\")",
            }
      }
      {...props}
    >
      {options.map((opt) => {
        const value = typeof opt === "string" ? opt : opt.value;
        const label = typeof opt === "string" ? opt : opt.label;
        return (
          <option key={value} value={value}>
            {label}
          </option>
        );
      })}
    </select>
  );
}

export function Suggest({
  id,
  value,
  onChange,
  options,
  placeholder,
  className,
}: {
  id?: string;
  value: string;
  onChange: (next: string) => void;
  options: string[];
  placeholder?: string;
  className?: string;
}) {
  const listId = `${id ?? "suggest"}-options`;
  return (
    <>
      <input
        id={id}
        list={options.length ? listId : undefined}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        autoComplete="off"
        className={cn("field", className)}
      />
      {options.length > 0 && (
        <datalist id={listId}>
          {options.map((o) => (
            <option key={o} value={o} />
          ))}
        </datalist>
      )}
    </>
  );
}

export function Switch({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-[1.15rem] w-8 shrink-0 items-center rounded-full transition-colors",
        checked ? "bg-ink" : "bg-rule",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      <span
        className={cn(
          "inline-block h-3.5 w-3.5 rounded-full bg-white shadow-sm transition-transform",
          checked ? "translate-x-[0.95rem]" : "translate-x-[0.15rem]",
        )}
      />
    </button>
  );
}

export function Field({
  label,
  hint,
  htmlFor,
  children,
  className,
  inline = false,
  invalid = false,
}: {
  label: React.ReactNode;
  hint?: React.ReactNode;
  htmlFor?: string;
  children: React.ReactNode;
  className?: string;
  inline?: boolean;
  invalid?: boolean;
}) {
  if (inline) {
    return (
      <div
        className={cn(
          "flex items-start justify-between gap-6 py-2.5",
          className,
        )}
      >
        <div className="min-w-0">
          <label htmlFor={htmlFor} className="text-base text-ink">
            {label}
          </label>
          {hint && (
            <p
              className={cn(
                "mt-0.5 max-w-prose text-tiny leading-relaxed",
                invalid ? "text-badInk" : "text-ink3",
              )}
            >
              {hint}
            </p>
          )}
        </div>
        <div className="shrink-0 pt-1">{children}</div>
      </div>
    );
  }
  return (
    <div className={cn("space-y-1.5", className)}>
      <label htmlFor={htmlFor} className="block text-base text-ink">
        {label}
      </label>
      {children}
      {hint && (
        <p
          className={cn(
            "max-w-prose text-tiny leading-relaxed",
            invalid ? "text-badInk" : "text-ink3",
          )}
        >
          {hint}
        </p>
      )}
    </div>
  );
}

export function Tag({
  children,
  className,
  tone = "neutral",
}: {
  children: React.ReactNode;
  className?: string;
  tone?: "neutral" | "accent" | "good" | "warn" | "bad";
}) {
  const tones = {
    neutral: "border-rule text-ink2",
    accent: "border-accent/30 text-accent",
    good: "border-good/30 text-goodInk",
    warn: "border-warn/40 text-warnInk",
    bad: "border-bad/30 text-badInk",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-px text-micro tracking-normal",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Dot({
  tone = "muted",
  pulse = false,
}: {
  tone?: "muted" | "accent" | "good" | "warn" | "bad";
  pulse?: boolean;
}) {
  const color = {
    muted: "bg-ink4",
    accent: "bg-accent",
    good: "bg-good",
    warn: "bg-warn",
    bad: "bg-bad",
  }[tone];
  return (
    <span
      className={cn(
        "inline-block h-1.5 w-1.5 rounded-full",
        color,
        pulse && "animate-breathe",
      )}
    />
  );
}

export function Note({
  tone = "info",
  title,
  children,
  className,
  icon,
}: {
  tone?: "info" | "good" | "warn" | "bad";
  title?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  icon?: React.ReactNode;
}) {
  const tones = {
    info: { rule: "border-l-accent", text: "text-accent" },
    good: { rule: "border-l-good", text: "text-goodInk" },
    warn: { rule: "border-l-warn", text: "text-warnInk" },
    bad: { rule: "border-l-bad", text: "text-badInk" },
  }[tone];

  return (
    <div className={cn("border-l-2 py-1 pl-4", tones.rule, className)}>
      <div className="flex gap-2">
        {icon && (
          <span className={cn("mt-[0.15rem] shrink-0", tones.text)}>
            {icon}
          </span>
        )}
        <div className="min-w-0">
          {title && (
            <p className={cn("text-base font-medium", tones.text)}>{title}</p>
          )}
          {children && (
            <div className="mt-0.5 max-w-readable text-sm leading-relaxed text-ink2">
              {children}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <Loader2 className={cn("h-4 w-4 animate-spin text-ink3", className)} />
  );
}

export function Empty({
  title,
  children,
  className,
}: {
  title: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("py-12 text-center", className)}>
      <p className="text-base text-ink2">{title}</p>
      {children && (
        <p className="mx-auto mt-1 max-w-prose text-sm leading-relaxed text-ink3">
          {children}
        </p>
      )}
    </div>
  );
}

export function CopyButton({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const [copied, setCopied] = React.useState(false);
  const { t } = useT();
  return (
    <Button
      variant="ghost"
      size="sm"
      className={className}
      onClick={() => {
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1400);
        });
      }}
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-goodInk" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
      {copied ? t("common.copied") : t("common.copy")}
    </Button>
  );
}

export function CodeBlock({
  code,
  className,
  maxHeight = "24rem",
  wrap = false,
}: {
  code: string;
  className?: string;
  maxHeight?: string;
  wrap?: boolean;
}) {
  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-lg bg-sunken",
        className,
      )}
    >
      <CopyButton
        text={code}
        className="absolute right-1.5 top-1.5 z-10 bg-sunken/90 opacity-0 transition-opacity group-hover:opacity-100"
      />
      <pre
        className={cn(
          "overflow-auto p-4 font-mono text-tiny leading-relaxed text-ink2",
          wrap && "whitespace-pre-wrap break-words",
        )}
        style={{ maxHeight }}
      >
        {code}
      </pre>
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  items,
  className,
}: {
  value: T;
  onChange: (next: T) => void;
  items: { value: T; label: React.ReactNode; count?: number }[];
  className?: string;
}) {
  return (
    <div className={cn("inline-flex items-center gap-4", className)}>
      {items.map((item) => (
        <button
          key={item.value}
          onClick={() => onChange(item.value)}
          className={cn(
            "-mb-px border-b-2 pb-2 text-sm transition-colors",
            value === item.value
              ? "border-ink font-medium text-ink"
              : "border-transparent text-ink3 hover:text-ink2",
          )}
        >
          {item.label}
          {item.count !== undefined && (
            <span className="ml-1.5 text-ink4">{item.count}</span>
          )}
        </button>
      ))}
    </div>
  );
}
