/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        page: "#FFFFFF",
        raised: "#FFFFFF",
        sunken: "#F7F7F8",
        rule: "#E5E5E5",
        ruleSoft: "#EEEEEE",

        ink: "#0B0B0B",
        ink2: "#52514E",
        ink3: "#898781",
        ink4: "#B3B1A9",

        accent: {
          DEFAULT: "#2A78D6",
          soft: "#86B6EF",
          wash: "#EEF4FD",
          deep: "#1C5CAB",
        },

        stage: {
          1: "#86B6EF",
          2: "#5598E7",
          3: "#2A78D6",
          4: "#1C5CAB",
          5: "#104281",
        },

        good: "#0CA30C",
        goodInk: "#006300",
        warn: "#FAB219",
        warnInk: "#8A5A00",
        bad: "#D03B3B",
        badInk: "#A32020",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      fontSize: {
        micro: ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.06em" }],
        tiny: ["0.75rem", { lineHeight: "1.1rem" }],
        sm: ["0.8125rem", { lineHeight: "1.35rem" }],
        base: ["0.875rem", { lineHeight: "1.6rem" }],
        lead: ["0.9375rem", { lineHeight: "1.7rem" }],
        h3: ["1rem", { lineHeight: "1.5rem", letterSpacing: "-0.005em" }],
        h2: ["1.125rem", { lineHeight: "1.6rem", letterSpacing: "-0.011em" }],
        h1: ["1.75rem", { lineHeight: "2.1rem", letterSpacing: "-0.021em" }],
        figure: ["2rem", { lineHeight: "2.25rem", letterSpacing: "-0.022em" }],
        hero: ["3rem", { lineHeight: "3.25rem", letterSpacing: "-0.028em" }],
      },
      spacing: {
        18: "4.5rem",
        22: "5.5rem",
      },
      maxWidth: {
        prose: "38rem",
        readable: "44rem",
      },
      boxShadow: {
        raise: "0 1px 2px rgba(0,0,0,0.03), 0 4px 14px -8px rgba(0,0,0,0.08)",
        pop: "0 8px 28px -10px rgba(11,11,11,0.20)",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        breathe: {
          "0%,100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
      },
      animation: {
        "fade-up": "fade-up 240ms cubic-bezier(0.2,0.7,0.3,1) both",
        breathe: "breathe 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
