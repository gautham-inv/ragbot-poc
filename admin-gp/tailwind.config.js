/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        "ink-50":  "#f2f6fc",
        "ink-100": "#e3ecf7",
        "ink-300": "#7da3cf",
        "ink-600": "#0a4a94",
        "ink-700": "#003777",
        "ink-900": "#00224a",
        "aqua-100": "#d7f0f5",
        "aqua-300": "#7ad3e3",
        "aqua-500": "#37BAD1",
        "canvas":      "#f7f8fa",
        "card":        "#ffffff",
        "rule":        "#e6e8ec",
        "rule-strong": "#d4d7dd",
        "ink-text":    "#0f1420",
        "ink-2":       "#4a5161",
        "ink-3":       "#838a97",
        "ink-4":       "#b1b6c0",
        "good":        "#0b7a4a",
        "good-bg":     "#e6f5ee",
        "bad":         "#b42323",
        "bad-bg":      "#fde7e7",
        "warn":        "#7a4d0b",
        "warn-bg":     "#fbf1df",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["var(--font-jbmono)", "monospace"],
      },
      boxShadow: {
        "soft-sm": "0 1px 2px rgba(15, 20, 32, 0.04)",
        "soft":    "0 4px 14px rgba(15, 20, 32, 0.06), 0 1px 2px rgba(15,20,32,0.04)",
        "soft-lg": "0 16px 40px rgba(15, 20, 32, 0.14), 0 2px 8px rgba(15,20,32,0.06)",
      },
      transitionTimingFunction: {
        ease: "cubic-bezier(.2,.7,.2,1)",
      },
    },
  },
  plugins: [],
};
