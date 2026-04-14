/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          // Brand blue: #01468B
          50: "#e6eef7",
          100: "#ccdeef",
          200: "#99bede",
          300: "#669dce",
          400: "#337dbd",
          500: "#01468B",
          600: "#01468B",
          700: "#013E7C",
          800: "#01356A",
          900: "#012C59"
        }
      }
    }
  },
  plugins: []
};
