/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./node_modules/@mawtech/glass-ui/**/*.{js,mjs}"
  ],
  theme: {
    extend: {
      colors: {
        glass: {
          cyan: '#00F0FF',
          violet: '#8B5CF6',
          pink: '#EC4899',
          dark: '#09090B',
          card: '#0F172A',
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', '"Segou UI"', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
