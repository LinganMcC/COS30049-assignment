import { createTheme } from '@mui/material/styles'

// One palette for the whole app. The two poles passed the colour-blindness
// checks (dataviz validator: protan ΔE 13.5, all contrast >= 3:1 on white).
export const ink = {
  paper: '#EEF1F4', // page background: cool ledger grey
  sheet: '#FFFFFF', // the sheet the passage sits on
  text: '#18243A', // navy ink
  muted: '#4A5568',
  rule: '#D5DBE3',
  human: '#2B66C2',
  ai: '#C2386F',
  neutral: '#8A94A6',
  error: '#B42318',
}

// Highlighter tints for the four bands, weakest to strongest on each side.
export const bandStyle = {
  Human: { tint: '#D9E4F6', pole: ink.human, side: 'human' },
  'Possibly AI': { tint: '#EDF2FA', pole: ink.human, side: 'human' },
  'Likely AI': { tint: '#F8E6ED', pole: ink.ai, side: 'ai' },
  'Very likely AI': { tint: '#EFC9D8', pole: ink.ai, side: 'ai' },
}

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: ink.text, contrastText: '#FFFFFF' },
    secondary: { main: ink.human },
    error: { main: ink.error },
    background: { default: ink.paper, paper: ink.sheet },
    text: { primary: ink.text, secondary: ink.muted },
    divider: ink.rule,
  },
  typography: {
    fontFamily: '"Instrument Sans", system-ui, sans-serif',
    fontSize: 15,
    h1: { fontFamily: '"Literata Variable", Georgia, serif', fontWeight: 600, fontSize: '2.35rem', lineHeight: 1.15, letterSpacing: '-0.01em' },
    h2: { fontFamily: '"Literata Variable", Georgia, serif', fontWeight: 600, fontSize: '1.5rem', lineHeight: 1.25 },
    h3: { fontFamily: '"Literata Variable", Georgia, serif', fontWeight: 600, fontSize: '1.15rem', lineHeight: 1.3 },
    body1: { lineHeight: 1.6 },
    body2: { lineHeight: 1.55 },
    button: { textTransform: 'none', fontWeight: 600, letterSpacing: 0 },
  },
  shape: { borderRadius: 6 },
  components: {
    MuiButtonBase: { defaultProps: { disableRipple: false } },
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: { root: { paddingInline: 18, minHeight: 42 } },
    },
    MuiPaper: { defaultProps: { elevation: 0 } },
    MuiLink: { defaultProps: { underline: 'always' } },
  },
})
