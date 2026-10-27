// Likelihood bands from the A1 design. The thresholds come from GET /api/config;
// these are the defaults used until it loads (or if it fails).
export const DEFAULT_CONFIG = {
  max_words: 1000,
  max_chars: 8000,
  min_sentence_words: 5,
  bands: [
    { name: 'Human', upper: 0.25 },
    { name: 'Possibly AI', upper: 0.5 },
    { name: 'Likely AI', upper: 0.75 },
    { name: 'Very likely AI', upper: 1.0 },
  ],
};

// CSS variable names from styles/tokens.css, one pair per band.
export const BAND_STYLE = {
  'Human': { color: 'var(--band-human)', soft: 'var(--band-human-soft)', key: 'human' },
  'Possibly AI': { color: 'var(--band-possibly)', soft: 'var(--band-possibly-soft)', key: 'possibly' },
  'Likely AI': { color: 'var(--band-likely)', soft: 'var(--band-likely-soft)', key: 'likely' },
  'Very likely AI': { color: 'var(--band-very)', soft: 'var(--band-very-soft)', key: 'very' },
};

export const percent = (p) => `${Math.round(p * 100)}%`;
