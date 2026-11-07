// Input rules and formatting shared by the components.
// The back end enforces the same limits; these checks are for usability only.

export const LIMITS = { minWords: 20, maxWords: 1000, maxChars: 20000 }

/**
 * Words for the input limits: whitespace-separated pieces containing a letter or digit,
 * so "résumé", "work-life", "don't" and "2024" each count once.
 * Same rule as count_input_words() in backend/app/schemas.py.
 */
export function countWords(text) {
  return text.split(/\s+/).filter((token) => /[\p{L}\p{N}]/u.test(token)).length
}

/** Returns a message to show beside the field, or null when the text can be sent. */
export function validateText(text) {
  if (!text.trim()) return 'Paste or type the text you want to check.'
  const words = countWords(text)
  if (words < LIMITS.minWords) {
    return `Add at least ${LIMITS.minWords - words} more word${LIMITS.minWords - words === 1 ? '' : 's'}. The check needs ${LIMITS.minWords} or more to be meaningful.`
  }
  if (words > LIMITS.maxWords) {
    return `This is ${words.toLocaleString()} words. Shorten it to ${LIMITS.maxWords.toLocaleString()} or fewer.`
  }
  if (text.length > LIMITS.maxChars) {
    return `This is ${text.length.toLocaleString()} characters. The limit is ${LIMITS.maxChars.toLocaleString()}.`
  }
  return null
}

/** A probability as a percentage. Never shows 0% or 100%: the model is never certain. */
export const pct = (p) => (p >= 0.995 ? '>99%' : p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`)

/** Axis labels, where 0% and 100% are exact positions rather than results. */
export const tickPct = (p) => `${Math.round(p * 100)}%`

/** Human-readable value for one feature. */
export function formatFeature(feature, value) {
  switch (feature) {
    case 'n_words':
      return `${Math.round(value)} words`
    case 'exclaim':
      return value >= 0.5 ? 'yes' : 'no'
    case 'starts_with_transition':
      return value >= 0.5 ? 'yes' : 'no'
    case 'rel_len_deviation':
      return `${Math.round(value * 100)}% from the average`
    case 'comma_rate':
    case 'apostrophe_rate':
      return `${value.toFixed(2)} per word`
    default:
      return `${Math.round(value * 100)}% of words`
  }
}

export const VERDICT_TEXT = {
  Human: 'Reads as human-written',
  'Possibly AI': 'Mostly reads as human-written',
  'Likely AI': 'Likely AI-generated',
  'Very likely AI': 'Very likely AI-generated',
}

/** CSS class suffix for a band, e.g. 'Very likely AI' -> 'very-likely-ai'. */
export const slug = (band) => band.toLowerCase().replace(/\s+/g, '-')
