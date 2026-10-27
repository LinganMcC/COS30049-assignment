import { useId, useState } from 'react';

const EXAMPLES = {
  ai: 'Renewable energy has become a central topic in discussions about the future of our planet. '
    + 'Solar and wind power offer clean, sustainable alternatives to fossil fuels, reducing greenhouse '
    + 'gas emissions and improving air quality. Furthermore, advances in battery storage technology '
    + 'have made it increasingly feasible to rely on these sources even when the sun is not shining.',
  human: 'Honestly I wasnt sure what to write for this one. My brother drives an old ute that drinks '
    + 'petrol like nothing else, and he reckons electric cars are a fad. I think hes wrong but I get '
    + 'why people are worried about the cost. Charging stations near us are pretty rare too.',
};

const countWords = (text) => (text.trim() ? text.trim().split(/\s+/).length : 0);

// Usability checks only - the back end enforces the same limits for security.
export function validate(text, config) {
  if (!text.trim()) return 'Enter some text to analyse.';
  const words = countWords(text);
  if (words > config.max_words) {
    return `Your text has ${words.toLocaleString()} words. The limit is ${config.max_words.toLocaleString()}.`;
  }
  if (text.length > config.max_chars) {
    return `Your text has ${text.length.toLocaleString()} characters. The limit is ${config.max_chars.toLocaleString()}.`;
  }
  if (words < config.min_sentence_words) {
    return `Enter at least one sentence of ${config.min_sentence_words} or more words.`;
  }
  return null;
}

export default function TextInputForm({ text, onTextChange, onSubmit, onClear, config, loading }) {
  const [error, setError] = useState(null);
  const [touched, setTouched] = useState(false);
  const fieldId = useId();
  const words = countWords(text);
  const overLimit = words > config.max_words || text.length > config.max_chars;

  const handleChange = (event) => {
    onTextChange(event.target.value);
    // Once the user has seen an error, re-check as they type so it clears promptly.
    if (touched) setError(validate(event.target.value, config));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setTouched(true);
    const problem = validate(text, config);
    setError(problem);
    if (!problem) onSubmit(text);
  };

  const handleClear = () => {
    onClear();
    setError(null);
    setTouched(false);
  };

  return (
    <form className="card input-card" onSubmit={handleSubmit} noValidate>
      <div className="field-header">
        <label htmlFor={fieldId} className="field-label">Text to check</label>
        <span
          id={`${fieldId}-count`}
          className={`word-count${overLimit ? ' over' : ''}`}
          aria-live="polite"
        >
          {words.toLocaleString()} / {config.max_words.toLocaleString()} words
        </span>
      </div>
      <p id={`${fieldId}-hint`} className="field-hint">
        Paste an essay, review or message in English. Each sentence of {config.min_sentence_words}+
        words is scored. Your text is not stored.
      </p>
      <textarea
        id={fieldId}
        className="text-input"
        value={text}
        onChange={handleChange}
        onBlur={() => touched && setError(validate(text, config))}
        rows={10}
        aria-invalid={Boolean(error)}
        aria-describedby={`${fieldId}-hint ${fieldId}-count${error ? ` ${fieldId}-error` : ''}`}
        disabled={loading}
        spellCheck="false"
      />
      {error && (
        <p id={`${fieldId}-error`} className="field-error" role="alert">{error}</p>
      )}
      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Analysing…' : 'Analyse text'}
        </button>
        <button type="button" className="btn btn-secondary" onClick={handleClear} disabled={loading}>
          Clear
        </button>
        <span className="examples">
          <span className="examples-label">Try an example:</span>
          <button type="button" className="btn-link" onClick={() => onTextChange(EXAMPLES.ai)} disabled={loading}>
            AI-style
          </button>
          <button type="button" className="btn-link" onClick={() => onTextChange(EXAMPLES.human)} disabled={loading}>
            Human-style
          </button>
        </span>
      </div>
    </form>
  );
}
