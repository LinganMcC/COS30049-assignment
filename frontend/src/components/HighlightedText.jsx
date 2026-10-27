import { Fragment, useEffect, useRef } from 'react';
import { BAND_STYLE, percent } from '../bands';

// The submitted text, one coloured span per sentence.
// Security: every sentence is rendered as a React text child, which React
// escapes, so "<script>" or "<img onerror=…>" in the input shows as literal
// text. We never use dangerouslySetInnerHTML to build the highlighting.
export default function HighlightedText({ sentences, selectedIndex, onSelect }) {
  const refs = useRef(new Map());

  // When a sentence is picked from the chart, bring it into view here too.
  useEffect(() => {
    if (selectedIndex == null) return;
    refs.current.get(selectedIndex)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [selectedIndex]);

  return (
    <section className="card" aria-labelledby="highlight-heading">
      <div className="card-header">
        <h2 id="highlight-heading">Sentence by sentence</h2>
        <BandLegend />
      </div>
      <p className="card-hint">Select a sentence to see its score in the chart.</p>
      <div className="highlighted-text">
        {sentences.map((s) => {
          if (!s.scored) {
            return (
              <Fragment key={s.index}>
                <span className="sentence unscored" title="Under 5 words – not scored">{s.text}</span>{' '}
              </Fragment>
            );
          }
          const style = BAND_STYLE[s.band];
          const selected = s.index === selectedIndex;
          return (
            <Fragment key={s.index}>
              <button
                type="button"
                ref={(el) => (el ? refs.current.set(s.index, el) : refs.current.delete(s.index))}
                className={`sentence${selected ? ' selected' : ''}`}
                style={{ '--band': style.color, '--band-soft': style.soft }}
                aria-pressed={selected}
                aria-label={`Sentence ${s.index + 1}: ${percent(s.score)}, ${s.band}. ${s.text}`}
                onClick={() => onSelect(selected ? null : s.index)}
              >
                {s.text}
              </button>{' '}
            </Fragment>
          );
        })}
      </div>
    </section>
  );
}

export function BandLegend() {
  return (
    <ul className="legend" aria-label="Colour key">
      {Object.entries(BAND_STYLE).map(([name, style]) => (
        <li key={name}>
          <span className="swatch" style={{ background: style.soft, borderColor: style.color }} aria-hidden="true" />
          {name}
        </li>
      ))}
      <li><span className="swatch swatch-unscored" aria-hidden="true" />Not scored</li>
    </ul>
  );
}
