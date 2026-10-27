import { BAND_STYLE, percent } from '../bands';

// The headline result: the most prominent element on the page.
export default function ResultSummary({ result }) {
  const { document: doc, sentences, excluded_sentences: excluded, characters_normalised: normalised } = result;
  const style = BAND_STYLE[doc.band];
  const flagged = sentences.filter((s) => s.scored && s.score > 0.5).length;

  return (
    <section className="card result-card" aria-labelledby="result-heading" style={{ '--band': style.color, '--band-soft': style.soft }}>
      <h2 id="result-heading" className="visually-hidden">Result</h2>
      <div className="result-main">
        <p className="result-score" aria-describedby="score-explainer">{percent(doc.score)}</p>
        <div>
          <p className="result-band">{doc.band}</p>
          <p id="score-explainer" className="result-explainer">
            estimated likelihood this text is AI-generated
          </p>
        </div>
      </div>

      <div className="range" aria-label={`Confidence range ${percent(doc.low)} to ${percent(doc.high)}`}>
        <div className="range-track" aria-hidden="true">
          <div className="range-fill" style={{ left: percent(doc.low), width: percent(doc.high - doc.low) }} />
          <div className="range-marker" style={{ left: percent(doc.score) }} />
        </div>
        <div className="range-labels" aria-hidden="true">
          <span>0%</span><span>Human</span><span>AI</span><span>100%</span>
        </div>
        <p className="range-text">
          Confidence range <strong>{percent(doc.low)} – {percent(doc.high)}</strong>. A wider range
          means fewer sentences or sentences that disagree.
        </p>
      </div>

      <dl className="result-stats">
        <div><dt>Sentences scored</dt><dd>{doc.scored_sentences}</dd></div>
        <div><dt>Likely AI or above</dt><dd>{flagged}</dd></div>
        <div><dt>Too short to score</dt><dd>{excluded}</dd></div>
      </dl>

      {normalised > 0 && (
        <p className="notice">
          We replaced {normalised} look-alike or invisible character{normalised === 1 ? '' : 's'} before
          scoring, so the text was judged as a reader would see it.
        </p>
      )}
      <p className="disclaimer">
        This is a statistical estimate, not proof of authorship. Use it to guide a closer read.
      </p>
    </section>
  );
}
