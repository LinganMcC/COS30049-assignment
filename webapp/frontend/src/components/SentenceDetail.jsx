import { formatFeature, pct } from '../lib/text.js'

/** Why one sentence scored as it did: the model's exact per-feature pushes. */
export default function SentenceDetail({ sentence, total }) {
  if (!sentence) {
    return (
      <aside className="detail detail--empty" aria-live="polite">
        <p>Select a sentence in the passage or the chart to see what pushed its score.</p>
      </aside>
    )
  }
  const max = Math.max(
    0.01,
    ...sentence.toward_ai.map((f) => Math.abs(f.contribution)),
    ...sentence.toward_human.map((f) => Math.abs(f.contribution)),
  )
  return (
    <aside className="detail" aria-live="polite" aria-labelledby="detail-title">
      <h3 id="detail-title">
        Sentence {sentence.index + 1} of {total}: {pct(sentence.score)}, {sentence.band}
      </h3>
      <blockquote className="detail__quote">{sentence.text}</blockquote>
      <div className="detail__cols">
        <FactorList title="Pushed toward AI" side="ai" factors={sentence.toward_ai} max={max} />
        <FactorList title="Pushed toward human" side="human" factors={sentence.toward_human} max={max} />
      </div>
      <p className="detail__note">
        Bars compare how strongly each feature pushed the model's internal sentence score (its log-odds,
        before it becomes a percentage), relative to a typical training sentence. They are not percentage
        points. The model learned from essays labelled as a whole, so a sentence score is an estimate,
        not a verified label for that sentence.
      </p>
    </aside>
  )
}

function FactorList({ title, side, factors, max }) {
  return (
    <div className="factors">
      <h4>{title}</h4>
      {factors.length === 0 ? (
        <p className="factors__none">Nothing notable.</p>
      ) : (
        <ul>
          {factors.map((f) => (
            <li key={f.feature}>
              <div className="factors__row">
                <span>{f.label}</span>
                <span className="factors__value">{formatFeature(f.feature, f.value)}</span>
              </div>
              <span className="factors__bar" aria-hidden="true">
                <span className={`factors__fill factors__fill--${side}`} style={{ width: `${(Math.abs(f.contribution) / max) * 100}%` }} />
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
