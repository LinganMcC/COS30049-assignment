import Alert from '@mui/material/Alert'
import { VERDICT_TEXT, formatFeature, pct } from '../lib/text.js'
import { bandStyle, ink } from '../theme.js'

const UNCERTAIN_POLE = ink.neutral

/** The verdict: the most prominent thing on screen once a check finishes. */
export default function ResultSummary({ result, stale }) {
  const { label, band, score, range, threshold, certainty, sentences, word_count: words, normalisation } = result
  const style = bandStyle[band]
  const flagged = sentences.filter((s) => s.score >= threshold).length
  const tampered = normalisation.invisible_removed + normalisation.homoglyphs_replaced + normalisation.marks_removed
  const uncertain = certainty === 'uncertain'
  const isAi = label === 'ai'
  const leans = isAi ? 'AI-generated' : 'human-written'

  return (
    <section className={`verdict${stale ? ' verdict--stale' : ''}`} aria-labelledby="verdict-title" aria-live="polite">
      {stale && (
        <p className="verdict__stale">You've edited the text since this check. Check it again to update the result.</p>
      )}
      {/* When the range crosses the decision line, the heading says so, so the
          headline and the explanation never disagree. */}
      <h2 id="verdict-title" className="verdict__title" style={{ '--pole': uncertain ? UNCERTAIN_POLE : style.pole }}>
        {uncertain ? 'Uncertain result' : VERDICT_TEXT[band]}
      </h2>
      {uncertain && <p className="verdict__lean">Leans {leans} ({band}), but not clearly.</p>}

      <p className="verdict__score">
        <span className="verdict__number">{pct(score)}</span>
        <span className="verdict__unit">estimated chance it was AI-generated</span>
      </p>

      <RangeBar score={score} low={range.low} high={range.high} threshold={threshold} />

      <p className="verdict__certainty">
        Sentence-agreement range {pct(range.low)}–{pct(range.high)}
        {uncertain
          ? `, which crosses the ${pct(threshold)} line, so under this heuristic the result is uncertain.`
          : `, entirely ${isAi ? 'above' : 'below'} the ${pct(threshold)} line.`}
      </p>
      <p className="verdict__note">
        The range widens when there are few sentences or their scores disagree. It is a guide to how
        settled the estimate is, not a statistical guarantee.
      </p>

      <dl className="verdict__facts">
        <div><dt>Sentences</dt><dd>{sentences.length}</dd></div>
        <div><dt>Words</dt><dd>{words}</dd></div>
        <div><dt>Read as AI</dt><dd>{flagged} of {sentences.length}</dd></div>
      </dl>

      {tampered > 0 && (
        <Alert severity="warning" variant="outlined" className="verdict__alert">
          Before scoring, we removed {normalisation.invisible_removed} invisible character
          {normalisation.invisible_removed === 1 ? '' : 's'} and {normalisation.marks_removed} stray accent mark
          {normalisation.marks_removed === 1 ? '' : 's'}, and replaced {normalisation.homoglyphs_replaced} look-alike
          letter{normalisation.homoglyphs_replaced === 1 ? '' : 's'} from other alphabets. These can come from
          copying text out of some apps, and are also used to hide text from detectors, so the score was
          calculated without them. Accented letters stored as single characters (é, ü, ệ) are kept; an accent
          stored as a separate mark after a plain letter is removed.
        </Alert>
      )}

      <div className="verdict__drivers">
        <h3>Features most linked to the sentence scores</h3>
        <ul>
          {result.style_summary.slice(0, 4).map((f) => (
            <li key={f.feature}>
              <span className={`push push--${f.contribution >= 0 ? 'ai' : 'human'}`}>
                {f.contribution >= 0 ? 'Toward AI' : 'Toward human'}
              </span>
              <span className="push__label">{f.label}</span>
              <span className="push__value">{formatFeature(f.feature, f.value)}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

function RangeBar({ score, low, high, threshold }) {
  const pos = (v) => `${(v * 100).toFixed(2)}%`
  const color = score >= threshold ? ink.ai : ink.human
  return (
    <figure className="range" aria-label={`Score ${pct(score)}. Likely range ${pct(low)} to ${pct(high)}. Decision line at ${pct(threshold)}.`} role="img">
      <div className="range__track">
        <span className="range__half range__half--human" />
        <span className="range__half range__half--ai" />
        <span className="range__span" style={{ left: pos(low), width: pos(high - low), background: color }} />
        <span className="range__threshold" style={{ left: pos(threshold) }} />
        <span className="range__dot" style={{ left: pos(score), borderColor: color }} />
      </div>
      <figcaption className="range__labels" aria-hidden="true">
        <span>Human-written</span>
        <span>{pct(threshold)} line</span>
        <span>AI-generated</span>
      </figcaption>
    </figure>
  )
}
