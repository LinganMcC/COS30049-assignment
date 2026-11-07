import { useEffect, useState } from 'react'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Skeleton from '@mui/material/Skeleton'
import { getModelInfo } from '../api/client.js'
import { pct } from '../lib/text.js'

/** Model facts come from GET /api/model-info, so this page always matches the deployed model. */
export default function AboutPage() {
  const [state, setState] = useState({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    getModelInfo({ signal: controller.signal })
      .then((info) => setState({ status: 'success', info }))
      .catch((err) => err.kind !== 'cancelled' && setState({ status: 'error', error: err }))
    return () => controller.abort()
  }, [attempt])

  return (
    <article className="page page--narrow about">
      <h1>How Inkprint decides</h1>
      <p className="lede">
        Inkprint doesn't compare your text with anything online. It measures how each sentence is built
        (word length, punctuation, parts of speech) and asks a model trained on 31,000 labelled essays how
        AI-like that pattern is.
      </p>

      <h2>From your text to a verdict</h2>
      <ol className="steps">
        <li><strong>Clean.</strong> Invisible characters and look-alike letters from other alphabets are removed (normal accents stay), then spacing and stray HTML are tidied exactly as they were for the training data.</li>
        <li><strong>Split.</strong> spaCy divides the text into sentences. Fragments without letters are shown but not scored.</li>
        <li><strong>Measure.</strong> Each sentence gets 15 style measurements.</li>
        <li><strong>Score.</strong> A Logistic Regression model turns the measurements into a probability for each sentence.</li>
        <li><strong>Combine.</strong> Sentence scores are averaged, giving longer sentences more weight (square root of their length). A sentence-agreement range is shown around the result: it widens when there are few sentences or their scores disagree.</li>
      </ol>

      {state.status === 'loading' && (
        <div aria-busy="true" aria-label="Loading model details">
          <Skeleton height={40} /><Skeleton height={180} /><Skeleton height={180} />
        </div>
      )}

      {state.status === 'error' && (
        <Alert severity="error" variant="outlined" action={<Button color="inherit" onClick={() => { setState({ status: 'loading' }); setAttempt((a) => a + 1) }}>Try again</Button>}>
          Model details are unavailable. {state.error.message}
        </Alert>
      )}

      {state.status === 'success' && <ModelDetails info={state.info} />}

      <h2>What a result can and can't tell you</h2>
      <ul className="limits">
        <li>
          <strong>Sentence scores are estimates.</strong> The model learned from whole essays labelled AI or human,
          with every sentence given its essay's label. A highlighted sentence shows how AI-like its style is,
          not a verified label for that sentence.
        </li>
        <li>
          <strong>The range is a guide, not a guarantee.</strong> It is calculated from how much the sentence scores
          disagree and how many there are. It has not been tested as a statistical confidence interval, so
          treat it as a sign of how settled the estimate is.
        </li>
        <li>
          <strong>Features are associations, not proof.</strong> The feature bars show which measurements pushed the
          model's sentence scores. No single writing habit shows that a text was AI-generated.
        </li>
        <li>
          <strong>It works best on student essays.</strong> Accuracy drops sharply on other kinds of writing (see the
          Q&amp;A column above).
        </li>
      </ul>
    </article>
  )
}

function ModelDetails({ info }) {
  const test = info.evaluation.drcat_test
  const hc3 = info.evaluation.hc3_external
  const rows = [
    ['Accuracy', 'accuracy'],
    ['Precision (AI)', 'precision'],
    ['Recall (AI)', 'recall'],
    ['F1 (AI)', 'f1'],
    ['ROC-AUC', 'roc_auc'],
  ]
  const features = [...info.features].sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient))
  const maxCoef = Math.max(...features.map((f) => Math.abs(f.coefficient)))

  return (
    <>
      <h2>How accurate it is</h2>
      <p>
        Measured once on essays the model never saw. The bracket is a 95% bootstrap interval. The second
        column is a different kind of writing (HC3 question answers), which is why the app warns that it is weaker outside student essays.
      </p>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr><th scope="col">Measure</th><th scope="col">Student essays (DRCAT test)</th><th scope="col">Q&amp;A answers (HC3)</th></tr>
          </thead>
          <tbody>
            {rows.map(([label, key]) => (
              <tr key={key}>
                <th scope="row">{label}</th>
                <td>{fmt(test[key])}</td>
                <td>{fmt(hc3[key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>The 15 measurements</h2>
      <p>Ordered by how strongly each moves the score. Bars show the model's weight for one standard deviation of change.</p>
      <ul className="feature-list">
        {features.map((f) => (
          <li key={f.feature}>
            <div className="feature-list__row">
              <span className="feature-list__name">{f.label}</span>
              <span className={`push push--${f.direction}`}>{f.direction === 'ai' ? 'Higher reads as AI' : 'Higher reads as human'}</span>
            </div>
            <span className="factors__bar" aria-hidden="true">
              <span className={`factors__fill factors__fill--${f.direction}`} style={{ width: `${(Math.abs(f.coefficient) / maxCoef) * 100}%` }} />
            </span>
            <p className="feature-list__meaning">{f.meaning}</p>
          </li>
        ))}
      </ul>

      <h2>Reading the result</h2>
      <ul className="bands">
        {info.bands.map((b, i) => (
          <li key={b.label}>
            <strong>{b.label}</strong>: {i === 0 ? `up to ${pct(b.upper)}` : `${pct(info.bands[i - 1].upper)} to ${pct(b.upper)}`}
          </li>
        ))}
      </ul>
      <p>
        The verdict uses the {pct(info.threshold)} line. Each check takes {info.limits.min_words}–{info.limits.max_words} words.
        Model version {info.version}.
      </p>
    </>
  )
}

const fmt = (m) => (m ? `${pct(m.estimate)} (${pct(m.ci_low)}–${pct(m.ci_high)})` : '–')
