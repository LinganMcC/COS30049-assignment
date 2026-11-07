import { useEffect, useRef, useState } from 'react'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import LinearProgress from '@mui/material/LinearProgress'
import { detectText } from '../api/client.js'
import { validateText } from '../lib/text.js'
import { useHistory } from '../state/HistoryContext.jsx'
import TextInputForm from '../components/TextInputForm.jsx'
import ResultSummary from '../components/ResultSummary.jsx'
import HighlightedPassage from '../components/HighlightedPassage.jsx'
import SentenceDetail from '../components/SentenceDetail.jsx'
import SentenceChart from '../components/SentenceChart.jsx'
import HistoryChart from '../components/HistoryChart.jsx'

/**
 * idle -> loading -> success | error
 * The page owns the text and the result; child components receive what they need as props.
 */
export default function DetectPage() {
  const history = useHistory()
  const [text, setText] = useState('')
  const [status, setStatus] = useState('idle')
  const [check, setCheck] = useState(null) // { id, text, result }
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [showScores, setShowScores] = useState(true)
  const controller = useRef(null)
  const resultRef = useRef(null)

  useEffect(() => () => controller.current?.abort(), [])

  async function runCheck() {
    // Every path that sends a request (the form and "Try again") validates first.
    const problem = validateText(text)
    if (problem) {
      setError({ kind: 'http', status: 422, message: problem })
      setStatus('error')
      return
    }
    controller.current?.abort()
    controller.current = new AbortController()
    const submitted = text
    setStatus('loading')
    setError(null)
    try {
      const result = await detectText(submitted, { signal: controller.current.signal })
      const id = history.add(result)
      setCheck({ id, text: submitted, result })
      const top = result.sentences.reduce((a, b) => (b.score > a.score ? b : a), result.sentences[0])
      setSelected(top.index)
      setStatus('success')
      requestAnimationFrame(() => resultRef.current?.focus({ preventScroll: window.innerWidth >= 960 }))
    } catch (err) {
      if (err.kind === 'cancelled') return
      setError(err)
      setStatus('error')
    }
  }

  // Clear returns the page to its starting state, including the session chart.
  function clearAll() {
    controller.current?.abort()
    history.clear()
    setText('')
    setCheck(null)
    setError(null)
    setSelected(null)
    setStatus('idle')
  }

  const fieldError = error?.status === 422 ? error.message : null
  const stale = Boolean(check) && text !== check.text
  const result = check?.result

  return (
    <div className="page">
      <header className="intro">
        <h1>Does this read like AI wrote it?</h1>
        <p className="lede">
          Paste a piece of English writing. Inkprint scores every sentence on 15 measures of writing style,
          shades the passage by how AI-like each sentence reads, and gives an overall estimate with a range showing how much the sentences agree.
        </p>
      </header>

      <div className="workspace">
        <div className="workspace__main">
          <TextInputForm
            text={text}
            onTextChange={setText}
            onSubmit={runCheck}
            onClear={clearAll}
            loading={status === 'loading'}
            serverError={fieldError}
          />
        </div>

        <div className="workspace__side" ref={resultRef} tabIndex={-1} aria-label="Result">
          {status === 'loading' && (
            <section className="status-card" aria-live="polite" aria-busy="true">
              <h2>Checking your text</h2>
              <LinearProgress aria-label="Checking" />
              <p>Splitting it into sentences and scoring each one. This usually takes under a second.</p>
            </section>
          )}

          {status === 'error' && error && error.status !== 422 && (
            <ErrorPanel error={error} onRetry={runCheck} />
          )}

          {(status === 'idle' || (status === 'error' && error?.status === 422)) && !result && (
            <section className="status-card status-card--idle">
              <h2>What you'll get</h2>
              <p>An overall estimate, every sentence shaded by its score, and the writing features linked to each score.</p>
              <p className="muted">The model was trained on student essays. It is weaker on other kinds of writing, and no detector is proof.</p>
            </section>
          )}

          {result && status !== 'loading' && !(status === 'error' && error?.status !== 422) && (
            <ResultSummary result={result} stale={stale} />
          )}
        </div>
      </div>

      {result && status !== 'loading' && (
        <div className={`analysis${stale ? ' analysis--stale' : ''}`}>
          <div className="analysis__reading">
            <HighlightedPassage
              passage={result.passage}
              sentences={result.sentences}
              selected={selected}
              onSelect={setSelected}
              showScores={showScores}
              onToggleScores={setShowScores}
            />
            <SentenceDetail sentence={selected == null ? null : result.sentences[selected]} total={result.sentences.length} />
          </div>
          <SentenceChart sentences={result.sentences} threshold={result.threshold} selected={selected} onSelect={setSelected} />
        </div>
      )}

      {history.items.length > 0 && (
        <HistoryChart items={history.items} currentId={check?.id} />
      )}
    </div>
  )
}

function ErrorPanel({ error, onRetry }) {
  const [seconds, setSeconds] = useState(error.retryAfter || 0)

  useEffect(() => {
    if (!seconds) return undefined
    const t = setTimeout(() => setSeconds((s) => s - 1), 1000)
    return () => clearTimeout(t)
  }, [seconds])

  const title = {
    network: "Can't reach the detector",
    timeout: 'The check took too long',
  }[error.kind] || {
    503: 'The detector is busy',
    413: 'That text is too large',
  }[error.status] || 'The check failed'

  return (
    <Alert severity="error" variant="outlined" className="error-panel">
      <strong>{title}</strong>
      <span>{error.message}</span>
      <Button variant="outlined" color="inherit" size="small" onClick={onRetry} disabled={seconds > 0} className="error-panel__retry">
        {seconds > 0 ? `Try again in ${seconds}s` : 'Try again'}
      </Button>
    </Alert>
  )
}
