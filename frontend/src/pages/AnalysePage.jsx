import { useEffect, useRef, useState } from 'react';
import { analyseText, getConfig } from '../api/client';
import { DEFAULT_CONFIG } from '../bands';
import HighlightedText from '../components/HighlightedText';
import ResultSummary from '../components/ResultSummary';
import SentenceChart from '../components/SentenceChart';
import SessionChart from '../components/SessionChart';
import StatusBanner from '../components/StatusBanner';
import TextInputForm from '../components/TextInputForm';
import { useSessionHistory } from '../hooks/useSessionHistory';

// status: 'idle' -> 'loading' -> 'success' | 'error'
export default function AnalysePage() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [text, setText] = useState('');
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [current, setCurrent] = useState(null);       // { number, result } on screen
  const [selected, setSelected] = useState(null);     // sentence index picked in text or chart
  const history = useSessionHistory();
  const resultRef = useRef(null);
  const lastSubmitted = useRef('');

  // Limits come from the back end so both sides agree; defaults cover a failure.
  useEffect(() => {
    getConfig().then(setConfig).catch(() => {});
  }, []);

  // After a new result arrives, move focus to it so keyboard and screen-reader
  // users land on the answer rather than staying on the button.
  useEffect(() => {
    if (status === 'success') resultRef.current?.focus();
  }, [status, current]);

  const submit = async (value) => {
    lastSubmitted.current = value;
    setStatus('loading');
    setError(null);
    try {
      const result = await analyseText(value);
      history.add(value, result);
      setCurrent({ number: history.entries.length + 1, result });
      setSelected(null);
      setStatus('success');
    } catch (err) {
      setError(err);
      setStatus('error');
    }
  };

  const openFromHistory = (number) => {
    const entry = history.entries.find((e) => e.number === number);
    if (!entry) return;
    setText(entry.text);
    setCurrent({ number, result: entry.result });
    setSelected(null);
    setError(null);
    setStatus('success');
  };

  const clear = () => {
    setText('');
    setCurrent(null);
    setSelected(null);
    setError(null);
    setStatus('idle');
  };

  return (
    <>
      <section className="intro">
        <h1>Was this written by AI?</h1>
        <p>
          Paste a piece of writing to see how likely each sentence is to be AI-generated, and an
          overall estimate with a confidence range.
        </p>
      </section>

      <TextInputForm
        text={text}
        onTextChange={setText}
        onSubmit={submit}
        onClear={clear}
        config={config}
        loading={status === 'loading'}
      />

      <StatusBanner status={status} error={error} onRetry={() => submit(lastSubmitted.current)} />

      {current && status !== 'loading' && (
        <div className="results" ref={resultRef} tabIndex={-1} aria-label="Analysis results">
          <ResultSummary result={current.result} />
          <div className="results-grid">
            <HighlightedText sentences={current.result.sentences} selectedIndex={selected} onSelect={setSelected} />
            <SentenceChart sentences={current.result.sentences} selectedIndex={selected} onSelect={setSelected} />
          </div>
        </div>
      )}

      <SessionChart
        entries={history.entries}
        activeNumber={current?.number}
        onOpen={openFromHistory}
        onClear={history.clear}
      />
    </>
  );
}
