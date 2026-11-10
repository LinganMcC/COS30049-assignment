import { useId, useRef, useState } from 'react'
import Button from '@mui/material/Button'
import { LIMITS, countWords, validateText } from '../lib/text.js'

const SAMPLES = {
  ai: 'Remote work has fundamentally transformed the modern workplace, offering both significant opportunities and notable challenges. Furthermore, employees now enjoy greater flexibility, which can lead to improved work-life balance and increased productivity. However, the absence of in-person interaction may weaken team cohesion and hinder spontaneous collaboration. Additionally, organisations must invest in reliable digital infrastructure to support distributed teams effectively. In conclusion, remote work represents a valuable evolution, provided that companies thoughtfully address its inherent limitations.',
  human: "Young adults and teens can not put their phone down for a second. Any notification they receive is so important to them. The first thing they do is check and see what popped up on their phone. The worst time for this to happen is when they're driving because they will still cheak no matter what they're doing. Using your phone while driving can cause you to get a ticket, cause you to hurt someone, and even cause you to hurt yourself.",
}
const MAX_FILE_BYTES = 100 * 1024

/**
 * The text box, its live word count, and the actions.
 * Validation runs here first (for usability); the back end checks again (for security).
 */
export default function TextInputForm({ text, onTextChange, onSubmit, onClear, loading, serverError, hasHistory }) {
  const id = useId()
  const fileInput = useRef(null)
  const [attempted, setAttempted] = useState(false)
  const [fileError, setFileError] = useState(null)

  const words = countWords(text)
  const localError = validateText(text)
  const overLimit = words > LIMITS.maxWords || text.length > LIMITS.maxChars
  const message = fileError || serverError || ((attempted || overLimit) ? localError : null)

  function handleSubmit(event) {
    event.preventDefault()
    setAttempted(true)
    if (validateText(text) || loading) return
    onSubmit()
  }

  function handleClear() {
    setAttempted(false)
    setFileError(null)
    onClear()
  }

  function handleFile(event) {
    const file = event.target.files?.[0]
    event.target.value = '' // let the same file be chosen again
    setFileError(null)
    if (!file) return
    const looksLikeText = file.type === 'text/plain' || (!file.type && file.name.toLowerCase().endsWith('.txt'))
    if (!looksLikeText) {
      setFileError('Choose a plain-text (.txt) file. For Word or PDF files, copy the text and paste it here.')
      return
    }
    if (file.size > MAX_FILE_BYTES) {
      setFileError('That file is larger than 100 KB. Paste the part you want to check instead.')
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      setAttempted(true)
      onTextChange(String(reader.result))
    }
    reader.onerror = () => setFileError('That file could not be read. Try pasting the text instead.')
    reader.readAsText(file)
  }

  const counterState = overLimit ? 'over' : words >= LIMITS.minWords ? 'ok' : 'under'

  return (
    <form className="input-form" onSubmit={handleSubmit} noValidate aria-busy={loading}>
      <div className="input-form__head">
        <label htmlFor={`${id}-text`} className="field-label">Text to check</label>
        <span id={`${id}-count`} className={`word-count word-count--${counterState}`} aria-live="polite">
          {words.toLocaleString()} / {LIMITS.maxWords.toLocaleString()} words
        </span>
      </div>
      <textarea
        id={`${id}-text`}
        className={`text-input${message ? ' text-input--invalid' : ''}`}
        value={text}
        onChange={(e) => {
          setFileError(null)
          onTextChange(e.target.value)
        }}
        rows={10}
        spellCheck="false"
        placeholder="Paste an essay, report or answer in English. At least 20 words; three or more sentences give a narrower range."
        aria-invalid={Boolean(message)}
        aria-describedby={`${id}-count ${id}-msg`}
      />
      <p id={`${id}-msg`} className="field-message" role={message ? 'alert' : undefined}>
        {message || ' '}
      </p>

      <div className="input-form__actions">
        <Button type="submit" variant="contained" size="large" disabled={loading}>
          {loading ? 'Checking…' : 'Check text'}
        </Button>
        <Button type="button" variant="outlined" onClick={handleClear} disabled={!text && !attempted && !hasHistory}>
          Clear
        </Button>
        <span className="input-form__spacer" />
        <Button type="button" variant="text" onClick={() => fileInput.current?.click()}>
          Open a .txt file
        </Button>
        <input ref={fileInput} type="file" accept=".txt,text/plain" hidden onChange={handleFile} aria-label="Open a text file" />
        <Button type="button" variant="text" onClick={() => { setAttempted(false); setFileError(null); onTextChange(SAMPLES.ai) }}>
          AI sample
        </Button>
        <Button type="button" variant="text" onClick={() => { setAttempted(false); setFileError(null); onTextChange(SAMPLES.human) }}>
          Student sample
        </Button>
      </div>
    </form>
  )
}
