import { useEffect, useRef } from 'react'
import { pct, slug } from '../lib/text.js'

const BANDS = ['Human', 'Possibly AI', 'Likely AI', 'Very likely AI']

/**
 * The analysed passage with every sentence tinted by its score.
 * All text is rendered as React text nodes, never as HTML, so nothing a user
 * pastes can become markup or script.
 */
export default function HighlightedPassage({ passage, sentences, selected, onSelect, showScores, onToggleScores }) {
  const refs = useRef(new Map())

  useEffect(() => {
    if (selected == null) return
    const node = refs.current.get(selected)
    if (node && document.activeElement !== node) {
      node.scrollIntoView({ block: 'nearest', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })
    }
  }, [selected])

  function onKeyDown(event, index) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(index)
    }
  }

  return (
    <section className="passage" aria-labelledby="passage-title">
      <div className="passage__head">
        <h2 id="passage-title">Sentence by sentence</h2>
        <label className="toggle">
          <input type="checkbox" checked={showScores} onChange={(e) => onToggleScores(e.target.checked)} />
          Show scores
        </label>
      </div>
      <ul className="legend" aria-label="Highlight key">
        {BANDS.map((band) => (
          <li key={band}><span className={`swatch band-${slug(band)}`} aria-hidden="true" />{band}</li>
        ))}
      </ul>

      <div className="passage__sheet">
        {passage.map((seg, i) => {
          if (seg.sentence_index == null) {
            return <span key={i} className="passage__gap">{seg.text}</span>
          }
          const s = sentences[seg.sentence_index]
          const isSelected = selected === s.index
          return (
            <span
              key={i}
              ref={(node) => (node ? refs.current.set(s.index, node) : refs.current.delete(s.index))}
              role="button"
              tabIndex={0}
              aria-pressed={isSelected}
              aria-label={`Sentence ${s.index + 1}: ${pct(s.score)}, ${s.band}. ${s.text}`}
              className={`sent band-${slug(s.band)}${isSelected ? ' sent--selected' : ''}`}
              onClick={() => onSelect(s.index)}
              onKeyDown={(e) => onKeyDown(e, s.index)}
            >
              {s.text}
              {showScores && <span className="sent__score" aria-hidden="true">{pct(s.score)}</span>}
            </span>
          )
        })}
      </div>
    </section>
  )
}

