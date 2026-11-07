import { useId, useState } from 'react'
import * as d3 from 'd3'
import { pct, tickPct } from '../lib/text.js'
import { ink } from '../theme.js'
import { TableToggle, Tooltip } from './chartKit.jsx'
import useWidth from '../lib/useWidth.js'

const HEIGHT = 220
const M = { top: 16, right: 12, bottom: 34, left: 44 }

/**
 * Question answered: which sentences pull this text toward AI, and how hard?
 * Bars grow up (AI) or down (human) from the 50% decision line, in reading order.
 * Select a bar to highlight that sentence in the passage.
 */
export default function SentenceChart({ sentences, threshold, selected, onSelect }) {
  const id = useId()
  const [frameRef, width] = useWidth()
  const [tip, setTip] = useState(null)
  const [table, setTable] = useState(false)

  const innerW = width - M.left - M.right
  const innerH = HEIGHT - M.top - M.bottom
  const x = d3.scaleBand().domain(sentences.map((s) => s.index)).range([0, innerW]).paddingInner(sentences.length > 40 ? 0.15 : 0.3).paddingOuter(0.1)
  const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0])
  const barW = Math.max(2, x.bandwidth())
  const labelEvery = Math.ceil(sentences.length / Math.max(1, Math.floor(innerW / 28)))

  function show(s) {
    const cx = M.left + x(s.index) + x.bandwidth() / 2
    setTip({
      x: cx,
      y: M.top + y(Math.max(s.score, threshold)) - 8,
      title: `Sentence ${s.index + 1}: ${pct(s.score)}`,
      lines: [s.band, s.text.length > 70 ? `${s.text.slice(0, 70)}…` : s.text],
    })
  }

  function onKeyDown(event, s) {
    const i = s.index
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(i)
    } else if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault()
      const next = Math.min(sentences.length - 1, Math.max(0, i + (event.key === 'ArrowRight' ? 1 : -1)))
      onSelect(next)
      document.getElementById(`${id}-bar-${next}`)?.focus()
    }
  }

  return (
    <figure className="chart" aria-labelledby={`${id}-title`}>
      <div className="chart__head">
        <h3 id={`${id}-title`}>Where the AI signal sits in this text</h3>
        <TableToggle open={table} onChange={setTable} controls={`${id}-table`} />
      </div>
      <p className="chart__sub">
        Each bar is one sentence, in reading order. Bars above the line read as AI; below it, as human.
        Select a bar to find the sentence. Use ← → to move between bars.
      </p>
      <div className="chart__frame" ref={frameRef} onMouseLeave={() => setTip(null)}>
        <svg width={width} height={HEIGHT} role="group" aria-label={`Bar chart of ${sentences.length} sentence scores`}>
          <g transform={`translate(${M.left},${M.top})`}>
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x2={innerW} className={t === threshold ? 'axis-threshold' : 'grid'} />
                <text x={-8} dy="0.32em" textAnchor="end" className="tick">{tickPct(t)}</text>
              </g>
            ))}
            <text x={innerW} y={y(threshold) - 6} textAnchor="end" className="tick tick--strong">decision line</text>
            {sentences.map((s) => {
              const up = s.score >= threshold
              const top = up ? y(s.score) : y(threshold)
              const h = Math.max(2, Math.abs(y(s.score) - y(threshold)))
              const isSel = selected === s.index
              return (
                <g
                  key={s.index}
                  id={`${id}-bar-${s.index}`}
                  className={`bar${isSel ? ' bar--selected' : ''}`}
                  tabIndex={0}
                  role="button"
                  aria-pressed={isSel}
                  aria-label={`Sentence ${s.index + 1}: ${pct(s.score)}, ${s.band}`}
                  onClick={() => onSelect(s.index)}
                  onKeyDown={(e) => onKeyDown(e, s)}
                  onMouseEnter={() => show(s)}
                  onFocus={() => show(s)}
                  onBlur={() => setTip(null)}
                >
                  {/* Hit area taller than the bar so small bars are easy to hover and tap. */}
                  <rect x={x(s.index) - 1} y={0} width={barW + 2} height={innerH} fill="transparent" />
                  <rect
                    x={x(s.index)}
                    y={top}
                    width={barW}
                    height={h}
                    rx={Math.min(3, barW / 2)}
                    fill={up ? ink.ai : ink.human}
                    opacity={selected == null || isSel ? 1 : 0.45}
                  />
                  {isSel && <rect x={x(s.index) - 3} y={top - 3} width={barW + 6} height={h + 6} rx={4} className="bar__ring" />}
                </g>
              )
            })}
            {sentences.map((s) =>
              s.index % labelEvery === 0 ? (
                <text key={s.index} x={x(s.index) + x.bandwidth() / 2} y={innerH + 18} textAnchor="middle" className="tick">
                  {s.index + 1}
                </text>
              ) : null,
            )}
            <text x={innerW / 2} y={innerH + 32} textAnchor="middle" className="tick tick--axis">sentence</text>
          </g>
        </svg>
        <Tooltip tip={tip} width={width} />
      </div>
      {table && (
        <div id={`${id}-table`} className="table-wrap">
          <table className="data-table">
            <caption className="sr-only">Sentence scores</caption>
            <thead><tr><th scope="col">#</th><th scope="col">Score</th><th scope="col">Band</th><th scope="col">Sentence</th></tr></thead>
            <tbody>
              {sentences.map((s) => (
                <tr key={s.index}><td>{s.index + 1}</td><td>{pct(s.score)}</td><td>{s.band}</td><td>{s.text}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  )
}
