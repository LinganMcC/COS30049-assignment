import { useId, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { VERDICT_TEXT, pct, tickPct } from '../lib/text.js'
import { ink } from '../theme.js'
import { TableToggle, Tooltip } from './chartKit.jsx'
import useWidth from '../lib/useWidth.js'

const HEIGHT = 230
const M = { top: 16, right: 16, bottom: 34, left: 44 }
const FILTERS = { all: 'All checks', ai: 'Read as AI', human: 'Read as human' }

const time = (iso) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

/**
 * Question answered: how does this text compare with the others checked on this page?
 * One dot per check (its score) with a line for its sentence-agreement range.
 * Select a dot to see that check's details; filter to one verdict.
 * Only scores are kept, never the text (see HistoryContext.jsx).
 */
export default function HistoryChart({ items, currentId }) {
  const id = useId()
  const [frameRef, width] = useWidth()
  const [tip, setTip] = useState(null)
  const [table, setTable] = useState(false)
  const [filter, setFilter] = useState('all')
  const [selectedId, setSelectedId] = useState(null)

  // Number checks in the order they were made, then filter (numbers stay stable).
  const numbered = useMemo(() => items.map((item, i) => ({ ...item, n: i + 1 })), [items])
  const shown = numbered.filter((it) => filter === 'all' || it.label === filter)
  const selected = numbered.find((it) => it.id === (selectedId ?? currentId)) || null

  const innerW = width - M.left - M.right
  const innerH = HEIGHT - M.top - M.bottom
  const x = d3.scalePoint().domain(shown.map((it) => it.n)).range([0, innerW]).padding(0.6)
  const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0])
  const threshold = items[0]?.threshold ?? 0.5

  function show(it) {
    setTip({
      x: M.left + x(it.n),
      y: M.top + y(it.range.high) - 10,
      title: `Check ${it.n}: ${pct(it.score)}`,
      lines: [it.band, `Range ${pct(it.range.low)}–${pct(it.range.high)}`, `${it.sentenceCount} sentences, ${it.wordCount} words`],
    })
  }

  if (items.length === 0) return null

  return (
    <figure className="chart" aria-labelledby={`${id}-title`}>
      <div className="chart__head">
        <h3 id={`${id}-title`}>Your checks on this page</h3>
        <TableToggle open={table} onChange={setTable} controls={`${id}-table`} />
      </div>
      <p className="chart__sub">
        Each dot is one check; its line is the sentence-agreement range. Select a dot to see its details.
        Only the scores are kept, never your text. Clear, reloading or closing the page removes them.
      </p>
      <div className="chart__controls" role="group" aria-label="Filter checks">
        {Object.entries(FILTERS).map(([key, label]) => (
          <button key={key} type="button" className={`chip${filter === key ? ' chip--on' : ''}`} aria-pressed={filter === key} onClick={() => setFilter(key)}>
            {label}
          </button>
        ))}
      </div>
      <div className="chart__frame" ref={frameRef} onMouseLeave={() => setTip(null)}>
        {shown.length === 0 ? (
          <p className="chart__empty">No checks match this filter yet.</p>
        ) : (
          <svg width={width} height={HEIGHT} role="group" aria-label={`${shown.length} checks with scores and ranges`}>
            <g transform={`translate(${M.left},${M.top})`}>
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <g key={t} transform={`translate(0,${y(t)})`}>
                  <line x2={innerW} className={t === threshold ? 'axis-threshold' : 'grid'} />
                  <text x={-8} dy="0.32em" textAnchor="end" className="tick">{tickPct(t)}</text>
                </g>
              ))}
              {shown.map((it) => {
                const color = it.score >= threshold ? ink.ai : ink.human
                const isSel = selected?.id === it.id
                return (
                  <g
                    key={it.id}
                    className="dot"
                    transform={`translate(${x(it.n)},0)`}
                    tabIndex={0}
                    role="button"
                    aria-pressed={isSel}
                    aria-label={`Check ${it.n}: ${pct(it.score)}, ${it.band}, range ${pct(it.range.low)} to ${pct(it.range.high)}. Show details.`}
                    onClick={() => setSelectedId(it.id)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedId(it.id) } }}
                    onMouseEnter={() => show(it)}
                    onFocus={() => show(it)}
                    onBlur={() => setTip(null)}
                  >
                    <rect x={-14} y={0} width={28} height={innerH} fill="transparent" />
                    <line y1={y(it.range.low)} y2={y(it.range.high)} stroke={color} strokeWidth={2} strokeLinecap="round" />
                    {isSel && <circle cy={y(it.score)} r={10} className="dot__ring" />}
                    <circle cy={y(it.score)} r={5.5} fill={color} stroke="#fff" strokeWidth={2} />
                    <text y={innerH + 18} textAnchor="middle" className={`tick${it.id === currentId ? ' tick--strong' : ''}`}>{it.n}</text>
                  </g>
                )
              })}
              <text x={innerW / 2} y={innerH + 32} textAnchor="middle" className="tick tick--axis">check</text>
            </g>
          </svg>
        )}
        <Tooltip tip={tip} width={width} />
      </div>

      {selected && (
        <dl className="check-detail" aria-live="polite">
          <div><dt>Check</dt><dd>{selected.n}{selected.id === currentId ? ' (current)' : ''}, at {time(selected.at)}</dd></div>
          <div><dt>Result</dt><dd>{selected.certainty === 'uncertain' ? 'Uncertain' : VERDICT_TEXT[selected.band]}</dd></div>
          <div><dt>Score</dt><dd>{pct(selected.score)} (range {pct(selected.range.low)}–{pct(selected.range.high)})</dd></div>
          <div><dt>Length</dt><dd>{selected.sentenceCount} sentences, {selected.wordCount} words</dd></div>
        </dl>
      )}

      {table && (
        <div id={`${id}-table`} className="table-wrap">
          <table className="data-table">
            <caption className="sr-only">Checks on this page</caption>
            <thead><tr><th scope="col">#</th><th scope="col">Time</th><th scope="col">Score</th><th scope="col">Range</th><th scope="col">Band</th><th scope="col">Sentences</th></tr></thead>
            <tbody>
              {shown.map((it) => (
                <tr key={it.id}>
                  <td>{it.n}</td><td>{time(it.at)}</td><td>{pct(it.score)}</td>
                  <td>{pct(it.range.low)}–{pct(it.range.high)}</td>
                  <td>{it.band}</td><td>{it.sentenceCount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  )
}
