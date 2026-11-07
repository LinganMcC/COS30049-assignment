// Small shared pieces for the two charts.

/** Tooltip positioned inside the chart frame. Content is React text, never HTML. */
export function Tooltip({ tip, width }) {
  if (!tip) return null
  const left = Math.min(Math.max(tip.x, 90), width - 90)
  return (
    <div className="chart-tip" style={{ left, top: tip.y }} role="status">
      <strong>{tip.title}</strong>
      {tip.lines.map((line) => (
        <span key={line}>{line}</span>
      ))}
    </div>
  )
}

/** "Show as table" switch: every chart has a text equivalent. */
export function TableToggle({ open, onChange, controls }) {
  return (
    <button type="button" className="link-button" aria-expanded={open} aria-controls={controls} onClick={() => onChange(!open)}>
      {open ? 'Hide table' : 'Show as table'}
    </button>
  )
}
