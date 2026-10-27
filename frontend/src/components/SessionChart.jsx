import {
  Bar, CartesianGrid, Cell, ComposedChart, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { BAND_STYLE, percent } from '../bands';

// Chart 2 answers: "how do the texts I've checked this session compare, and how
// sure is each verdict?" Each bar is one document; the whisker is its
// confidence range. Clicking a bar (or a row in the list) reopens that result.
export default function SessionChart({ entries, activeNumber, onOpen, onClear }) {
  if (entries.length === 0) return null;

  const data = entries.map((e) => {
    const { score, low, high, band } = e.result.document;
    return {
      number: e.number,
      label: `#${e.number}`,
      score: score * 100,
      // ErrorBar takes [distance below, distance above] the bar's top.
      range: [(score - low) * 100, (high - score) * 100],
      low, high, band,
      text: e.label,
    };
  });

  return (
    <section className="card chart-card" aria-labelledby="session-chart-heading">
      <div className="card-header">
        <h2 id="session-chart-heading">How do your texts compare?</h2>
        <button type="button" className="btn-link" onClick={onClear}>Clear history</button>
      </div>
      <p className="card-hint">Document score and confidence range for every text checked in this session.</p>

      <div className="chart-frame">
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 24, left: 4 }}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis
              dataKey="label"
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              label={{ value: 'Text checked', position: 'insideBottom', offset: -14, fill: 'var(--text-muted)', fontSize: 12 }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tickFormatter={(v) => `${v}%`}
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              width={44}
              label={{ value: 'Document score', angle: -90, position: 'insideLeft', offset: 8, fill: 'var(--text-muted)', fontSize: 12, dy: 50 }}
            />
            <ReferenceLine y={50} stroke="var(--text-muted)" strokeDasharray="4 4" />
            <Tooltip content={<SessionTooltip />} cursor={{ fill: 'var(--surface-muted)' }} />
            <Bar dataKey="score" name="Document score" maxBarSize={56} radius={[4, 4, 0, 0]} onClick={(bar) => onOpen(bar.number)} cursor="pointer">
              {data.map((d) => (
                <Cell
                  key={d.number}
                  fill={BAND_STYLE[d.band].color}
                  fillOpacity={d.number === activeNumber ? 1 : 0.45}
                />
              ))}
              <ErrorBar dataKey="range" width={10} strokeWidth={2} stroke="var(--text)" />
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* The same data as a list, so every text can be reopened by keyboard. */}
      <ol className="history-list">
        {data.map((d) => (
          <li key={d.number}>
            <button
              type="button"
              className={`history-item${d.number === activeNumber ? ' active' : ''}`}
              aria-current={d.number === activeNumber ? 'true' : undefined}
              onClick={() => onOpen(d.number)}
            >
              <span className="history-number">#{d.number}</span>
              <span className="history-label">{d.text}</span>
              <span className="history-score" style={{ color: BAND_STYLE[d.band].color }}>
                {percent(d.score / 100)}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

function SessionTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>Text #{d.number}: {percent(d.score / 100)}</strong> · {d.band}
      <p>Range {percent(d.low)} – {percent(d.high)}</p>
      <p className="chart-tooltip-muted">{d.text}</p>
    </div>
  );
}
