import { useMemo, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { BAND_STYLE, percent } from '../bands';

// Chart 1 answers: "which sentences pushed this document's score up or down?"
// Interactions: band filter buttons, hover/focus tooltips, and clicking a bar
// selects that sentence in the highlighted text (and vice versa).
export default function SentenceChart({ sentences, selectedIndex, onSelect }) {
  const [hidden, setHidden] = useState(() => new Set());

  const data = useMemo(() => sentences
    .filter((s) => s.scored && !hidden.has(s.band))
    .map((s) => ({ index: s.index, label: `${s.index + 1}`, score: s.score * 100, band: s.band, text: s.text })),
  [sentences, hidden]);

  const toggle = (band) => setHidden((prev) => {
    const next = new Set(prev);
    next.has(band) ? next.delete(band) : next.add(band);
    return next;
  });

  return (
    <section className="card chart-card" aria-labelledby="sentence-chart-heading">
      <div className="card-header">
        <h2 id="sentence-chart-heading">Which sentences look AI-written?</h2>
      </div>
      <p className="card-hint">AI likelihood for each scored sentence, in reading order. Click a bar to find it in the text.</p>

      <div className="filter-row" role="group" aria-label="Show bands">
        {Object.entries(BAND_STYLE).map(([band, style]) => (
          <button
            key={band}
            type="button"
            className="filter-chip"
            aria-pressed={!hidden.has(band)}
            style={{ '--band': style.color, '--band-soft': style.soft }}
            onClick={() => toggle(band)}
          >
            <span className="swatch" aria-hidden="true" />{band}
          </button>
        ))}
      </div>

      {data.length === 0 ? (
        <p className="chart-empty">No sentences match the selected bands.</p>
      ) : (
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data} margin={{ top: 8, right: 8, bottom: 24, left: 4 }}>
              <CartesianGrid vertical={false} stroke="var(--border)" />
              <XAxis
                dataKey="label"
                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                label={{ value: 'Sentence number', position: 'insideBottom', offset: -14, fill: 'var(--text-muted)', fontSize: 12 }}
              />
              <YAxis
                domain={[0, 100]}
                ticks={[0, 25, 50, 75, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                width={44}
                label={{ value: 'AI likelihood', angle: -90, position: 'insideLeft', offset: 8, fill: 'var(--text-muted)', fontSize: 12, dy: 40 }}
              />
              <ReferenceLine y={50} stroke="var(--text-muted)" strokeDasharray="4 4" />
              <Tooltip content={<SentenceTooltip />} cursor={{ fill: 'var(--surface-muted)' }} />
              <Bar dataKey="score" name="AI likelihood" radius={[4, 4, 0, 0]} onClick={(bar) => onSelect(bar.index)} cursor="pointer">
                {data.map((d) => (
                  <Cell
                    key={d.index}
                    fill={BAND_STYLE[d.band].color}
                    fillOpacity={selectedIndex == null || selectedIndex === d.index ? 1 : 0.35}
                    stroke={selectedIndex === d.index ? 'var(--text)' : 'none'}
                    strokeWidth={2}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

function SentenceTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>Sentence {d.index + 1}: {percent(d.score / 100)}</strong> · {d.band}
      <p>{d.text.length > 140 ? `${d.text.slice(0, 140)}…` : d.text}</p>
    </div>
  );
}
