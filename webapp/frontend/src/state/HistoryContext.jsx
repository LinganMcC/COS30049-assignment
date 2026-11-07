// Scores from the checks made on this page, for the comparison chart.
//
// Privacy: only numbers are kept (score, range, verdict, counts). The submitted
// text and the sentence results are NOT kept, and nothing is written to
// browser storage: everything here disappears on reload or when the tab closes.
import { createContext, useCallback, useContext, useMemo, useState } from 'react'

const MAX_ITEMS = 30
const HistoryContext = createContext(null)

export function HistoryProvider({ children }) {
  const [items, setItems] = useState([])

  const add = useCallback((result) => {
    const item = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      at: new Date().toISOString(),
      score: result.score,
      range: result.range,
      threshold: result.threshold,
      label: result.label,
      band: result.band,
      certainty: result.certainty,
      sentenceCount: result.sentence_count,
      wordCount: result.word_count,
    }
    setItems((prev) => [...prev, item].slice(-MAX_ITEMS))
    return item.id
  }, [])

  const clear = useCallback(() => setItems([]), [])

  const value = useMemo(() => ({ items, add, clear }), [items, add, clear])
  return <HistoryContext.Provider value={value}>{children}</HistoryContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useHistory() {
  const ctx = useContext(HistoryContext)
  if (!ctx) throw new Error('useHistory must be used inside HistoryProvider')
  return ctx
}
