import { useCallback, useState } from 'react';

// Analyses from this visit only. Kept in React state - never localStorage or the
// server - so nothing the user pasted outlives the tab (A1 Req 11: no stored text).
export function useSessionHistory() {
  const [entries, setEntries] = useState([]);

  const add = useCallback((text, result) => {
    const trimmed = text.trim();
    setEntries((prev) => [...prev, {
      number: prev.length + 1,   // also the id: entries are only ever appended
      label: trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed,
      checkedAt: new Date(),
      text,
      result,
    }]);
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return { entries, add, clear };
}
