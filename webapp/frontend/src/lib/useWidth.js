import { useEffect, useRef, useState } from 'react'

/** Width of a container, updated when it resizes (so charts fit phones and desktops). */
export default function useWidth(initial = 320) {
  const ref = useRef(null)
  const [width, setWidth] = useState(initial)
  useEffect(() => {
    const node = ref.current
    if (!node) return undefined
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(260, Math.floor(entry.contentRect.width))))
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return [ref, width]
}

