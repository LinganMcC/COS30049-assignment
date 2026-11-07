import { Link, useLocation } from 'react-router-dom'
import Button from '@mui/material/Button'

export default function NotFoundPage() {
  const { pathname } = useLocation()
  return (
    <section className="page page--narrow status-page" aria-labelledby="nf-title">
      <p className="status-page__code" aria-hidden="true">404</p>
      <h1 id="nf-title">There's no page at this address</h1>
      {/* React renders the path as text, so a crafted URL cannot inject markup. */}
      <p className="lede">
        <code className="path">{pathname}</code> doesn't exist. The link may be mistyped or out of date.
      </p>
      <div className="status-page__actions">
        <Button component={Link} to="/" variant="contained">Check some text</Button>
        <Button component={Link} to="/about" variant="outlined">How it works</Button>
      </div>
    </section>
  )
}
