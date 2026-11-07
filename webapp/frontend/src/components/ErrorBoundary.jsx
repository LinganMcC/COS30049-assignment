import { Component } from 'react'
import { Link } from 'react-router-dom'
import Button from '@mui/material/Button'

/** Catches render errors anywhere below it so the user never gets a blank page. */
export default class ErrorBoundary extends Component {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error, info) {
    // Log to the browser console for developers. No user text is included.
    console.error('Inkprint view crashed:', error?.name, info?.componentStack?.split('\n')[1]?.trim())
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <section className="page page--narrow status-page" aria-labelledby="crash-title">
        <h1 id="crash-title">This page stopped working</h1>
        <p className="lede">
          Something in the page failed to display. Your text was not sent anywhere new.
          Reload to start again, or go back to the checker.
        </p>
        <div className="status-page__actions">
          <Button variant="contained" onClick={() => window.location.reload()}>Reload the page</Button>
          <Button component={Link} to="/" variant="outlined" onClick={() => this.setState({ failed: false })}>
            Go to the checker
          </Button>
        </div>
      </section>
    )
  }
}
