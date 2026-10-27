import { Component } from 'react';

// Catches any rendering error below it, so a bug shows this page instead of a
// blank screen. Error boundaries must be class components in React.
export class ErrorBoundary extends Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error, info) {
    // Developer console only - the user never sees the stack.
    console.error('Unexpected UI error', error, info.componentStack);
  }

  render() {
    return this.state.failed ? <ErrorFallback /> : this.props.children;
  }
}

export default function ErrorFallback() {
  return (
    <section className="card message-page" role="alert">
      <p className="message-code">Oops</p>
      <h1>Something went wrong</h1>
      <p>The page hit an unexpected problem. Your text was not sent anywhere else.</p>
      <button type="button" className="btn btn-primary" onClick={() => window.location.assign('/')}>
        Reload the detector
      </button>
    </section>
  );
}
