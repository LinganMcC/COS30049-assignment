// Loading and error states. role="status" is announced politely by screen
// readers; role="alert" interrupts, which suits a failed request.
export default function StatusBanner({ status, error, onRetry }) {
  if (status === 'loading') {
    return (
      <div className="banner banner-info" role="status">
        <span className="spinner" aria-hidden="true" />
        Analysing your text sentence by sentence…
      </div>
    );
  }

  if (status === 'error' && error) {
    const canRetry = ['server_busy', 'analysis_timeout', 'network_error', 'timeout', 'internal_error']
      .includes(error.code);
    return (
      <div className="banner banner-error" role="alert">
        <div>
          <strong>Analysis failed.</strong> {error.message}
          {error.retryAfter && ` You can try again in about ${error.retryAfter} seconds.`}
          {error.details?.length > 0 && (
            <ul className="error-details">
              {error.details.map((d) => <li key={`${d.field}-${d.message}`}>{d.message}</li>)}
            </ul>
          )}
        </div>
        {canRetry && (
          <button type="button" className="btn btn-secondary btn-small" onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    );
  }

  return null;
}
