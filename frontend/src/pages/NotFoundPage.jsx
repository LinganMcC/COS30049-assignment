import { Link, useLocation } from 'react-router-dom';

export default function NotFoundPage() {
  const { pathname } = useLocation();
  return (
    <section className="card message-page">
      <p className="message-code">404</p>
      <h1>Page not found</h1>
      {/* pathname is user-controlled, so it is shown as escaped text only. */}
      <p>There is no page at <code>{pathname}</code>.</p>
      <Link to="/" className="btn btn-primary">Back to the detector</Link>
    </section>
  );
}
