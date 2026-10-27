import { Link, Route, Routes } from 'react-router-dom';
import AnalysePage from './pages/AnalysePage';
import { ErrorBoundary } from './pages/ErrorFallback';
import NotFoundPage from './pages/NotFoundPage';

export default function App() {
  return (
    <>
      <a href="#main" className="skip-link">Skip to main content</a>
      <header className="site-header">
        <div className="container header-inner">
          <Link to="/" className="brand">
            <img src="/favicon.svg" alt="" width="28" height="28" />
            <span>AI Content Detector</span>
          </Link>
          <span className="header-tag">Sentence-level analysis</span>
        </div>
      </header>

      <main id="main" className="container main">
        {/* Inside <main> so the header stays usable even if a page crashes. */}
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<AnalysePage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </ErrorBoundary>
      </main>

      <footer className="site-footer">
        <div className="container">
          COS30049 Group 4 · Scores are estimates, not proof of authorship · Text is never stored
        </div>
      </footer>
    </>
  );
}
