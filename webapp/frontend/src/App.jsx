import { Route, Routes, useLocation } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import DetectPage from './pages/DetectPage.jsx'
import AboutPage from './pages/AboutPage.jsx'
import NotFoundPage from './pages/NotFoundPage.jsx'

export default function App() {
  const location = useLocation()
  return (
    <Layout>
      {/* Keyed on the path so the fallback clears when the user navigates away. */}
      <ErrorBoundary key={location.pathname}>
        <Routes>
          <Route path="/" element={<DetectPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </ErrorBoundary>
    </Layout>
  )
}
