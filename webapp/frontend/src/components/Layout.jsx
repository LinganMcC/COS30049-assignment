import { NavLink } from 'react-router-dom'

export default function Layout({ children }) {
  return (
    <div className="shell">
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site-header">
        <div className="site-header__inner">
          <NavLink to="/" className="wordmark" aria-label="Inkprint home">
            <span className="wordmark__glyph" aria-hidden="true">
              <span /><span /><span />
            </span>
            Inkprint
          </NavLink>
          <nav aria-label="Main">
            <NavLink to="/" end className="nav-link">Check text</NavLink>
            <NavLink to="/about" className="nav-link">How it works</NavLink>
          </nav>
        </div>
      </header>
      <main id="main" className="site-main" tabIndex={-1}>
        {children}
      </main>
      <footer className="site-footer">
        <p>
          Inkprint estimates; it does not prove. Treat a result as a reason to look closer,
          never as evidence on its own. Your text is analysed and then discarded: it is not stored
          or logged. The page keeps only the scores, until you press Clear, reload or close it.
        </p>
        <p>COS30049 Computing Technology Innovation Project · Assignment 3</p>
      </footer>
    </div>
  )
}
