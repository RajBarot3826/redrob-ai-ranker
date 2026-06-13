import { useState, useEffect } from 'react'
import './index.css'

/* ═══════════════════════════════════════════════════════════ */
/* Utility Helpers                                            */
/* ═══════════════════════════════════════════════════════════ */

function scoreColor(score) {
  if (score >= 0.7) return 'high'
  if (score >= 0.4) return 'medium'
  return 'low'
}

function rankClass(rank) {
  if (rank <= 3) return 'top3'
  if (rank <= 10) return 'top10'
  return 'normal'
}

/* ═══════════════════════════════════════════════════════════ */
/* AnimatedCounter — Counts up from 0 to value               */
/* ═══════════════════════════════════════════════════════════ */

function AnimatedCounter({ value, suffix = '', duration = 1500 }) {
  const [count, setCount] = useState(0)
  useEffect(() => {
    let start = 0
    const end = typeof value === 'number' ? value : parseInt(value)
    if (isNaN(end)) { setCount(value); return }
    const step = Math.max(1, Math.floor(end / (duration / 16)))
    const timer = setInterval(() => {
      start += step
      if (start >= end) { setCount(end); clearInterval(timer) }
      else setCount(start)
    }, 16)
    return () => clearInterval(timer)
  }, [value, duration])
  return <>{typeof count === 'number' ? count.toLocaleString() : count}{suffix}</>
}

/* ═══════════════════════════════════════════════════════════ */
/* CandidateModal — Detailed view of a single candidate       */
/* ═══════════════════════════════════════════════════════════ */

function CandidateModal({ candidate, onClose }) {
  if (!candidate) return null
  const c = candidate
  const s = c.scores

  const dimensions = [
    { key: 'semantic_similarity', label: 'Semantic' },
    { key: 'title_alignment', label: 'Title Fit' },
    { key: 'career_quality', label: 'Career' },
    { key: 'skills_match', label: 'Skills' },
    { key: 'experience_band', label: 'Experience' },
    { key: 'location', label: 'Location' },
    { key: 'availability', label: 'Availability' },
    { key: 'engagement', label: 'Engagement' },
    { key: 'profile_trust', label: 'Trust' },
    { key: 'market_signals', label: 'Market' },
  ]

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>

        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span className={`rank-badge ${rankClass(c.rank)}`}>{c.rank}</span>
            <div>
              <h2>{c.name}</h2>
              <div className="subtitle">{c.title} at {c.company}</div>
            </div>
            <div style={{ marginLeft: 'auto' }}>
              <span className={`score-value ${scoreColor(c.score)}`} style={{ fontSize: '2rem' }}>
                {(c.score * 100).toFixed(1)}
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>/100</span>
            </div>
          </div>
          <div className="modal-meta">
            <span className="meta-item">📍 {c.location}, {c.country}</span>
            <span className="meta-item">🕐 {c.years_of_experience} years</span>
            <span className="meta-item">🆔 {c.candidate_id}</span>
          </div>
        </div>

        {/* Score Breakdown */}
        <h3 style={{ marginBottom: 8 }}>📊 Score Breakdown</h3>
        <div className="score-grid">
          {dimensions.map(d => (
            <div className="score-card" key={d.key}>
              <div className="label">{d.label}</div>
              <div className={`value ${scoreColor(s[d.key] || 0)}`}>
                {((s[d.key] || 0) * 100).toFixed(0)}
              </div>
            </div>
          ))}
        </div>

        {/* Skills */}
        {c.skills && c.skills.length > 0 && (
          <>
            <h3 style={{ marginTop: 24, marginBottom: 4 }}>🛠 Skills</h3>
            <div className="skills-tags">
              {c.skills.map(sk => <span className="skill-tag" key={sk}>{sk}</span>)}
            </div>
          </>
        )}

        {/* Career Timeline */}
        {c.career && c.career.length > 0 && (
          <>
            <h3 style={{ marginTop: 24, marginBottom: 4 }}>📈 Career</h3>
            <div className="career-timeline">
              {c.career.map((role, i) => (
                <div className={`career-item ${role.current ? 'current' : ''}`} key={i}>
                  <div className="role">{role.title}</div>
                  <div className="company">{role.company}</div>
                  <div className="duration">{role.duration}{role.current ? ' · Current' : ''}</div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Reasoning */}
        <h3 style={{ marginTop: 24, marginBottom: 4 }}>💡 Ranking Reasoning</h3>
        <div className="reasoning-box">{c.reasoning}</div>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════ */
/* Main App                                                   */
/* ═══════════════════════════════════════════════════════════ */

function App() {
  const [data, setData] = useState(null)
  const [selectedCandidate, setSelectedCandidate] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/results.json')
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { console.error(e); setLoading(false) })
  }, [])

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="stat-value" style={{ fontSize: '2rem' }}>Loading...</div>
          <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>Preparing ranking dashboard</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>Failed to load data</div>
  }

  const { metadata, jd_summary, jd_requirements, results } = data

  return (
    <>
      {/* ── HERO ── */}
      <section className="hero">
        <h1>RedRob AI Ranker</h1>
        <p className="tagline">
          Intelligent Candidate Discovery & Ranking — semantic understanding beyond keyword matching
        </p>

        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-value"><AnimatedCounter value={metadata.total_candidates} /></div>
            <div className="stat-label">Candidates Analyzed</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">10</div>
            <div className="stat-label">Scoring Dimensions</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">&lt;<AnimatedCounter value={Math.ceil(metadata.runtime_seconds / 60)} /></div>
            <div className="stat-label">Minutes Runtime</div>
          </div>
          <div className="stat-card">
            <div className="stat-value"><AnimatedCounter value={metadata.total_scored} /></div>
            <div className="stat-label">Top Ranked</div>
          </div>
        </div>
      </section>

      {/* ── ARCHITECTURE PIPELINE ── */}
      <section className="section">
        <h2 className="section-title"><span className="icon">🏗️</span> 3-Stage Hybrid Pipeline</h2>
        <div className="pipeline-flow">
          <div className="pipeline-stage">
            <div className="stage-number">1</div>
            <div className="stage-title">Coarse Filter</div>
            <div className="stage-desc">Honeypot detection, keyword-stuffer traps, services-only careers, non-tech titles</div>
            <div className="stage-flow">100K → ~5K</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">2</div>
            <div className="stage-title">Semantic Ranking</div>
            <div className="stage-desc">Sentence-BERT embeddings + cosine similarity + 10-dimensional feature scoring</div>
            <div className="stage-flow">~5K → Top 200</div>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">
            <div className="stage-number">3</div>
            <div className="stage-title">Fine Re-Ranking</div>
            <div className="stage-desc">Weighted composite scoring with title/career gating + reasoning generation</div>
            <div className="stage-flow">200 → Top 100</div>
          </div>
        </div>
      </section>

      {/* ── JD PANEL ── */}
      <section className="section">
        <h2 className="section-title"><span className="icon">📋</span> Job Description</h2>
        <div className="jd-panel">
          <div className="jd-text">{jd_summary}</div>
          <div className="jd-tags">
            {jd_requirements && jd_requirements.map((req, i) => (
              <span className="jd-tag" key={i}>{req}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── RANKED CANDIDATES ── */}
      <section className="section">
        <h2 className="section-title"><span className="icon">🏆</span> Ranked Candidates</h2>
        <div style={{ overflowX: 'auto' }}>
          <table className="candidates-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Candidate</th>
                <th>Title</th>
                <th>Company</th>
                <th>Location</th>
                <th>Experience</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {results.map(c => (
                <tr key={c.candidate_id} onClick={() => setSelectedCandidate(c)}>
                  <td><span className={`rank-badge ${rankClass(c.rank)}`}>{c.rank}</span></td>
                  <td style={{ fontWeight: 600 }}>{c.name}</td>
                  <td>{c.title}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{c.company}</td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{c.location}</td>
                  <td>{c.years_of_experience} yrs</td>
                  <td>
                    <div className="score-cell">
                      <span className={`score-value ${scoreColor(c.score)}`}>
                        {(c.score * 100).toFixed(1)}
                      </span>
                      <div className="score-bar-bg">
                        <div
                          className={`score-bar-fill ${scoreColor(c.score)}`}
                          style={{ width: `${c.score * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── MODAL ── */}
      {selectedCandidate && (
        <CandidateModal
          candidate={selectedCandidate}
          onClose={() => setSelectedCandidate(null)}
        />
      )}
    </>
  )
}

export default App
