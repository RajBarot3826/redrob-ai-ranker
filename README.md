# RedRob AI Ranker — Intelligent Candidate Discovery & Ranking System

> **India Runs Data & AI Challenge** — Hack2Skill × Redrob AI  
> Ranks 100K candidates for the Senior AI Engineer role using semantic understanding,
> not keyword matching.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRE-COMPUTATION (one-time, ~20 min)              │
│  all-MiniLM-L6-v2 embeddings for 100K candidates + JD             │
│  Saved to: precomputed/embeddings.npy, jd_embedding.npy           │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│              STAGE 1: COARSE FILTER (< 30 sec)                     │
│  10-rule honeypot detector (catches ~80 impossible profiles)       │
│  Title-based gating (penalizes keyword-stuffer traps)             │
│  Career quality gate (penalizes services-only careers)            │
│  100K → viable candidates                                         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│          STAGE 2: 10-DIMENSIONAL SCORING (< 2 min)                 │
│  Semantic similarity (cosine, pre-computed embeddings)             │
│  + Title alignment · Career quality · Skills match                │
│  + Experience band · Location · Education                         │
│  + Availability · Engagement · Profile trust · Market signals     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│          STAGE 3: FINAL RANKING + REASONING (< 30 sec)             │
│  Weighted composite score with multiplicative gating              │
│  Per-candidate reasoning (specific, honest, varied)               │
│  → Top 100 candidates with scores + reasoning                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.10
pip install -r requirements.txt
```

### Step 1: Pre-compute embeddings (one-time, ~20 min)
```bash
python scripts/precompute_embeddings.py \
    --candidates ./candidates.jsonl \
    --jd ./data/jd_query.txt \
    --output ./precomputed
```

### Step 2: Generate submission (< 5 min)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Step 3: Validate
```bash
python validate_submission.py submission.csv
```

## 📁 Project Structure

```
├── rank.py                          # Main CLI entry point
├── app.py                           # Streamlit sandbox demo
├── requirements.txt                 # Python dependencies
├── submission_metadata.yaml         # Hackathon submission metadata
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # All JD-specific constants & weights
│   ├── data_loader.py               # JSONL streaming loader
│   ├── honeypot_detector.py         # 10-rule honeypot detection
│   ├── feature_scorer.py            # Title, career, skills, experience, location
│   ├── behavioral_scorer.py         # Availability, engagement, trust, market
│   ├── embedder.py                  # Sentence-BERT embeddings + FAISS
│   ├── ranker.py                    # 3-stage ranking pipeline
│   └── reasoning_generator.py       # Per-candidate reasoning
│
├── scripts/
│   └── precompute_embeddings.py     # Pre-compute 100K embeddings
│
├── data/
│   └── jd_query.txt                 # Condensed JD for embedding
│
├── precomputed/                     # (generated)
│   ├── embeddings.npy
│   ├── jd_embedding.npy
│   └── candidate_ids.json
│
└── frontend/                        # React dashboard (presentation)
```

## 🎯 Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Semantic Similarity | 25% | Embedding-based JD-candidate match |
| Title Alignment | 20% | Current/past role relevance (anti-keyword-stuffer) |
| Career Quality | 15% | Product vs services companies, ML experience |
| Skills Match | 15% | Skill relevance with trust validation |
| Experience Band | 5% | Fit within 5-9 year sweet spot |
| Location | 5% | Pune/Noida/India preference |
| Availability | 5% | Recency, notice period, open-to-work |
| Engagement | 5% | Response rate, interview completion |
| Profile Trust | 3% | Verification, GitHub, completeness |
| Market Signals | 2% | Recruiter saves, search appearances |

## 🛡️ Anti-Trap Measures

1. **Keyword Stuffer Detection**: Non-tech titles with AI skills → 85-90% score penalty
2. **Honeypot Detection**: 10 rules catching impossible profiles (expert skills with 0 months, date paradoxes, etc.)
3. **Services-Only Career Penalty**: Entire career at TCS/Infosys/Wipro → heavy penalty (per JD)
4. **Behavioral Gating**: Inactive 6+ months or <10% response rate → significant downweight
5. **Title-Career Gating**: Multiplicative (not additive) — prevents traps from sneaking through

## 🖥️ Sandbox Demo

Run the Streamlit app locally:
```bash
streamlit run app.py
```

Upload `sample_candidates.json` from the hackathon bundle to see the ranking in action.

## ⏱️ Performance

| Metric | Value |
|--------|-------|
| Pre-computation (one-time) | ~20 min |
| Ranking (100K candidates) | ~2-3 min |
| Memory usage | ~3 GB |
| Embedding model | all-MiniLM-L6-v2 (22M params) |
| Embedding dimensions | 384 |

## 📝 License

Built for the India Runs Data & AI Challenge by Redrob AI × Hack2Skill.
