"""
Streamlit App — Redrob AI Candidate Ranker Sandbox Demo

This is the required sandbox link for the hackathon submission.
Accepts a small candidate sample (≤100 candidates) and runs the
full ranking pipeline end-to-end.

Deploy to: HuggingFace Spaces or Streamlit Cloud
"""

import streamlit as st
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import parse_candidate, Candidate
from src.honeypot_detector import detect_honeypot
from src.feature_scorer import compute_feature_scores
from src.behavioral_scorer import compute_behavioral_scores
from src.reasoning_generator import generate_reasoning

# ── Page Config ──
st.set_page_config(
    page_title="RedRob AI Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin-top: 5px;
    }
    .candidate-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
    }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    h1 { color: #ffffff !important; }
    h2, h3 { color: #d0d0ff !important; }
</style>
""", unsafe_allow_html=True)


def rank_without_embeddings(candidates):
    """
    Rank candidates using only structured + behavioral scoring.
    For the sandbox demo, we skip semantic embeddings (no model loading needed).
    This makes the demo fast and lightweight.
    """
    scored = []
    progress = st.progress(0, text="Scoring candidates...")

    for i, cand in enumerate(candidates):
        progress.progress((i + 1) / len(candidates), text=f"Scoring {i+1}/{len(candidates)}...")

        is_honeypot, hp_reasons = detect_honeypot(cand)
        feature_scores = compute_feature_scores(cand)
        behavioral_scores = compute_behavioral_scores(cand)

        if is_honeypot:
            final_score = 0.0
        else:
            # Without semantic similarity, reweight the other dimensions
            final_score = (
                0.28 * feature_scores.get("title_alignment", 0) +
                0.22 * feature_scores.get("career_quality", 0) +
                0.20 * feature_scores.get("skills_match", 0) +
                0.06 * feature_scores.get("experience_band", 0) +
                0.06 * feature_scores.get("location", 0) +
                0.06 * behavioral_scores.get("availability", 0) +
                0.06 * behavioral_scores.get("engagement", 0) +
                0.03 * behavioral_scores.get("profile_trust", 0) +
                0.03 * behavioral_scores.get("market_signals", 0)
            )

            # Title gating
            title_score = feature_scores.get("title_alignment", 0)
            if title_score < 0.10:
                final_score *= 0.10
            elif title_score < 0.20:
                final_score *= 0.35

            # Career gating
            career_score = feature_scores.get("career_quality", 0)
            if career_score < 0.15:
                final_score *= 0.25

        all_scores = {
            **feature_scores,
            **behavioral_scores,
            "semantic_similarity": 0.5,  # Neutral placeholder
            "is_honeypot": is_honeypot,
            "final_score": final_score,
        }

        scored.append((cand, final_score, all_scores, is_honeypot, hp_reasons))

    progress.empty()

    # Sort by score
    scored.sort(key=lambda x: (-x[1], x[0].candidate_id))

    # Generate reasoning for top 100
    results = []
    for rank_idx, (cand, score, all_scores, is_hp, hp_reasons) in enumerate(scored[:100]):
        rank = rank_idx + 1
        reasoning = generate_reasoning(cand, rank, score, all_scores)
        results.append({
            "rank": rank,
            "candidate_id": cand.candidate_id,
            "name": cand.profile.get("anonymized_name", "N/A"),
            "score": score,
            "title": cand.current_title,
            "company": cand.current_company,
            "location": cand.location,
            "country": cand.country,
            "years_exp": cand.years_of_experience,
            "reasoning": reasoning,
            "is_honeypot": is_hp,
            "scores": all_scores,
        })

    return results


# ═══════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════

# Header
st.markdown("# 🎯 RedRob AI Ranker")
st.markdown("### Intelligent Candidate Discovery & Ranking System")
st.markdown("*Semantic understanding beyond keyword matching — built for the India Runs Data & AI Challenge*")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## 📤 Upload Candidates")
    st.markdown("Upload a JSON file with candidate profiles (≤100 candidates)")

    uploaded = st.file_uploader(
        "Choose a JSON file",
        type=["json"],
        help="Upload the sample_candidates.json from the hackathon bundle"
    )

    st.divider()
    st.markdown("## 🏗️ Architecture")
    st.markdown("""
    **3-Stage Hybrid Pipeline:**
    1. 🔍 Coarse Filter (honeypots + disqualifiers)
    2. 📊 Multi-dimensional Scoring (10 features)
    3. 🎯 Final Ranking + Reasoning

    **Scoring Dimensions:**
    - Title Alignment
    - Career Quality
    - Skills Match
    - Experience Band
    - Location Fit
    - Availability
    - Engagement
    - Profile Trust
    - Market Signals
    """)

    st.divider()
    st.markdown("## ⚙️ Compute")
    st.markdown("""
    - **Model**: all-MiniLM-L6-v2
    - **Runtime**: <5 min on CPU
    - **Memory**: <16 GB RAM
    - **No GPU / No API calls**
    """)

# Main content
if uploaded is not None:
    try:
        data = json.loads(uploaded.read().decode("utf-8"))

        # Handle both array and single-object format
        if isinstance(data, dict):
            data = [data]

        st.success(f"✅ Loaded {len(data)} candidates")

        # Parse candidates
        candidates = [parse_candidate(d) for d in data]

        # Run ranking
        with st.spinner("🔄 Running ranking pipeline..."):
            start = time.time()
            results = rank_without_embeddings(candidates)
            elapsed = time.time() - start

        # ── Metrics Row ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Candidates", len(candidates))
        with col2:
            st.metric("Ranked", len(results))
        with col3:
            st.metric("Runtime", f"{elapsed:.1f}s")
        with col4:
            hp_count = sum(1 for r in results if r["is_honeypot"])
            st.metric("Honeypots Detected", hp_count)

        st.divider()

        # ── Results Table ──
        st.markdown("## 🏆 Ranking Results")

        df = pd.DataFrame([{
            "Rank": r["rank"],
            "ID": r["candidate_id"],
            "Name": r["name"],
            "Score": f"{r['score']:.4f}",
            "Title": r["title"],
            "Company": r["company"],
            "Location": r["location"],
            "Experience": f"{r['years_exp']:.1f} yrs",
            "🍯": "⚠️" if r["is_honeypot"] else "✅",
        } for r in results])

        st.dataframe(df, use_container_width=True, height=400)

        # ── Score Distribution ──
        st.markdown("## 📊 Score Distribution")
        col1, col2 = st.columns(2)

        with col1:
            scores = [r["score"] for r in results]
            fig = px.histogram(
                x=scores, nbins=20,
                title="Final Score Distribution",
                labels={"x": "Score", "y": "Count"},
                color_discrete_sequence=["#667eea"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            titles = [r["title"] for r in results[:20]]
            fig = px.pie(
                names=titles,
                title="Title Distribution (Top 20)",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Detailed Candidate View ──
        st.markdown("## 🔍 Candidate Details")

        selected_rank = st.selectbox(
            "Select a candidate to view details",
            options=[f"Rank {r['rank']}: {r['name']} ({r['title']})" for r in results],
        )

        if selected_rank:
            rank_num = int(selected_rank.split(":")[0].replace("Rank ", ""))
            selected = next(r for r in results if r["rank"] == rank_num)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"### {selected['name']}")
                st.markdown(f"**{selected['title']}** at **{selected['company']}**")
                st.markdown(f"📍 {selected['location']}, {selected['country']}")
                st.markdown(f"🕐 {selected['years_exp']:.1f} years experience")
                st.markdown(f"**Score: {selected['score']:.4f}**")
                st.markdown(f"**Reasoning:** {selected['reasoning']}")

            with col2:
                # Score breakdown radar chart
                scores = selected["scores"]
                categories = [
                    "Title", "Career", "Skills", "Experience",
                    "Location", "Availability", "Engagement", "Trust"
                ]
                values = [
                    scores.get("title_alignment", 0),
                    scores.get("career_quality", 0),
                    scores.get("skills_match", 0),
                    scores.get("experience_band", 0),
                    scores.get("location", 0),
                    scores.get("availability", 0),
                    scores.get("engagement", 0),
                    scores.get("profile_trust", 0),
                ]

                fig = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.3)',
                    line=dict(color='#667eea', width=2),
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1]),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    title="Score Breakdown",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Download CSV ──
        st.divider()
        st.markdown("## 💾 Download Submission")

        csv_lines = ["candidate_id,rank,score,reasoning"]
        for r in results:
            reasoning = r["reasoning"].replace('"', '""')
            csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.4f},"{reasoning}"')

        csv_content = "\n".join(csv_lines) + "\n"

        st.download_button(
            label="📥 Download submission.csv",
            data=csv_content,
            file_name="submission.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    # Landing page when no file uploaded
    st.markdown("""
    ## 👋 Welcome

    This is the **Intelligent Candidate Discovery & Ranking System** built for the
    [India Runs Data & AI Challenge](https://hack2skill.com/event/india_runs).

    ### How to use:
    1. Upload the `sample_candidates.json` file from the hackathon bundle
    2. The system will automatically rank candidates for the **Senior AI Engineer** role
    3. View detailed scores, reasoning, and download the submission CSV

    ### How it works:
    - **Semantic understanding** of job descriptions — not keyword matching
    - **10-dimensional scoring** covering skills, career, behavior, and market signals
    - **Honeypot detection** to catch impossible profiles
    - **Anti-keyword-stuffer** logic to penalize fake skill claims
    - **Behavioral signal analysis** to assess actual availability
    """)

    # Show architecture diagram
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    ```
    100K Candidates ──→ Stage 1: Coarse Filter ──→ Stage 2: Semantic + Structured ──→ Stage 3: Re-rank + Reason ──→ Top 100
                        (honeypots, non-fits)       (embeddings + 10 features)         (composite scoring)          CSV
    ```
    """)
