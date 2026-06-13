"""
Reasoning generator — Produces per-candidate justification text.
Each reasoning must be:
  - Specific to the candidate's actual profile data
  - Connected to JD requirements
  - Honest about gaps
  - Substantively different from other candidates (no templates)
  - Consistent with the assigned rank
"""

from src.data_loader import Candidate
from src.config import SERVICES_COMPANIES, PREFERRED_CITIES, GOOD_INDIAN_CITIES


def _normalize(text: str) -> str:
    return text.lower().strip()


def _get_relevant_skills(candidate: Candidate) -> list:
    """Extract the most relevant AI/ML skills from the candidate."""
    ai_keywords = {
        "ml", "ai", "machine learning", "deep learning", "nlp",
        "natural language", "embeddings", "sentence-transformers",
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "pytorch", "tensorflow",
        "transformers", "bert", "gpt", "llm", "rag",
        "xgboost", "lightgbm", "recommendation", "ranking",
        "information retrieval", "semantic search", "vector",
        "fine-tuning", "lora", "qlora", "peft",
        "data science", "neural network", "feature engineering",
        "python", "spark", "airflow", "docker", "kubernetes",
        "mlops", "mlflow", "wandb",
    }
    relevant = []
    for skill in candidate.skills:
        name_lower = _normalize(skill.name)
        if any(kw in name_lower for kw in ai_keywords):
            relevant.append(skill.name)
    return relevant[:8]  # Cap at 8 for brevity


def _is_services_only(candidate: Candidate) -> bool:
    """Check if entire career is at services/consulting companies."""
    if not candidate.career_history:
        return False
    for entry in candidate.career_history:
        company_lower = _normalize(entry.company)
        if not any(sc in company_lower for sc in SERVICES_COMPANIES):
            return False
    return True


def _get_location_fit(candidate: Candidate) -> str:
    """Describe location fit."""
    location = _normalize(candidate.location)
    country = _normalize(candidate.country)
    willing = candidate.redrob_signals.willing_to_relocate

    if any(city in location for city in PREFERRED_CITIES):
        return "located in preferred hiring region"
    elif any(city in location for city in GOOD_INDIAN_CITIES):
        base = "in India"
        return f"{base}, willing to relocate" if willing else base
    elif country == "india":
        return "in India, willing to relocate" if willing else "in India"
    else:
        return f"based outside India ({candidate.country})" + (", open to relocation" if willing else "")


def _describe_career_highlights(candidate: Candidate) -> str:
    """Describe key career aspects."""
    parts = []
    career = candidate.career_history

    if career:
        current = career[0]
        parts.append(f"currently {current.title} at {current.company}")

        # Check for ML/AI in descriptions
        for entry in career:
            desc_lower = _normalize(entry.description)
            if any(kw in desc_lower for kw in [
                "machine learning", "ranking", "retrieval", "embedding",
                "recommendation", "nlp", "deep learning", "model",
                "inference", "pipeline", "feature engineer",
            ]):
                if entry.company != current.company:
                    parts.append(f"prior ML-relevant work at {entry.company}")
                break

    return "; ".join(parts) if parts else "limited career information"


def generate_reasoning(
    candidate: Candidate,
    rank: int,
    final_score: float,
    scores: dict,
) -> str:
    """
    Generate a 1-2 sentence reasoning for why this candidate is ranked here.

    Requirements (from submission_spec.md Stage 4):
    - Reference specific facts from the candidate's profile
    - Connect to specific JD requirements
    - Acknowledge honest concerns where they exist
    - No hallucination (only mention what's actually in the profile)
    - Must be substantively different from other candidates
    - Tone must match the rank
    """
    yrs = candidate.years_of_experience
    title = candidate.current_title
    company = candidate.current_company
    relevant_skills = _get_relevant_skills(candidate)
    location_fit = _get_location_fit(candidate)
    signals = candidate.redrob_signals

    # Build the reasoning based on rank tier
    parts = []

    # Core identification
    parts.append(f"{title} with {yrs:.1f} yrs experience at {company}")

    # Top 10: Emphasize strengths
    if rank <= 10:
        if relevant_skills:
            parts.append(f"strong AI/ML skill set ({', '.join(relevant_skills[:5])})")
        career_desc = _describe_career_highlights(candidate)
        if career_desc:
            parts.append(career_desc)
        parts.append(location_fit)
        if signals.recruiter_response_rate >= 0.5:
            parts.append(f"highly responsive ({signals.recruiter_response_rate:.0%} recruiter response rate)")
        if signals.notice_period_days <= 30:
            parts.append(f"{signals.notice_period_days}-day notice period")

    # Top 11-30: Balanced view
    elif rank <= 30:
        if relevant_skills:
            parts.append(f"relevant skills: {', '.join(relevant_skills[:4])}")
        parts.append(location_fit)
        # Note concerns
        if _is_services_only(candidate):
            parts.append("note: career primarily at services companies")
        if signals.recruiter_response_rate < 0.3:
            parts.append(f"low recruiter response rate ({signals.recruiter_response_rate:.0%})")
        elif signals.recruiter_response_rate >= 0.4:
            parts.append(f"responsive ({signals.recruiter_response_rate:.0%} response rate)")

    # Top 31-60: More concerns noted
    elif rank <= 60:
        if relevant_skills:
            parts.append(f"has {len(relevant_skills)} relevant skills")
        else:
            parts.append("limited directly relevant AI/ML skills")
        if _is_services_only(candidate):
            parts.append("entire career at services/consulting firms")
        if signals.recruiter_response_rate < 0.2:
            parts.append(f"low responsiveness ({signals.recruiter_response_rate:.0%})")
        days_inactive = max((Candidate._parse_date_safe(signals.last_active_date) - Candidate._parse_date_safe("2026-06-01")).days, 0) if hasattr(Candidate, '_parse_date_safe') else 0
        if yrs < 4 or yrs > 12:
            parts.append(f"experience ({yrs:.1f} yrs) outside ideal 5-9yr band")

    # Top 61-100: Honest about why they're lower
    else:
        if relevant_skills:
            parts.append(f"some relevant skills ({', '.join(relevant_skills[:3])})")
        else:
            parts.append("few directly relevant AI/ML skills")

        # Note main gaps
        gaps = []
        if _is_services_only(candidate):
            gaps.append("services-only career")
        if yrs < 3:
            gaps.append(f"limited experience ({yrs:.1f} yrs)")
        elif yrs > 15:
            gaps.append(f"overqualified ({yrs:.1f} yrs)")
        if signals.recruiter_response_rate < 0.15:
            gaps.append("very low responsiveness")
        title_lower = _normalize(title)
        if any(kw in title_lower for kw in ["marketing", "hr", "accountant", "sales", "customer support"]):
            gaps.append("non-technical current role")

        if gaps:
            parts.append("concerns: " + ", ".join(gaps))

    # Assemble into 1-2 sentences
    reasoning = "; ".join(parts) + "."

    # Ensure reasonable length (1-2 sentences, not too long)
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning


# Standalone helper for date parsing in reasoning (avoid circular import)
def _parse_date_safe(date_str: str):
    """Parse date string safely."""
    from datetime import datetime, date as date_type
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return date_type(2020, 1, 1)
