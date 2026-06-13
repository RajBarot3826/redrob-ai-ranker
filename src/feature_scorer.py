"""
Feature scorer — Computes structured scores for each candidate based on
title alignment, career quality, skills match, experience band, location,
and education relevance.

These are the core "understanding" signals that go beyond keyword matching.
"""

from typing import Dict
from src.data_loader import Candidate
from src.config import (
    TITLE_TIERS, TITLE_DEFAULT_SCORE,
    SERVICES_COMPANIES, KNOWN_PRODUCT_COMPANIES,
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, RELEVANT_SKILLS, NON_TECH_SKILLS,
    PROFICIENCY_WEIGHT,
    IDEAL_EXP_MIN, IDEAL_EXP_MAX,
    PREFERRED_CITIES, GOOD_INDIAN_CITIES,
)


def _normalize(text: str) -> str:
    """Lowercase and strip text for matching."""
    return text.lower().strip()


def _skill_in_set(skill_name: str, skill_set: set) -> bool:
    """Check if a skill name matches any entry in a skill set (case-insensitive)."""
    name_lower = _normalize(skill_name)
    for target in skill_set:
        if target in name_lower or name_lower in target:
            return True
    return False


# ============================================================================
# Title Alignment Score
# ============================================================================

def score_title_alignment(candidate: Candidate) -> float:
    """
    Score how well the candidate's current title aligns with
    the Senior AI Engineer role. This is the primary anti-keyword-stuffer signal.

    Returns a score between 0.0 and 1.0.
    """
    title = _normalize(candidate.current_title)

    # Check career history titles too — look for ML/AI titles anywhere
    career_titles = [_normalize(e.title) for e in candidate.career_history]
    all_titles = [title] + career_titles

    best_score = 0.0
    matched_any = False

    for t in all_titles:
        for keywords, score in TITLE_TIERS:
            for kw in keywords:
                if kw in t:
                    matched_any = True
                    # Current title gets full weight, past titles get 70%
                    multiplier = 1.0 if t == title else 0.70
                    candidate_score = score * multiplier
                    best_score = max(best_score, candidate_score)

    # If no tier matched at all, use the default
    if not matched_any:
        best_score = TITLE_DEFAULT_SCORE

    # Boost: if current title is non-tech but has senior AI titles in past
    # They're probably transitioning — slight boost
    if best_score < 0.1 and any(
        any(kw in t for kw in ["ml", "ai", "machine learning", "data scientist"])
        for t in career_titles
    ):
        best_score = max(best_score, 0.25)

    return min(best_score, 1.0)


# ============================================================================
# Career Quality Score
# ============================================================================

def score_career_quality(candidate: Candidate) -> float:
    """
    Score career trajectory quality:
    - Product company experience vs services-only
    - Relevance of industry and roles
    - Career progression signals
    - Penalize entire-career-at-services (JD explicit disqualifier)

    Returns a score between 0.0 and 1.0.
    """
    if not candidate.career_history:
        return 0.1

    career = candidate.career_history
    total_months = sum(e.duration_months for e in career)
    if total_months == 0:
        return 0.1

    # Classify each role
    services_months = 0
    product_months = 0
    tech_role_months = 0
    ml_role_months = 0

    for entry in career:
        company_lower = _normalize(entry.company)
        title_lower = _normalize(entry.title)
        desc_lower = _normalize(entry.description)

        months = entry.duration_months

        # Company type classification
        is_services = any(sc in company_lower for sc in SERVICES_COMPANIES)
        is_product = any(pc in company_lower for pc in KNOWN_PRODUCT_COMPANIES)

        if is_services:
            services_months += months
        elif is_product:
            product_months += months
        else:
            # Unknown companies — treat as neutral/product (could be startups)
            product_months += months * 0.7

        # Role relevance
        is_tech_role = any(kw in title_lower for kw in [
            "engineer", "developer", "scientist", "architect",
            "swe", "sde", "programmer", "coder",
        ])
        is_ml_role = any(kw in title_lower for kw in [
            "ml", "ai", "machine learning", "data scientist",
            "nlp", "deep learning", "research",
        ])

        # Also check description for ML/production signals
        desc_has_ml = any(kw in desc_lower for kw in [
            "machine learning", "deep learning", "model training",
            "embeddings", "recommendation", "ranking", "retrieval",
            "nlp", "neural network", "transformer", "bert",
            "inference", "fine-tun", "feature engineer",
        ])
        desc_has_production = any(kw in desc_lower for kw in [
            "production", "deployed", "real users", "a/b test",
            "scale", "pipeline", "infrastructure", "monitoring",
            "microservice", "api", "latency", "throughput",
        ])

        if is_ml_role or desc_has_ml:
            ml_role_months += months
        elif is_tech_role or desc_has_production:
            tech_role_months += months

    # --- Compute sub-scores ---

    # Product vs services ratio
    services_ratio = services_months / max(total_months, 1)
    all_services = services_ratio > 0.95  # Entire career at services = penalty

    if all_services:
        company_score = 0.10  # Harsh penalty per JD
    elif services_ratio > 0.7:
        company_score = 0.25
    elif services_ratio > 0.4:
        company_score = 0.50
    else:
        company_score = 0.70 + 0.30 * (product_months / max(total_months, 1))

    # ML/AI role experience ratio
    ml_ratio = ml_role_months / max(total_months, 1)
    tech_ratio = (ml_role_months + tech_role_months) / max(total_months, 1)

    if ml_ratio > 0.5:
        role_score = 0.90 + 0.10 * min(ml_ratio, 1.0)
    elif ml_ratio > 0.25:
        role_score = 0.65 + 0.25 * ml_ratio
    elif tech_ratio > 0.5:
        role_score = 0.40 + 0.25 * tech_ratio
    else:
        role_score = 0.10 + 0.30 * tech_ratio

    # Career length and progression
    num_roles = len(career)
    if num_roles >= 2:
        # Check for progression (more senior titles over time)
        progression_score = 0.5  # Neutral
        recent_title = _normalize(career[0].title)
        if any(kw in recent_title for kw in ["senior", "lead", "staff", "principal"]):
            progression_score = 0.7
    else:
        progression_score = 0.4

    # Weighted combination
    score = (
        0.40 * company_score +
        0.45 * role_score +
        0.15 * progression_score
    )

    return min(max(score, 0.0), 1.0)


# ============================================================================
# Skills Match Score
# ============================================================================

def score_skills_match(candidate: Candidate) -> float:
    """
    Score how well the candidate's skills match the JD requirements.
    Uses weighted matching with proficiency, endorsements, and duration as trust signals.

    The JD explicitly warns about keyword stuffers — high AI skill count
    with non-tech titles is a RED FLAG, not a positive signal.

    Returns a score between 0.0 and 1.0.
    """
    if not candidate.skills:
        return 0.0

    must_have_score = 0.0
    must_have_max = 0.0
    nice_to_have_score = 0.0
    nice_to_have_max = 0.0
    relevant_score = 0.0
    relevant_max = 0.0
    non_tech_count = 0

    for skill in candidate.skills:
        name = skill.name
        prof_weight = PROFICIENCY_WEIGHT.get(skill.proficiency, 0.2)

        # Trust multiplier: endorsements + duration validate the skill claim
        endorse_trust = min(skill.endorsements / 20.0, 1.0) * 0.3
        duration_trust = min(skill.duration_months / 36.0, 1.0) * 0.3
        trust = 0.4 + endorse_trust + duration_trust  # 0.4 base + up to 0.6

        skill_value = prof_weight * trust

        if _skill_in_set(name, MUST_HAVE_SKILLS):
            must_have_score += skill_value
            must_have_max += 1.0
        elif _skill_in_set(name, NICE_TO_HAVE_SKILLS):
            nice_to_have_score += skill_value
            nice_to_have_max += 1.0
        elif _skill_in_set(name, RELEVANT_SKILLS):
            relevant_score += skill_value
            relevant_max += 1.0

        if _skill_in_set(name, NON_TECH_SKILLS):
            non_tech_count += 1

    # Normalize each category
    must_have_norm = (must_have_score / max(must_have_max, 3.0))  # Expect ~3 must-haves
    nice_to_have_norm = (nice_to_have_score / max(nice_to_have_max, 3.0))
    relevant_norm = (relevant_score / max(relevant_max, 3.0))

    # Cap at 1.0
    must_have_norm = min(must_have_norm, 1.0)
    nice_to_have_norm = min(nice_to_have_norm, 1.0)
    relevant_norm = min(relevant_norm, 1.0)

    # Weighted combination (must-haves most important)
    raw_score = (
        0.55 * must_have_norm +
        0.30 * nice_to_have_norm +
        0.15 * relevant_norm
    )

    # ── Anti-keyword-stuffer penalty ──
    # If candidate has many AI skills but a non-tech title,
    # the skills are likely stuffed / exaggerated
    title_lower = _normalize(candidate.current_title)
    is_non_tech_title = any(kw in title_lower for kw in [
        "marketing", "hr", "accountant", "sales",
        "customer support", "graphic designer", "content writer",
        "civil engineer", "mechanical engineer", "operations manager",
    ])

    if is_non_tech_title and must_have_score > 0:
        # Severe penalty: skills don't match career reality
        raw_score *= 0.15

    # Penalty for mostly non-tech skills
    total_skills = len(candidate.skills)
    if total_skills > 0:
        non_tech_ratio = non_tech_count / total_skills
        if non_tech_ratio > 0.7:
            raw_score *= 0.30

    # Check skill assessment validation
    assessments = candidate.redrob_signals.skill_assessment_scores
    if assessments:
        # Reward candidates who backed up claims with assessments
        avg_assessment = sum(assessments.values()) / len(assessments)
        assessment_bonus = min(avg_assessment / 100.0, 1.0) * 0.1
        raw_score += assessment_bonus

    return min(max(raw_score, 0.0), 1.0)


# ============================================================================
# Experience Band Score
# ============================================================================

def score_experience_band(candidate: Candidate) -> float:
    """
    Score how well experience years fit the 5-9 year sweet spot.
    The JD says this is flexible but 5-9 is where hires typically land.

    Returns a score between 0.0 and 1.0.
    """
    yrs = candidate.years_of_experience

    if IDEAL_EXP_MIN <= yrs <= IDEAL_EXP_MAX:
        return 1.0
    elif 4.0 <= yrs < IDEAL_EXP_MIN:
        return 0.75
    elif IDEAL_EXP_MAX < yrs <= 12.0:
        return 0.70
    elif 3.0 <= yrs < 4.0:
        return 0.50
    elif 12.0 < yrs <= 15.0:
        return 0.55
    elif 2.0 <= yrs < 3.0:
        return 0.30
    elif yrs > 15.0:
        return 0.40
    else:
        return 0.15  # < 2 years


# ============================================================================
# Location Score
# ============================================================================

def score_location(candidate: Candidate) -> float:
    """
    Score location fit. JD prefers Pune/Noida, India (hybrid).
    Willing to relocate is a bonus.

    Returns a score between 0.0 and 1.0.
    """
    location = _normalize(candidate.location)
    country = _normalize(candidate.country)
    willing = candidate.redrob_signals.willing_to_relocate

    # Check preferred cities
    if any(city in location for city in PREFERRED_CITIES):
        return 1.0

    # Check other good Indian cities
    if country == "india" or any(city in location for city in GOOD_INDIAN_CITIES):
        if any(city in location for city in GOOD_INDIAN_CITIES):
            return 0.85 if willing else 0.70
        # India but not a major city
        return 0.65 if willing else 0.50

    # Outside India
    if willing:
        return 0.30
    return 0.15


# ============================================================================
# Education Relevance Score
# ============================================================================

def score_education(candidate: Candidate) -> float:
    """
    Score education relevance (minor signal).
    CS/ML/AI degrees are preferred; tier matters slightly.

    Returns a score between 0.0 and 1.0.
    """
    if not candidate.education:
        return 0.3  # No education listed — neutral

    best_score = 0.0

    for edu in candidate.education:
        field = _normalize(edu.field_of_study)
        degree = _normalize(edu.degree)
        tier = edu.tier

        # Field relevance
        if any(kw in field for kw in [
            "machine learning", "artificial intelligence", "ai",
            "data science", "nlp", "natural language",
        ]):
            field_score = 1.0
        elif any(kw in field for kw in [
            "computer science", "computer engineering",
            "software engineering", "information technology",
            "computational", "informatics",
        ]):
            field_score = 0.8
        elif any(kw in field for kw in [
            "electrical", "electronics", "mathematics",
            "statistics", "physics",
        ]):
            field_score = 0.5
        elif any(kw in field for kw in [
            "mechanical", "civil", "chemical",
            "commerce", "arts", "humanities",
        ]):
            field_score = 0.2
        else:
            field_score = 0.3

        # Degree level bonus
        if any(d in degree for d in ["ph.d", "phd", "doctorate"]):
            degree_mult = 1.1
        elif any(d in degree for d in ["m.tech", "mtech", "m.s", "ms", "m.e", "me", "m.sc", "msc", "master"]):
            degree_mult = 1.0
        elif any(d in degree for d in ["b.tech", "btech", "b.e", "be", "b.s", "bs", "b.sc", "bsc", "bachelor"]):
            degree_mult = 0.9
        else:
            degree_mult = 0.8

        # Tier bonus (minor)
        tier_mult = {
            "tier_1": 1.15,
            "tier_2": 1.05,
            "tier_3": 0.95,
            "tier_4": 0.85,
            "unknown": 0.90,
        }.get(tier, 0.90)

        edu_score = field_score * degree_mult * tier_mult
        best_score = max(best_score, edu_score)

    return min(best_score, 1.0)


# ============================================================================
# Composite Feature Scorer
# ============================================================================

def compute_feature_scores(candidate: Candidate) -> Dict[str, float]:
    """
    Compute all structured feature scores for a candidate.

    Returns dict with all individual scores.
    """
    return {
        "title_alignment":  score_title_alignment(candidate),
        "career_quality":   score_career_quality(candidate),
        "skills_match":     score_skills_match(candidate),
        "experience_band":  score_experience_band(candidate),
        "location":         score_location(candidate),
        "education":        score_education(candidate),
    }
