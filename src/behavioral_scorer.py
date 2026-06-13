"""
Behavioral scorer — Scores candidates based on Redrob platform signals.
These indicate whether a candidate is actually reachable, engaged, and hireable.

The JD explicitly says: "A perfect-on-paper candidate who hasn't logged in
for 6 months and has a 5% recruiter response rate is, for hiring purposes,
not actually available. Down-weight them appropriately."
"""

from datetime import date, datetime
from typing import Dict
from src.data_loader import Candidate
from src.config import REFERENCE_DATE


def _parse_date(date_str: str) -> date:
    """Parse a date string to a date object. Returns epoch on failure."""
    if not date_str:
        return date(2020, 1, 1)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return date(2020, 1, 1)


def _days_since(date_str: str) -> int:
    """Days between the date and REFERENCE_DATE."""
    d = _parse_date(date_str)
    return max((REFERENCE_DATE - d).days, 0)


# ============================================================================
# Availability Score
# ============================================================================

def score_availability(candidate: Candidate) -> float:
    """
    How available and reachable is this candidate?

    Signals: open_to_work, last_active_date, notice_period, preferred_work_mode.
    """
    signals = candidate.redrob_signals
    score = 0.0

    # Open to work flag (strong signal)
    if signals.open_to_work_flag:
        score += 0.25

    # Recency of activity (critical — inactive = unreachable)
    days_inactive = _days_since(signals.last_active_date)
    if days_inactive <= 7:
        score += 0.30   # Active in last week
    elif days_inactive <= 30:
        score += 0.25   # Active in last month
    elif days_inactive <= 90:
        score += 0.15   # Active in last quarter
    elif days_inactive <= 180:
        score += 0.08   # Semi-active
    else:
        score += 0.02   # Dormant — big penalty

    # Notice period (JD prefers <30 days, can buy out up to 30)
    notice = signals.notice_period_days
    if notice <= 15:
        score += 0.25
    elif notice <= 30:
        score += 0.20   # Sweet spot for JD
    elif notice <= 60:
        score += 0.12
    elif notice <= 90:
        score += 0.07
    else:
        score += 0.03   # 90+ days = harder to hire

    # Preferred work mode alignment (JD is hybrid, Pune/Noida)
    mode = signals.preferred_work_mode
    if mode in ("hybrid", "flexible"):
        score += 0.15
    elif mode == "onsite":
        score += 0.12
    elif mode == "remote":
        score += 0.08

    # Willing to relocate bonus
    if signals.willing_to_relocate:
        score += 0.05

    return min(score, 1.0)


# ============================================================================
# Engagement Score
# ============================================================================

def score_engagement(candidate: Candidate) -> float:
    """
    How engaged is this candidate with the recruiting process?

    High engagement = actually responsive and reliable.
    """
    signals = candidate.redrob_signals
    score = 0.0

    # Recruiter response rate (most important engagement signal)
    rr = signals.recruiter_response_rate
    if rr >= 0.70:
        score += 0.35
    elif rr >= 0.50:
        score += 0.25
    elif rr >= 0.30:
        score += 0.15
    elif rr >= 0.10:
        score += 0.08
    else:
        score += 0.02   # <10% = effectively unreachable

    # Response time
    rt = signals.avg_response_time_hours
    if rt <= 12:
        score += 0.20
    elif rt <= 24:
        score += 0.18
    elif rt <= 48:
        score += 0.14
    elif rt <= 72:
        score += 0.10
    elif rt <= 168:  # Within a week
        score += 0.05
    else:
        score += 0.02

    # Interview completion rate
    icr = signals.interview_completion_rate
    if icr >= 0.80:
        score += 0.20
    elif icr >= 0.60:
        score += 0.15
    elif icr >= 0.40:
        score += 0.10
    else:
        score += 0.03

    # Offer acceptance rate
    oar = signals.offer_acceptance_rate
    if oar >= 0:  # Has offer history
        if oar >= 0.70:
            score += 0.15
        elif oar >= 0.40:
            score += 0.10
        else:
            score += 0.05
    else:
        score += 0.05  # No history — neutral

    # Applications submitted (shows active job search)
    apps = signals.applications_submitted_30d
    if 1 <= apps <= 10:
        score += 0.10  # Actively looking but not desperately spray-and-pray
    elif apps > 10:
        score += 0.05  # Too many = less targeted
    else:
        score += 0.03

    return min(score, 1.0)


# ============================================================================
# Profile Trust Score
# ============================================================================

def score_profile_trust(candidate: Candidate) -> float:
    """
    How trustworthy / credible is this profile?

    Verification signals + completeness + external validation.
    """
    signals = candidate.redrob_signals
    score = 0.0

    # Profile completeness
    completeness = signals.profile_completeness_score / 100.0
    score += 0.25 * completeness

    # Verification flags
    if signals.verified_email:
        score += 0.12
    if signals.verified_phone:
        score += 0.12
    if signals.linkedin_connected:
        score += 0.15

    # GitHub activity (strong external validation for engineering roles)
    github = signals.github_activity_score
    if github >= 0:  # Has GitHub linked
        if github >= 60:
            score += 0.20
        elif github >= 30:
            score += 0.15
        elif github >= 10:
            score += 0.10
        else:
            score += 0.05
    else:
        score += 0.02  # No GitHub — slightly negative for engineering role

    # Endorsements received (social validation)
    endorsements = signals.endorsements_received
    if endorsements >= 50:
        score += 0.10
    elif endorsements >= 20:
        score += 0.07
    elif endorsements >= 5:
        score += 0.04
    else:
        score += 0.01

    # Connection count (network size)
    connections = signals.connection_count
    if connections >= 500:
        score += 0.06
    elif connections >= 200:
        score += 0.04
    elif connections >= 50:
        score += 0.02

    return min(score, 1.0)


# ============================================================================
# Market Signals Score
# ============================================================================

def score_market_signals(candidate: Candidate) -> float:
    """
    Market demand indicators — how much are recruiters already interested?

    These are proxy signals for candidate quality from the market.
    """
    signals = candidate.redrob_signals
    score = 0.0

    # Saved by recruiters (strong demand signal)
    saved = signals.saved_by_recruiters_30d
    if saved >= 20:
        score += 0.40
    elif saved >= 10:
        score += 0.30
    elif saved >= 5:
        score += 0.20
    elif saved >= 1:
        score += 0.10
    else:
        score += 0.02

    # Search appearances (visibility)
    appearances = signals.search_appearance_30d
    if appearances >= 200:
        score += 0.25
    elif appearances >= 100:
        score += 0.20
    elif appearances >= 50:
        score += 0.15
    elif appearances >= 20:
        score += 0.10
    else:
        score += 0.03

    # Profile views (interest)
    views = signals.profile_views_received_30d
    if views >= 30:
        score += 0.20
    elif views >= 15:
        score += 0.15
    elif views >= 5:
        score += 0.10
    elif views >= 1:
        score += 0.05
    else:
        score += 0.02

    # Account tenure (established presence)
    days_on_platform = _days_since(signals.signup_date)
    if days_on_platform >= 365:
        score += 0.15
    elif days_on_platform >= 180:
        score += 0.10
    elif days_on_platform >= 90:
        score += 0.07
    else:
        score += 0.03

    return min(score, 1.0)


# ============================================================================
# Composite Behavioral Scorer
# ============================================================================

def compute_behavioral_scores(candidate: Candidate) -> Dict[str, float]:
    """
    Compute all behavioral signal scores for a candidate.
    """
    return {
        "availability":    score_availability(candidate),
        "engagement":      score_engagement(candidate),
        "profile_trust":   score_profile_trust(candidate),
        "market_signals":  score_market_signals(candidate),
    }
