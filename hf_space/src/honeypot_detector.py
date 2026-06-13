"""
Honeypot detector — identifies ~80 candidates with subtly impossible profiles.
These candidates have fabricated data that a naive keyword matcher would rank highly.
>10% honeypot rate in top-100 = DISQUALIFICATION.

Detection rules based on the submission_spec.md description:
- "8 years of experience at a company founded 3 years ago"
- "expert proficiency in 10 skills with 0 years used"
- Impossible date/duration combinations
"""

from typing import List, Dict, Tuple
from src.data_loader import Candidate


def detect_honeypot(candidate: Candidate) -> Tuple[bool, List[str]]:
    """
    Check if a candidate is a honeypot.

    Returns:
        (is_honeypot: bool, reasons: List[str])
    """
    red_flags = []

    # ── Rule 1: Expert proficiency with 0 months duration ──
    expert_zero_duration = 0
    for skill in candidate.skills:
        if skill.proficiency == "expert" and skill.duration_months == 0:
            expert_zero_duration += 1
    if expert_zero_duration >= 3:
        red_flags.append(
            f"Expert proficiency with 0 months duration in {expert_zero_duration} skills"
        )

    # ── Rule 2: Many advanced/expert skills with 0 endorsements ──
    advanced_zero_endorsements = 0
    for skill in candidate.skills:
        if skill.proficiency in ("expert", "advanced") and skill.endorsements == 0:
            advanced_zero_endorsements += 1
    if advanced_zero_endorsements >= 5:
        red_flags.append(
            f"{advanced_zero_endorsements} advanced/expert skills with 0 endorsements"
        )

    # ── Rule 3: Career duration exceeds years_of_experience significantly ──
    total_career_months = sum(e.duration_months for e in candidate.career_history)
    reported_years = candidate.years_of_experience
    career_years = total_career_months / 12.0
    if reported_years > 0 and career_years > 0:
        # If total career months (non-overlapping) is way more than reported experience
        if career_years > reported_years * 2.0 and career_years - reported_years > 5:
            red_flags.append(
                f"Career months ({total_career_months}) imply {career_years:.1f} yrs "
                f"but reported only {reported_years} yrs"
            )

    # ── Rule 4: Duration impossibly long for a single role ──
    for entry in candidate.career_history:
        if entry.duration_months > 360:  # > 30 years in one role
            red_flags.append(
                f"Impossible duration: {entry.duration_months} months at {entry.company}"
            )

    # ── Rule 5: Start date after end date ──
    for entry in candidate.career_history:
        if entry.end_date and entry.start_date:
            if entry.end_date < entry.start_date:
                red_flags.append(
                    f"End date before start date at {entry.company}: "
                    f"{entry.start_date} to {entry.end_date}"
                )

    # ── Rule 6: Too many expert skills relative to experience ──
    expert_count = sum(1 for s in candidate.skills if s.proficiency == "expert")
    if expert_count >= 8 and reported_years < 3:
        red_flags.append(
            f"{expert_count} expert skills but only {reported_years} years experience"
        )

    # ── Rule 7: High skill assessment scores with beginner proficiency ──
    assessment_scores = candidate.redrob_signals.skill_assessment_scores
    for skill in candidate.skills:
        skill_name = skill.name
        if skill_name in assessment_scores:
            score = assessment_scores[skill_name]
            if skill.proficiency == "beginner" and score > 90:
                red_flags.append(
                    f"Assessment score {score} but beginner proficiency for {skill_name}"
                )

    # ── Rule 8: Contradictory title vs career description ──
    # Detect when current_title completely mismatches all career descriptions
    current_title_lower = candidate.current_title.lower()
    if candidate.career_history:
        current_role = candidate.career_history[0]  # Most recent
        desc_lower = current_role.description.lower()

        # Title says one thing, description describes completely different field
        title_is_tech = any(kw in current_title_lower for kw in [
            "engineer", "developer", "scientist", "ml", "ai", "data"
        ])
        desc_is_nontech = any(kw in desc_lower for kw in [
            "accounting", "marketing campaign", "sales quota", "recruitment",
            "brand identity", "packaging design", "customer support team lead"
        ]) and not any(kw in desc_lower for kw in [
            "machine learning", "deep learning", "model", "algorithm",
            "neural", "embedding", "pipeline", "inference"
        ])

        if title_is_tech and desc_is_nontech:
            red_flags.append(
                f"Title '{candidate.current_title}' but description is non-technical"
            )

    # ── Rule 9: Implausible combination — very high metrics but very low activity ──
    signals = candidate.redrob_signals
    if (signals.saved_by_recruiters_30d > 50 and
            signals.profile_views_received_30d == 0):
        red_flags.append(
            f"50+ recruiter saves but 0 profile views in 30 days"
        )

    # ── Rule 10: Skill duration exceeds total experience ──
    for skill in candidate.skills:
        if skill.duration_months > 0 and reported_years > 0:
            max_possible_months = reported_years * 12 + 24  # Some buffer
            if skill.duration_months > max_possible_months * 1.5:
                red_flags.append(
                    f"Skill '{skill.name}' duration {skill.duration_months} months "
                    f"exceeds possible experience ({reported_years} yrs)"
                )

    # ── Threshold: 2+ red flags = honeypot ──
    is_honeypot = len(red_flags) >= 2

    return is_honeypot, red_flags


def flag_honeypots(candidates: List[Candidate]) -> Dict[str, List[str]]:
    """
    Flag all honeypot candidates in the dataset.

    Returns:
        Dict mapping candidate_id → list of red flag reasons
        (only for candidates flagged as honeypots)
    """
    honeypots = {}
    for candidate in candidates:
        is_hp, reasons = detect_honeypot(candidate)
        if is_hp:
            honeypots[candidate.candidate_id] = reasons
    return honeypots
