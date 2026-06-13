"""
Data loader for candidate profiles from JSONL files.
Streams candidates line-by-line for memory efficiency with 465MB files.
"""

import json
import gzip
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator
from pathlib import Path


@dataclass
class CareerEntry:
    """A single position in a candidate's career history."""
    company: str
    title: str
    start_date: str
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class Education:
    """A single education entry."""
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"


@dataclass
class Skill:
    """A single skill with metadata."""
    name: str
    proficiency: str
    endorsements: int
    duration_months: int = 0


@dataclass
class Certification:
    """A professional certification."""
    name: str
    issuer: str
    year: int


@dataclass
class Language:
    """Language proficiency."""
    language: str
    proficiency: str


@dataclass
class RedrobSignals:
    """Platform behavioral signals."""
    profile_completeness_score: float = 0.0
    signup_date: str = ""
    last_active_date: str = ""
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: Dict[str, float] = field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_range_inr_lpa: Dict[str, float] = field(default_factory=dict)
    preferred_work_mode: str = "flexible"
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


@dataclass
class Candidate:
    """Complete candidate profile with all sections."""
    candidate_id: str
    profile: dict
    career_history: List[CareerEntry]
    education: List[Education]
    skills: List[Skill]
    certifications: List[Certification]
    languages: List[Language]
    redrob_signals: RedrobSignals
    _raw: dict = field(default_factory=dict, repr=False)

    def get_text_for_embedding(self) -> str:
        """Build a rich text representation for semantic embedding."""
        parts = []

        # Profile headline and summary
        headline = self.profile.get("headline", "")
        summary = self.profile.get("summary", "")
        if headline:
            parts.append(headline)
        if summary:
            parts.append(summary)

        # Career descriptions (richest signal)
        for entry in self.career_history:
            role_text = f"{entry.title} at {entry.company}"
            if entry.description:
                role_text += f": {entry.description}"
            parts.append(role_text)

        # Skills list
        skill_names = [s.name for s in self.skills]
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))

        # Education
        for edu in self.education:
            parts.append(f"{edu.degree} in {edu.field_of_study} from {edu.institution}")

        # Certifications
        for cert in self.certifications:
            parts.append(f"Certified: {cert.name} by {cert.issuer}")

        return " . ".join(parts)

    @property
    def current_title(self) -> str:
        return self.profile.get("current_title", "")

    @property
    def current_company(self) -> str:
        return self.profile.get("current_company", "")

    @property
    def years_of_experience(self) -> float:
        return self.profile.get("years_of_experience", 0.0)

    @property
    def location(self) -> str:
        return self.profile.get("location", "")

    @property
    def country(self) -> str:
        return self.profile.get("country", "")


def parse_candidate(data: dict) -> Candidate:
    """Parse a raw JSON dict into a Candidate object."""
    career = []
    for e in data.get("career_history", []):
        career.append(CareerEntry(
            company=e.get("company", ""),
            title=e.get("title", ""),
            start_date=e.get("start_date", ""),
            end_date=e.get("end_date"),
            duration_months=e.get("duration_months", 0),
            is_current=e.get("is_current", False),
            industry=e.get("industry", ""),
            company_size=e.get("company_size", ""),
            description=e.get("description", ""),
        ))

    education = []
    for e in data.get("education", []):
        education.append(Education(
            institution=e.get("institution", ""),
            degree=e.get("degree", ""),
            field_of_study=e.get("field_of_study", ""),
            start_year=e.get("start_year", 0),
            end_year=e.get("end_year", 0),
            grade=e.get("grade"),
            tier=e.get("tier", "unknown"),
        ))

    skills = []
    for s in data.get("skills", []):
        skills.append(Skill(
            name=s.get("name", ""),
            proficiency=s.get("proficiency", "beginner"),
            endorsements=s.get("endorsements", 0),
            duration_months=s.get("duration_months", 0),
        ))

    certs = [Certification(**c) for c in data.get("certifications", [])]
    langs = [Language(**l) for l in data.get("languages", [])]

    sig_data = data.get("redrob_signals", {})
    signals = RedrobSignals(
        profile_completeness_score=sig_data.get("profile_completeness_score", 0),
        signup_date=sig_data.get("signup_date", ""),
        last_active_date=sig_data.get("last_active_date", ""),
        open_to_work_flag=sig_data.get("open_to_work_flag", False),
        profile_views_received_30d=sig_data.get("profile_views_received_30d", 0),
        applications_submitted_30d=sig_data.get("applications_submitted_30d", 0),
        recruiter_response_rate=sig_data.get("recruiter_response_rate", 0),
        avg_response_time_hours=sig_data.get("avg_response_time_hours", 0),
        skill_assessment_scores=sig_data.get("skill_assessment_scores", {}),
        connection_count=sig_data.get("connection_count", 0),
        endorsements_received=sig_data.get("endorsements_received", 0),
        notice_period_days=sig_data.get("notice_period_days", 0),
        expected_salary_range_inr_lpa=sig_data.get("expected_salary_range_inr_lpa", {}),
        preferred_work_mode=sig_data.get("preferred_work_mode", "flexible"),
        willing_to_relocate=sig_data.get("willing_to_relocate", False),
        github_activity_score=sig_data.get("github_activity_score", -1),
        search_appearance_30d=sig_data.get("search_appearance_30d", 0),
        saved_by_recruiters_30d=sig_data.get("saved_by_recruiters_30d", 0),
        interview_completion_rate=sig_data.get("interview_completion_rate", 0),
        offer_acceptance_rate=sig_data.get("offer_acceptance_rate", -1),
        verified_email=sig_data.get("verified_email", False),
        verified_phone=sig_data.get("verified_phone", False),
        linkedin_connected=sig_data.get("linkedin_connected", False),
    )

    return Candidate(
        candidate_id=data["candidate_id"],
        profile=data.get("profile", {}),
        career_history=career,
        education=education,
        skills=skills,
        certifications=certs,
        languages=langs,
        redrob_signals=signals,
        _raw=data,
    )


def stream_candidates(path: str) -> Generator[Candidate, None, None]:
    """Stream candidates from JSONL file one at a time (memory efficient)."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt" if p.suffix == ".gz" else "r"

    with opener(p, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield parse_candidate(data)


def load_candidates(path: str) -> List[Candidate]:
    """Load all candidates into memory from a JSONL file."""
    return list(stream_candidates(path))


def load_candidate_texts(path: str) -> tuple:
    """Load candidate IDs and embedding texts. Returns (ids, texts) lists."""
    ids = []
    texts = []
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt" if p.suffix == ".gz" else "r"

    with opener(p, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            candidate = parse_candidate(data)
            ids.append(candidate.candidate_id)
            texts.append(candidate.get_text_for_embedding())

    return ids, texts
