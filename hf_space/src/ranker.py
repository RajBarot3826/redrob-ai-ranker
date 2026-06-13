"""
Ranker — The main ranking engine that combines all scoring dimensions
into a final composite score and produces the top-100 ranking.

Architecture: 3-stage hybrid pipeline
  Stage 1: Coarse filter (honeypot + disqualifier elimination)
  Stage 2: Full scoring (semantic + structured + behavioral)
  Stage 3: Final ranking with reasoning generation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from src.data_loader import Candidate
from src.honeypot_detector import detect_honeypot
from src.feature_scorer import compute_feature_scores
from src.behavioral_scorer import compute_behavioral_scores
from src.reasoning_generator import generate_reasoning
from src.config import SCORING_WEIGHTS


def compute_final_score(
    feature_scores: Dict[str, float],
    behavioral_scores: Dict[str, float],
    semantic_similarity: float,
    is_honeypot: bool,
) -> float:
    """
    Compute the final composite score for a candidate.

    Uses weighted combination of all scoring dimensions,
    with honeypot penalty as a hard gate (score → 0).
    """
    if is_honeypot:
        return 0.0

    weights = SCORING_WEIGHTS

    # Combine all scores
    raw_score = (
        weights["semantic_similarity"] * semantic_similarity +
        weights["title_alignment"]     * feature_scores.get("title_alignment", 0) +
        weights["career_quality"]      * feature_scores.get("career_quality", 0) +
        weights["skills_match"]        * feature_scores.get("skills_match", 0) +
        weights["experience_band"]     * feature_scores.get("experience_band", 0) +
        weights["location"]           * feature_scores.get("location", 0) +
        weights["availability"]        * behavioral_scores.get("availability", 0) +
        weights["engagement"]          * behavioral_scores.get("engagement", 0) +
        weights["profile_trust"]       * behavioral_scores.get("profile_trust", 0) +
        weights["market_signals"]      * behavioral_scores.get("market_signals", 0)
    )

    # ── Title-based gating ──
    # The JD is explicit: non-tech titles with AI keywords = trap.
    # Apply a multiplicative penalty based on title alignment.
    title_score = feature_scores.get("title_alignment", 0)
    if title_score < 0.10:
        # Marketing Manager, HR Manager, Accountant, etc.
        raw_score *= 0.10  # Devastating penalty
    elif title_score < 0.20:
        raw_score *= 0.35
    elif title_score < 0.35:
        raw_score *= 0.60

    # ── Career quality gating ──
    # All-services career is an explicit disqualifier
    career_score = feature_scores.get("career_quality", 0)
    if career_score < 0.15:
        raw_score *= 0.25

    return min(max(raw_score, 0.0), 1.0)


def rank_candidates(
    candidates: List[Candidate],
    semantic_similarities: np.ndarray,
    candidate_id_to_idx: Dict[str, int],
    top_k: int = 100,
    show_progress: bool = True,
) -> List[Tuple[Candidate, int, float, str, Dict]]:
    """
    Run the full ranking pipeline on all candidates.

    Args:
        candidates: List of all candidate objects
        semantic_similarities: Pre-computed cosine similarities (N,)
        candidate_id_to_idx: Mapping from candidate_id to embedding index
        top_k: Number of top candidates to return
        show_progress: Show progress bar

    Returns:
        List of (candidate, rank, score, reasoning, all_scores) tuples
        sorted by score descending.
    """
    scored_candidates = []

    iterator = tqdm(candidates, desc="Scoring candidates") if show_progress else candidates

    for candidate in iterator:
        # ── Stage 1: Honeypot check ──
        is_honeypot, hp_reasons = detect_honeypot(candidate)

        # ── Stage 2: Compute all scores ──
        feature_scores = compute_feature_scores(candidate)
        behavioral_scores = compute_behavioral_scores(candidate)

        # Get semantic similarity
        idx = candidate_id_to_idx.get(candidate.candidate_id)
        if idx is not None and idx < len(semantic_similarities):
            sem_sim = float(semantic_similarities[idx])
        else:
            sem_sim = 0.0

        # Normalize semantic similarity to [0, 1] range
        # Cosine similarity with normalized vectors is already in [-1, 1]
        # Map to [0, 1] for scoring
        sem_sim_normalized = (sem_sim + 1.0) / 2.0

        # ── Stage 3: Final composite score ──
        final_score = compute_final_score(
            feature_scores=feature_scores,
            behavioral_scores=behavioral_scores,
            semantic_similarity=sem_sim_normalized,
            is_honeypot=is_honeypot,
        )

        all_scores = {
            **feature_scores,
            **behavioral_scores,
            "semantic_similarity": sem_sim_normalized,
            "is_honeypot": is_honeypot,
            "honeypot_reasons": hp_reasons if is_honeypot else [],
            "final_score": final_score,
        }

        scored_candidates.append((candidate, final_score, all_scores))

    # Sort by final score descending
    scored_candidates.sort(key=lambda x: (-x[1], x[0].candidate_id))

    # Take top K and assign ranks
    top_candidates = scored_candidates[:top_k]

    results = []
    for rank_idx, (candidate, score, all_scores) in enumerate(top_candidates):
        rank = rank_idx + 1

        # Generate reasoning
        reasoning = generate_reasoning(
            candidate=candidate,
            rank=rank,
            final_score=score,
            scores=all_scores,
        )

        results.append((candidate, rank, score, reasoning, all_scores))

    return results


def format_submission_csv(
    results: List[Tuple[Candidate, int, float, str, Dict]],
) -> str:
    """
    Format the ranking results as a submission CSV string.

    Format: candidate_id,rank,score,reasoning
    Scores must be monotonically non-increasing as rank increases.
    """
    lines = ["candidate_id,rank,score,reasoning"]

    for candidate, rank, score, reasoning, _ in results:
        # Escape reasoning for CSV (double-quote if contains comma or quote)
        reasoning_escaped = reasoning.replace('"', '""')
        if ',' in reasoning_escaped or '"' in reasoning or '\n' in reasoning:
            reasoning_escaped = f'"{reasoning_escaped}"'

        lines.append(f"{candidate.candidate_id},{rank},{score:.4f},{reasoning_escaped}")

    return "\n".join(lines) + "\n"
