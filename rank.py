#!/usr/bin/env python3
"""
rank.py — Main CLI entry point for the Redrob Candidate Ranking System.

Produces a submission CSV with the top 100 candidates ranked for the
Senior AI Engineer role at Redrob AI.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints (from submission_spec.md):
    - ≤ 5 minutes wall-clock
    - ≤ 16 GB RAM
    - CPU only (no GPU)
    - No network access during ranking
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_candidates
from src.embedder import load_precomputed, compute_similarities
from src.ranker import rank_candidates, format_submission_csv


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Senior AI Engineer role at Redrob AI."
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out", "-o",
        default="submission.csv",
        help="Output path for the submission CSV (default: submission.csv)"
    )
    parser.add_argument(
        "--precomputed", "-p",
        default="precomputed",
        help="Directory with pre-computed embeddings (default: precomputed/)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=100,
        help="Number of top candidates to return (default: 100)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    start_time = time.time()

    # ── Step 1: Load pre-computed embeddings ──
    if not args.quiet:
        print(f"Loading pre-computed embeddings from {args.precomputed}/...")

    precomputed_dir = Path(args.precomputed)
    if not precomputed_dir.exists():
        print(f"ERROR: Pre-computed directory '{args.precomputed}' not found.")
        print("Run: python scripts/precompute_embeddings.py first")
        sys.exit(1)

    embeddings, embedding_ids, jd_embedding = load_precomputed(args.precomputed)

    # Build ID → index mapping
    id_to_idx = {cid: i for i, cid in enumerate(embedding_ids)}

    t1 = time.time()
    if not args.quiet:
        print(f"  Loaded {len(embedding_ids)} embeddings in {t1 - start_time:.1f}s")

    # ── Step 2: Compute semantic similarities ──
    if not args.quiet:
        print("Computing semantic similarities...")

    similarities = compute_similarities(embeddings, jd_embedding)

    t2 = time.time()
    if not args.quiet:
        print(f"  Computed similarities in {t2 - t1:.1f}s")

    # ── Step 3: Load candidates ──
    if not args.quiet:
        print(f"Loading candidates from {args.candidates}...")

    candidates = load_candidates(args.candidates)

    t3 = time.time()
    if not args.quiet:
        print(f"  Loaded {len(candidates)} candidates in {t3 - t2:.1f}s")

    # ── Step 4: Run ranking pipeline ──
    if not args.quiet:
        print(f"Running ranking pipeline (top {args.top_k})...")

    results = rank_candidates(
        candidates=candidates,
        semantic_similarities=similarities,
        candidate_id_to_idx=id_to_idx,
        top_k=args.top_k,
        show_progress=not args.quiet,
    )

    t4 = time.time()
    if not args.quiet:
        print(f"  Ranked candidates in {t4 - t3:.1f}s")

    # ── Step 5: Write submission CSV ──
    csv_content = format_submission_csv(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_content)

    total_time = time.time() - start_time

    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"SUBMISSION GENERATED: {out_path}")
        print(f"{'='*60}")
        print(f"  Candidates ranked: {len(results)}")
        print(f"  Score range: {results[0][2]:.4f} (rank 1) → {results[-1][2]:.4f} (rank {len(results)})")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Within 5-min budget: {'✓ YES' if total_time < 300 else '✗ NO'}")
        print()

        # Print top-10 summary
        print("Top 10 candidates:")
        print(f"  {'Rank':<5} {'ID':<15} {'Score':<8} {'Title':<35} {'Company'}")
        print(f"  {'-'*5} {'-'*15} {'-'*8} {'-'*35} {'-'*20}")
        for cand, rank, score, reasoning, _ in results[:10]:
            print(f"  {rank:<5} {cand.candidate_id:<15} {score:<8.4f} {cand.current_title[:35]:<35} {cand.current_company}")

        # Honeypot check
        honeypot_count = sum(1 for _, _, _, _, scores in results if scores.get("is_honeypot", False))
        print(f"\n  Honeypots in top-{args.top_k}: {honeypot_count} ({honeypot_count/len(results)*100:.1f}%)")
        if honeypot_count > 10:
            print("  ⚠ WARNING: Honeypot rate exceeds 10%! Risk of disqualification.")
        else:
            print("  ✓ Honeypot rate within safe limits.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
