#!/usr/bin/env python3
"""
Pre-compute embeddings for all candidates and the JD.

This runs ONCE before the ranking step and can take 15-20 minutes on CPU.
The ranking step only loads the pre-computed arrays (no model needed).

Usage:
    python scripts/precompute_embeddings.py \
        --candidates ./candidates.jsonl \
        --jd ./data/jd_query.txt \
        --output ./precomputed
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_candidate_texts
from src.embedder import (
    load_model,
    compute_embeddings,
    save_embeddings,
    save_jd_embedding,
    build_faiss_index,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute candidate and JD embeddings."
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--jd", "-j",
        default="data/jd_query.txt",
        help="Path to JD query text file"
    )
    parser.add_argument(
        "--output", "-o",
        default="precomputed",
        help="Output directory for embeddings (default: precomputed/)"
    )
    parser.add_argument(
        "--model", "-m",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=256,
        help="Encoding batch size (default: 256)"
    )

    args = parser.parse_args()
    start_time = time.time()

    # ── Step 1: Load model ──
    print(f"Loading model: {args.model}...")
    model = load_model(args.model)
    t1 = time.time()
    print(f"  Model loaded in {t1 - start_time:.1f}s")

    # ── Step 2: Load candidate texts ──
    print(f"Loading candidate texts from {args.candidates}...")
    candidate_ids, texts = load_candidate_texts(args.candidates)
    t2 = time.time()
    print(f"  Loaded {len(candidate_ids)} candidates in {t2 - t1:.1f}s")

    # ── Step 3: Compute candidate embeddings ──
    print(f"Computing {len(texts)} candidate embeddings (batch_size={args.batch_size})...")
    print("  This may take 15-20 minutes on CPU...")
    candidate_embeddings = compute_embeddings(
        model, texts,
        batch_size=args.batch_size,
        show_progress=True,
        normalize=True,
    )
    t3 = time.time()
    print(f"  Computed embeddings in {t3 - t2:.1f}s ({(t3 - t2)/60:.1f} min)")

    # ── Step 4: Save candidate embeddings ──
    save_embeddings(candidate_embeddings, candidate_ids, args.output)

    # ── Step 5: Compute and save JD embedding ──
    print(f"Computing JD embedding from {args.jd}...")
    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"ERROR: JD file not found: {args.jd}")
        sys.exit(1)

    jd_text = jd_path.read_text(encoding="utf-8").strip()
    jd_embedding = compute_embeddings(
        model, [jd_text],
        batch_size=1,
        show_progress=False,
        normalize=True,
    )
    save_jd_embedding(jd_embedding[0], args.output)

    # ── Step 6: Build FAISS index (optional, for demo) ──
    print("Building FAISS index...")
    faiss_path = str(Path(args.output) / "faiss.index")
    try:
        build_faiss_index(candidate_embeddings, faiss_path)
    except ImportError:
        print("  FAISS not available — skipping index build (not required for ranking)")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PRE-COMPUTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output directory: {args.output}/")
    print(f"  Embeddings shape: {candidate_embeddings.shape}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nYou can now run:")
    print(f"  python rank.py --candidates {args.candidates} --out submission.csv")


if __name__ == "__main__":
    main()
