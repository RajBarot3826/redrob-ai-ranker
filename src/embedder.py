"""
Embedder — Handles sentence-transformer embeddings and FAISS indexing.
Uses all-MiniLM-L6-v2 for fast CPU inference (384-dim embeddings).

Pre-computation is done offline (can take 15-20 min for 100K candidates).
At ranking time, only loads pre-computed numpy arrays — no model needed.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load the sentence-transformer model. Only needed for pre-computation."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def compute_embeddings(
    model,
    texts: List[str],
    batch_size: int = 256,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Args:
        model: SentenceTransformer model
        texts: List of text strings to embed
        batch_size: Batch size for encoding
        show_progress: Show tqdm progress bar
        normalize: L2-normalize embeddings (for cosine similarity via dot product)

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    candidate_ids: List[str],
    output_dir: str,
):
    """
    Save embeddings and candidate IDs to disk.

    Saves:
        - embeddings.npy: (N, dim) float32 array
        - candidate_ids.json: ordered list of candidate IDs
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "embeddings.npy", embeddings)
    with open(out / "candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)

    print(f"Saved {len(candidate_ids)} embeddings to {output_dir}")
    print(f"  embeddings.npy: {embeddings.shape}, {embeddings.nbytes / 1e6:.1f} MB")


def save_jd_embedding(embedding: np.ndarray, output_dir: str):
    """Save the JD embedding separately."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "jd_embedding.npy", embedding)
    print(f"Saved JD embedding to {output_dir}/jd_embedding.npy")


def load_precomputed(precomputed_dir: str):
    """
    Load pre-computed embeddings and candidate IDs.

    Returns:
        (embeddings: np.ndarray, candidate_ids: List[str], jd_embedding: np.ndarray)
    """
    d = Path(precomputed_dir)

    embeddings = np.load(d / "embeddings.npy")
    with open(d / "candidate_ids.json", "r") as f:
        candidate_ids = json.load(f)
    jd_embedding = np.load(d / "jd_embedding.npy")

    assert len(candidate_ids) == embeddings.shape[0], \
        f"ID count ({len(candidate_ids)}) != embedding count ({embeddings.shape[0]})"

    return embeddings, candidate_ids, jd_embedding


def compute_similarities(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between all candidates and the JD.

    Since embeddings are L2-normalized, dot product = cosine similarity.

    Args:
        candidate_embeddings: (N, dim) normalized float32 array
        jd_embedding: (dim,) or (1, dim) normalized float32 array

    Returns:
        (N,) array of cosine similarities
    """
    if jd_embedding.ndim == 2:
        jd_embedding = jd_embedding[0]

    # Dot product with normalized vectors = cosine similarity
    similarities = candidate_embeddings @ jd_embedding

    return similarities


def build_faiss_index(
    embeddings: np.ndarray,
    output_path: Optional[str] = None,
):
    """
    Build a FAISS flat index for exact nearest-neighbor search.
    Useful for the interactive demo where new JDs can be queried.

    Args:
        embeddings: (N, dim) float32 array (normalized)
        output_path: Optional path to save the index

    Returns:
        FAISS index
    """
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
    index.add(embeddings)

    if output_path:
        faiss.write_index(index, output_path)
        print(f"Saved FAISS index to {output_path}")

    return index
