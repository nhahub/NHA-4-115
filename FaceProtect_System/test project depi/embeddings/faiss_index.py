import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("FaissIndex")


@dataclass
class SearchResult:
    person_id: str
    image_path: str
    distance: float
    similarity: float
    similarity_percent: float


def _require_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError("FAISS is not installed. Install with: pip install faiss-cpu") from e


def _validate_paths(emb_path: str, meta_path: str) -> None:
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return vectors / norms


def build_faiss_index(
    emb_path: str,
    meta_path: str,
    index_dir: str,
    normalize_for_cosine: bool = True,
) -> str:
    os.makedirs(index_dir, exist_ok=True)
    _validate_paths(emb_path, meta_path)

    faiss = _require_faiss()

    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape (N, D), got: {embeddings.shape}")

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Metadata length ({len(metadata)}) does not match embeddings count ({embeddings.shape[0]})"
        )

    vectors = embeddings.astype("float32", copy=False)

    if normalize_for_cosine:
        vectors = _normalize_vectors(vectors).astype("float32", copy=False)

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = os.path.join(index_dir, "faiss.index")
    metadata_path = os.path.join(index_dir, "metadata.csv")
    info_path = os.path.join(index_dir, "index_info.json")

    faiss.write_index(index, index_path)
    metadata.to_csv(metadata_path, index=False)

    info = {
        "dim": int(dim),
        "ntotal": int(index.ntotal),
        "index_type": "IndexFlatIP",
        "metric": "cosine_similarity",
        "normalize_for_cosine": bool(normalize_for_cosine),
        "embeddings_path": emb_path,
        "metadata_path": meta_path,
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info("FAISS index built successfully: %s", index_path)

    return index_path


def load_faiss_index(index_dir: str) -> Tuple[Any, pd.DataFrame]:
    faiss = _require_faiss()

    index_path = os.path.join(index_dir, "faiss.index")
    metadata_path = os.path.join(index_dir, "metadata.csv")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    index = faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_path)

    logger.info("FAISS index loaded successfully: %s", index_path)

    return index, metadata


def get_or_build_faiss_index(
    emb_path: str,
    meta_path: str,
    index_dir: str,
    rebuild: bool = False,
    normalize_for_cosine: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    index_path = os.path.join(index_dir, "faiss.index")
    metadata_path = os.path.join(index_dir, "metadata.csv")

    if rebuild or not os.path.exists(index_path) or not os.path.exists(metadata_path):
        build_faiss_index(
            emb_path=emb_path,
            meta_path=meta_path,
            index_dir=index_dir,
            normalize_for_cosine=normalize_for_cosine,
        )

    return load_faiss_index(index_dir)


def search_faiss(
    query_embedding: np.ndarray,
    index,
    metadata: pd.DataFrame,
    top_k: int = 5,
) -> List[SearchResult]:
    if query_embedding is None:
        raise ValueError("Query embedding is None")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if metadata.empty:
        raise ValueError("Metadata is empty")

    top_k = min(top_k, len(metadata))

    vec = query_embedding.astype("float32", copy=False)

    if vec.ndim == 1:
        vec = vec.reshape(1, -1)

    vec = _normalize_vectors(vec).astype("float32", copy=False)

    scores, indices = index.search(vec, top_k)

    results: List[SearchResult] = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata.iloc[int(idx)]

        similarity = float(score)
        similarity = max(0.0, min(1.0, similarity))
        distance = 1.0 - similarity

        results.append(
            SearchResult(
                person_id=str(row.get("person_id", "unknown")),
                image_path=str(row.get("image_path", "")),
                distance=float(distance),
                similarity=float(similarity),
                similarity_percent=float(similarity * 100),
            )
        )

    return results


def decide_match(
    results: List[SearchResult],
    min_similarity: float = 0.70,
) -> Dict[str, Any]:
    if not results:
        return {
            "decision": "REJECT",
            "level": "No Match",
            "person_id": None,
            "distance": None,
            "similarity": 0.0,
            "similarity_percent": 0.0,
        }

    best = results[0]

    if best.similarity >= 0.90:
        level = "Strong Match"
    elif best.similarity >= min_similarity:
        level = "Match"
    elif best.similarity >= 0.50:
        level = "Weak Match"
    else:
        level = "No Match"

    accepted = best.similarity >= min_similarity

    return {
        "decision": "ACCEPT" if accepted else "REJECT",
        "level": level,
        "person_id": best.person_id if accepted else None,
        "distance": best.distance,
        "similarity": best.similarity,
        "similarity_percent": best.similarity_percent,
    }


def results_to_dict(results: List[SearchResult]) -> List[Dict[str, Any]]:
    return [asdict(result) for result in results]


def run_vector_search_pipeline(
    query_embedding: np.ndarray,
    index,
    metadata: pd.DataFrame,
    top_k: int = 5,
    min_similarity: float = 0.70,
) -> Dict[str, Any]:
    results = search_faiss(
        query_embedding=query_embedding,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    decision = decide_match(
        results=results,
        min_similarity=min_similarity,
    )

    return {
        "decision": decision,
        "top_k_results": results_to_dict(results),
    }