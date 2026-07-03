from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import uuid
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from embeddings.faiss_index import (
    build_faiss_index,
    get_or_build_faiss_index,
    run_vector_search_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("Phase4VectorDB")


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        logger.warning("Config file not found. Using default values.")
        return {}

    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    except Exception as e:
        logger.warning("Failed to load config file. Using default values. Error: %s", e)
        return {}


def fix_path(path: str) -> str:
    path = path.strip().strip('"').strip("'")
    path = path.replace("\\", "/")
    return os.path.normpath(path)


def safe_image_path(original_path: str) -> str:
    original_path = fix_path(original_path)

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Query image not found: {original_path}")

    safe_dir = "temp_safe"
    os.makedirs(safe_dir, exist_ok=True)

    ext = os.path.splitext(original_path)[1].lower()

    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    safe_path = os.path.join(safe_dir, f"query_{uuid.uuid4().hex}{ext}")
    shutil.copy(original_path, safe_path)

    return safe_path


def get_query_embedding(image_path: str) -> np.ndarray:
    try:
        from deepface import DeepFace

        result = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
        )

        if not result:
            raise RuntimeError("No embedding returned from DeepFace")

        return np.asarray(result[0]["embedding"])

    except Exception as e:
        raise RuntimeError(f"Failed to extract query embedding: {e}") from e


def get_config_value(config: Dict[str, Any], keys: list[str], default: Any) -> Any:
    current: Any = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: FAISS Vector Database and Nearest Neighbor Search"
    )

    parser.add_argument(
        "--query_image",
        type=str,
        default=None,
        help="Path to query image",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of nearest neighbors",
    )

    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Rebuild FAISS index before searching",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.70,
        help="Minimum similarity required for ACCEPT decision",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    emb_path = get_config_value(
        config,
        ["outputs", "embeddings_path"],
        "outputs/embeddings/embeddings.npy",
    )

    meta_path = get_config_value(
        config,
        ["outputs", "metadata_path"],
        "outputs/embeddings/metadata.csv",
    )

    index_dir = get_config_value(
        config,
        ["faiss", "index_dir"],
        "outputs/embeddings/faiss_index",
    )

    top_k = args.top_k or int(
        get_config_value(
            config,
            ["faiss", "top_k"],
            5,
        )
    )

    logger.info("[Phase4] Embeddings path: %s", emb_path)
    logger.info("[Phase4] Metadata path: %s", meta_path)
    logger.info("[Phase4] FAISS index dir: %s", index_dir)
    logger.info("[Phase4] Top-K: %s", top_k)
    logger.info("[Phase4] Minimum similarity: %.2f", args.min_similarity)

    if args.rebuild_index:
        logger.info("[Phase4] Rebuilding FAISS index...")

        build_faiss_index(
            emb_path=emb_path,
            meta_path=meta_path,
            index_dir=index_dir,
            normalize_for_cosine=True,
        )

    index, metadata = get_or_build_faiss_index(
        emb_path=emb_path,
        meta_path=meta_path,
        index_dir=index_dir,
        rebuild=args.rebuild_index,
        normalize_for_cosine=True,
    )

    if not args.query_image:
        logger.info("[Phase4] No query image provided. Index is ready.")

        print(
            json.dumps(
                {
                    "status": "index_ready",
                    "index_dir": index_dir,
                    "total_vectors": int(index.ntotal),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    safe_path = safe_image_path(args.query_image)

    logger.info("[Phase4] Extracting query embedding...")
    query_embedding = get_query_embedding(safe_path)

    result = run_vector_search_pipeline(
        query_embedding=query_embedding,
        index=index,
        metadata=metadata,
        top_k=top_k,
        min_similarity=args.min_similarity,
    )

    output = {
        "query_image": args.query_image,
        "safe_query_image": safe_path,
        "top_k": top_k,
        "min_similarity": args.min_similarity,
        **result,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()