import os
import argparse
import logging
import json

import numpy as np

from utils.config_loader import load_config
from utils.logger import get_logger
from models.model_loader import load_model
from embeddings.faiss_index import build_faiss_index, load_faiss_index, search_faiss

logger = logging.getLogger("Search")


def _load_threshold(report_path: str):
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report.get("threshold")


def _embed_one_image(model, image_path: str):
    # Project models expose `extract(batch_paths)`.
    # It returns (embeddings, valid_paths)
    embeddings, valid_paths = model.extract([image_path])
    if not embeddings:
        raise RuntimeError(f"No embedding extracted for: {image_path}")
    # embeddings corresponds to valid_paths ordering
    return np.asarray(embeddings[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K results")
    parser.add_argument("--rebuild_index", action="store_true", help="Rebuild FAISS index")
    args = parser.parse_args()

    config = load_config("config.yaml")
    logger2 = get_logger("SearchMain", log_level=config['logging']['log_level'], log_dir=config['logging']['log_dir'])

    emb_dir = config['embeddings']['output_dir']
    emb_path = os.path.join(emb_dir, "embeddings.npy")
    meta_path = os.path.join(emb_dir, "metadata.csv")

    faiss_conf = config.get("faiss", {})
    index_dir = faiss_conf.get("index_dir", "outputs/embeddings/faiss_index")

    top_k = args.top_k if args.top_k is not None else int(faiss_conf.get("top_k", 5))

    index_exists = os.path.exists(os.path.join(index_dir, "faiss.index")) and os.path.exists(
        os.path.join(index_dir, "metadata.csv")
    )

    if args.rebuild_index or not index_exists:
        logger2.info("Building FAISS index...")
        build_faiss_index(
            emb_path=emb_path,
            meta_path=meta_path,
            index_dir=index_dir,
            normalize_for_cosine=True,
        )

    logger2.info("Loading FAISS index...")
    index, metadata = load_faiss_index(index_dir)

    model = load_model(config)
    logger2.info(f"Embedding input image: {args.image}")
    query_emb = _embed_one_image(model, args.image)

    results = search_faiss(
        query_embedding=query_emb,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    # Optional verification gate using evaluation threshold
    report_path = os.path.join(config['evaluation']['output_reports_dir'], "evaluation_report.json")
    threshold = _load_threshold(report_path)

    best = results[0] if results else None
    verdict = None
    if best is not None and threshold is not None:
        verdict = "ACCEPT" if best.distance <= float(threshold) else "REJECT"

    payload = {
        "query_image": args.image,
        "top_k": top_k,
        "threshold": threshold,
        "best_match": None if best is None else {
            "person_id": best.person_id,
            "image_path": best.image_path,
            "cosine_distance": best.distance,
        },
        "verdict": verdict,
        "top_k_results": [
            {
                "person_id": r.person_id,
                "image_path": r.image_path,
                "cosine_distance": r.distance,
            }
            for r in results
        ],
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

