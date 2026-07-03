import os
import shutil
import uuid
import numpy as np
from deepface import DeepFace

from embeddings.faiss_index import (
    get_or_build_faiss_index,
    run_vector_search_pipeline,
)

EMBEDDINGS_PATH = "outputs/embeddings/embeddings.npy"
METADATA_PATH = "outputs/embeddings/metadata.csv"
INDEX_DIR = "outputs/embeddings/faiss_index"
MIN_SIMILARITY = 0.70
TOP_K = 5


def fix_path(path):
    path = path.strip().strip('"').strip("'")
    path = path.replace("\\", "/")
    return os.path.normpath(path)


def safe_image_path(original_path):
    original_path = fix_path(original_path)

    if not os.path.exists(original_path):
        print("❌ Image not found!\n")
        return None

    safe_dir = "temp_safe"
    os.makedirs(safe_dir, exist_ok=True)

    ext = os.path.splitext(original_path)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    safe_path = os.path.join(safe_dir, f"query_{uuid.uuid4().hex}{ext}")

    try:
        shutil.copy(original_path, safe_path)
        return safe_path
    except Exception as e:
        print(f"\n❌ Error copying image:\n{e}")
        return None


def get_embedding(img_path):
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
        )
        return np.asarray(result[0]["embedding"])
    except Exception as e:
        print(f"\n❌ Error processing image:\n{e}")
        return None


def print_results(result):
    decision = result["decision"]
    top_results = result["top_k_results"]

    print("\n🔍 Top Matches:")

    for i, item in enumerate(top_results, start=1):
        print(
            f"{i}. {item['person_id']} | "
            f"Similarity: {item['similarity_percent']:.2f}% | "
            f"Distance: {item['distance']:.4f}"
        )

    print("\n🎯 Result:", decision["level"])
    print("✅ Decision:", decision["decision"])
    print(f"📊 Similarity: {decision['similarity_percent']:.2f}%")

    if decision["person_id"]:
        print("👤 Person:", decision["person_id"])
    else:
        print("👤 Person: Unknown")

    print()


def main():
    print("🚀 PRO Face Vector Search Demo")
    print(f"✅ Minimum Similarity Required: {MIN_SIMILARITY * 100:.0f}%")

    index, metadata = get_or_build_faiss_index(
        emb_path=EMBEDDINGS_PATH,
        meta_path=METADATA_PATH,
        index_dir=INDEX_DIR,
        rebuild=False,
        normalize_for_cosine=True,
    )

    print("✅ FAISS Search Engine Ready\n")

    while True:
        img_path = input("📂 Enter image path (or type 'exit'): ")

        if img_path.strip().lower() == "exit":
            print("👋 Exiting system...")
            break

        safe_path = safe_image_path(img_path)

        if safe_path is None:
            continue

        query_embedding = get_embedding(safe_path)

        if query_embedding is None:
            continue

        result = run_vector_search_pipeline(
            query_embedding=query_embedding,
            index=index,
            metadata=metadata,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
        )

        print_results(result)


if __name__ == "__main__":
    main()