import numpy as np

def normalize_l2(embeddings):
    """Applies L2 normalization to a 2D numpy array of embeddings."""
    embeddings = np.array(embeddings)
    if len(embeddings) == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return embeddings / norms
