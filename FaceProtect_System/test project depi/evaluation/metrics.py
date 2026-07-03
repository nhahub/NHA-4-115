import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def compute_distances(emb1, emb2, metric="cosine"):
    """Computes pairwise distance/similarity between two sets of embeddings."""
    if metric == "cosine":
        # Cosine distance = 1 - Cosine Similarity
        sim = cosine_similarity(emb1, emb2)
        return 1.0 - sim
    elif metric == "euclidean":
        return euclidean_distances(emb1, emb2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def generate_pairs(metadata, max_pairs=10000):
    """Generates positive and negative pairs for evaluation."""
    identities = metadata['person_id'].values
    indices = np.arange(len(identities))
    
    pos_pairs = []
    neg_pairs = []
    
    # Simple pairing strategy for demonstration
    for i in range(len(indices)):
        same_idx = np.where(identities == identities[i])[0]
        same_idx = same_idx[same_idx != i]
        if len(same_idx) > 0:
            pos_pairs.append((i, np.random.choice(same_idx)))
            
        diff_idx = np.where(identities != identities[i])[0]
        if len(diff_idx) > 0:
            neg_pairs.append((i, np.random.choice(diff_idx)))
            
        if len(pos_pairs) + len(neg_pairs) > max_pairs * 2:
            break
            
    min_len = min(len(pos_pairs), len(neg_pairs), max_pairs)
    
    if min_len == 0:
        return [], [], [], []
        
    np.random.shuffle(pos_pairs)
    np.random.shuffle(neg_pairs)
    
    pos_pairs = pos_pairs[:min_len]
    neg_pairs = neg_pairs[:min_len]
    
    pos_idx1 = [p[0] for p in pos_pairs]
    pos_idx2 = [p[1] for p in pos_pairs]
    
    neg_idx1 = [p[0] for p in neg_pairs]
    neg_idx2 = [p[1] for p in neg_pairs]
    
    return pos_idx1, pos_idx2, neg_idx1, neg_idx2
