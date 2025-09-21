import numpy as np
from collections import defaultdict
from itertools import combinations

# ----------------------------
# Evaluation Metrics Functions
# ----------------------------

def precision_at_k(recommended, ground_truth, k):
    """Precision@K = relevant items in top-K / K"""
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(ground_truth))
    return hits / k


def hit_at_k(recommended, ground_truth, k):
    """Hit@K = 1 if any ground-truth item is in top-K, else 0"""
    recommended_k = recommended[:k]
    return 1.0 if len(set(recommended_k) & set(ground_truth)) > 0 else 0.0


def ndcg_at_k(recommended, ground_truth, k):
    """Normalized Discounted Cumulative Gain"""
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)  # +2 because log starts at 1
    # Ideal DCG
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(recommended, item_features):
    """
    Diversity = average dissimilarity between recommended items.
    item_features: dict {item_id: feature_vector (numpy array)}
    """
    if len(recommended) <= 1:
        return 0.0

    sims = []
    for i, j in combinations(recommended, 2):
        if i in item_features and j in item_features:
            vec_i, vec_j = item_features[i], item_features[j]
            sim = np.dot(vec_i, vec_j) / (
                np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-9
            )
            sims.append(sim)
    if not sims:
        return 0.0
    return 1 - np.mean(sims)  # Higher = more diverse


# ----------------------------
# Example Run (Simulated Data)
# ----------------------------
if __name__ == "__main__":
    # Ground truth: movies the user actually liked
    ground_truth = [101, 102, 103]

    # Recommendations from your model
    recommended = [101, 105, 106, 107, 108]

    # Fake item features (e.g., genre embeddings) for diversity
    item_features = {
        101: np.array([1, 0, 0]),
        102: np.array([0, 1, 0]),
        103: np.array([0, 0, 1]),
        105: np.array([1, 1, 0]),
        106: np.array([0, 1, 1]),
        107: np.array([1, 0, 1]),
        108: np.array([0.5, 0.5, 0]),
    }

    K = 5
    print("📊 Evaluation Results")
    print("--------------------")
    print(f"Precision@{K}: {precision_at_k(recommended, ground_truth, K):.4f}")
    print(f"Hit@{K}:       {hit_at_k(recommended, ground_truth, K):.4f}")
    print(f"NDCG@{K}:      {ndcg_at_k(recommended, ground_truth, K):.4f}")
    print(f"Diversity:     {intra_list_diversity(recommended, item_features):.4f}")
