"""
Day 56: E-commerce Product Recommendations - Project
Topic: Collaborative filtering and content-based recommendation
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Generate synthetic user-item ratings ---
np.random.seed(42)
n_users, n_items = 100, 50
users = [f"User_{i}" for i in range(n_users)]
items = [f"Item_{i}" for i in range(n_items)]

ratings_matrix = np.zeros((n_users, n_items))
for u in range(n_users):
    rated = np.random.choice(n_items, size=np.random.randint(5, 20), replace=False)
    for item in rated:
        ratings_matrix[u, item] = np.random.randint(1, 6)

ratings_df = pd.DataFrame(ratings_matrix, index=users, columns=items)
print(f"Ratings matrix: {n_users} users x {n_items} items")
print(f"Sparsity: {(ratings_matrix == 0).mean():.2%}")
print(f"Rating distribution: {pd.Series(ratings_matrix[ratings_matrix > 0]).describe().to_dict()}")

# --- Collaborative Filtering (User-based) ---
user_similarity = cosine_similarity(ratings_matrix)
np.fill_diagonal(user_similarity, 0)

def user_based_predict(user_idx, item_idx, k=5):
    similar_users = np.argsort(user_similarity[user_idx])[::-1][:k]
    weighted_sum = 0
    similarity_sum = 0
    for sim_user in similar_users:
        if ratings_matrix[sim_user, item_idx] > 0:
            weighted_sum += user_similarity[user_idx, sim_user] * ratings_matrix[sim_user, item_idx]
            similarity_sum += user_similarity[user_similarity[user_idx, sim_user] > 0][0]
    return weighted_sum / max(similarity_sum, 1e-10) if similarity_sum > 0 else 0

# --- Matrix Factorization (SVD) ---
svd = TruncatedSVD(n_components=10, random_state=42)
user_factors = svd.fit_transform(ratings_matrix)
item_factors = svd.components_.T
predicted_ratings = np.dot(user_factors, item_factors)

print("\nSVD Reconstruction Error:", f"{np.linalg.norm(ratings_matrix - svd.inverse_transform(user_factors)):.4f}")

# --- Top-N Recommendations ---
def recommend(user_idx, n=5):
    user_pred = predicted_ratings[user_idx]
    already_rated = np.where(ratings_matrix[user_idx] > 0)[0]
    recommendations = [(items[i], user_pred[i]) for i in np.argsort(user_pred)[::-1] if i not in already_rated][:n]
    return recommendations

print(f"\nRecommendations for User_0:")
for item, score in recommend(0):
    print(f"  {item}: predicted rating {score:.2f}")

# --- Content-Based Filtering ---
product_data = pd.DataFrame({
    'item_id': items,
    'description': [f"{'premium' if i%3==0 else 'budget'} {'electronics' if i%2==0 else 'clothing'} item with feature_{i}" for i in range(n_items)],
    'category': [['Electronics', 'Clothing'][i%2] for i in range(n_items)],
})

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(product_data['description'])
item_similarity = cosine_similarity(tfidf_matrix)

def content_based_recommend(item_idx, n=5):
    similar = np.argsort(item_similarity[item_idx])[::-1][1:n+1]
    return [(items[i], item_similarity[item_idx, i]) for i in similar]

print(f"\nContent-Based Recommendations for Item_0:")
for item, score in content_based_recommend(0):
    print(f"  {item}: similarity {score:.4f}")

print("\nKey Concepts:")
print("- Collaborative Filtering: recommends based on similar users' preferences")
print("- Content-Based: recommends based on item feature similarity")
print("- Matrix Factorization (SVD): decomposes ratings matrix into latent factors")
