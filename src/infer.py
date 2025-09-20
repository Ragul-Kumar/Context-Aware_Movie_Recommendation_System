# src/infer.py
import torch
import pandas as pd
import datetime
from src.models.mf import MF

# ------------------------------
# Load features and movies
# ------------------------------
ratings = pd.read_parquet("data/processed/features.parquet")
movies = pd.read_csv(
    "data/raw/ml-1m/movies.dat",
    sep="::", engine="python",
    header=None, names=["movieId", "title", "genres"],
    encoding="latin-1"
)

# ------------------------------
# Load trained model
# ------------------------------
n_users = ratings['user_id_enc'].nunique()
n_items = ratings['movie_id_enc'].nunique()
model = MF(n_users, n_items)
model.load_state_dict(torch.load("saved_model/context_mf.pth"))
model.eval()

# ------------------------------
# Recommend top N movies
# ------------------------------
def recommend_top_n(user_id, top_k=10, genre=None):
    # Encode user
    user_enc = ratings.loc[ratings['userId'] == user_id, 'user_id_enc'].iloc[0]
    all_items = ratings['movie_id_enc'].unique()

    users = torch.tensor([user_enc] * len(all_items))
    items = torch.tensor(all_items)

    # Predict scores using the trained MF model
    with torch.no_grad():
        scores = model(users, items).numpy()

    # Create DataFrame to map scores to movie info
    score_df = pd.DataFrame({
        'movie_id_enc': all_items,
        'score': scores
    })

    recommended = ratings[['movie_id_enc', 'movieId']].drop_duplicates()
    recommended = recommended.merge(movies, on='movieId')
    recommended = recommended.merge(score_df, on='movie_id_enc')

    # Filter by genre if provided
    if genre:
        recommended = recommended[recommended['genres'].str.contains(genre, case=False, na=False)]

    # Sort by predicted score and pick top-K
    recommended = recommended.sort_values('score', ascending=False).head(top_k)
    return recommended[['movieId', 'title', 'genres']]

# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    # Get current time
    now = datetime.datetime.now()
    print(f"\n📅 Current date & time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get user input
    user_id = int(input("Enter User ID: "))
    genre_input = input("Enter preferred genre (or press Enter to skip): ").strip()

    # Get recommendations
    recs = recommend_top_n(user_id=user_id, top_k=10, genre=genre_input if genre_input else None)

    # Pretty print results
    print("\n🎬 Top Recommendations:")
    for i, row in enumerate(recs.itertuples(), 1):
        genres = row.genres.replace("|", ", ")
        print(f"{i}. {row.title} — {genres}")
