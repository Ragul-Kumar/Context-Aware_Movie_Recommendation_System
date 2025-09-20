# run_pipeline.py
import os
import torch
import pandas as pd
import datetime
from src.data.load_data import load_movielens
from src.data.preprocess import preprocess_ratings, save_processed_data, encode_ids
from src.features.make_features import create_user_item_embeddings, create_context_features
from src.models.mf import MF

# ------------------------------
# Config
# ------------------------------
TOP_K = 10
SAVED_MODEL_DIR = "saved_model"
FEATURES_PATH = "data/processed/features.parquet"

# ------------------------------
# Step 1: Load raw data
# ------------------------------
print("1️⃣ Loading raw data...")
users, movies, ratings = load_movielens()
print(f"Users: {len(users)}, Movies: {len(movies)}, Ratings: {len(ratings)}")

# ------------------------------
# Step 2: Preprocess ratings
# ------------------------------
print("2️⃣ Preprocessing ratings...")
ratings = preprocess_ratings(ratings)
save_processed_data(ratings)

# ------------------------------
# Step 3: Encode user/movie IDs
# ------------------------------
print("3️⃣ Encoding IDs...")
ratings, user_encoder, movie_encoder = encode_ids(ratings)

# Save encoders for later
import pickle
os.makedirs("models", exist_ok=True)
with open("models/user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)
with open("models/movie_encoder.pkl", "wb") as f:
    pickle.dump(movie_encoder, f)

# ------------------------------
# Step 4: Feature engineering
# ------------------------------
print("4️⃣ Feature engineering...")
ratings, _, _ = create_user_item_embeddings(ratings)
ratings = create_context_features(ratings)

# Save features
os.makedirs("data/processed", exist_ok=True)
ratings.to_parquet(FEATURES_PATH, index=False)
print(f"Features saved to {FEATURES_PATH}")

# ------------------------------
# Step 5: Train MF model
# ------------------------------
print("5️⃣ Training MF model...")
num_users = ratings['user_id_enc'].nunique()
num_items = ratings['movie_id_enc'].nunique()
model = MF(num_users=num_users, num_items=num_items, emb_size=64)

# Training parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id_enc'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_id_enc'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

dataset = RatingsDataset(ratings)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

for epoch in range(5):
    total_loss = 0
    for users_batch, movies_batch, ratings_batch in loader:
        optimizer.zero_grad()
        preds = model(users_batch, movies_batch)
        loss = criterion(preds, ratings_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/5 - Avg Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, "context_mf.pth"))
print(f"MF model saved to {SAVED_MODEL_DIR}/context_mf.pth")

# ------------------------------
# Step 6: Recommend movies
# ------------------------------
print("6️⃣ Generating Top-N recommendations...")

# Get inputs
user_id = int(input("Enter User ID: "))
genre_input = input("Enter preferred genre (or press Enter to skip): ").strip()

all_items = ratings['movie_id_enc'].unique()
users_tensor = torch.tensor([ratings.loc[ratings['userId']==user_id,'user_id_enc'].iloc[0]] * len(all_items))
items_tensor = torch.tensor(all_items)

with torch.no_grad():
    scores = model(users_tensor, items_tensor).numpy()

score_df = pd.DataFrame({'movie_id_enc': all_items, 'score': scores})
recommended = ratings[['movie_id_enc','movieId']].drop_duplicates()
recommended = recommended.merge(movies, on='movieId')
recommended = recommended.merge(score_df, on='movie_id_enc')

# Filter by genre if given
if genre_input:
    recommended = recommended[recommended['genres'].str.contains(genre_input, case=False, na=False)]

recommended = recommended.sort_values('score', ascending=False).head(TOP_K)

# Print current time and recommendations
now = datetime.datetime.now()
print(f"\n📅 Current date & time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print("\n🎬 Top Recommendations:")
for i, row in enumerate(recommended.itertuples(),1):
    genres = row.genres.replace("|", ", ")
    print(f"{i}. {row.title} — {genres}")
