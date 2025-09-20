import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def create_user_item_embeddings(ratings):
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    
    ratings['user_id_enc'] = user_enc.fit_transform(ratings['userId'])
    ratings['movie_id_enc'] = item_enc.fit_transform(ratings['movieId'])
    
    return ratings, user_enc, item_enc

def create_context_features(ratings):
    ratings['hour_norm'] = ratings['hour'] / 23
    ratings['day_of_week_norm'] = ratings['day_of_week'] / 6
    return ratings

if __name__ == "__main__":
    ratings = pd.read_parquet("data/processed/ratings_processed.parquet")
    ratings, user_enc, item_enc = create_user_item_embeddings(ratings)
    ratings = create_context_features(ratings)
    
    os.makedirs("data/processed", exist_ok=True)
    ratings.to_parquet("data/processed/features.parquet", index=False)
    print("Feature engineering complete.")
