import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_ratings(ratings):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['hour'] = ratings['timestamp'].dt.hour
    ratings['day_of_week'] = ratings['timestamp'].dt.dayofweek
    ratings['month'] = ratings['timestamp'].dt.month
    return ratings

def save_processed_data(ratings, path="data/processed/"):
    os.makedirs(path, exist_ok=True)
    ratings.to_parquet(os.path.join(path, "ratings_processed.parquet"), index=False)

def encode_ids(ratings):
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings['user_id_enc'] = user_encoder.fit_transform(ratings['userId'])
    ratings['movie_id_enc'] = movie_encoder.fit_transform(ratings['movieId'])

    return ratings, user_encoder, movie_encoder

if __name__ == "__main__":
    ratings = pd.read_csv("data/raw/ml-1m/ratings.dat", sep="::", engine='python',
                          header=None, names=['userId','movieId','rating','timestamp'],
                          encoding='latin-1')
    ratings = preprocess_ratings(ratings)
    save_processed_data(ratings)
    print("Preprocessing complete.")
