import pandas as pd
import os

def load_movielens(path="data/raw/ml-1m"):
    users = pd.read_csv(
        os.path.join(path, "users.dat"),
        sep="::", engine='python', header=None,
        names=['userId', 'gender', 'age', 'occupation', 'zip'],
        encoding='latin-1'
    )
    
    movies = pd.read_csv(
        os.path.join(path, "movies.dat"),
        sep="::", engine='python', header=None,
        names=['movieId', 'title', 'genres'], encoding='latin-1'
    )
    
    ratings = pd.read_csv(
        os.path.join(path, "ratings.dat"),
        sep="::", engine='python', header=None,
        names=['userId', 'movieId', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    
    return users, movies, ratings

if __name__ == "__main__":
    users, movies, ratings = load_movielens()
    print(users.head())
    print(movies.head())
    print(ratings.head())
