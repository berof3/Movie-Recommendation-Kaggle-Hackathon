import os
import re

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def extract_release_year(title):
    """Pulls the 4-digit year out of a "Title (YYYY)" string; NaN if absent/malformed."""
    match = _YEAR_RE.search(str(title))
    return int(match.group(1)) if match else np.nan


def clean_title(title):
    """Strips the trailing " (YYYY)" from a movie title."""
    return _YEAR_RE.sub("", str(title)).strip()


def clean_budget(value):
    """Converts a "$30,000,000" style string into a float; NaN if missing/unparseable."""
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"[^\d.]", "", str(value))
    return float(digits) if digits else np.nan


def encode_genres(movies):
    """Multi-hot encodes the pipe-delimited genres column into one `genre_<name>` column each."""
    exploded = (
        movies[["movieId", "genres"]]
        .assign(genres=movies["genres"].replace("(no genres listed)", np.nan).str.split("|"))
        .explode("genres")
        .dropna(subset=["genres"])
        .reset_index(drop=True)  # explode() keeps duplicate index labels; crosstab needs unique ones
    )
    dummies = pd.crosstab(exploded["movieId"], exploded["genres"])
    dummies.columns = [f"genre_{c}" for c in dummies.columns]
    return dummies


def build_movie_features(movies, imdb_data):
    """Builds one feature row per movie: parsed title/year, multi-hot genres, and cleaned
    IMDb metadata (director/runtime/budget) where available, with missing-data flags since
    IMDb coverage is partial (~40% of movies)."""
    features = movies[["movieId"]].copy()
    features["clean_title"] = movies["title"].apply(clean_title)
    features["release_year"] = movies["title"].apply(extract_release_year)

    genre_dummies = encode_genres(movies)
    features = features.merge(genre_dummies, on="movieId", how="left")
    genre_cols = list(genre_dummies.columns)
    features[genre_cols] = features[genre_cols].fillna(0).astype(int)

    imdb_clean = imdb_data[["movieId", "director", "runtime", "budget"]].copy()
    imdb_clean["budget"] = imdb_clean["budget"].apply(clean_budget)
    features = features.merge(imdb_clean, on="movieId", how="left")
    features["has_imdb_data"] = features["movieId"].isin(imdb_data["movieId"]).astype(int)

    return features


def build_genome_features(genome_scores, genome_tags, top_n=50):
    """Pivots the tag genome into a movieId x tag relevance matrix, kept to the `top_n` tags
    with the highest variance in relevance (i.e. most discriminative between movies).
    Coverage is partial (~22% of movies have any genome data) - meant as an optional signal
    for content-based modeling, not merged into the main movie feature table."""
    tag_variance = genome_scores.groupby("tagId")["relevance"].var().sort_values(ascending=False)
    top_tag_ids = tag_variance.head(top_n).index

    pivot = genome_scores[genome_scores["tagId"].isin(top_tag_ids)].pivot(
        index="movieId", columns="tagId", values="relevance"
    )
    tag_names = genome_tags.set_index("tagId")["tag"]
    pivot.columns = [f"tag_{tag_names.get(c, c)}" for c in pivot.columns]
    return pivot.reset_index()


def run_preprocessing():
    """Loads raw competition data, builds engineered feature tables, and writes them to
    data/processed/. Skips work if the outputs already exist."""
    movies_out = os.path.join(PROCESSED_DIR, "movies_features.csv")
    genome_out = os.path.join(PROCESSED_DIR, "genome_features.csv")

    if os.path.exists(movies_out) and os.path.exists(genome_out):
        print(f"Processed features already present in {PROCESSED_DIR}, skipping.")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw data...")
    movies = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    imdb_data = pd.read_csv(os.path.join(RAW_DIR, "imdb_data.csv"))

    print("Building movie features (genres, release year, IMDb metadata)...")
    movie_features = build_movie_features(movies, imdb_data)
    movie_features.to_csv(movies_out, index=False)
    print(f"Saved {movie_features.shape} -> {movies_out}")

    print("Building genome tag features (top-variance tags)...")
    genome_scores = pd.read_csv(os.path.join(RAW_DIR, "genome_scores.csv"))
    genome_tags = pd.read_csv(os.path.join(RAW_DIR, "genome_tags.csv"))
    genome_features = build_genome_features(genome_scores, genome_tags)
    genome_features.to_csv(genome_out, index=False)
    print(f"Saved {genome_features.shape} -> {genome_out}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
