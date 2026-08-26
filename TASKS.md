# Task List

Tracks progress through the ML pipeline for the ALX Movie Recommendation Kaggle Hackathon 2026.
See `README.md` for full methodology context.

- [x] **1. Data ingestion** — `src/data_loader.py`
      Downloads competition data via Kaggle API into `data/raw/` (skips if already present).
- [ ] **2. Exploratory Data Analysis** — `notebooks/01_eda.ipynb`
      Rating distribution, matrix sparsity, user/item activity, genre breakdown, tag genome overview.
- [x] **3. Feature engineering / preprocessing** — `src/preprocessing.py`
      Builds `data/processed/movies_features.csv` (multi-hot genres, release year, cleaned IMDb
      director/runtime/budget, `has_imdb_data` flag — full movie coverage) and
      `data/processed/genome_features.csv` (top-50 highest-variance tag relevance scores,
      ~22% movie coverage, kept separate as an optional content-based signal).
- [x] **4a. Collaborative filtering (matrix factorization)** — `src/models.py`
      Custom Funk-SVD-style model (bias-aware, mini-batch SGD, early stopping) trained on
      `train.csv`. Holdout val RMSE: GlobalMeanBaseline 1.0611, BiasBaseline 0.8649,
      **MatrixFactorization 0.8535** (50 factors, reg=0.015, early-stopped at epoch 31/35).
      Trained models saved to `models/*.pkl` (gitignored). Known gap: 12.7% of test.csv movies
      never appear in train.csv (item cold start) - CF alone falls back to global mean + user
      bias for those, which is exactly where content-based features should help.
- [ ] **4b. Content-based filtering** — `src/models.py`
      Use `data/processed/movies_features.csv` / `genome_features.csv` to cover the item
      cold-start gap and (later) blend with the CF model.
- [ ] **5. Evaluation** — `src/evaluation.py`
      K-fold cross-validation, RMSE tracking against the competition metric.
- [ ] **6. Reporting** — `reports/`, `figures/`
      Final performance analysis and visualizations.

## Notes
- Update this file (check off items, add sub-tasks as they emerge) as work progresses — it's the
  source of truth for "what's next" across sessions.
