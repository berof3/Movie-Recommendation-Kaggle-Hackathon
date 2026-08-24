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
- [ ] **4. Modeling** — `src/models.py`
      Collaborative filtering (SVD/ALS) and content-based filtering.
- [ ] **5. Evaluation** — `src/evaluation.py`
      K-fold cross-validation, RMSE tracking against the competition metric.
- [ ] **6. Reporting** — `reports/`, `figures/`
      Final performance analysis and visualizations.

## Notes
- Update this file (check off items, add sub-tasks as they emerge) as work progresses — it's the
  source of truth for "what's next" across sessions.
