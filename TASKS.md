# Task List

Tracks progress through the ML pipeline for the ALX Movie Recommendation Kaggle Hackathon 2026.
See `README.md` for full methodology context.

- [x] **1. Data ingestion** — `src/data_loader.py`
      Downloads competition data via Kaggle API into `data/raw/` (skips if already present).
- [x] **2. Exploratory Data Analysis** — `notebooks/01_eda.ipynb`
      Rating distribution, matrix sparsity, user/item activity, genre breakdown, tag genome
      overview. Executed end to end against the real data (10M ratings, 62,423 movies): 99.87%
      matrix sparsity, rating mean 3.53, most-rated movie Shawshank Redemption (32,831 ratings).
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
- [x] **4b. Content-based filtering** — `src/models.py`
      Ridge regression on movie genres/year/runtime/budget + the user's own taste profile
      (overall average, per-genre average shrunk toward it), predicting the *residual* from the
      user's average rather than the rating directly (avoids a collinearity/leakage trap - see
      commit history). Validated on a true item cold-start holdout (12.7% of movies' ratings
      entirely withheld, matching test.csv's real cold-start rate): GlobalMean 1.0663,
      BiasBaseline/MF-fallback 0.9816, **ContentBased 0.9275** - the improvement this model
      exists for. On regular (non-cold) rows it scores 0.9223, worse than MatrixFactorization's
      0.8535, which is expected: it has no per-item learned factor, only metadata. Not yet used
      genome_features.csv (only ~22% movie coverage) or blended with MF - candidates for later.
      Saved to `models/content_based.pkl` (gitignored).
- [x] **4c. Hybrid blend** — `src/models.py` (`HybridModel`, `build_hybrid_model`)
      Blends MF + ContentBased predictions with a weight that scales with how many training
      ratings the target movie had: `alpha(count) = max_alpha * count / (count + k)`
      (max_alpha=0.8, k=10) - same shrinkage idea as BiasBaseline/genre affinity. Weight curve
      grounded in a grid search across rating-count buckets on held-out data (0 ratings -> best
      alpha~0.3 but noisy/small-n; 1-5 -> 0.45; 5-20 -> 0.60; 20-100 -> 0.70; 100+ -> 0.80).
      Validated system-level estimate (87.3% regular / 12.7% cold rows, matching test.csv's real
      composition): pure MF 0.8713 -> **Hybrid 0.8598** (~1.3% relative RMSE improvement).
      Not persisted separately (would duplicate the ~90MB MF pickle) - built on demand via
      `build_hybrid_model(train_ratings)`, which loads the two saved models and recomputes the
      (cheap) rating-count table.
- [x] **5. Evaluation** — `src/evaluation.py`
      5-fold CV (row-level, shared splits across models for a fair comparison), full rigor
      including MatrixFactorization at full convergence per fold (chunked/checkpointed to
      `models/cv/` - each fold takes ~30-45 min). Results (mean ± std across 5 folds):

      | Model | mean RMSE | std |
      |---|---|---|
      | GlobalMeanBaseline | 1.0611 | 0.0006 |
      | BiasBaseline | 0.8654 | 0.0005 |
      | ContentBasedModel | 0.9245 | 0.0006 |
      | MatrixFactorization | 0.8663 | 0.0022 |

      All low-variance/robust across folds. MatrixFactorization's CV mean (0.8663) is a bit
      higher than the earlier single-holdout number (0.8535) - expected and honest: CV here uses
      80/20 splits (less training data per fold) vs. the original 95/5 holdout. HybridModel
      wasn't re-run through full CV (would mean re-deriving cold-start-holdout numbers per fold
      too); its validated system-level estimate (0.8598, see 4c) stands as the reported number.
      `generate_submission()` produces the actual Kaggle submission (`submission.csv`, gitignored
      - regenerable, not source) from the HybridModel trained on all of train.csv; verified its
      format exactly matches `sample_submission.csv` (same columns, row count, Id set).
- [x] **6. Reporting** — `reports/final_report.md`, `figures/`, `notebooks/02_results_report.ipynb`
      Written summary of the full pipeline, methodology, and results (including the real bugs
      hit and fixed along the way), with three figures: 5-fold CV model comparison, hybrid
      blending's impact per scenario, and the blend-weight curve with its empirical grounding.
      Honest limitations section (genome features unused, hybrid not re-run through full CV,
      noisy middle blend-weight buckets, no true ensembling beyond one blend weight).

## Pipeline complete
All 6 stages done. See `reports/final_report.md` for the full write-up and `submission.csv`
(gitignored, regenerate via `src/evaluation.py:generate_submission()`) for the Kaggle deliverable.

## Notes
- Update this file (check off items, add sub-tasks as they emerge) as work progresses — it's the
  source of truth for "what's next" across sessions.
