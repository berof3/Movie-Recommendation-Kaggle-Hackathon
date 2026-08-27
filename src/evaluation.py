import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from models import (BiasBaseline, ContentBasedModel, GlobalMeanBaseline, HybridModel,
                     MatrixFactorization, load_model, rmse)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CV_DIR = os.path.join("models", "cv")  # gitignored, like everything under models/
N_SPLITS = 5
SEED = 42


def k_fold_indices(ratings, n_splits=N_SPLITS, seed=SEED):
    """Row-level K-fold split, shared across all models for a fair, apples-to-apples comparison
    per fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(ratings))


def summarize(name, scores):
    scores = np.array(scores, dtype=float)
    print(f"{name:<20} mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores, 4).tolist()}")
    return scores.mean(), scores.std()


def evaluate_cheap_models(ratings, movie_features, n_splits=N_SPLITS):
    """K-fold CV for the models cheap enough to refit from scratch every fold (seconds, not
    tens of minutes): GlobalMeanBaseline, BiasBaseline, ContentBasedModel."""
    folds = k_fold_indices(ratings, n_splits=n_splits)
    results = {"GlobalMeanBaseline": [], "BiasBaseline": [], "ContentBasedModel": []}

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        train = ratings.iloc[train_idx].reset_index(drop=True)
        val = ratings.iloc[val_idx].reset_index(drop=True)
        print(f"\n=== fold {fold_i} ===  train={len(train):,}  val={len(val):,}")

        t0 = time.time()
        gm = GlobalMeanBaseline().fit(train)
        r = rmse(val["rating"], gm.predict(val["userId"], val["movieId"]))
        results["GlobalMeanBaseline"].append(r)
        print(f"GlobalMeanBaseline  RMSE={r:.4f}  ({time.time()-t0:.0f}s)")

        t0 = time.time()
        bias = BiasBaseline().fit(train)
        r = rmse(val["rating"], bias.predict(val["userId"], val["movieId"]))
        results["BiasBaseline"].append(r)
        print(f"BiasBaseline        RMSE={r:.4f}  ({time.time()-t0:.0f}s)")

        t0 = time.time()
        cb = ContentBasedModel().fit(train, movie_features)
        r = rmse(val["rating"], cb.predict(val["userId"], val["movieId"]))
        results["ContentBasedModel"].append(r)
        print(f"ContentBasedModel   RMSE={r:.4f}  ({time.time()-t0:.0f}s)")

    print("\n--- Cheap-model CV summary ---")
    summary = {name: summarize(name, scores) for name, scores in results.items()}
    return results, summary


def run_mf_cv_chunk(ratings, epochs_per_chunk=4, n_splits=N_SPLITS):
    """Does ONE bounded chunk of MatrixFactorization CV work (one fold's next few epochs) and
    returns. Meant to be called repeatedly (e.g. from a short foreground command) until it
    reports every fold done - a single fold's full training takes tens of minutes, too long for
    one uninterrupted call to be reliable in this environment (see TASKS.md / commit history).
    Checkpoints to models/cv/mf_fold_{i}.pkl between calls, keyed on the same K-fold split
    k_fold_indices() produces (fixed seed, so folds line up call to call)."""
    os.makedirs(CV_DIR, exist_ok=True)
    folds = k_fold_indices(ratings, n_splits=n_splits)

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        ckpt_path = os.path.join(CV_DIR, f"mf_fold_{fold_i}.pkl")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                mf = pickle.load(f)
            if mf.stopped_early_:
                continue
        else:
            mf = MatrixFactorization(n_factors=50, n_epochs=epochs_per_chunk, lr=0.01, reg=0.015,
                                      batch_size=100_000, patience=4, seed=SEED)

        train = ratings.iloc[train_idx].reset_index(drop=True)
        val = ratings.iloc[val_idx].reset_index(drop=True)

        mf.n_epochs = epochs_per_chunk
        t0 = time.time()
        mf.fit(train, val=val, warm_start=True)
        print(f"[fold {fold_i}] chunk took {time.time()-t0:.0f}s  epochs_trained_={mf.epochs_trained_}  "
              f"stopped_early_={mf.stopped_early_}  best_val_rmse_={mf.best_val_rmse_:.4f}")

        with open(ckpt_path, "wb") as f:
            pickle.dump(mf, f)
        return False  # more work remains (this fold, or later ones)
    else:
        print("ALL FOLDS DONE")
        return True


def mf_cv_summary(n_splits=N_SPLITS):
    """Reads back the checkpointed per-fold MF models (once run_mf_cv_chunk reports done) and
    reports their best_val_rmse_ - the same aggregation as evaluate_cheap_models but for the
    already-completed MatrixFactorization folds."""
    scores = []
    for fold_i in range(n_splits):
        ckpt_path = os.path.join(CV_DIR, f"mf_fold_{fold_i}.pkl")
        with open(ckpt_path, "rb") as f:
            mf = pickle.load(f)
        if not mf.stopped_early_:
            raise RuntimeError(f"fold {fold_i} hasn't finished training yet")
        scores.append(mf.best_val_rmse_)
    summarize("MatrixFactorization", scores)
    return scores


def generate_submission(hybrid_model, test_path=None, out_path="submission.csv"):
    """Generates the actual Kaggle submission file from a fitted HybridModel, matching
    sample_submission.csv's format: Id = "{userId}_{movieId}", rating = prediction."""
    test_path = test_path or os.path.join(RAW_DIR, "test.csv")
    test = pd.read_csv(test_path)
    preds = hybrid_model.predict(test["userId"], test["movieId"])
    submission = pd.DataFrame({
        "Id": test["userId"].astype(str) + "_" + test["movieId"].astype(str),
        "rating": preds,
    })
    submission.to_csv(out_path, index=False)
    print(f"Saved submission -> {out_path}  ({len(submission):,} rows)")
    return submission


if __name__ == "__main__":
    ratings = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    movie_features = pd.read_csv(os.path.join(PROCESSED_DIR, "movies_features.csv"), low_memory=False)

    evaluate_cheap_models(ratings, movie_features)

    print("\n--- MatrixFactorization CV ---")
    print("(this is the slow part - each fold takes ~30-45 min to converge; run_mf_cv_chunk is "
          "checkpointed to models/cv/, so this loop can be killed and re-run to resume)")
    done = False
    while not done:
        done = run_mf_cv_chunk(ratings)
    mf_cv_summary()

    print("\n--- Generating final Kaggle submission (HybridModel, trained on all of train.csv) ---")
    hybrid = HybridModel(load_model("matrix_factorization"), load_model("content_based"),
                          ratings.groupby("movieId").size())
    generate_submission(hybrid)
