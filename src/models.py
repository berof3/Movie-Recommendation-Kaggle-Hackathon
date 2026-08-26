import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
GENRE_PREFIX = "genre_"


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_val_split(ratings, val_frac=0.05, seed=42):
    """Random holdout split used only for training-time monitoring/early stopping.
    Formal cross-validation for model comparison lives in src/evaluation.py."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(ratings))
    n_val = int(len(ratings) * val_frac)
    val_idx, train_idx = shuffled[:n_val], shuffled[n_val:]
    return ratings.iloc[train_idx].reset_index(drop=True), ratings.iloc[val_idx].reset_index(drop=True)


def movie_holdout_split(ratings, movie_frac=0.05, seed=42):
    """Holds out ALL ratings for a random subset of movies (rather than a random subset of
    rows) - simulates true item cold start, i.e. exactly the situation where ~12.7% of
    test.csv's movies never appear in train.csv at all. A random row-level holdout can't
    measure this: with 10M+ rows spread over ~48k movies, almost every movie keeps some
    training rows regardless of which rows are held out."""
    rng = np.random.default_rng(seed)
    movie_ids = ratings["movieId"].unique()
    n_cold = int(len(movie_ids) * movie_frac)
    cold_ids = set(rng.choice(movie_ids, size=n_cold, replace=False))
    is_cold = ratings["movieId"].isin(cold_ids)
    return ratings[~is_cold].reset_index(drop=True), ratings[is_cold].reset_index(drop=True)


def _batch_counts(idx, n_rows):
    """How many times each row index appears in this mini-batch."""
    return np.bincount(idx, minlength=n_rows)


def _scatter_add_1d(target, idx, values, inv_count):
    """target[idx] += mean(values) per duplicate index, using the precomputed 1/count so a
    popular movie appearing hundreds of times in one batch gets one averaged-size update
    instead of hundreds of stacked ones (which is what caused training to diverge to NaN when
    summed raw)."""
    delta = np.bincount(idx, weights=values, minlength=len(target)) * inv_count
    target += delta


def _scatter_add_2d(target, idx, values, inv_count):
    """Same as _scatter_add_1d but for a (n, k) factor matrix. The sum-then-divide-by-count is
    done via a sparse matrix multiply (optimized C code) rather than a Python-level loop, since
    this is the hot path in training."""
    n_rows, m = target.shape[0], len(idx)
    indicator = coo_matrix((np.ones(m), (idx, np.arange(m))), shape=(n_rows, m)).tocsr()
    summed = indicator @ values
    target += summed * inv_count[:, None]


class GlobalMeanBaseline:
    """Predicts the training set's global average rating for everything. The floor any real
    model needs to beat - if a fancier model can't clear this, it isn't learning anything."""

    def fit(self, ratings):
        self.mean_ = float(ratings["rating"].mean())
        return self

    def predict(self, user_ids, movie_ids):
        return np.full(len(user_ids), self.mean_)


class BiasBaseline:
    """pred(u, i) = global_mean + user_bias[u] + item_bias[i], fit by alternating closed-form
    ridge regression. Captures "this user tends to rate high/low" and "this movie tends to be
    rated high/low" without any latent-factor interaction - a much stronger baseline than the
    global mean, and the number MatrixFactorization needs to beat to prove the extra complexity
    (latent factors) is earning its keep."""

    def __init__(self, reg=10.0, n_iters=10):
        self.reg = reg
        self.n_iters = n_iters

    def fit(self, ratings):
        self.user_ids_, u = np.unique(ratings["userId"].to_numpy(), return_inverse=True)
        self.movie_ids_, i = np.unique(ratings["movieId"].to_numpy(), return_inverse=True)
        r = ratings["rating"].to_numpy(dtype=np.float64)
        n_users, n_movies = len(self.user_ids_), len(self.movie_ids_)

        self.mean_ = float(r.mean())
        b_u = np.zeros(n_users)
        b_i = np.zeros(n_movies)

        for _ in range(self.n_iters):
            resid_i = r - self.mean_ - b_u[u]
            b_i = np.bincount(i, weights=resid_i, minlength=n_movies) / (
                np.bincount(i, minlength=n_movies) + self.reg
            )

            resid_u = r - self.mean_ - b_i[i]
            b_u = np.bincount(u, weights=resid_u, minlength=n_users) / (
                np.bincount(u, minlength=n_users) + self.reg
            )

        self.b_u_, self.b_i_ = b_u, b_i
        return self

    def _encode(self, ids, id_array):
        ids = np.asarray(ids)
        idx = np.clip(np.searchsorted(id_array, ids), 0, len(id_array) - 1)
        valid = id_array[idx] == ids
        return idx, valid

    def predict(self, user_ids, movie_ids):
        u_idx, u_valid = self._encode(user_ids, self.user_ids_)
        i_idx, i_valid = self._encode(movie_ids, self.movie_ids_)

        preds = np.full(len(u_idx), self.mean_)
        preds[u_valid] += self.b_u_[u_idx[u_valid]]
        preds[i_valid] += self.b_i_[i_idx[i_valid]]
        return np.clip(preds, 0.5, 5.0)


class MatrixFactorization:
    """Bias-aware matrix factorization - the "SVD"-style collaborative filtering approach from
    Funk's Netflix Prize formulation:

        pred(u, i) = mean + b_u[u] + b_i[i] + p_u[u] . q_i[i]

    Fit by mini-batch stochastic gradient descent with L2 regularization on every learned term.
    Users/movies unseen during training (cold start) fall back gracefully: unknown-unknown ->
    global mean, unknown movie -> mean + user bias only, unknown user -> mean + item bias only.
    """

    def __init__(self, n_factors=50, n_epochs=40, lr=0.01, reg=0.015, batch_size=100_000, seed=42,
                 patience=4):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        # early stopping: with `val` passed to fit(), stop once `patience` epochs pass with no
        # val RMSE improvement, and keep the best-seen parameters rather than the final epoch's -
        # needed because low regularization lets train RMSE keep improving well past the point
        # where val RMSE stops (i.e. after the model starts overfitting).
        self.patience = patience

    def fit(self, ratings, val=None, verbose=True, warm_start=False):
        """Trains for self.n_epochs more epochs. With warm_start=True, continues from this
        instance's existing parameters and early-stopping bookkeeping instead of reinitializing -
        lets a long run be split into several short, resumable fit() calls (e.g. across separate
        foreground commands) rather than needing one uninterrupted long-running process. Returns
        self; check self.stopped_early_ to see whether early stopping already fired (in which
        case further warm_start=True calls are a no-op)."""
        rng = np.random.default_rng(self.seed + getattr(self, "epochs_trained_", 0))

        fresh = not warm_start or not hasattr(self, "p_")
        if fresh:
            self.user_ids_, u = np.unique(ratings["userId"].to_numpy(), return_inverse=True)
            self.movie_ids_, i = np.unique(ratings["movieId"].to_numpy(), return_inverse=True)
            n_users, n_movies = len(self.user_ids_), len(self.movie_ids_)

            self.mean_ = float(ratings["rating"].to_numpy(dtype=np.float64).mean())
            self.b_u_ = np.zeros(n_users)
            self.b_i_ = np.zeros(n_movies)
            self.p_ = rng.normal(0, 0.1, (n_users, self.n_factors))
            self.q_ = rng.normal(0, 0.1, (n_movies, self.n_factors))

            self.epochs_trained_ = 0
            self._best_val_rmse = np.inf
            self._best_state = None
            self._epochs_no_improve = 0
            self.stopped_early_ = False
        else:
            # re-encode with the existing (already-fitted) id mappings, not a fresh np.unique -
            # the same training set must be passed on every resume call for indices to line up.
            u, u_valid = self._encode(ratings["userId"].to_numpy(), self.user_ids_)
            i, i_valid = self._encode(ratings["movieId"].to_numpy(), self.movie_ids_)
            if not (u_valid.all() and i_valid.all()):
                raise ValueError("warm_start=True requires the same `ratings` used on the first fit() call")
            n_users, n_movies = len(self.user_ids_), len(self.movie_ids_)

        if self.stopped_early_:
            if verbose:
                print("Early stopping already fired on a previous fit() call - skipping.")
            return self

        r = ratings["rating"].to_numpy(dtype=np.float64)
        n = len(r)

        best_val_rmse = self._best_val_rmse
        best_state = self._best_state
        epochs_no_improve = self._epochs_no_improve

        start_epoch = self.epochs_trained_ + 1
        for epoch in range(start_epoch, start_epoch + self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start:start + self.batch_size]
                bu, bi, ru = u[batch], i[batch], r[batch]

                cnt_u = _batch_counts(bu, n_users)
                cnt_i = _batch_counts(bi, n_movies)
                inv_cnt_u = 1.0 / np.maximum(cnt_u, 1)
                inv_cnt_i = 1.0 / np.maximum(cnt_i, 1)

                pred = self.mean_ + self.b_u_[bu] + self.b_i_[bi] + np.einsum(
                    "ij,ij->i", self.p_[bu], self.q_[bi]
                )
                err = ru - pred

                _scatter_add_1d(self.b_u_, bu, self.lr * (err - self.reg * self.b_u_[bu]), inv_cnt_u)
                _scatter_add_1d(self.b_i_, bi, self.lr * (err - self.reg * self.b_i_[bi]), inv_cnt_i)
                _scatter_add_2d(self.p_, bu, self.lr * (err[:, None] * self.q_[bi] - self.reg * self.p_[bu]), inv_cnt_u)
                _scatter_add_2d(self.q_, bi, self.lr * (err[:, None] * self.p_[bu] - self.reg * self.q_[bi]), inv_cnt_i)

            train_rmse_val = None
            val_rmse_val = None
            if verbose:
                train_pred = self.predict(ratings["userId"].to_numpy(), ratings["movieId"].to_numpy())
                train_rmse_val = rmse(r, train_pred)

            if val is not None:
                val_pred = self.predict(val["userId"].to_numpy(), val["movieId"].to_numpy())
                val_rmse_val = rmse(val["rating"], val_pred)

                if val_rmse_val < best_val_rmse - 1e-5:
                    best_val_rmse = val_rmse_val
                    epochs_no_improve = 0
                    best_state = (self.b_u_.copy(), self.b_i_.copy(), self.p_.copy(), self.q_.copy())
                else:
                    epochs_no_improve += 1

            self.epochs_trained_ = epoch

            if verbose:
                msg = f"epoch {epoch:>3} (total)  train RMSE={train_rmse_val:.4f}"
                if val_rmse_val is not None:
                    msg += f"  val RMSE={val_rmse_val:.4f}"
                    if epochs_no_improve == 0:
                        msg += "  (best)"
                print(msg)

            if val is not None and epochs_no_improve >= self.patience:
                self.stopped_early_ = True
                if verbose:
                    print(f"No val improvement for {self.patience} epochs - stopping early, "
                          f"restoring best (val RMSE={best_val_rmse:.4f}).")
                break

        self._best_val_rmse = best_val_rmse
        self._best_state = best_state
        self._epochs_no_improve = epochs_no_improve

        if best_state is not None:
            self.b_u_, self.b_i_, self.p_, self.q_ = best_state
            self.best_val_rmse_ = best_val_rmse

        return self

    def _encode(self, ids, id_array):
        ids = np.asarray(ids)
        idx = np.clip(np.searchsorted(id_array, ids), 0, len(id_array) - 1)
        valid = id_array[idx] == ids
        return idx, valid

    def predict(self, user_ids, movie_ids):
        u_idx, u_valid = self._encode(user_ids, self.user_ids_)
        i_idx, i_valid = self._encode(movie_ids, self.movie_ids_)

        preds = np.full(len(u_idx), self.mean_)
        preds[u_valid] += self.b_u_[u_idx[u_valid]]
        preds[i_valid] += self.b_i_[i_idx[i_valid]]

        both = u_valid & i_valid
        preds[both] += np.einsum("ij,ij->i", self.p_[u_idx[both]], self.q_[i_idx[both]])
        return np.clip(preds, 0.5, 5.0)


class ContentBasedModel:
    """Predicts a rating from movie content features (genres, release year, IMDb metadata) and
    the user's own historical taste profile - their overall average rating, and their average
    rating within each genre, shrunk toward their overall average when they have few ratings in
    that genre - via ridge regression.

    Unlike MatrixFactorization, this needs zero training ratings for a given movie to make a
    prediction - only that movie's metadata - so it covers the item cold-start gap (~12.7% of
    test.csv movies never appear in train.csv) that pure collaborative filtering can't: a movie
    with no training ratings still has a genre/year/runtime/budget feature row, so the model can
    still fall back on "users who liked similar movies liked this one too" rather than just the
    dataset-wide average.
    """

    def __init__(self, alpha=5.0, genre_shrinkage=5.0):
        self.alpha = alpha
        self.genre_shrinkage = genre_shrinkage

    def fit(self, ratings, movie_features):
        self.genre_cols_ = [c for c in movie_features.columns if c.startswith(GENRE_PREFIX)]
        mf = movie_features.set_index("movieId")

        self.year_mean_, self.year_std_ = mf["release_year"].mean(), mf["release_year"].std()
        self.runtime_mean_, self.runtime_std_ = mf["runtime"].mean(), mf["runtime"].std()
        log_budget = np.log1p(mf["budget"])
        self.log_budget_mean_, self.log_budget_std_ = log_budget.mean(), log_budget.std()

        movie_table = mf[self.genre_cols_].fillna(0).astype(float).copy()
        movie_table["release_year_norm"] = ((mf["release_year"] - self.year_mean_) / self.year_std_).fillna(0.0)
        movie_table["has_year"] = mf["release_year"].notna().astype(float)
        movie_table["runtime_norm"] = ((mf["runtime"] - self.runtime_mean_) / self.runtime_std_).fillna(0.0)
        movie_table["has_runtime"] = mf["runtime"].notna().astype(float)
        movie_table["budget_norm"] = ((log_budget - self.log_budget_mean_) / self.log_budget_std_).fillna(0.0)
        movie_table["has_budget"] = mf["budget"].notna().astype(float)
        movie_table["has_imdb_data"] = mf["has_imdb_data"].astype(float)
        self.movie_table_ = movie_table  # indexed by movieId, one row per movie, all numeric

        # --- user taste profile, derived purely from training ratings ---
        r = ratings["rating"].to_numpy(dtype=np.float64)
        self.global_mean_ = float(r.mean())

        self.user_ids_, u = np.unique(ratings["userId"].to_numpy(), return_inverse=True)
        n_users = len(self.user_ids_)
        user_sum = np.bincount(u, weights=r, minlength=n_users)
        user_cnt = np.bincount(u, minlength=n_users)
        self.user_avg_ = user_sum / np.maximum(user_cnt, 1)

        genre_lookup = self.movie_table_[self.genre_cols_].reindex(ratings["movieId"]).to_numpy()
        weighted = genre_lookup * r[:, None]
        indicator = coo_matrix((np.ones(len(u)), (u, np.arange(len(u)))), shape=(n_users, len(u))).tocsr()
        genre_sum = indicator @ weighted
        genre_cnt = indicator @ genre_lookup

        # Bayesian shrinkage: a user's genre average pulls toward their overall average until
        # they've rated enough movies in that genre for the genre-specific number to be trusted.
        # This full (non-leave-one-out) version is what gets stored for predicting on genuinely
        # new rows later - see the leave-one-out version just below for why *fitting* Ridge needs
        # something different.
        k = self.genre_shrinkage
        self.user_genre_avg_ = (genre_sum + k * self.user_avg_[:, None]) / (genre_cnt + k)

        user_avg_train = self._user_avg(ratings["userId"].to_numpy())

        # Leave-one-out for the Ridge *training* features specifically: genre_sum/genre_cnt
        # above are aggregated over ALL of a user's training ratings, so using them as-is to
        # build a training row's own feature lets that row's own rating leak into its own
        # feature (severe for a user with only 1-2 ratings in a genre - the "average" is then
        # nearly the row's own target). Subtracting each row's own contribution before computing
        # its feature removes that leak; self.user_genre_avg_ (used for prediction on rows that
        # were never part of the aggregate to begin with) doesn't need this adjustment.
        loo_sum = genre_sum[u] - weighted
        loo_cnt = genre_cnt[u] - genre_lookup
        loo_avg = (loo_sum + k * user_avg_train[:, None]) / (loo_cnt + k)
        g_sum = genre_lookup.sum(axis=1)
        genre_affinity_train = user_avg_train.copy()
        has_signal = g_sum > 0
        genre_affinity_train[has_signal] = (
            (loo_avg[has_signal] * genre_lookup[has_signal]).sum(axis=1) / g_sum[has_signal]
        )

        other_cols = ["release_year_norm", "has_year", "runtime_norm", "has_runtime",
                      "budget_norm", "has_budget", "has_imdb_data"]
        other_features_train = self.movie_table_[other_cols].reindex(
            ratings["movieId"]).fillna(0.0).to_numpy()
        X = np.column_stack([
            genre_affinity_train - user_avg_train,
            genre_lookup,
            other_features_train,
        ])
        # Regress the residual after subtracting the user's own average, rather than including
        # user_avg as a free-standing feature: user_avg and genre_affinity are highly correlated
        # (genre_affinity falls back to user_avg whenever there's no/weak genre signal), and
        # giving Ridge two near-duplicate features let it fit a huge, near-cancelling coefficient
        # pair that matched training noise but blew up on validation (train RMSE 0.51, val RMSE
        # 1.13 - worse than predicting the global mean for everyone). Factoring the deterministic
        # per-user baseline out algebraically leaves Ridge a much smaller, better-behaved problem:
        # only "how should this movie's genre/year/budget shift the prediction relative to what
        # this user usually rates".
        residual = r - user_avg_train
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X, residual)
        return self

    def _user_avg(self, user_ids):
        user_ids = np.asarray(user_ids)
        u_idx = np.clip(np.searchsorted(self.user_ids_, user_ids), 0, len(self.user_ids_) - 1)
        u_valid = self.user_ids_[u_idx] == user_ids
        return np.where(u_valid, self.user_avg_[u_idx], self.global_mean_)

    def _build_features(self, user_ids, movie_ids):
        user_ids = np.asarray(user_ids)
        movie_ids = np.asarray(movie_ids)

        u_idx = np.clip(np.searchsorted(self.user_ids_, user_ids), 0, len(self.user_ids_) - 1)
        u_valid = self.user_ids_[u_idx] == user_ids
        user_avg = self._user_avg(user_ids)

        genre_features = self.movie_table_[self.genre_cols_].reindex(movie_ids).fillna(0.0).to_numpy()
        g_sum = genre_features.sum(axis=1)

        genre_affinity = user_avg.copy()  # default: no genre signal -> fall back to user's overall average
        has_signal = u_valid & (g_sum > 0)
        if has_signal.any():
            per_row_genre_avg = self.user_genre_avg_[u_idx[has_signal]]
            g = genre_features[has_signal]
            genre_affinity[has_signal] = (per_row_genre_avg * g).sum(axis=1) / g_sum[has_signal]

        other_cols = ["release_year_norm", "has_year", "runtime_norm", "has_runtime",
                      "budget_norm", "has_budget", "has_imdb_data"]
        other_features = self.movie_table_[other_cols].reindex(movie_ids).fillna(0.0).to_numpy()

        return np.column_stack([
            genre_affinity - user_avg,  # this genre's pull relative to the user's own baseline
            genre_features,
            other_features,
        ])

    def predict(self, user_ids, movie_ids):
        X = self._build_features(user_ids, movie_ids)
        preds = self._user_avg(user_ids) + self.model_.predict(X)
        return np.clip(preds, 0.5, 5.0)


class HybridModel:
    """Blends MatrixFactorization and ContentBasedModel predictions, weighting MF more heavily
    the more training ratings its target movie had. Grid-searching the optimal blend weight per
    rating-count bucket on held-out data showed a clear, monotonic pattern: ~0 for movies unseen
    in training (MF has literally no item-specific signal there, matching the earlier cold-start
    finding), rising smoothly to ~0.8 for movies with 100+ training ratings (see TASKS.md for the
    full bucket table). That's fit here with a saturating curve rather than a lookup table -
    the same shrinkage idea already used in BiasBaseline and ContentBasedModel's genre affinity:

        alpha(count) = max_alpha * count / (count + k)

    k=10 means a movie needs about 10 training ratings for its blend weight to reach half of
    max_alpha.
    """

    def __init__(self, mf_model, cb_model, movie_rating_counts, max_alpha=0.8, k=10.0):
        self.mf_model = mf_model
        self.cb_model = cb_model
        self.movie_rating_counts = movie_rating_counts  # pd.Series of training rating count, indexed by movieId
        self.max_alpha = max_alpha
        self.k = k

    def _alpha(self, movie_ids):
        counts = pd.Series(np.asarray(movie_ids)).map(self.movie_rating_counts).fillna(0.0).to_numpy()
        return self.max_alpha * counts / (counts + self.k)

    def predict(self, user_ids, movie_ids):
        alpha = self._alpha(movie_ids)
        mf_preds = self.mf_model.predict(user_ids, movie_ids)
        cb_preds = self.cb_model.predict(user_ids, movie_ids)
        return np.clip(alpha * mf_preds + (1 - alpha) * cb_preds, 0.5, 5.0)


def build_hybrid_model(ratings, mf_model=None, cb_model=None, **kwargs):
    """Convenience constructor: loads the saved MF/CB models if not given, and computes
    movie_rating_counts fresh from `ratings` (cheap - a single groupby, no need to persist it
    separately). Note HybridModel is not itself saved via save_model(): pickling it would
    duplicate the ~90MB matrix_factorization.pkl unnecessarily, since HybridModel just holds
    references to the two already-saved models plus a small count table rebuilt on demand."""
    mf_model = mf_model if mf_model is not None else load_model("matrix_factorization")
    cb_model = cb_model if cb_model is not None else load_model("content_based")
    counts = ratings.groupby("movieId").size()
    return HybridModel(mf_model, cb_model, counts, **kwargs)


def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model -> {path}")


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training():
    """Trains and saves all three models end to end. Note: with the tuned defaults, the
    MatrixFactorization fit typically runs ~30-35 epochs before early stopping fires, which can
    take on the order of an hour uninterrupted on the full ~9.5M-row training set - this is meant
    to be run as a standalone script, not inside a short-lived interactive session (see
    MatrixFactorization.fit's warm_start=True option to checkpoint/resume across shorter runs
    instead, which is how these numbers were actually produced during development)."""
    print("Loading ratings...")
    ratings = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    train, val = train_val_split(ratings, val_frac=0.05)
    print(f"train: {len(train):,}  val (holdout): {len(val):,}")

    print("\n--- GlobalMeanBaseline ---")
    global_model = GlobalMeanBaseline().fit(train)
    global_rmse = rmse(val["rating"], global_model.predict(val["userId"], val["movieId"]))
    print(f"val RMSE={global_rmse:.4f}")

    print("\n--- BiasBaseline ---")
    bias_model = BiasBaseline().fit(train)
    bias_rmse = rmse(val["rating"], bias_model.predict(val["userId"], val["movieId"]))
    print(f"val RMSE={bias_rmse:.4f}")

    print("\n--- MatrixFactorization ---")
    mf_model = MatrixFactorization().fit(train, val=val)

    print("\n--- ContentBasedModel ---")
    movie_features = pd.read_csv(os.path.join(PROCESSED_DIR, "movies_features.csv"), low_memory=False)
    cb_model = ContentBasedModel().fit(train, movie_features)
    cb_rmse = rmse(val["rating"], cb_model.predict(val["userId"], val["movieId"]))
    print(f"val RMSE={cb_rmse:.4f}  (on random rows - most already have a trained item factor in "
          f"MatrixFactorization; ContentBasedModel's real value is on item cold start, see "
          f"src/models.py's movie_holdout_split for that comparison)")

    save_model(mf_model, "matrix_factorization")
    save_model(bias_model, "bias_baseline")

    # retrain content-based on ALL of train.csv (not just this holdout split) for the saved,
    # deployable model - cheap to do since fitting is a single Ridge solve, not an SGD loop
    save_model(ContentBasedModel().fit(ratings, movie_features), "content_based")

    print("\n--- Summary (val RMSE, lower is better) ---")
    print(f"{'GlobalMeanBaseline':<22} {global_rmse:.4f}")
    print(f"{'BiasBaseline':<22} {bias_rmse:.4f}")
    print(f"{'MatrixFactorization':<22} {mf_model.best_val_rmse_:.4f}  "
          f"({mf_model.epochs_trained_} epochs, stopped_early={mf_model.stopped_early_})")
    print(f"{'ContentBasedModel':<22} {cb_rmse:.4f}  (random rows; see cold-start note above)")


if __name__ == "__main__":
    run_training()
