import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

RAW_DIR = "data/raw"
MODELS_DIR = "models"


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

    save_model(mf_model, "matrix_factorization")
    save_model(bias_model, "bias_baseline")

    print("\n--- Summary (val RMSE, lower is better) ---")
    print(f"{'GlobalMeanBaseline':<22} {global_rmse:.4f}")
    print(f"{'BiasBaseline':<22} {bias_rmse:.4f}")
    print(f"{'MatrixFactorization':<22} {mf_model.best_val_rmse_:.4f}  "
          f"({mf_model.epochs_trained_} epochs, stopped_early={mf_model.stopped_early_})")


if __name__ == "__main__":
    run_training()
