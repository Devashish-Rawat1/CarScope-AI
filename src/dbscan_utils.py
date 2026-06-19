"""
dbscan_utils.py
----------------
sklearn's DBSCAN has no `.predict()` method for unseen data -- it's a pure
clustering algorithm that only knows about the points it was fit on. That's a
problem for a live fraud-scoring tool, where a single new listing needs an
anomaly verdict in real time.

DBSCANNoveltyDetector works around this with the standard trick: after fitting
DBSCAN once on the training feature space, it builds a fast nearest-neighbor
index over only the *core* points DBSCAN identified. A new point is then
treated as "in a cluster" (normal) if it falls within `eps` of any core point,
and as "noise" (anomaly) otherwise -- which is exactly DBSCAN's own definition
of cluster membership, just applied to a point that wasn't part of the
original fit.

`.predict()` returns -1 for anomalies and 1 for normal points, matching the
convention used by IsolationForest and LocalOutlierFactor, so all three models
can be treated identically by downstream code.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class DBSCANNoveltyDetector:
    def __init__(self, eps=1.5, min_samples=3, n_jobs=-1):
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.dbscan_ = None
        self.nn_index_ = None
        self.core_labels_ = None

    def fit(self, X):
        self.dbscan_ = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=self.n_jobs)
        self.dbscan_.fit(X)

        core_idx = self.dbscan_.core_sample_indices_
        if len(core_idx) == 0:
            raise ValueError(
                "DBSCAN found zero core points with the given eps/min_samples -- "
                "every training point would be treated as noise. Increase eps or "
                "lower min_samples and refit."
            )

        core_points = self.dbscan_.components_
        self.core_labels_ = self.dbscan_.labels_[core_idx]
        self.nn_index_ = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(core_points)
        return self

    def predict(self, X):
        """-1 = noise/anomaly, 1 = normal (falls within eps of a real cluster's core point)."""
        if self.nn_index_ is None:
            raise RuntimeError("DBSCANNoveltyDetector must be fit before calling predict().")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        distances, _ = self.nn_index_.kneighbors(X, n_neighbors=1)
        distances = distances.ravel()
        return np.where(distances <= self.eps, 1, -1)

    @property
    def labels_(self):
        """Cluster labels assigned to the original training data (sklearn DBSCAN convention, -1 = noise)."""
        return self.dbscan_.labels_

    @property
    def noise_fraction_(self):
        return float(np.mean(self.dbscan_.labels_ == -1))
