from typing import Sequence
from tqdm import tqdm
import warnings

import numpy as np
from numpy.typing import ArrayLike


class MatrixFactorization:
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 7, seed: int = None):
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        rng = np.random.default_rng(seed)
        self.p = rng.random((n_users, n_factors))
        self.bu = rng.random(n_users)
        self.q = rng.random((n_items, n_factors))
        self.bi = rng.random(n_items)
        
    def _check(self, u: int, i: int = None) -> None:
        if not 0 <= u < len(self.p):
            raise ValueError(f'User ID {u} out of range [0, {len(self.p)})')
        if i is not None and not 0 <= i < len(self.q):
            raise ValueError(f'Item ID {i} out of range [0, {len(self.q)})')
        
    def compute_prediction(self, u: int, i: int) -> float:
        return np.dot(self.p[u], self.q[i]) + self.bu[u] + self.bi[i]
    
    def predict(self, u: int, i: int) -> float:
        self._check(u, i)
        return self.compute_prediction(u, i)
        
    def fit(
        self,
        U: ArrayLike | Sequence[int],
        I: ArrayLike | Sequence[int],
        ratings: ArrayLike | Sequence[float],
        epochs: int = 10,
        learning_rate: float = 0.001,
        regularization: float = 0.1,
        verbose: int = 0,
    ):
        if min(U) < 0 and max(U) >= self.n_users:
            raise ValueError('User IDs must be in the range [0, len(U))')
        if min(I) < 0 or max(I) >= self.n_items:
            raise ValueError('Item IDs must be in the range [0, len(I))')
        
        
        for epoch in tqdm(range(epochs), desc='Training', unit='epoch', disable=verbose != 1):
            pbar = tqdm(
                zip(U, I, ratings),
                total=len(ratings),
                desc=f'Epoch {epoch + 1}',
                unit='it',
                mininterval=0.2,
                disable=verbose < 2,
            )
            for u, i, rating in pbar:
                prediction = self.predict(u, i)
                error = rating - prediction
                
                self.p[u] += learning_rate * (error * self.q[i] - regularization * self.p[u])
                self.q[i] += learning_rate * (error * self.p[u] - regularization * self.q[i])
                self.bu[u] += learning_rate * (error - regularization * self.bu[u])
                self.bi[i] += learning_rate * (error - regularization * self.bi[i])
                
    def predict_all(self, u: int) -> np.ndarray:
        self._check(u)
        return np.dot(self.p[u], self.q.T) + self.bu[u] + self.bi

    
    def recommend(
        self,
        u: int,
        N: int = 10,
        *,
        sorted: bool = False,
        predictions: np.ndarray = None,
    ) -> np.ndarray:
        if predictions is None:
            predictions = self.predict_all(u)
        if len(predictions) < N:
            warnings.warn(f'Not enough items in dataset for {N} recommendations ({len(predictions)}).', RuntimeWarning)
            recommendations = np.arange(len(predictions))
            if sorted:
                recommendations = recommendations[np.argsort(-predictions[recommendations])]
            return recommendations
        
        recommendations = np.argpartition(predictions, -N)[-N:]
        if sorted:
            recommendations = recommendations[np.argsort(-predictions[recommendations])]
        return recommendations