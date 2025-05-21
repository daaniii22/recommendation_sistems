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
        verbose: bool = False,
    ):
        if min(U) < 0 and max(U) >= self.n_users:
            raise ValueError('User IDs must be in the range [0, len(U))')
        if min(I) < 0 or max(I) >= self.n_items:
            raise ValueError('Item IDs must be in the range [0, len(I))')
        
        iterable = range(epochs) if not verbose else tqdm(range(epochs), desc='Training')
        for _ in iterable:
            for u, i, rating in zip(U, I, ratings):
                prediction = self.predict(u, i)
                error = rating - prediction
                
                self.p[u] += learning_rate * (error * self.q[i] - regularization * self.p[u])
                self.q[i] += learning_rate * (error * self.p[u] - regularization * self.q[i])
                self.bu[u] += learning_rate * (error - regularization * self.bu[u])
                self.bi[i] += learning_rate * (error - regularization * self.bi[i])
                
    def predict_all(self, u: int) -> np.ndarray:
        self._check(u)
        return np.dot(self.p[u], self.q.T) + self.bu[u] + self.bi
    
    def get_recommendations(self, u: int, n: int = 10) -> np.ndarray:
        predictions = self.predict_all(u)
        if len(predictions) < n:
            warnings.warn(f'Not enough items in dataset for {n} recommendations ({len(predictions)}).', RuntimeWarning)
            return np.arange(len(predictions))
        
        recommendations = np.argpartition(predictions, -n)[-n:]
        return recommendations