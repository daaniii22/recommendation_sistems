from typing import Sequence
from tqdm import tqdm
import warnings

import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

def get_mae(true: np.ndarray, predictions: np.ndarray) -> float:
    ae = np.abs(true - predictions)
    return ae.mean()


class BernoulliFactorization:
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        min_rating: int = 0,
        max_rating: int = 10,
        n_factors: int = 5,
        seed: int = None,
    ):
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.scores = np.arange(min_rating, max_rating + 1)
        rng = np.random.default_rng(seed)
        self.U = rng.random((len(self.scores), n_users, n_factors))
        self.V = rng.random((len(self.scores), n_items, n_factors))
        
    def _check(self, u: int, i: int = None) -> None:
        if not 0 <= u < len(self.U[0]):
            raise ValueError(f'User ID {u} out of range [0, {len(self.U[0])})')
        if i is not None and not 0 <= i < len(self.V[0]):
            raise ValueError(f'Item ID {i} out of range [0, {len(self.V[0])})')
        
    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
        
        
    def compute_prediction(self, u: int, i: int) -> np.ndarray:
        logits = np.einsum('sf,sf->s', self.U[:,u], self.V[:,i])
        return self.__sigmoid(logits)
    
    def predict_proba(self, u: int, i: int) -> np.ndarray:
        self._check(u, i)
        unnormalized_prediction = self.compute_prediction(u, i)
        prediction = unnormalized_prediction / unnormalized_prediction.sum()
        return prediction
    
    def predict(self, u: int, i: int) -> int:
        self._check(u, i)
        logits = np.einsum('sf,sf->s', self.U[:,u], self.V[:,i])
        return self.scores[np.argmax(logits)]
    
    def predict_all_proba(self, u: int) -> np.ndarray:
        self._check(u)
        logits = np.einsum('sf,sif->is', self.U[:,u], self.V)
        unnormalized_predictions = self.__sigmoid(logits)
        predictions = unnormalized_predictions / unnormalized_predictions.sum(axis=1, keepdims=True)
        return predictions
    
    def predict_all(self, u: int) -> np.ndarray:
        self._check(u)
        logits = np.einsum('sf,sif->is', self.U[:,u], self.V)
        return self.scores[np.argmax(logits, axis=1)]
    
    def predict_many(self, U: np.array, I: np.ndarray) -> np.ndarray:
        if len(U) != len(I):
            raise ValueError('Input arrays must have the same length')
        if min(U) < 0 or max(U) >= self.n_users:
            raise ValueError(f'User IDs must be in the range [0, {self.n_users})')
        if min(I) < 0 or max(I) >= self.n_items:
            raise ValueError(f'Item IDs must be in the range [0, {self.n_items})')
        
        logits = np.einsum('sef,sef->es', self.U[:,U], self.V[:,I])
        return self.scores[np.argmax(logits, axis=-1)]
        
    def recommend(
        self,
        u: int,
        N: int = 10,
        *,
        sorted: bool = False,
        predictions: np.ndarray = None
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
    
    
    def validate(
        self,
        U: ArrayLike | Sequence[int],
        I: ArrayLike | Sequence[int],
        ratings: ArrayLike | Sequence[int],
        *,
        metric: Callable[[np.ndarray, np.ndarray], float] = get_mae
    ) -> float:
        if len(U) != len(ratings):
            raise ValueError('Input arrays must have the same length')
        
        predictions = self.predict_many(U, I)
        return metric(ratings, predictions)
        
        
    def fit(
        self,
        U: ArrayLike | Sequence[int],
        I: ArrayLike | Sequence[int],
        ratings: ArrayLike | Sequence[int],
        epochs: int = 10,
        learning_rate: float = 0.0001,
        regularization: float = 0.1,
        verbose: int = 0,
        validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        validation_metric: Callable[[np.ndarray, np.ndarray], float] = get_mae,
    ):
        if len(U) != len(I) or len(U) != len(ratings):
            raise ValueError('Input arrays must have the same length')
        if min(U) < 0 or max(U) >= self.n_users:
            raise ValueError(f'User IDs must be in the range [0, {self.n_users})')
        if min(I) < 0 or max(I) >= self.n_items:
            raise ValueError(f'Item IDs must be in the range [0, {self.n_items})')
        
        history = []
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
                prediction = self.compute_prediction(u, i)
                positive = self.scores == rating
                delta = (positive * (1 - prediction)) - ((1 - positive) * prediction)
                
                deltaU = (self.V[:,i] * delta[:, None])
                deltaU -= regularization * self.U[:,u]
                deltaU *= learning_rate
                self.U[:,u] += deltaU
                
                deltaV = (self.U[:,u] * delta[:, None])
                deltaV -= regularization * self.V[:,i]
                deltaV *= learning_rate
                self.V[:,i] += deltaV
                
            train_score = self.validate(U, I, ratings, metric=validation_metric)
            if validation_data is not None:
                val_score = self.validate(*validation_data, metric=validation_metric)
                tqdm.write(f'Epoch {epoch + 1} - train score: {train_score:.4f} - validation score: {val_score:.4f}')
            elif verbose >= 2:
                tqdm.write(f'Epoch {epoch + 1} - training score: {train_score:.4f}')
                
            history.append(train_score if validation_data is None else val_score)          
            
        return history