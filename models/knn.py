from typing import Literal
import numpy as np
import warnings

class KNN:
    def __init__(
        self,
        k: int,
        similarity_kind: Literal['correlation', 'jmsd'] = 'jmsd',
    ):
        self.k = k
        self.similarity = similarity_kind
        
        
    def _check(self, ratings_u: np.ndarray) -> None:
        if not hasattr(self, 'ratings'):
            raise ValueError('Model not fitted yet')
        if not len(ratings_u.shape) == 1:
            raise ValueError('User ratings must be a 1D numpy array')
        if not ratings_u.shape[0] == self.ratings.shape[1]:
            raise ValueError('User ratings must have the same number of items as the training data')
        
    
    def _correlation_similarities(self, ratings_u: np.ndarray) -> np.ndarray:
        all_ratings = self.ratings
        valids = ~np.isnan(all_ratings)
        valids_u = ~np.isnan(ratings_u)
        intersections = valids_u & valids
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = all_ratings.mean(axis=1, where=valids)
            mean_u = ratings_u[valids_u].mean()
            
        errors = all_ratings - means[:, np.newaxis]
        errors_u = ratings_u - mean_u
        numerator = (errors * errors_u).sum(axis=1, where=intersections)
        denominator = np.sqrt(
            (errors**2).sum(axis=1, where=intersections)
            *
            (np.broadcast_to(errors_u, all_ratings.shape)**2).sum(axis=1, where=intersections)
        )
            
        denominator[denominator == 0] = np.nan
        similarities = numerator / denominator
        return similarities
        
    
    def _jmsd_similarities(self, ratings_u: np.ndarray) -> np.ndarray:
        all_ratings = self.ratings.copy()

        valids = ~np.isnan(all_ratings)
        valids_u = ~np.isnan(ratings_u)
        union_sizes = (valids_u | valids).sum(axis=1)
        intersections = valids_u & valids
        intersection_sizes = intersections.sum(axis=1, dtype=np.float64)
        intersection_sizes[intersection_sizes == 0] = np.nan
        
        jaccard_similiarties = intersection_sizes/union_sizes

        all_ratings[intersections] = (all_ratings[intersections] - self.MIN_RATING) / (self.MAX_RATING - self.MIN_RATING)
        all_ratings[intersections] = ((np.broadcast_to(ratings_u, all_ratings.shape)[intersections] - all_ratings[intersections])**2)
        all_ratings[~intersections] = 0
        similarities = jaccard_similiarties * (1 - all_ratings.sum(axis=1) / intersection_sizes)
        return similarities
    
    def _get_neighbors(self, similarities: np.ndarray) -> np.ndarray:
        if len(similarities) <= self.k:
            # Not enough neighbors
            similarities = np.where(~np.isnan(similarities), similarities, 0)
            neighbors = np.arange(len(similarities))
            neighbors = neighbors[similarities[neighbors] != 0]
            return neighbors

        similarities = np.where(~np.isnan(similarities), similarities, 0)
        neighbors = np.argpartition(similarities, -self.k)[-self.k:]
        return neighbors[similarities[neighbors] != 0]

    
    def _average_prediction(self, item: np.integer | int, neighbors: np.ndarray) -> np.ndarray:
        values = self.ratings[:,item][neighbors]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.nan
        return valid_values.mean()
    
    def _average_predictions(self, neighbors: np.ndarray) -> np.ndarray:
        values = self.ratings[neighbors] 
        valids = ~np.isnan(values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return values.mean(axis=0, where=valids)
    
        
    def _weighted_average_prediction(self, item: np.integer | int, neighbors: np.ndarray, similarities: np.ndarray) -> np.ndarray:
        neighbors_ratings = self.ratings[:, item][neighbors]
        valid_ratings = ~np.isnan(neighbors_ratings)
        
        neighbors_ratings = neighbors_ratings[valid_ratings]
        if len(neighbors_ratings) == 0:
            return np.nan
        
        neighbors_similarities = similarities[neighbors][valid_ratings]
        return (neighbors_similarities * neighbors_ratings).sum() / neighbors_similarities.sum()
    
    def _weighted_average_predictions(self, neighbors: np.ndarray, similarities: np.ndarray) -> np.ndarray:
        neighbors_ratings = self.ratings[neighbors]
        valids = ~np.isnan(neighbors_ratings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return (neighbors_ratings * similarities[neighbors, np.newaxis]).sum(axis=0, where=valids)\
                / np.broadcast_to(similarities[neighbors, np.newaxis], valids.shape).sum(axis=0, where=valids)
                
    
    def _deviation_from_mean_prediction(self, item: np.integer | int, neighbors: np.ndarray, ratings_u: np.ndarray) -> np.ndarray:
        mean_u = ratings_u[~np.isnan(ratings_u)]
        mean_u = mean_u.mean() if len(mean_u) > 0 else np.nan
        neighbors_ratings = self.ratings[neighbors]
        valid_neighbors = ~np.isnan(neighbors_ratings[:,item])
        neighbors_ratings = neighbors_ratings[valid_neighbors]
        mean_neighbors = neighbors_ratings.mean(axis=1, where=~np.isnan(neighbors_ratings))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return mean_u + (neighbors_ratings[:,item] - mean_neighbors).mean()
    
    def _deviation_from_mean_predictions(self, neighbors: np.ndarray, ratings_u: np.ndarray) -> np.ndarray:
        mean_u = ratings_u[~np.isnan(ratings_u)]
        mean_u = mean_u.mean() if len(mean_u) > 0 else np.nan
        neighbors_ratings = self.ratings[neighbors]
        mean_neighbors = neighbors_ratings.mean(axis=1, where=~np.isnan(neighbors_ratings))
        valid_neighbors = ~np.isnan(neighbors_ratings)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return mean_u + (neighbors_ratings - mean_neighbors[:, np.newaxis]).mean(axis=0, where=valid_neighbors)
        
        
        
    def fit(self, ratings: np.ndarray):
        try:
            self.ratings = ratings.astype(np.float64)
            if self.similarity == 'jmsd':
                self.MIN_RATING = ratings[~np.isnan(ratings)].min()
                self.MAX_RATING = ratings[~np.isnan(ratings)].max()
        except ValueError:
            raise ValueError('Ratings must be a 2D numpy array of floats')
        
        
    def predict(self, ratings_u: np.ndarray, item: int, *, prediction_mode: Literal['average', 'weighted_average', 'deviation_from_mean'] = 'weighted_average'):
        self._check(ratings_u)
        if self.similarity == 'correlation':
            similarities = self._correlation_similarities(ratings_u)
        else:
            similarities = self._jmsd_similarities(ratings_u)
            
        neighbors = self._get_neighbors(similarities)
        if prediction_mode == 'average':
            return self._average_prediction(item, neighbors)
        elif prediction_mode == 'weighted_average':
            return self._weighted_average_prediction(item, neighbors, similarities)
        elif prediction_mode == 'deviation_from_mean':
            return self._deviation_from_mean_prediction(item, neighbors, ratings_u)
        else:
            raise ValueError('Invalid prediction mode')
    
    def predict_all(self, ratings_u: np.ndarray, *, prediction_mode: Literal['average', 'weighted_average', 'deviation_from_mean'] = 'weighted_average'):
        self._check(ratings_u)
        if self.similarity == 'correlation':
            similarities = self._correlation_similarities(ratings_u)
        else:
            similarities = self._jmsd_similarities(ratings_u)
            
        neighbors = self._get_neighbors(similarities)
        if prediction_mode == 'average':
            return self._average_predictions(neighbors)
        elif prediction_mode == 'weighted_average':
            return self._weighted_average_predictions(neighbors, similarities)
        elif prediction_mode == 'deviation_from_mean':
            return self._deviation_from_mean_predictions(neighbors, ratings_u)
        else:
            raise ValueError('Invalid prediction mode')
    
    
    def recommend(
        self,
        ratings_u: np.ndarray,
        N: int,
        *,
        prediction_mode: Literal['average', 'weighted_average', 'deviation_from_mean'] = 'weighted_average',
        predictions: np.ndarray = None,
        sorted: bool = False,
    ) -> np.ndarray:
        if predictions is None:
            predictions = self.predict_all(ratings_u, prediction_mode=prediction_mode)
        if len(predictions) < N:
            warnings.warn(f'Not enough items in dataset for {N} recommendations ({len(predictions)}).', RuntimeWarning)
            recommendations = np.arange(len(predictions))
            recommendations = recommendations[~np.isnan(predictions[recommendations])]
            if sorted:
                recommendations = recommendations[np.argsort(-predictions[recommendations])]
            return recommendations
        

        predictions = np.where(~np.isnan(predictions), predictions, 0)
        recommendations = np.argpartition(predictions, -N)[-N:]
        recommendations = recommendations[predictions[recommendations] != 0]
        if sorted:
            recommendations = recommendations[np.argsort(-predictions[recommendations])]
        return recommendations
        