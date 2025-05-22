import tensorflow as tf
from keras import models, layers, regularizers, optimizers
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings

def int_to_binary_vector(x, nbits=19):
    x = tf.cast(tf.squeeze(x, axis=-1), tf.int32)
    bits = [tf.bitwise.bitwise_and(tf.bitwise.right_shift(x, i), 1) for i in reversed(range(nbits))]
    return tf.stack(bits, axis=1)

class NeuralNetwork:
    def __init__(
        self,
        num_users,
        num_items,
        min_rating: int = 0,
        max_rating: int = 10,
        latent_dim=20,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.latent_dim = latent_dim
        self.model = self._build_model()

    def _build_model(self):
        # Inputs
        user_input = layers.Input(shape=(1,))
        item_input = layers.Input(shape=(1,))

        # Embeddings
        # user_embedding = layers.Embedding(self.num_users, self.latent_dim, lora_rank=2, embeddings_regularizer=regularizers.l2(0.0001))(user_input)
        # item_embedding = layers.Embedding(self.num_items, self.latent_dim, lora_rank=2, embeddings_regularizer=regularizers.l2(0.0001))(item_input)
        
        user_binary = layers.Lambda(int_to_binary_vector, output_shape=(19,))(user_input)
        item_binary = layers.Lambda(int_to_binary_vector, output_shape=(19,))(item_input)
        
        user_data_recovery = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(user_binary)
        item_data_recovery = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(item_binary)
        
        
        user_lazy_embedding = layers.Dense(self.latent_dim, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(user_data_recovery)
        item_lazy_embedding = layers.Dense(self.latent_dim, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(item_data_recovery)

        user_vec = layers.Flatten()(user_lazy_embedding)
        item_vec = layers.Flatten()(item_lazy_embedding)

        # MLP
        mlp_input = layers.Concatenate()([user_vec, item_vec])
        mlp_dense = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(mlp_input)
        #mlp_dense = layers.BatchNormalization()(mlp_dense)
        mlp_dense = layers.Dropout(0.4)(mlp_dense)
        mlp_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(mlp_dense)
        #mlp_dense = layers.BatchNormalization()(mlp_dense)

        output = layers.Dense(1, activation='sigmoid')(mlp_dense)

        # Modelo final
        model = models.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=optimizers.Lamb(learning_rate=1e-3), loss='mse', metrics=['mae'])
        return model

    def _normalize(self, ratings):
        return (ratings - self.min_rating) / (self.max_rating - self.min_rating)
    
    def _denormalize(self, ratings):
        return ratings * (self.max_rating - self.min_rating) + self.min_rating

    def fit(self, U, I, y_train, *args, epochs: int = 10, verbose: str = 'auto', validation_data=None, batch_size=512, **kwargs):
        y_train_norm = self._normalize(y_train)
        if validation_data is not None:
            U_val, I_val, y_val = validation_data
            y_val_norm = self._normalize(y_val)
            validation_data = ([U_val, I_val], y_val_norm)
        self.model.fit([U, I], y_train_norm, *args, epochs=epochs, verbose=verbose, validation_data=validation_data, batch_size=batch_size, **kwargs)

    def predict(self, X_test, batch_size=512):
        y_pred_norm = self.model.predict([X_test[0], X_test[1]], verbose=0, batch_size=batch_size)
        return self._denormalize(y_pred_norm.flatten())
    
    def recommend(self, user_id, N=10, sorted=False, predictions=None, batch_size=512):
        if predictions is None:
            user_input = np.array([user_id] * self.num_items)
            item_input = np.arange(self.num_items)
            predictions = self.predict([user_input, item_input], batch_size=batch_size)
            
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
    