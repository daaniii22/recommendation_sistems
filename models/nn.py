from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from sklearn.metrics import mean_absolute_error
import numpy as np

class NeuralNetwork:
    def __init__(self, num_users, num_items, latent_dim=20, epochs=10):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.model = self._build_model()

    def _build_model(self):
        # Inputs
        user_input = Input(shape=(1,))
        item_input = Input(shape=(1,))

        # Embeddings
        user_embedding = Embedding(self.num_users, self.latent_dim)(user_input)
        item_embedding = Embedding(self.num_items, self.latent_dim)(item_input)

        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)

        # GMF
        gmf_vec = Dot(axes=1)([user_vec, item_vec])

        # MLP
        mlp_input = Concatenate()([user_vec, item_vec])
        mlp_dense = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(mlp_input)
        mlp_dense = BatchNormalization()(mlp_dense)
        mlp_dense = Dropout(0.4)(mlp_dense)
        mlp_dense = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(mlp_dense)
        mlp_dense = BatchNormalization()(mlp_dense)

        # Fusion GMF + MLP
        fusion = Concatenate()([gmf_vec, mlp_dense])
        output = Dense(1, activation='sigmoid')(fusion)

        # Modelo final
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, X_train, y_train):
        # Normalizar ratings a [0, 1]
        y_train_norm = y_train / 10.0
        self.model.fit([X_train[0], X_train[1]], y_train_norm, epochs=self.epochs, verbose=1)

    def predict(self, X_test):
        y_pred_norm = self.model.predict([X_test[0], X_test[1]], verbose=0)
        return y_pred_norm.flatten() * 10  # Reescalar a [0, 10]

    def evaluate(self, X_test, y_test):
        y_test_norm = y_test / 10.0
        y_test_orig = y_test_norm * 10
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test_orig, y_pred)
        print(f"Mean Absolute Error: {mae:.4f}")
        return mae
