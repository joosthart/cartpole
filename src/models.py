
import numpy as np
import tensorflow as tf

class DenseModel:

    def __init__(self, input_shape, output_shape, num_hidden_nodes, lr,
                 hidden_activation='relu', hidden_initializer='random_normal',
                 output_initializer='random_normal'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_hidden_nodes = num_hidden_nodes
        self.lr = lr
        self.hidden_activation = hidden_activation
        self.hidden_initializer = hidden_initializer
        self.output_initializer = output_initializer

        self.set_model()

    def save(self, fn):
        self.model.save(fn)

    def load(self, fn):
        self.model = tf.keras.models.load_model(fn)

    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def get_weights(self):
        return self.model.get_weights()

    def set_model(self):
        in_layer = tf.keras.layers.InputLayer(
            self.input_shape, name='input'
        )
        out_layer = tf.keras.layers.Dense(
            self.output_shape,
            kernel_initializer=self.output_initializer,
            name = 'output'
        )
        
        self.model = tf.keras.Sequential()

        self.model.add(in_layer)
        for idx, n_nodes in enumerate(self.num_hidden_nodes):
            self.model.add(
                tf.keras.layers.Dense(
                    n_nodes,
                    activation=self.hidden_activation,
                    kernel_initializer=self.hidden_initializer,
                    use_bias=True,
                    name='dense_layer_{}'.format(idx)
                )
            )
        self.model.add(out_layer)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mean_squared_error'
        )

    def predict(self, states):
        return self.model.predict(np.atleast_2d(states))

    def train(self, x, y):
        return self.model.train_on_batch(x, y)

