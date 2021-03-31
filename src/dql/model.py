
import numpy as np
import tensorflow as tf

class DenseModel:
    """Dense fully connected neural network model
    """
    def __init__(self, input_shape, output_shape, num_hidden_nodes, lr,
                 hidden_activation='relu', hidden_initializer='random_normal',
                 output_initializer='random_normal'):
        """
        Args:
            input_shape (int): shape of input data
            output_shape (int): shape of output data
            num_hidden_nodes (list[int]): list of integers stating the number of 
                input nodes.
            lr (float): learning rate
            hidden_activation (str, optional): Activation function of input 
                layers. Defaults to 'relu'.
            hidden_initializer (str, optional): Initializer of hidden layers. 
                Defaults to 'random_normal'.
            output_initializer (str, optional): Initializer of output layer. 
                Defaults to 'random_normal'.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_hidden_nodes = num_hidden_nodes
        self.lr = lr
        self.hidden_activation = hidden_activation
        self.hidden_initializer = hidden_initializer
        self.output_initializer = output_initializer

        self.set_model()

    def save(self, fn):
        """Save keras model.

        Args:
            fn (str): Filename of model folder.
        """
        self.model.save(fn)

    def load(self, fn):
        """Load keras model.

        Args:
            fn (str): Filename of model folder.
        """
        self.model = tf.keras.models.load_model(fn, compile=False)

    def set_weights(self, weights):
        """Set weights of the model.
        """
        self.model.set_weights(weights)
    
    def get_weights(self):
        """get the weights of the model.
        """
        return self.model.get_weights()

    def set_model(self):
        """ Create model.
        """
        # input layer
        in_layer = tf.keras.layers.InputLayer(
            self.input_shape, name='input'
        )
        # output layer
        out_layer = tf.keras.layers.Dense(
            self.output_shape,
            kernel_initializer=self.output_initializer,
            name = 'output'
        )
        
        # Initialize sequential model
        self.model = tf.keras.Sequential()

        # Add layerss to the model
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

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mean_squared_error'
        )

    def predict(self, states):
        """Get a prediction from the model.
        """
        return self.model.predict(np.atleast_2d(states))

    def train(self, x, y):
        """Train the model for one epoch on a batch of data.

        Args:
            x (numpy.array): Input train data.
            y (numpy.array): Output train data, matching input datat.

        Returns:
            float: Training loss
        """
        return self.model.train_on_batch(x, y)

