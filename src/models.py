
import tensorflow as tf

class DenseModel():

    def __init__(self, num_states, num_hidden_nodes, num_actions, 
                 hidden_activation='relu', hidden_initializer='glorot_uniform',
                 output_activation='sigmoid', 
                 output_initializer='glorot_uniform'):
        self.input_layer = tf.keras.layers.InputLayer((num_states,))
        self.hidden_layers = []
        for n_nodes in num_hidden_nodes:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    n_nodes,
                    activation=hidden_activation,
                    kernel_initializer=hidden_initializer
                )
            )
        
        self.output_layer = tf.keras.layers.Dense(
            num_actions,
            activation=output_activation,
            kernel_initializer=output_initializer
        )
    
    # @tf.function
    # def call(self, inputs):
    #     z = self.input_layer(inputs)
    #     for layer in self.hidden_layers:
    #         z = layer(z)
    #     output = self.output_layer(z)
    #     return output