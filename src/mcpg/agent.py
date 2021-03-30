import tensorflow as tf
import numpy as np

class Agency():
    def __init__(self, env, lr, gamma, normalize, n_hidden, seed):
        tf.random.set_seed(seed)

        #Set (Hyper-)parameters
        self.LR               = lr
        self.GAMMA            = gamma
        self.NORM             = normalize
        self.num_hidden_layers = n_hidden
        self.action_space     = env.action_space.n
        self.state_space      = env.observation_space.shape[0]

        #Initialize Policy
        self.model = self.make_policy()

        #for storing
        self.episode_states      = []
        self.episode_probs       = []
        self.episode_rewards     = []
        self.episode_action_diff = []
        self.episode = 0

    def make_policy(self):
        """Initialize Policy Network"""

        #placeholders
        In_states        = tf.keras.layers.InputLayer(self.state_space, 
                                            name = 'input_states')
        Out_actions       = tf.keras.layers.Dense(self.action_space, 
                                            activation = 'softmax', 
                                            name = 'output_actions')

        model = tf.keras.Sequential()
        model.add(In_states)

        for idx in range(self.num_hidden_layers):
            model.add(
                tf.keras.layers.Dense(  
                    24,
                    activation= 'relu',
                    kernel_initializer='random_normal',
                    use_bias=True,
                    name='dense{}'.format(idx+1)
                )
            )

        #Create Output layer
        model.add(Out_actions)

        model.compile(
            optimizer= tf.keras.optimizers.Adam(self.LR),
            loss="categorical_crossentropy"
        )

        return model
    def make_move (self, S):
        """Perform action based on the probabillity distribution"""
       
        #get action probabillities
        probs = self.model.predict(S.reshape([1, 4])).flatten()
        action = np.random.choice(self.action_space, 1, p = probs)[0]
        return action, probs

    def save_step(self, S, A, R, P):
        """Store the state, action, reward, and Prob. of the action """
        
        action_ = np.zeros(self.action_space)
        action_[A] = 1
        self.episode_action_diff.append(action_ - P)
        self.episode_probs.append(P)
        self.episode_rewards.append(R)
        self.episode_states.append(S)

    def train(self, last_results):
        """Perform the update rule"""

        states        = np.vstack(np.array(self.episode_states))
        action_diff   = np.vstack(np.array(self.episode_action_diff))            
        state_rewards = self.get_reward_trace()   

        action_diff   = action_diff * state_rewards.reshape(-1,1)
        action_diff   = np.vstack([action_diff])+self.episode_probs
        loss = self.model.train_on_batch(states, action_diff)

        #Should do over last 100 results
        if len(last_results) < 100:
          mean = np.mean(last_results)
          std  = np.std(last_results) 
        else:
          mean = np.mean(last_results[-100:])
          std  = np.std(last_results[-100:])

        self.episode_states      = []
        self.episode_probs       = []
        self.episode_rewards     = []
        self.episode_action_diff = []

        self.episode += 1

        return loss, mean, std


    def get_reward_trace(self):
        """Get the Sum over the rewards in the episode, see Alg. 3 on page 47"""
        grad   = np.zeros_like(self.episode_rewards)
        epochs = range(len(self.episode_rewards))
        R      = 0

        for t in reversed(epochs):
            R       = self.episode_rewards[t] + self.GAMMA*R
            grad[t] = R
        
        # Perform Baseline Subtraction. 
        if self.NORM == True:
          grad = grad - np.mean(grad)
          grad = grad / np.std(grad)
          return grad
        else:
          return grad