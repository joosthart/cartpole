

from tqdm import tqdm
import gym
import numpy as np
import tensorflow as tf

from src.dql.model import DenseModel

SEED = 42

class ExperienceBuffer:
    """Experience Buffer used for experience replay by DeepQLearning
    """
    def __init__(self, max_buffer_size, obs_space):
        self.max_buffer_size = int(max_buffer_size)
        self.obs_space = int(obs_space)

        self.states = np.zeros((self.max_buffer_size, self.obs_space))
        self.actions = np.zeros(self.max_buffer_size, dtype=int)
        self.rewards = np.zeros(self.max_buffer_size)
        self.next_states = np.zeros((self.max_buffer_size, self.obs_space))
        self.done_flags = np.zeros(self.max_buffer_size, dtype=bool)

        self.row = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done_flag):
        """Add a state to buffer
        """
        self.states[self.row] = state
        self.actions[self.row] = action
        self.rewards[self.row] = reward
        self.next_states[self.row] = next_state
        self.done_flags[self.row] = done_flag

        self.size = max(self.size, self.row)
        self.row = (self.row + 1) % self.max_buffer_size
    
    def get_size(self):
        """Get the current size of the buffer
        """
        return self.size

    def get_batch(self, batch_size):
        """Get a random batch from the buffer
        """
        if batch_size > self.get_size:
            raise ValueError(
                'Batch size is too large. Batch size should be less than or '
                'equal to current buffer size.'
            )

        idx = np.random.choice(
            min(self.size, self.max_buffer_size), batch_size, replace=False
        )
        return (
            self.states[idx], 
            self.actions[idx], 
            self.rewards[idx],
            self.next_states[idx],
            self.done_flags[idx]
        )

class DeepQLearning:
    """Deep Q-Learning agent for solving the CartPole problem.
    """
    def __init__(
            self, num_hidden_states=[8,8], discount=0.95, max_buffer_size=1e6, 
            min_buffer_size=1e3, batch_size=64, lr=0.01, update_freq=10, 
            epsilon=0.2, min_epsilon=0.1, epsilon_decay=0.99, 
            hidden_activation='relu', hidden_initializer='random_normal', 
            output_initializer='random_normal', log_dir='./log/'):

        # init environment
        self.env = gym.make('CartPole-v0')
        # Set random seeds
        self.env.seed(SEED)
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.discount = discount
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Neural Networks
        self.model = DenseModel(
            input_shape=self.num_states,
            output_shape=self.num_actions,
            num_hidden_nodes=num_hidden_states,
            lr=lr,
            hidden_activation=hidden_activation,
            hidden_initializer=hidden_initializer,
            output_initializer=output_initializer
        )

        self.target_model = DenseModel(
            input_shape=self.num_states,
            output_shape=self.num_actions,
            num_hidden_nodes=num_hidden_states,
            lr=lr,
            hidden_activation=hidden_activation,
            hidden_initializer=hidden_initializer,
            output_initializer=output_initializer
        )

        # Initialize tensorboard
        self.summary_writer = tf.summary.create_file_writer(log_dir)

        if self.min_buffer_size < self.batch_size:
            raise ValueError((
                'batch size should be smaller than or equal to minimal buffer'
                'size.'
            ))
        
        # Initialize buffer
        self.memory = ExperienceBuffer(self.max_buffer_size, self.num_states)

    def _update_epsilon(self):
        """ Update epsilon with epsilon decay. If new epsilon is smaller than 
        minimal epsilon, returns minimal epsilon.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def _get_action(self, state, simulate=False):
        """Get an action for a given state.

        Args:
            state (numpy.array): environment current observation state
            simulate (bool, optional): If false, use epsilon-greedy 
                exploitation. Else 100% exploitation. Defaults to False.

        Returns:
            int: action
        """
        # Explore
        if np.random.random() < self.epsilon and not simulate:
            return self.env.action_space.sample()
        # Exploit
        else:
            return np.argmax(self.model.predict(state), axis=1)[0]

    def _update_model(self):
        """Update target model with training model weights.
        """
        self.target_model.set_weights(self.model.get_weights())

    def _train_one_batch(self):
        """Train on one batch of data. In this method the target values are 
        calculated and the training is started.

        Returns:
            float: Traing loss
        """
        # Get a batch of data
        states, actions, rewards, next_states, done_flags = \
            self.memory.get_batch(self.batch_size)

        # Get current state Q-values from trainig model
        q_values = self.model.predict(states)
        # Get target Q-values using target model
        q_values_next = self.target_model.predict(next_states)

        # Calculate target Q-values using Bellman's equation
        q_values[range(len(q_values)), actions] = \
            rewards + self.discount*np.max(q_values_next, axis=1)
        q_values[done_flags, actions[done_flags]] = rewards[done_flags]

        # Train train model on one batch of data
        loss = self.model.train(states, q_values)
        
        return loss

    def train(self, num_epochs, verbose=True, render_every=500):
        """Main training method

        Args:
            num_epochs (int): Number of epochs to train for
            verbose (bool, optional): Verbose tag. Defaults to True.
            render_every (int, optional): Frequency to display the algorith 
                performace visiually. Defaults to 500.
        """
        self.total_training_reward = []
        self.total_training_loss = []
        self.total_epsilon = []

        if verbose:
            pbar = tqdm(range(num_epochs))
        else:
            pbar = range(num_epochs)
        for epoch in pbar:
            # Reset environment
            state = self.env.reset()

            epoch_reward = 0
            done = False
            epoch_loss = []
            while not done:
                if verbose and epoch != 0 and epoch % render_every == 0:
                    self.env.render()

                # get next action
                action = self._get_action(state)
                # make move
                next_state, reward, done, _ = self.env.step(action)

                epoch_reward += reward
                # Add state to experience buffer
                self.memory.add(
                    state=state, 
                    action=action, 
                    reward=reward, 
                    next_state=next_state,
                    done_flag=done
                )
                # If experience buffer large enough, train on one batch of data.
                if self.memory.get_size() > self.min_buffer_size:
                    loss = self._train_one_batch()
                    epoch_loss.append(loss) 

                state = next_state

            self.total_training_reward.append(epoch_reward)
            self.total_epsilon.append(self.epsilon)

            # Update epsilon
            if self.memory.get_size() > self.min_buffer_size:
                self._update_epsilon()

            self.total_training_loss.append(epoch_loss)
            
            # Write performance to Tensorboard
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    'epoch reward', 
                    epoch_reward, 
                    step=epoch
                )

                tf.summary.scalar(
                    'average loss', 
                    np.mean(epoch_loss), 
                    step=epoch
                )

                tf.summary.scalar(
                    'epsilon', 
                    self.epsilon, 
                    step=epoch
                )
            
            if epoch % self.update_freq == 0:
                self._update_model()

            if verbose:
                pbar.set_postfix({
                    'reward': '{:.1f}'.format(
                        self.total_training_reward[-1]
                    ),
                    'loss': '{:.3f}'.format(
                        np.mean(self.total_training_loss[-1])
                    ),
                    'epsilon': '{:.3f}'.format(
                        np.mean(self.epsilon)
                    )
                })

    def save(self, fn):
        """Save model

        Args:
            fn (str): path to save model
        """
        self.model.save(fn)

    def load(self, fn):
        """Load model

        Args:
            fn (str): path to save model
        """
        self.model.load(fn)

    def simulate(self, n_simulations, verbose=True, maxsteps=500):
        """Simualte the current model performance visually

        Args:
            n_simulations (int): Number of simulations
            verbose (bool, optional): Vebose flag. Defaults to True.
            maxsteps (int, optional): Maximal steps of episode. Defaults to 500.

        Returns:
            list[int]: obtained rewards
        """
        rewards = []
        for i in range(n_simulations):
            # Reset environment
            state = self.env.reset()
            sim_reward = 0
            step = 0 
            done = False
            self.epsilon = 0.1 
            while not done and step < maxsteps:
                if verbose:
                    self.env.render()
                action = self._get_action(state, simulate=False)
                state, reward, done, _ = self.env.step(action)
                sim_reward += reward
                step += 1
            if verbose:          
                print('Reward run {}: {}'.format(i+1, sim_reward))
            rewards.append(sim_reward)
        return rewards
        