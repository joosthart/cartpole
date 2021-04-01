import pickle
import random
from tqdm import tqdm

import gym
import numpy as np
import tensorflow as tf

SEED = 42

class TabularQLearning:
    """Tabular Q-Learning agent for solving the CartPole problem.
    """
    def __init__(self, epsilon=1, discount=0.95, lr=0.1, epsilon_decay=0.999, 
                 save_progress=True, log_dir='log_tql', **kwargs):
        self.epsilon = epsilon
        self.max_epsilon = epsilon
        self.min_epsilon = 1e-4
        self.discount = discount
        self.lr = lr
        self.epsilon_decay = epsilon_decay

        self.step_size = [
            kwargs.get('position_step_size', 0.48),
            kwargs.get('velocity_step_size', 0.4),
            kwargs.get('angle_step_size', 0.0418),
            kwargs.get('angular_velocity_step_size', 0.4)
        ]

        self.save_progress = save_progress
        # Initialize tensorboard
        if self.save_progress:
            self.summary_writer = tf.summary.create_file_writer(log_dir)

        # Init env
        self.env = gym.make(kwargs.get('environment', 'CartPole-v0'))
        self.env.seed(SEED)
        
        self.table = {}

        # set numpy seed
        np.random.seed(SEED)

    def _bellman(self, state, next_state, action, reward):
        """Calculate Q-values using Bellman's equation to update table.

        Args:
            state (numpy.array): environment current observation state
            next_state (numpy.array): environment next state observation
            action (int): Move taken
            reward (int): Reward for action at current state

        Returns:
            float: Q-value
        """
        q0 = self._get_q_value(state, action)
        q1 = self._get_q_value(next_state, self._get_best_action(next_state))
        return (1-self.lr)*q0 + self.lr*(reward + self.discount*q1)

    def _discretize(self, state):
        """Discretize state

        Args:
            state (numpy.array): environment current observation state

        Returns:
            numpy.array: Discretized state
        """
        state = state.copy()
        for idx, (s, ds) in enumerate(zip(state, self.step_size)):
            state[idx] = round(
                (s + 0.5*ds)//ds * ds,
                len(str(ds).split('.')[-1])
            )
        return state

    def _add_state(self, state):
        """Add a state tot the Q-learning table

        Args:
            state (numpy.array): environment current observation state
        """
        action_space = list(range(self.env.action_space.n))
        random.shuffle(action_space)  # Random action is the first in the dict
        self.table[tuple(state)] = {
            k: 0 for k in action_space
        }

    def _set_q_value(self, state, action, score):
        """ Set Q-value to in table
        """
        self.table[tuple(state)][action] = score

    def _get_q_value(self, state, action):
        """ Add Q-value in table
        """
        return self.table[tuple(state)][action]

    def _get_best_action(self, state):
        """ Get action with highest Q-value for a given state
        """
        return max(
            self.table[tuple(state)],
            key=self.table[tuple(state)].get
        )

    def _update_epsilon(self):
        """ Update epsilon with epsilon decay. If new epsilon is smaller than 
        minimal epsilon, returns minimal epsilon.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

    def train(self, epochs, maxsteps=10000, render_every=500, verbose=True):
        """ Main training method

        Args:
            epochs (int): Number of Epochs to train for
            maxsteps (int, optional): Maximal environment steps. Defaults to 
                10000.
            render_every (int, optional): Frequency to display the algorith 
                performace visiually. Defaults to 500.
            verbose (bool, optional): Verbose flag. Defaults to True.
        """
        self.total_training_reward = []
        self.total_epsilon = []

        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        
        for i in pbar:
            # Reset environment
            state = self.env.reset()

            # Discretize state
            state_disc = self._discretize(state)
            if tuple(state_disc) not in self.table.keys():
                self._add_state(state_disc)

            epoch_reward = 0
            step = 0
            done = False
            while not done and step < maxsteps:

                if verbose and i != 0 and i % render_every == 0:
                    self.env.render()

                # Epsilon-greedy exploration implementation
                # Exploit
                if random.uniform(0, 1) > self.epsilon:
                    if tuple(state_disc) not in self.table.keys():
                        self._add_state(state_disc)

                    action = self._get_best_action(state_disc)
                # Explore
                else:
                    action = self.env.action_space.sample()

                # make move
                next_state, reward, done, _ = self.env.step(action)
                # Discretize state
                next_state_disc = self._discretize(next_state)
                if tuple(next_state_disc) not in self.table.keys():
                    self._add_state(next_state_disc)

                # Calculate new Q-value using bellman's equation
                new_score = self._bellman(
                    state_disc,
                    next_state_disc,
                    action,
                    reward
                )
                
                # Add new Q-value to table
                self._set_q_value(state_disc, action, new_score)

                epoch_reward += reward
                state_disc = next_state_disc
                step += 1

            # Update epsilon
            self._update_epsilon()

            self.total_training_reward.append(epoch_reward)
            self.total_epsilon.append(self.epsilon)

            # Write performance to Tensorboard
            if self.save_progress:
                with self.summary_writer.as_default():
                    tf.summary.scalar(
                        'epoch reward', 
                        epoch_reward, 
                        step=i
                    )

                    tf.summary.scalar(
                        'epsilon', 
                        self.epsilon, 
                        step=i
                    )

            if verbose and i != 0 and i % 100 == 0:
                pbar.set_postfix({
                    'mean reward': '{:.1f}'.format(
                        np.mean(self.total_training_reward[-100:])
                    )
                })

    def save(self, fn):
        """Save model

        Args:
            fn (str): path to save model
        """
        savedict = self.__dict__.copy()
        savedict.pop('env')

        with open(fn, 'wb') as f:
            pickle.dump(savedict, f)

    def load(self, fn):
        """Load model

        Args:
            fn (str): path to save model
        """
        with open(fn, 'rb') as f:
            loaddict = pickle.load(f)

        for k, v in loaddict.items():
            setattr(self, k, v)

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
            # Discretize state
            state_disc = self._discretize(state)
            done = False
            step = 0
            sim_reward = 0
            while not done and step < maxsteps:
                if verbose:
                    self.env.render()
                if tuple(state_disc) in self.table.keys():
                    action = self._get_best_action(state_disc)
                else:
                    action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                state_disc = self._discretize(next_state)
                sim_reward += reward
                step += 1
            if verbose:          
                print('Reward run {}: {}'.format(i+1, sim_reward))
            rewards.append(sim_reward)
        return rewards