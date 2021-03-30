import pickle
import random
from tqdm import tqdm

import gym
import numpy as np
import tensorflow as tf

SEED = 42

class TabularQLearning:

    def __init__(self, epsilon=1, discount=0.95, lr=0.1, e_folding=None, 
                 save_progress=True, log_dir='log_tql', **kwargs):
        self.epsilon = epsilon
        self.max_epsilon = epsilon
        self.min_epsilon = 1e-4
        self.discount = discount
        self.lr = lr
        self.e_folding = e_folding

        self.step_size = [
            kwargs.get('position_step_size', 0.48),
            kwargs.get('velocity_step_size', 0.4),
            kwargs.get('angle_step_size', 0.0418),
            kwargs.get('angular_velocity_step_size', 0.4)
        ]

        self.save_progress = save_progress
        if self.save_progress:
            self.summary_writer = tf.summary.create_file_writer(log_dir)

        self.env = gym.make('CartPole-v0')
        self.env.seed(SEED)
        self.table = {}

        np.random.seed(SEED)

    def _bellman(self, state, next_state, action, reward):
        q0 = self._get_q_value(state, action)
        q1 = self._get_q_value(next_state, self._get_best_action(next_state))
        return (1-self.lr)*q0 + self.lr*(reward + self.discount*q1)

    def _discretize(self, state):
        state = state.copy()
        for idx, (s, ds) in enumerate(zip(state, self.step_size)):
            state[idx] = round(
                (s + 0.5*ds)//ds * ds,
                len(str(ds).split('.')[-1])
            )
        return state

    def _add_state(self, state):
        action_space = list(range(self.env.action_space.n))
        random.shuffle(action_space)  # Random action is the first in the dict
        self.table[tuple(state)] = {
            k: 0 for k in action_space
        }

    def _set_q_value(self, state, action, score):
        self.table[tuple(state)][action] = score

    def _get_q_value(self, state, action):
        return self.table[tuple(state)][action]

    def _get_best_action(self, state):
        return max(
            self.table[tuple(state)],
            key=self.table[tuple(state)].get
        )

    def _update_epsilon(self, epoch):
        self.epsilon = \
            self.min_epsilon + \
            (self.max_epsilon - self.min_epsilon)*np.exp(-epoch/self.e_folding)

    def train(self, epochs, maxsteps=10000, render_every=500, verbose=True):

        if not self.e_folding:
            self.e_folding = epochs / 10

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

                # Exploit
                if random.uniform(0, 1) > self.epsilon:
                    if tuple(state_disc) not in self.table.keys():
                        self._add_state(state_disc)

                    action = self._get_best_action(state_disc)
                # Explore
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)

                next_state_disc = self._discretize(next_state)
                if tuple(next_state_disc) not in self.table.keys():
                    self._add_state(next_state_disc)

                new_score = self._bellman(
                    state_disc,
                    next_state_disc,
                    action,
                    reward
                )

                self._set_q_value(state_disc, action, new_score)

                epoch_reward += reward
                state_disc = next_state_disc
                step += 1

            self._update_epsilon(i)

            self.total_training_reward.append(epoch_reward)
            self.total_epsilon.append(self.epsilon)

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
        savedict = self.__dict__.copy()
        savedict.pop('env')

        with open(fn, 'wb') as f:
            pickle.dump(savedict, f)

    def load(self, fn):
        with open(fn, 'rb') as f:
            loaddict = pickle.load(f)

        for k, v in loaddict.items():
            setattr(self, k, v)

    def simulate(self, maxsteps=100):
        done = False
        step = 0

        # Reset environment
        state = self.env.reset()
        # Discretize state
        state_disc = self._discretize(state)
        while not done and step < maxsteps:

            self.env.render()

            if tuple(state_disc) in self.table.keys():
                action = self._get_best_action(state_disc)
            else:
                action = self.env.action_space.sample()

            next_state, _, done, _ = self.env.step(action)

            state_disc = self._discretize(next_state)
            step += 1