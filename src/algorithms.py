import pickle
import random
import sys
from tqdm import tqdm
import os
import time

import gym
import numpy as np
import tensorflow as tf

from src.models import DenseModel

SEED = 42

class TabularQLearning:

    def __init__(self, epsilon=1, discount=0.95, lr=0.1, e_folding=None, **kwargs):
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

        self.env = gym.make('CartPole-v1')
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

    def train(self, epochs, maxsteps=10000, render_every=500):

        if not self.e_folding:
            self.e_folding = epochs / 10

        self.total_training_reward = []
        self.total_epsilon = []

        pbar = tqdm(range(epochs))
        for i in pbar:
            step = 0
            done = False

            # Reset environment
            state = self.env.reset()

            # Discretize state
            state_disc = self._discretize(state)
            if tuple(state_disc) not in self.table.keys():
                self._add_state(state_disc)

            epoch_reward = 0
            while not done and step < maxsteps:

                if i != 0 and i % render_every == 0:
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

            if i != 0 and i % 100 == 0:
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


class ExperienceBuffer:

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
        self.states[self.row] = state
        self.actions[self.row] = action
        self.rewards[self.row] = reward
        self.next_states[self.row] = next_state
        self.done_flags[self.row] = done_flag

        self.size = max(self.size, self.row)
        self.row = (self.row + 1) % self.max_buffer_size
    
    def get_size(self):
        return self.size

    def get_batch(self, batch_size):
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

    def __init__(
            self, num_hidden_states=[8,8], discount=0.95, max_buffer_size=1e6, 
            min_buffer_size=1e3, batch_size=64, lr=0.01, update_freq=10, 
            epsilon=0.2, min_epsilon=0.1, epsilon_decay=0.99, 
            hidden_activation='relu', hidden_initializer='random_normal', 
            output_initializer='random_normal', log_dir='./log/'):

        self.env = gym.make('CartPole-v0')
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

        self.summary_writer = tf.summary.create_file_writer(log_dir)

        if self.min_buffer_size < self.batch_size:
            raise ValueError((
                'batch size should be smaller than or equal to minimal buffer'
                'size.'
            ))
        
        self.memory = ExperienceBuffer(self.max_buffer_size, self.num_states)

    def _update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def _get_action(self, state, simulate=False):
        # Explore
        if np.random.random() < self.epsilon and not simulate:
            return self.env.action_space.sample()
        # Exploit
        else:
            return np.argmax(self.model.predict(state), axis=1)[0]

    def _bellman(self, next_q, reward):
        return reward + self.discount*next_q

    def _update_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_one_batch(self):

        states, actions, rewards, next_states, done_flags = \
            self.memory.get_batch(self.batch_size)

        q_values = self.model.predict(states)
        q_values_next = self.target_model.predict(next_states)

        q_values[range(len(q_values)), actions] = \
            rewards + self.discount*np.max(q_values_next, axis=1)
        q_values[done_flags, actions[done_flags]] = rewards[done_flags]

        loss = self.model.train(states, q_values)
        
        return loss

    def train(self, num_epochs, verbose=True, render_every=500):

        self.total_training_reward = []
        self.total_training_loss = []
        self.total_epsilon = []

        if verbose:
            pbar = tqdm(range(num_epochs))
        else:
            pbar = range(num_epochs)
        for epoch in pbar:
            state = self.env.reset()
            done = False
            epoch_reward = 0
            epoch_loss = []
            while not done:
                if verbose and epoch != 0 and epoch % render_every == 0:
                    self.env.render()
                action = self._get_action(state)

                next_state, reward, done, _ = self.env.step(action)

                epoch_reward += reward

                # if done:
                #     reward = -100

                self.memory.add(
                    state=state, 
                    action=action, 
                    reward=reward, 
                    next_state=next_state,
                    done_flag=done
                )

                if self.memory.get_size() > self.min_buffer_size:
                    loss = self.train_one_batch()
                    epoch_loss.append(loss) 

                state = next_state

            self.total_training_reward.append(epoch_reward)
            self.total_epsilon.append(self.epsilon)

            self._update_epsilon()
            self.total_training_loss.append(epoch_loss)
            
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
        self.model.save(fn)

    def load(self, fn):
        self.model.load(fn)

    def simulate(self, n_simulations, verbose=True, maxsteps=500):

        for i in range(n_simulations):
            # Reset environment
            state = self.env.reset()
            step = 0
            done = False
            while not done and step < maxsteps:
                self.env.render()
                action = self._get_action(state, simulate=True)
                state, _, done, _ = self.env.step(action)
                step += 1
            if verbose:          
                print('Reward run {}: {}'.format(i+1, step))
