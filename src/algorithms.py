import pickle
import random
import sys
from tqdm import tqdm

import gym
import numpy as np
import tensorflow as tf

from src.models import DenseModel


class TabularQLearning():

    def __init__(self, render=False, epsilon=1, discount=0.95, lr=0.1, e_folding=None, **kwargs):
        self.render = render
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

        self.show_every = kwargs.get('show_every', sys.maxsize)

        self.env = gym.make('CartPole-v0')
        self.table = {}

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

    def train(self, epochs, maxsteps=10000):

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

                if i != 0 and i % self.show_every == 0:
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

            if i != 0 and i % 1000 == 0:
                pbar.set_postfix({
                    'mean reward': '{:.1f}'.format(
                        np.mean(self.total_training_reward[-1000:])
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


class DeepQLearning():

    def __init__(self, num_states, num_actions, num_hidden_states, discount, 
                 max_buffer_size, min_buffer_size, batch_size, lr, epsilon=1, 
                 e_folding=None, hidden_activation='relu', 
                 hidden_initializer='glorot_uniform', 
                 output_activation='sigmoid', 
                 output_initializer='glorot_uniform'):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount = discount
        self.epsilon = epsilon
        self.max_epsilon = epsilon
        self.min_epsilon = 1e-4
        self.e_folding = e_folding

        self.optimizer = tf.optimizers.Adam(lr)

        self.model = DenseModel(
            num_states, 
            num_hidden_states, 
            num_actions,
            hidden_activation, 
            hidden_initializer,
            output_activation,
            output_initializer
        )

        self.target_model = DenseModel(
            num_states, 
            num_hidden_states, 
            num_actions,
            hidden_activation, 
            hidden_initializer,
            output_activation,
            output_initializer
        )

        self.exp_buffer = np.array([]) # s, a, r, s', done

        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size

        if self.min_buffer_size < self.batch_size:
            raise ValueError((
                'batch size should be smaller than or equal to minimal buffer'
                'size'
            ))

        self.env = gym.make('CartPole-v0')

    def _update_epsilon(self, epoch):
        self.epsilon = \
            self.min_epsilon + \
            (self.max_epsilon - self.min_epsilon)*np.exp(-epoch/self.e_folding)
    
    def _get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state)[0])
    
    def _bellman(self, next_q, reward):
        return reward + self.discount*next_q

    def _add_to_buffer(self, exp):
        if self.exp_buffer.shape[0] >= self.max_buffer_size:
            self.exp_buffer = np.delete(self.exp_buffer, 0, axis=0)

        self.exp_buffer = np.append(self.exp_buffer, exp)
        
    def _update_model(self):
        self.model.trainable_variables = \
            self.target_model.trainable_variables.copy()

    def predict(self, obs, model=None):
        if not model:
            return self.model(np.atleast_2d(obs))
        else:
            return model(np.atleast_2d(obs))

    def train_one_batch(self):
        if self.exp_buffer.shape[0] < self.min_buffer_size:
            return False
        
        sel = np.random.randint(0, self.exp_buffer.shape[0], self.batch_size)
        sample = self.exp_buffer[sel,:]

        states = sample[:,0]
        actions = sample[:,1]
        rewards = sample[:,2]
        next_states = sample[:,3]
        done_flags = sample[:,4]

        next_q = np.max(self.predict(next_states, self.target_model), axis=1) 

        true_q = np.where(
            done_flags, rewards, self._bellman(next_q, rewards)
        )

        with tf.GradientTape() as tape:
            pred_q = tf.math.reduce_sum(
                self.predict(states, self.model) * \
                tf.one_hot(actions, self.num_actions)
            )

            loss = tf.math.reduce_mean((true_q - pred_q)**2)
        
        model_variables = self.model.trainable_variables
        grad = tape.gradient(loss, model_variables)

        self.optimizer.apply_gradients(zip(grad, model_variables))

        return loss

    def train(self):
        

    def load(self):
        pass

    def simulate(self):
        pass
