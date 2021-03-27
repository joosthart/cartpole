import pickle
import random
import sys
from numpy.__config__ import show
from tqdm import tqdm

import gym
import numpy as np
import tensorflow as tf

from src.models import DenseModel

SEED = 42

class TabularQLearning:

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


class ExperienceBuffer:

    def __init__(self, max_buffer_size, obs_space):
        self.max_buffer_size = int(max_buffer_size)
        self.obs_space = int(obs_space)

        self.states = np.zeros((self.max_buffer_size, self.obs_space))
        self.actions = np.zeros(self.max_buffer_size)
        self.rewards = np.zeros(self.max_buffer_size)
        self.next_states = np.zeros((self.max_buffer_size, self.obs_space))
        self.done_flags = np.zeros(self.max_buffer_size)

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
            min(self.size, self.max_buffer_size), batch_size, replace=True
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
            self, num_hidden_states=[8,8], discount=0.99,
            max_buffer_size=1e6, min_buffer_size=1e2, batch_size=32, lr=1e-2,
            update_freq=50, epsilon=1, min_epsilon=0.1, e_folding=None,
            hidden_activation='relu', hidden_initializer='random_normal',
            output_activation='sigmoid', output_initializer='random_normal',
            show_every=10, log_dir='./log/'):

        self.env = gym.make('CartPole-v0')
        self.env.seed(SEED)

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.discount = discount
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.epsilon = epsilon
        self.max_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.e_folding = e_folding
        self.show_every = show_every

        self.model = DenseModel(
            input_shape=self.num_states,
            output_shape=self.num_actions,
            num_hidden_nodes=num_hidden_states,
            lr=lr,
            hidden_activation=hidden_activation,
            hidden_initializer=hidden_initializer,
            output_activation=output_activation,
            output_initializer=output_initializer
        )

        self.target_model = DenseModel(
            input_shape=self.num_states,
            output_shape=self.num_actions,
            num_hidden_nodes=num_hidden_states,
            lr=lr,
            hidden_activation=hidden_activation,
            hidden_initializer=hidden_initializer,
            output_activation=output_activation,
            output_initializer=output_initializer
        )

        self.summary_writer = tf.summary.create_file_writer(log_dir)

        if self.min_buffer_size < self.batch_size:
            raise ValueError((
                'batch size should be smaller than or equal to minimal buffer'
                'size.'
            ))
        
        self.memory = ExperienceBuffer(self.max_buffer_size, self.num_states)

    def _update_epsilon(self, episode):
        # self.epsilon = \
        #     self.min_epsilon + \
        #     (self.max_epsilon - self.min_epsilon) * \
        #     np.exp(-episode/self.e_folding)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= 0.9999
        else:
            self.epsilon = self.min_epsilon

    def _get_action(self, state):
        # Explore
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # Exploit
        else:
            return np.argmax(self.model.predict(np.atleast_2d(state)), axis=1)[0]

    def _bellman(self, next_q, reward):
        return reward + self.discount*next_q

    def _update_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, obs, model=None):  
        features = np.atleast_2d(obs.astype('float32'))
        if not model:
            return self.model(features)
        else:
            return model(features)

    def train_one_batch(self):

        states, actions, rewards, next_states, done_flags = \
            self.memory.get_batch(self.batch_size)

        y_target = self.model.predict(states)

        for i in range(states.shape[0]):
            if done_flags[i]:
                y_target[i, int(actions[i])] = rewards[i]
            else:
                y_next = np.max(
                    self.target_model.predict(states)
                )
                y_target[i, int(actions[i])] = self._bellman(y_next[i], rewards[i])

        loss = self.model.train(states, y_target)

        return loss

    def train(self, num_epochs):

        if not self.e_folding:
            self.e_folding = num_epochs / 1000

        self.total_training_reward = []
        self.total_training_loss = []
        self.total_epsilon = []
        
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            state = self.env.reset()
            done = False
            epoch_reward = 0 
            while not done:
                # if epoch % 500 == 0:
                #     self.env.render()

                action = self._get_action(state)

                next_state, reward, done, _ = self.env.step(action)

                epoch_reward += reward

                if done:
                    reward = -200

                self.memory.add(
                    state=state, 
                    action=action, 
                    reward=reward, 
                    next_state=next_state,
                    done_flag=done
                )

                state = next_state

            self.total_training_reward.append(epoch_reward)
            self.total_epsilon.append(self.epsilon)

            if self.memory.get_size() > self.min_buffer_size:
                self._update_epsilon(epoch)
                
                epoch_loss = self.train_one_batch()
                self.total_training_loss.append(epoch_loss)
                
                # with self.summary_writer.as_default():
                #     tf.summary.scalar(
                #         'epoch reward', 
                #         epoch_reward, 
                #         step=epoch
                #     )
                #     tf.summary.scalar(
                #         'running avg reward(100)', 
                #         np.mean(self.total_training_reward[-100:]), step=epoch
                #     )
                #     tf.summary.scalar(
                #         'average loss', 
                #         epoch_loss, 
                #         step=epoch
                #     )
            
            if epoch % self.update_freq == 0:
                self._update_model()

            if epoch % 100 == 0:
                pbar.set_postfix({
                    'mean reward': '{:.1f}'.format(
                        np.mean(self.total_training_reward[-100:])
                    ),
                    'max reward': '{:.1f}'.format(
                        np.max(self.total_training_reward[-100:])
                    ),
                    'min reward': '{:.1f}'.format(
                        np.min(self.total_training_reward[-100:])
                    ),
                    'mean_loss': '{:.1f}'.format(
                        np.mean(self.total_training_loss[-100:])
                    )
                })

    def load(self):
        pass



# class DeepQLearning:

    #     def __init__(
    #             self, num_hidden_states=[200,200], discount=0.99,
    #             max_buffer_size=1e5, min_buffer_size=1e2, batch_size=32, lr=1e-2,
    #             update_freq=25, epsilon=1, min_epsilon=1e-4, e_folding=None,
    #             hidden_activation='relu', hidden_initializer='glorot_uniform',
    #             output_activation='sigmoid', output_initializer='glorot_uniform',
    #             show_every=10):

    #         self.env = gym.make('CartPole-v0')

    #         self.num_states = self.env.observation_space.shape[0]
    #         self.num_actions = self.env.action_space.n

    #         self.discount = discount
    #         self.max_buffer_size = max_buffer_size
    #         self.min_buffer_size = min_buffer_size
    #         self.batch_size = batch_size
    #         self.update_freq = update_freq
    #         self.epsilon = epsilon
    #         self.max_epsilon = epsilon
    #         self.min_epsilon = min_epsilon
    #         self.e_folding = e_folding
    #         self.show_every = show_every

    #         self.optimizer = tf.optimizers.Adam(lr)

    #         self.model = DenseModel(
    #             self.num_states,
    #             self.num_actions,
    #             num_hidden_states,
    #             hidden_activation,
    #             hidden_initializer,
    #             output_activation,
    #             output_initializer
    #         )

    #         self.target_model = DenseModel(
    #             self.num_states,
    #             self.num_actions,
    #             num_hidden_states,
    #             hidden_activation,
    #             hidden_initializer,
    #             output_activation,
    #             output_initializer
    #         )

    #         if self.min_buffer_size < self.batch_size:
    #             raise ValueError((
    #                 'batch size should be smaller than or equal to minimal buffer'
    #                 'size.'
    #             ))
            
    #         self.memory = ExperienceBuffer(self.max_buffer_size, self.num_states)

    #     def _update_epsilon(self, episode):
    #         self.epsilon = \
    #             self.min_epsilon + \
    #             (self.max_epsilon - self.min_epsilon) * \
    #             np.exp(-episode/self.e_folding)

    #     def _get_action(self, state):
    #         # Explore
    #         if np.random.random() < self.epsilon:
    #             return self.env.action_space.sample()
    #         # Exploit
    #         else:
    #             return np.argmax(self.predict(state)[0])

    #     def _bellman(self, next_q, reward):
    #         return reward + self.discount*next_q

    #     def _add_to_buffer(self, exp):

    #         for idx, k in enumerate(self.exp_buffer.keys()):
    #             if len(self.exp_buffer[k]) >= self.max_buffer_size:
    #                 self.exp_buffer[k].pop()
    #             self.exp_buffer[k].append(exp[idx])

    #         # if self.exp_buffer.shape[0]==0:
    #         #     self.exp_buffer = np.asarray(exp)
    #         # elif self.exp_buffer.shape[0] >= self.max_buffer_size:
    #         #     self.exp_buffer = np.delete(self.exp_buffer, 0, axis=0)
    #         # else:
    #         #     self.exp_buffer = np.vstack((self.exp_buffer, np.asarray(exp)))

    #     def _update_model(self):


    #         self.model.trainable_variables = \
    #             self.target_model.trainable_variables.copy()

    #     def predict(self, obs, model=None):  
    #         features = np.atleast_2d(obs.astype('float32'))
    #         if not model:
    #             return self.model(features)
    #         else:
    #             return model(features)

    #     def train_one_batch(self):

    #         if len(self.exp_buffer['states']) <= self.min_buffer_size:
    #             return False


    #         sel = np.random.randint(
    #             0, len(self.exp_buffer['states']), self.batch_size
    #         )

    #         states = np.asarray(self.exp_buffer['states'])[sel]
    #         actions = np.asarray(self.exp_buffer['actions'])[sel]
    #         rewards = np.asarray(self.exp_buffer['rewards'])[sel]
    #         next_states = np.asarray(self.exp_buffer['next_states'])[sel]
    #         done_flags = np.asarray(self.exp_buffer['terminal_state_flags'])[sel]

    #         pred = self.predict(next_states, self.target_model)
    #         next_q = np.max(pred, axis=1)

    #         true_q = np.where(
    #             done_flags, rewards, self._bellman(next_q, rewards)
    #         )

    #         with tf.GradientTape() as tape:
    #             pred_q = tf.math.reduce_sum(
    #                 self.predict(states, self.model) *
    #                 tf.one_hot(actions, self.num_actions)
    #             )

    #             loss = tf.math.reduce_mean((true_q - pred_q)**2)

    #         model_variables = self.model.trainable_variables
    #         grad = tape.gradient(loss, model_variables)

    #         self.optimizer.apply_gradients(zip(grad, model_variables))

    #         return loss

    #     def simulate_episode(self, render=False):

    #         done = False

    #         # Reset environment
    #         state = self.env.reset()

            

    #         episode_reward = 0
    #         episode_loss = []
    #         burnin = False

    #         i = 0
    #         while not done:
    #             if render:
    #                 self.env.render()

    #             action = self._get_action(state)
                
    #             next_state, reward, done, _ = self.env.step(action)

    #             loss = self.train_one_batch()
    #             if not loss:
    #                 burnin = True
    #             else:
    #                 episode_loss.append(loss)
    #                 episode_reward += reward

    #             exp = [state, action, reward, next_state, done]
    #             self._add_to_buffer(exp)

    #             state = next_state

    #             if not burnin and i != 0 and i % self.update_freq == 0:
    #                 print('update')
    #                 self._update_model()
                
    #             i += 1

    #         # print(episode_reward, episode_loss)

    #         print(i)

    #         if burnin:
    #             return None, None
    #         else:
    #             return episode_reward, episode_loss

    #     def train(self, episodes):

    #         if not self.e_folding:
    #             self.e_folding = episodes / 10

    #         self.total_training_reward = []
    #         self.total_training_loss = []

    #         pbar = tqdm(range(episodes))
    #         for i in pbar:
    #             burnin = False
                
    #             if i != 0 and i % self.show_every == 0:
    #                 episode_reward, episode_loss = self.simulate_episode(True)
    #             else:
    #                 episode_reward, episode_loss = self.simulate_episode()

    #             if not episode_reward or not episode_loss:
    #                 burnin = True
    #             else:
    #                 self.total_training_reward.append(episode_reward)
    #                 self.total_training_loss.append(np.mean(episode_loss))

    #             if burnin:
    #                 pbar.set_postfix({'burnin...':''})
    #             elif i != 0 and i % 10 == 0:
    #                 pbar.set_postfix({
    #                     'mean reward': '{:.1f}'.format(
    #                         np.mean(self.total_training_reward[-10:])
    #                     )
    #                 })

    #             self._update_epsilon(i)

    #     def load(self):
    #         pass
