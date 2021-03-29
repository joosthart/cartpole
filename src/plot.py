
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms import TabularQLearning

plt.style.use('seaborn-darkgrid')

def plot_tabular_q_learning_performance(model_fn, save_prefix, show=False):

    estimator = TabularQLearning()
    
    estimator.load(model_fn)

    total_training_reward = pd.Series(estimator.total_training_reward)


    plt.figure()
    plt.plot(
        total_training_reward.rolling(1000).mean()
    )
    plt.fill_between(
        range(len(total_training_reward)), 
        total_training_reward.rolling(1000).mean() - total_training_reward.rolling(1000).std(), 
        total_training_reward.rolling(1000).mean() + total_training_reward.rolling(1000).std(),
        alpha=0.5,
        label='Standard deviation'
    )
    plt.axis(xmin=1000, xmax=10000-1)
    plt.xlabel('epoch')
    plt.ylabel('Rolling mean reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_prefix + 'rolling_mean_reward.jpg', dpi=300)


    plt.figure()
    plt.plot(estimator.total_epsilon)
    plt.xlabel('Epoch')
    plt.ylabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig(save_prefix + 'epsilon.jpg', dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

def plot_dqn_performance(model, save_prefix, window_size=10, show=False):


    total_training_reward = pd.Series(model.total_training_reward)
    total_training_loss = pd.Series(model.total_training_loss)

    total_training_loss = total_training_loss.apply(np.nan_to_num)

    plt.figure()
    plt.plot(
        total_training_reward.rolling(window_size).mean()
    )

    plt.fill_between(
        range(len(total_training_reward)), 
        total_training_reward.rolling(window_size).mean() - total_training_reward.rolling(window_size).std(), 
        total_training_reward.rolling(window_size).mean() + total_training_reward.rolling(window_size).std(),
        alpha=0.5,
        label='Standard deviation'
    )
    plt.axis(xmin=window_size, xmax=len(total_training_reward)-1)
    plt.xlabel('epoch')
    plt.ylabel('Rolling mean reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_prefix + 'rolling_mean_reward.jpg', dpi=300)

    plt.figure()
    plt.plot(
        total_training_loss.rolling(window_size).mean()
    )
    plt.fill_between(
        range(len(total_training_loss)), 
        total_training_loss.rolling(window_size).mean() - total_training_loss.rolling(window_size).std(), 
        total_training_loss.rolling(window_size).mean() + total_training_loss.rolling(window_size).std(),
        alpha=0.5,
        label='Standard deviation'
    )
    plt.axis(xmin=window_size, xmax=len(total_training_loss)-1)
    plt.xlabel('epoch')
    plt.ylabel('Rolling mean loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_prefix + 'rolling_mean_loss.jpg', dpi=300)

    plt.figure()
    plt.plot(model.total_epsilon)
    plt.xlabel('Epoch')
    plt.ylabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig(save_prefix + 'epsilon.jpg', dpi=300)

    if show:
        plt.show()
    else:
        plt.close()