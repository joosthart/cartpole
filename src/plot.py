
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
        range(10000), 
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