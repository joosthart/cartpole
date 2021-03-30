

import tensorflow as tf
import matplotlib.pyplot as plt

from src.algorithms import TabularQLearning, DeepQLearning
from src.plot import plot_dqn_performance, plot_tabular_q_learning_performance


def test():
    # estimator = TabularQLearning()

    # estimator.train(30000, render_every=1e9)

    # estimator.save('./models/test.pickle')

    # # pprint(estimator.table)
    
    # plot_tabular_q_learning_performance(
    #     './models/test.pickle',
    #     './output/tql_discount=0.95_lr=0.1_efolding=None_'
    # )

    estimator = DeepQLearning(epsilon=1, epsilon_decay=0.99)

    estimator.train(int(3e2), render_every=10, verbose=True)

    # estimator.load('models/dqn_first')
    # estimator.simulate(5)

    plot_dqn_performance(estimator, 'dqn')



if __name__ == '__main__':
    test()
    