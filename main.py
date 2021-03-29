

import tensorflow as tf
import matplotlib.pyplot as plt

from src.algorithms import TabularQLearning, DeepQLearning
from src.plot import plot_dqn_performance


def test():
    # estimator = TabularQLearning(render=False, show_every=1000)

    # # estimator.train(10000)

    # # estimator.save('./models/test.pickle')

    # # pprint(estimator.table)
    
    # plot_tabular_q_learning_performance(
    #     './models/works_pretty_good.pickle',
    #     './output/tql_discount=0.95_lr=0.1_efolding=None_'
    # )

    estimator = DeepQLearning(log_dir='./log/testen_met_thijs/')

    estimator.train(400, render_every=400)

    estimator.save('models/dqn_first')

    plot_dqn_performance(estimator, 'dqn')



if __name__ == '__main__':
    test()
    