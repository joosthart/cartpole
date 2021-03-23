

from src.algorithms import TabularQLearning
from src.plot import plot_tabular_q_learning_performance

def test():
    estimator = TabularQLearning(render=False, show_every=1000)

    # estimator.train(10000)

    # estimator.save('./models/test.pickle')

    # pprint(estimator.table)
    
    plot_tabular_q_learning_performance(
        './models/works_pretty_good.pickle',
        './output/tql_discount=0.95_lr=0.1_efolding=None_'
    )

if __name__ == '__main__':
    test()
    