from multiprocessing import Pool
import os


from src.algorithms import DeepQLearning
from src.plot import plot_dqn_performance

def train_one_model(params):
    
    model = DeepQLearning(**params)
    model.train(int(50), verbose=False)
    model.save(params['log_dir'].replace('log', 'models'))

    plot_dqn_performance(model, params['log_dir'].replace('log', 'output'))
    

def dql(n_cores):
    
    grid = {
        'num_hidden_states': [[4], [8], [4,4], [8,8], [64], [64,64]],
        'lr': [1e-1, 1e-2, 1e-3],
        'discount': [0.9, 0.95, 0.99],
        'update_freq': [1, 5, 10, 15],
        'batch_size': [32, 64, 128]
    }


    trials = []
    for n in grid['num_hidden_states']:
        for l in grid['lr']:
            for d in grid['discount']:
                for f in grid['update_freq']:
                    for b in grid['batch_size']:
                        fn = (
                            'num_hidden_states={}_'
                            'lr={}_'
                            'discount={}_'
                            'update_freq={}_'
                            'batch_size={}'
                        )
                        fn = fn.format('_'.join([str(x) for x in n]), l,d,f,b)
                        if os.path.exists(os.path.join('log', fn)):
                            continue
                        else:
                            trials.append({
                                'lr': l,
                                'discount': d,
                                'update_freq': f,
                                'batch_size': b,
                                'log_dir': os.path.join('log', fn)
                            })

    with Pool(n_cores) as p:
        p.map(train_one_model, trials)

if __name__ == '__main__':
    dql(4)
