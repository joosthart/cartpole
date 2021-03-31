from multiprocessing import Pool
import os

from src.dql.agent import DeepQLearning
from src.tql.agent import TabularQLearning

def train_one_dqn_model(params):
    
    model = DeepQLearning(**params)
    model.train(int(150), verbose=False)
    model.save(params['log_dir'].replace('dql', 'models/dql'))
    

def dql(n_cores):
    
    grid = {
        'num_hidden_states': [[4], [8], [4,4], [8,8], [64], [64,64]],
        'lr': [1e-1, 1e-2, 1e-3],
        'discount': [0.9, 0.95, 0.99],
        'update_freq': [1, 5, 10, 15],
        'batch_size': [32, 64, 128]
    }


    # TODO fix the for-loops; temporary solution
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
                        if os.path.exists(os.path.join('log_dql', fn)):
                            continue
                        else:
                            trials.append({
                                'lr': l,
                                'discount': d,
                                'update_freq': f,
                                'batch_size': b,
                                'log_dir': os.path.join(
                                    'log', 'dql', fn
                                )
                            })

    with Pool(n_cores) as p:
        p.map(train_one_dqn_model, trials)

def train_one_tql_model(params):

    model = TabularQLearning(**params)
    model.train(int(2e4), verbose=False)
    model.save(params['log_dir'].replace('tql', 'models/tql') + '.pickle')

def tql(n_cores):
    
    grid = {
        'epsilon': [0.5, 1],
        'min_epsilon': [1e-2, 5e-1],
        'epsilon_decay': [0.9, 0.99],
        'lr': [1e-1, 1e-2],
        'discount': [0.9, 0.99],
        'position_step_size': [0.48, 0.24],
        'velocity_step_size': [0.4, 0.2],
        'angle_step_size': [0.0418, 0.0209],
        'angular_velocity_step_size': [0.4, 0.2]
        
    }

    # TODO fix the for-loops; temporary solution
    trials = []
    for e in grid['epsilon']:
        for em in grid['min_epsilon']:
            for ed in grid['epsilon_decay']:
                for l in grid['lr']:
                    for d in grid['discount']:
                        for ps in grid['position_step_size']:
                            for vs in grid['velocity_step_size']:
                                for ans in grid['angle_step_size']:
                                    for avs in grid['angular_velocity_step_size']:
                                        fn = (
                                            'epsilon={}_'
                                            'min_epsilon={}_'
                                            'epsilon_decay={}_'
                                            'lr={}_'
                                            'discount={}_'
                                            'position_step_size={}_'
                                            'velocity_step_size={}_'
                                            'angle_step_size={}_'
                                            'angular_velocity_step_size={}'

                                        )
                                        fn = fn.format(
                                            e, em, ed, l, d, ps, vs, ans, avs
                                        )
                                        if os.path.exists(
                                                os.path.join('log_tql', fn)
                                            ):
                                            continue
                                        else:
                                            trials.append({
                                                'epsilon': e,
                                                'min_epsilon': em,
                                                'epsilon_decay': ed,
                                                'lr': l,
                                                'discount': d,
                                                'position_step_size': ps,
                                                'velocity_step_size': vs,
                                                'angle_step_size': ans,
                                                'angular_velocity_step_size': avs,
                                                'log_dir': os.path.join(
                                                    'log','tql', fn
                                                )
                                            })

    with Pool(n_cores) as p:
        p.map(train_one_tql_model, trials)
