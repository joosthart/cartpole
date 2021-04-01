from multiprocessing import Pool
import os

from src.dql.agent import DeepQLearning
from src.tql.agent import TabularQLearning
from src.mcpg.agent import *
from src.plot import *

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

###############
#### MCPGs ####
###############

def mcpg_tuning(gamma):
    """Run hyper parameter search
    
    Args:
        gamma(float): gamma Hyperparameter, given when multiprocessing
    """
    #Set Hyper-parameter space
    
    #initialize last three Hyperparameters
    lrs = np.array([0.01, 0.001])
    normalizes = [True, False]
    n_hidden_layers = [2,0]
    
    #run grid search
    for lr in lrs:
        for normalize in normalizes:
            for n_hidden in n_hidden_layers:
                last_results, running_mean, running_loss, running_std = MDP(lr, gamma, normalize, n_hidden)
                
                #save
                np.save('./output/mcpg/last_results2000e_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                                          gamma,
                                                                                          normalize,
                                                                                          n_hidden), last_results)
                np.save('./output/mcpg/running_mean2000e_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                                          gamma,
                                                                                          normalize,
                                                                                          n_hidden), running_mean)
                np.save('./output/mcpg/running_loss2000e_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                                          gamma,
                                                                                          normalize,
                                                                                          n_hidden), running_loss)
                np.save('./output/mcpg/running_std2000e_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                                          gamma,
                                                                                          normalize,
                                                                                          n_hidden), running_std)

def load_results_mcpg(direc):
    """ Function that Loads all hyperparmeter runs for the MCPG implementation
    
    Args:
        direc(str): file location where runs are stored
        
    Returns:
        list: floats with global std of a all hyperparamter combinations
        list: floats with global mean of a all hyperparamter combinations
        ndim-array: the combination of hyperparameters used for a given index in the array (for broadcasting perpusses)
        list: booleans if a running mean with reward of 200 is achieved in the simulation
        list: ints with the amount of running means with a value of 200 per simulation
    """
    
    #Set used hyper paramter values
    gms = np.array([0.99,0.95,0.999])
    lrs = np.array([0.01, 0.001])
    normalizes = np.array([True, False], dtype = bool)
    n_hidden_layers = [2,0]
    
    n_tests = 1
    n_tests *= len(gms)
    n_tests *= len(lrs)
    n_tests *= len(normalizes)
    n_tests *= len(n_hidden_layers)
    
    #allocate memory
    stds           = np.zeros(n_tests)
    means          = np.zeros(n_tests)
    params         = np.zeros((n_tests,4), dtype = np.float)
    succeed        = np.empty(n_tests, dtype = np.bool)
    amount_succeed = np.empty(n_tests, dtype = np.int)
    i = 0
    
    #load all Hyperparameter runs
    for gamma in gms:
        for lr in lrs:
            for normalize in normalizes:
                for n_hidden in n_hidden_layers:
                    mean_local = np.load(direc + 'running_mean2000e_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                                                                        gamma,
                                                                                                                        normalize,
                                                                                                                        n_hidden))
                    
                    stds[i]           = np.std(mean_local)
                    means[i]          = np.mean(mean_local)
                    params[i]         = np.array([lr,
                                                  gamma,
                                                  normalize,
                                                  n_hidden])
                    
                    amount_succeed[i] = sum(mean_local > 199)
                    if amount_succeed[i] != 0:
                        succeed[i] = True
                    else:
                        succeed[i] = False
                    i+=1
    return stds, means, params, succeed, amount_succeed


def make_figures_mcpg():
    """ Generetes figures shown in the report
    """
    
    #initialize file names
    direc = './output/mcpg/'
    mean_str = 'running_mean2000e_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'
    std_str  = 'running_std2000e_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'

    #get files
    stds, means, params, succeed, amount_succeed = load_results_mcpg(direc)
    
    #Make plots with highest mean and most succeeded epochs
    plot_mcpg(mean_str, std_str, params[np.argmax(means)], direc)
    plot_mcpg(mean_str, std_str, params[np.argmax(amount_succeed)], direc)
    
    #Get figure of longer episode number runs
    mean_str = 'running_mean10000e_highestmean_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'
    std_str  = 'running_std10000e_highestmean_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'
    plot_mcpg(mean_str, std_str, params[np.argmax(means)], direc)
    
    mean_str = 'running_mean10000e_mostsucceed_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'
    std_str  = 'running_std10000e_mostsucceed_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'
    plot_mcpg(mean_str, std_str, params[np.argmax(amount_succeed)], direc)


def mcpg(n_cores = 3):
    """Run all done experiments.
    
    Args:
        n_cores(int, optional): number of cores used for multiprocessing
    """

    #Set First Hyper-parameter
    gms = np.array([0.99,0.95,0.999])
    pool = Pool(n_cores)

    #run tuning
    pool.map(mcpg_tuning, gms)
    
    
    # final two runs with long 10.000 episodes instead of 2 (hardcoded)
    #First
    lr, gamma, normalize, n_hidden = [0.01, 0.99, True, 0]
    last_results, running_mean, running_loss, running_std = MDP(lr, gamma, normalize, n_hidden, 10000)
                
    #save
    np.save('./output/mcpg/last_results10000e_highestmean_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), last_results)
    np.save('./output/mcpg/running_mean10000e_highestmean_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), running_mean)
    np.save('./output/mcpg/running_loss10000e_highestmean_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), running_loss)
    np.save('./output/mcpg/running_std10000e_highestmean_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), running_std)
    
    #Second
    lr, gamma, normalize, n_hidden = [0.01, 0.999, True, 2]
    last_results, running_mean, running_loss, running_std = MDP(lr, gamma, normalize, n_hidden, 10000)
                
    #save
    np.save('./output/mcpg/last_results10000e_mostsucceed_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), last_results)
    np.save('./output/mcpg/running_mean10000e_mostsucceed_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), running_mean)
    np.save('./output/mcpg/running_loss10000e_mostsucceed_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), running_loss)
    np.save('./output/mcpg/running_std10000e_mostsucceed_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden), running_std)
    
    make_figures_mcpg()
    
                                                                
