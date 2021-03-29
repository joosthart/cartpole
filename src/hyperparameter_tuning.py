from multiprocessing import Pool
import os


from src.algorithms import DeepQLearning

def train_one_model(params):
    
    model = DeepQLearning(**params)
    model.train(int(200), verbose=False)

def placeholder():
    
    grid = {
        ''
        'lr': [1e-1, 1e-2, 1e-3],
        'discount': [0.7, 0.8, 0.9],
        'update_freq': [1, 5, 10, 15],
        'batch_size': [32, 64, 128]
    }


    trials = []
    for l in grid['lr']:
        for d in grid['discount']:
            for f in grid['update_freq']:
                for b in grid['batch_size']:
                    if os.path.exists('./log/lr={}_discount={}_update_freq={}_batch_size={}'.format(l,d,f,b)):
                        continue
                    else:
                        trials.append({
                            'lr': l,
                            'discount': d,
                            'update_freq': f,
                            'batch_size': b,
                            'log_dir': './log/lr={}_discount={}_update_freq={}_batch_size={}'.format(l,d,f,b)
                        })

    with Pool(4) as p:
        p.map(train_one_model, trials)

if __name__ == '__main__':
    placeholder()
