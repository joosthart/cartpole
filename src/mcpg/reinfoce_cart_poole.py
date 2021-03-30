from src.mcpg.agent import *
import multiprocessing as mp
import gym

import matplotlib.pyplot as plt

#Set global variables
Number_episodes = 5
seed            = 42
env             = gym.make('CartPole-v0')
env.seed(seed)

def MDP(lr = 0.01, gamma = 0.99, normalize = True, n_hidden = 3):
    #initialize policy
    agent  = Agency(env, lr, gamma, normalize, n_hidden, seed)

    #allocate memory
    running_mean = []
    running_loss = []
    running_std  = []
    last_results = []

    #begin training
    for i_episode in range(Number_episodes):

        #reset variables
        s_old        = env.reset() 
        cum_reward   = 0
        run          = 0
        done         = False

        #begin episode run
        while not done:
            #Get and make action from probabillity distribution
            action = agent.make_move(s_old) #take action given state
            s_new, reward, done, info = env.step(action)

            #store run
            agent.save_step(s_old, action, reward)

            #update
            cum_reward += reward
            run        += 1
            s_old       = s_new
        
        #Store episode
        last_results.append(cum_reward)
        
        #Train network
        loss, mean, std = agent.train(last_results)
        running_mean.append(mean)
        running_std.append(std)
        running_loss.append(loss)

        #Visualize
        if i_episode % 10 == 0:
          print('EPISODE {}'.format(i_episode), 
                '\t| REWARD: {:.0f}'.format(cum_reward),
                '\t| RUNNING MEAN: {:.0f}'.format(mean))

    return last_results, running_mean, running_loss, running_std

def visualize(mean, std,lr,gamma, normalize, n_hidden):
  
  plt.figure()
  plt.title("lr{}_gamma{}_normalize{}_hiddenlayers{}".format(lr,gamma, normalize, n_hidden))
  plt.xlabel('Evolution (runs)')
  plt.ylabel('Running Mean Reward')
  plt.plot(mean, label = 'Runnning Mean')
  plt.fill_between(np.arange(len(mean)), mean + std, mean-std, alpha = 0.6, label = '1$\sigma$')
  plt.tight_layout()
  plt.savefig('./save/mcpg_lr{}_gamma{}_normalize{}_hiddenlayers{}.pdf')
  plt.close()


def hyperparameter_tuning(gamma):
  """Run hyper parameter search"""
  #Set Hyper-parameter space
  lrs = np.array([0.01, 0.001])
  normalizes = [True, False]
  n_hidden_layers = [2,0]

  for lr in lrs:
    for normalize in normalizes:
      for n_hidden in n_hidden_layers:
        last_results, running_mean, running_loss, running_std = MDP(lr, gamma, normalize, n_hidden)
        
        print()
        print("Running lr{}_gamma{}_normalize{}_hiddenlayers{}".format(lr, 
                                                                      gamma,
                                                                      normalize, 
                                                                      n_hidden))
        
        print()
        visualize(running_mean, running_std, lr,gamma, normalize, n_hidden)

        #save
        np.save('./save/last_results_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr, 
                                                                                  gamma,
                                                                                  normalize, 
                                                                                  n_hidden), last_results)
        np.save('./save/running_mean_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr, 
                                                                                  gamma,
                                                                                  normalize, 
                                                                                  n_hidden), running_mean)
        np.save('./save/running_loss_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr, 
                                                                                  gamma,
                                                                                  normalize, 
                                                                                  n_hidden), running_loss)
        np.save('./save/running_std_lr{}_gamma{}_normalize{}_hiddenlayers{}.npy'.format(lr, 
                                                                                  gamma,
                                                                                  normalize, 
                                                                                  n_hidden), running_std)


def main():
  """Run all done experiments"""

  #Set Hyper-parameter space
  gms = np.array([0.99,0.95,0.999])
  pool = mp.Pool(3)

  #run 
  pool.map(hyperparameter_tuning, gms)    

if __name__ == '__main__':
  main()