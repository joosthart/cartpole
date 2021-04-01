
from src.dql.agent import DeepQLearning
from src.tql.agent import TabularQLearning

def dql(episodes, model_path = (
            'demo/dql/'
            'num_hidden_states=64_lr=0.01_discount=0.9_update_freq=1_batch_size=32'
        ),
        max_steps = 5000):
        

    agent = DeepQLearning()
    agent.load(model_path)
    agent.env._max_episode_steps = max_steps
    agent.simulate(episodes, verbose=True)

def tql(episodes, model_path = (
            'demo/tql/'
            'best_model_epsilon=1_min_epsilon=0.01_epsilon_decay=0.999_lr=0.1_discount=0.99_position_step_size=0.48_velocity_step_size=0.4_angle_step_size=0.0209_angular_velocity_step_size=0.4.pickle'
        ),
        max_steps = 5000):
        
    agent = TabularQLearning()
    agent.load(model_path)
    agent.env._max_episode_steps = max_steps
    agent.simulate(episodes, verbose=True)
