
from src.dql.agent import DeepQLearning
from src.tql.agent import TabularQLearning

def dql(episodes, model_path = (
            'demo/dql/'
            'num_hidden_states=64_lr=0.01_discount=0.9_update_freq=1_batch_size=32'
        )):
        

    agent = DeepQLearning()
    agent.load(model_path)

    agent.simulate(episodes, verbose=True)

def tql(episodes, model_path = (
            'demo/dql/'
            'num_hidden_states=64_lr=0.01_discount=0.9_update_freq=1_batch_size=32'
        )):
        
    agent = TabularQLearning()
    agent.load(model_path)

    agent.simulate(episodes, verbose=True)
