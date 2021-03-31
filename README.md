# cartpole

The repository contains three reinforcement learning algorithm which solve the [CartPole](https://gym.openai.com/envs/CartPole-v0/) problem. Using the minimal CLI, simple grid search can be run on the different algorithms. Furthermore, simulations can be run which illustrate the performance of the models.

## installation
Install the necessary packages using:
```
pip install -r requirements.txt
```

## Usage
All experiments can be run using:
```
python main.py -r all
```
The progress of the Tabular Q-Learning and Deep Q-Learning algorithm runs can be displayed in Tensorboard. To start Tensorboard, run:
```
tensorboard --log-dir ./log/
```
**Note:** The grid search algorithm is far from optimized and might take several hours to run.



With the following command, 5 episodes of the DQL agent will be shown:
```
python main.py --simulate DQL --episodes 5
```
All options can be displayed using:
```
python main.py --help
```
## Acknoledgments
We had help from some very good blogs and code examples. Below a list of the most helpful ones.

- https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
- https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
- https://www.mlprojecttutorials.com/reinforcement%20learning/cartpole/
- https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
- https://medium.com/@dey.ritajit/learning-cart-pole-and-lunar-lander-through-reinforce-9191fa21decc
