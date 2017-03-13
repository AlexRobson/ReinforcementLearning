"""
This is an experiment in the cartpole experiment in reinforcement learning.
It uses the AI gym by OpenAi
"""

import gym
import agentnet
import itertools
import numpy as np

act = itertools.cycle((0,1))
env = gym.make('CartPole-v0')
env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
#        action = env.action_space.sample()
        action = next(act)
#        observation, reward, done, info = env.step(action)
        if False:
            print("Episode finished after {} timesteps".format(t+1))
            break

def InferValue(observation):
    pass

def policy(observation, action_space):
    """
    Use the observations of the present state to assess a policy
    """
    def greedy(actionvalue, actionspace):
        """
        Choose the action that yeilds the greatest value
        """
        return actionspace[np.argmax(actionvalue)]


    actionvalue = [InferValue(observation, act) for act in action_space]
    return greedy(actionvalue)

