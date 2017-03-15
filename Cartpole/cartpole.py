"""
This is an experiment in the cartpole experiment in reinforcement learning.
It uses the AI gym by OpenAi
"""

import gym
#import agentnet
import lasagne
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import itertools
import numpy as np
import pdb
"""
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
"""
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

def PolicyNetwork(input_var):
    """
    This sets up a network in Lasagne that decides on what move to play
    """
    network = lasagne.layers.InputLayer(shape=(None, 4), input_var=input_var, name='Input')
    network = lasagne.layers.DenseLayer(incoming=network, num_units=100, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
    network = lasagne.layers.DenseLayer(incoming=network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    return network

def weighted_choice(P, random_sample, random_var):
#    s_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)
    x = T.switch(T.lt(P, random_var), 1, 0)
    return theano.function([P, random_var], x)


def run(choose_action, D_train):

    number_of_episodes = 100
    env = gym.make('CartPole-v0')
    env.reset()

    for _ in range(number_of_episodes):
        memory = []
        lossplot = []
        observation = env.reset()
        for t in range(1000):
            memory.append(observation)
            action = choose_action(observation)
            obs, reward, done, info = env.step(action)
            if done:
                # Backpropagate the results
                lossplot.append(D_train(memory, np.tile(reward, (len(memory),))))
                break

def TrainNetwork():



    observations = T.matrix('observations')
    reward_var = T.vector('reward')
    random_var = T.vector('random')
    srng = RandomStreams(seed=42)


    D_network = PolicyNetwork(observations)
    D_params = lasagne.layers.get_all_params(D_network, trainable=True)

    P_act = lasagne.layers.get_output(D_network)


    D_obj = lasagne.objectives.binary_crossentropy(P_act, reward_var).mean()
    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-4, beta1=0.5)
    D_train = theano.function([observations, reward_var], D_obj, updates=D_updates, name='D_training')



    rv_u = srng.uniform(size=(1,))
    random_sampler = theano.function([], rv_u)
    choose_action = weighted_choice(lasagne.layers.get_output(D_network), random_sampler, random_var)

    run(choose_action, D_train)

if __name__=='__main__':
    TrainNetwork()







