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
import matplotlib.pyplot as plt
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
    network = lasagne.layers.InputLayer(shape=(None,4), input_var=input_var, name='Input')
    network = lasagne.layers.DenseLayer(incoming=network, num_units=100, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
    network = lasagne.layers.DenseLayer(incoming=network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    return network



def run(choose_action, random_sampler, D_train, D_params):

    number_of_episodes = 10000
    env = gym.make('CartPole-v0')
    env.reset()
    lossplot = []
    rewardplot = []
    weightplot = []
    for N in range(number_of_episodes):

#        print "New episode"]
        if N % 100 == 0:
            print("Running {}th episode".format(N))
        memory_obs = []
        creward = 0
        obs = env.reset()
        for t in range(1000):
            memory_obs.append(obs)
#            pdb.set_trace()
            action = choose_action(obs.astype('float32').reshape(1, 4), random_sampler())
            obs, reward, done, info = env.step(action[0][0])
            creward += reward
            if done:
                # Backpropagate the results
                lossplot.append(
                        D_train(np.array(memory_obs).astype('float32'),
                                np.tile(creward, (len(memory_obs),)).astype('float32')))
                rewardplot.append(creward)
                weightplot.append(np.median(D_params[1].get_value()))
                break

    # Investigate the trained model
    obs = env.reset()
    for t in range(1000):
        env.render()
        action = choose_action(obs.astype('float32').reshape(1, 4), random_sampler())
        obs, reward, done, info = env.step(action[0][0])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    plt.plot(lossplot)
    plt.plot(rewardplot)
    plt.legend(['Loss', 'Reward'])
    plt.show()
    plt.plot(weightplot)
    plt.show()

def TrainNetwork():



    observations = T.matrix('observations')
    reward_var = T.vector('reward')
    random_var = T.vector('random')
    srng = RandomStreams(seed=42)


    D_network = PolicyNetwork(observations)
    D_params = lasagne.layers.get_all_params(D_network, trainable=True)

    P_act = lasagne.layers.get_output(D_network)
    f_act = theano.function([observations], P_act)


#    D_obj = lasagne.objectives.binary_crossentropy(P_act, reward_var).mean()

    # Set up an objective function that backpropagates 1 if it wins, 0 otherwise
    # This creates a vector of 'correct actions'
    D_obj = lasagne.objectives.binary_crossentropy(
            P_act, T.switch(T.gt(reward_var, 200), P_act, 1-P_act)
    ).mean()*(200-reward_var.mean())

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-4, beta1=0.5)
    D_train = theano.function([observations, reward_var], D_obj, updates=D_updates, name='D_training')



    rv_u = srng.uniform(size=(1,))
    random_sampler = theano.function([], rv_u)
    D_out = T.switch(T.lt(lasagne.layers.get_output(D_network), random_var), int(1) ,int(0))


    choose_action = theano.function([observations, random_var], D_out, name='weighted_choice')
#    pdb.set_trace()
    run(choose_action, random_sampler, D_train, D_params)


if __name__=='__main__':
    TrainNetwork()







