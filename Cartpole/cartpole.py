"""
This is an experiment in the cartpole experiment in reinforcement learning.
It uses the AI gym by OpenAi
"""
import gym
import lasagne
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import matplotlib.pyplot as plt
def PolicyNetwork(input_var):
    """
    This sets up a network in Lasagne that decides on what move to play
    """
    network = lasagne.layers.InputLayer(shape=(None,4), input_var=input_var, name='Input')
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=200,
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotNormal(gain=1))
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=1,
                                        W=lasagne.init.GlorotNormal(),
                                        nonlinearity=lasagne.nonlinearities.sigmoid)
    return network


def run(choose_action, random_sampler, D_train, D_params):

    Rgoal = 50
    number_of_episodes = 10000
    env = gym.make('CartPole-v0')
    env.reset()
    lossplot = []
    rewardplot = []
    weightplot = []
    for N in range(number_of_episodes):

        if N % 100 == 0:
            print("Running {}th episode".format(N))
        memory_obs = []
        memory_act = []
        creward = 0
        obs = env.reset()
        for t in range(1000):
            memory_obs.append(obs)
            action = choose_action(obs.astype('float32').reshape(1, 4), random_sampler())
            memory_act.append(action[0][0])
            obs, reward, done, info = env.step(action[0][0])
            creward += reward
            if done:
                # Backpropagate the results
                lossplot.append(
                        D_train(np.array(memory_obs).astype('float32'),
                                np.array(memory_act).astype('int8'),                   # Actions
                                np.tile(creward, (len(memory_obs),)).astype('float32'),   # Reward
                                np.ones((len(memory_obs),),dtype='int8')*np.int8(Rgoal))) # Goal
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
    action_var = T.vector('actions')
    srng = RandomStreams(seed=42)
    Rgoal = T.vector('goal')

    D_network = PolicyNetwork(observations)
    D_params = lasagne.layers.get_all_params(D_network, trainable=True)

    P_act = lasagne.layers.get_output(D_network)
    f_act = theano.function([observations], P_act)

    # Set up an objective function that uses fake action labels.
    # This creates a vector of 'correct actions' (action taken when a game is won)
    D_obj = T.switch(T.gt(reward_var, Rgoal),
                     lasagne.objectives.binary_crossentropy(P_act, action_var),
                     lasagne.objectives.binary_crossentropy(P_act, 1-action_var)
                     ).mean()

    D_obj = -lasagne.objectives.binary_crossentropy(P_act, action_var).mean()*(Rgoal-reward_var).mean()

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-4, beta1=0.5)
    D_train = theano.function([observations, action_var, reward_var, Rgoal], D_obj, updates=D_updates, name='D_training')

    rv_u = srng.uniform(size=(1,))
    random_sampler = theano.function([], rv_u)
    D_out = T.switch(T.lt(lasagne.layers.get_output(D_network), random_var), int(0) ,int(1))

    choose_action = theano.function([observations, random_var], D_out, name='weighted_choice')
    run(choose_action, random_sampler, D_train, D_params)


if __name__=='__main__':
    TrainNetwork()