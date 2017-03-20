"""
This is an experiment in the cartpole experiment in reinforcement learning.
It uses the AI gym by OpenAi
"""
import gym
from gym import wrappers
import lasagne
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import pdb
import matplotlib.pyplot as plt
def PolicyNetwork(input_var):

    """
    This sets up a network in Lasagne that decides on what move to play
    """
    from lasagne.layers import batch_norm
    from lasagne.layers import DenseLayer
    from lasagne.layers import InputLayer
    from lasagne.nonlinearities import rectify, sigmoid
    from lasagne.init import GlorotNormal
    network = InputLayer(shape=(None,4), input_var=input_var, name='Input')
    network = (DenseLayer(incoming=network,
                                        num_units=50,
                                        nonlinearity=rectify,
                                        W=GlorotNormal(gain=1))
                         )
    network = (DenseLayer(incoming=network,
                          num_units=50,
                          nonlinearity=rectify,
                          W=GlorotNormal(gain=1))
               )
    network = DenseLayer(incoming=network,
                                        num_units=2,
                                        W=GlorotNormal(),
                                        nonlinearity=rectify)
    network = lasagne.layers.ReshapeLayer(network, (-1, 2))
    return network


def RunEpisode(env, get_action_reward, choose_action):

    expected_reward = []
    actual_reward = []
    observations = []
    obs = env.reset()
    for t in range(1000):
        # Assess options
        pred_reward = get_action_reward(obs.astype('float32').reshape(1,4))
        # Decide
        action = choose_action(obs.astype('float32').reshape(1, 4))
        expected_reward.append(pred_reward)
        # Take action and observe
        obs, reward, done, info = env.step(action)
        actual_reward.append(reward)
        observations.append(obs)

        if done:
            break

    return observations, expected_reward, actual_reward

def trainmodel(get_action_reward, choose_action, D_train, D_params):

    Rgoal = 100
    eps_per_update = 1
    Rtol = 195
    req_number = 100
    env = gym.make('CartPole-v0')
    env.reset()
    lossplot = []
    rewardplot = []
    weightplot = []
    bestreward = 0
    running_score = []
    N = 0
    needs_more_training = True
    while needs_more_training:
        N += 1
        if N % 100 == 0:
            print("Running {}th update".format(N))

        for _ in range(eps_per_update):
            observations, expected_reward, actual_reward = RunEpisode(env, get_action_reward, choose_action)
            creward = np.sum(actual_reward)
            rewardplot.append(creward)
            running_score.append(creward)
            if len(running_score)>req_number:
                running_score.pop(0)
                print(np.mean(running_score))
                if np.mean(running_score)>Rtol:
                    needs_more_training=False

            if creward / eps_per_update > bestreward:
                bestreward = creward / eps_per_update

        lossplot.append(
            D_train(np.array(observations).astype('float32'),
                    np.array(actual_reward).astype('int8')))

        weightplot.append(np.median(D_params[1].get_value()))


def runmodel(choose_action, number_of_episodes=1, monitor=False):

    env = gym.make('CartPole-v0')
    if monitor:
        env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
    env.reset()
    for i_ep in range(number_of_episodes):
        obs = env.reset()
        for t in range(1000):
            env.render()
            action = choose_action(obs.astype('float32').reshape(1, 4))
            obs, reward, done, info = env.step(action[0][0])
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


def prepare_functions():

    observations = T.matrix('observations')
#    random_var = T.vector('random')
    action_var = T.vector('actions')
    srng = RandomStreams(seed=42)
    Rgoal = T.vector('goal')
    expected_reward = T.matrix('expected')
    actual_reward = T.vector('actual')

    D_network = PolicyNetwork(observations)
    D_params = lasagne.layers.get_all_params(D_network, trainable=True)

    expected_action_rewards = lasagne.layers.get_output(D_network)
    action_reward = T.max(expected_action_rewards)

    # The expected_reward for action is the sum of all rewards subsequent to that action
    # The actual_reward for the action is the total reward of the episode
    D_obj = lasagne.objectives.squared_error(action_reward, actual_reward).mean()

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-4)
    D_train = theano.function([observations, actual_reward], D_obj, updates=D_updates, name='D_training')

    rv_u = srng.uniform(size=(1,))
#    random_sampler = theano.function([], rv_u)
    get_action_reward = theano.function([observations], action_reward)
#    D_out = T.switch(T.lt(expected_action_rewards[:,0], expected_action_rewards[:,1]), int(0), int(1))
    D_out = T.argmax(expected_action_rewards)

    choose_action = theano.function([observations], D_out, name='greedy_choice')

    return get_action_reward, choose_action, D_train, D_params, D_network


def savemodel(network, filename):
    np.savez(filename, *lasagne.layers.get_all_param_values(network))

def initmodel(network, filename):
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)


if __name__=='__main__':
    get_action_reward, choose_action, D_train, D_params, D_network = prepare_functions()
    if True:
        trainmodel(get_action_reward, choose_action, D_train, D_params)
        savemodel(D_network, 'D_network.npz')
    else:
        initmodel(D_network, 'D_network.npz')
        runmodel(choose_action)

