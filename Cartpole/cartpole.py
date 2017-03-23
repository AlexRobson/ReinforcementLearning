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
from functools import partial

def ValueNetwork(input_var):

    """
    This sets up a network in Lasagne that decides on what move to play
    """
    n_actions = 2

    from lasagne.layers import batch_norm
    from lasagne.layers import DenseLayer
    from lasagne.layers import InputLayer
    from lasagne.nonlinearities import rectify, sigmoid, softmax
    from lasagne.init import GlorotNormal
    network = InputLayer(shape=(None,4), input_var=input_var, name='Input')
    network = (DenseLayer(incoming=network,
                                        num_units=200,
                                        nonlinearity=rectify,
                                        W=GlorotNormal(gain=1))
                         )
    network = (DenseLayer(incoming=network,
                          num_units=200,
                          nonlinearity=rectify,
                          W=GlorotNormal(gain=1))
               )
    network = DenseLayer(incoming=network,
                                        num_units=n_actions,
                                        W=GlorotNormal(),
                                        nonlinearity=rectify)
    network = lasagne.layers.ReshapeLayer(network, (-1, n_actions))
    return network


def RunEpisode(env, get_Q_values, policy):

    alfa = 1
    gamma = 0.99
    expected_reward = []
    actual_reward = []
    observations = []
    obs = env.reset()
    for t in range(1000):
        # Assess options

        Q_s = np.max(get_Q_values(obs.astype('float32').reshape(1,4)), axis=1)[0] # Prediction
        action = policy(obs.astype('float32').reshape(1, 4))[0]
        new_obs, reward, done, info = env.step(action)
        if not done:
            Q_sdash = reward+np.max(get_Q_values(obs.astype('float32').reshape(1,4)), axis=1)[0]
        else:
            Q_sdash = reward

        obs = new_obs

        # Book-keeping
        expected_reward.append(Q_s)
        actual_reward.append(Q_sdash)
        observations.append(obs)

        if done:
            break

    return observations, expected_reward, actual_reward

def trainmodel(get_Q_values, policy, D_train, D_params):

    Rgoal = 100
    eps_per_update = 1
    Rtol = 195
    req_number = 100
    env = gym.make('CartPole-v0')
    env.reset()
    lossplot = []
    rewardplot = []
    expect_reward = []
    weightplot = []
    bestreward = 0
    running_score = []
    N = 0
    needs_more_training = True
    while needs_more_training:
        N += 1
        if N % 100 == 0:
            print("Running {}th update".format(N))
            if N==2000:
                needs_more_training = False

        for _ in range(eps_per_update):
            # Execution
            observations, expected_reward, actual_reward = RunEpisode(env, get_q_values, policy)

            # Bookkeeping
            creward = np.sum(actual_reward)

            expect_reward.append(expected_reward)
            rewardplot.append(creward)

            running_score.append(creward)
            # Stopping condition
            if len(running_score)>req_number:
                running_score.pop(0)
                print(np.mean(running_score))
                if np.mean(running_score)>Rtol:
                    needs_more_training=False




        observations = np.array(observations).astype('float32')
        actual_reward = np.expand_dims(np.array(actual_reward).astype('float32'), axis=1)

        lossplot.append(D_train(observations, actual_reward))


        weightplot.append(np.median(D_params[1].get_value()))

    return lossplot, rewardplot, expect_reward, weightplot


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
    actual_reward = T.matrix('actual')

    D_network = ValueNetwork(observations)
    D_params = lasagne.layers.get_all_params(D_network, trainable=True)

    q_values = lasagne.layers.get_output(D_network)
    policy = partial(T.argmax, axis=1)

    prediction = T.max(q_values, axis=1, keepdims=True)
    get_q_values = theano.function([observations], q_values)
    get_prediction = theano.function([observations], prediction)

    # The expected_reward for action is the sum of all rewards subsequent to that action
    # The actual_reward for the action is the total reward of the episode
    def normalise(X):
        # X - T.mean(X,keepdims=True,axis=0)) / T.sum(X, keepdims=True,axis=0)
        return X


    D_obj = lasagne.objectives.squared_error(normalise(prediction),
                                             normalise(actual_reward)
                                             )\
            .mean()

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-4)
    D_train = theano.function([observations, actual_reward], D_obj, updates=D_updates, name='D_training')

    policy_action = theano.function([observations], policy(q_values), name='greedy_choice')
    return get_q_values, policy_action, D_train, D_params, D_network


def savemodel(network, filename):
    np.savez(filename, *lasagne.layers.get_all_param_values(network))


def initmodel(network, filename):
    with np.load(filename) as f:
        cumulative_reward = np.cumsum(actual_reward)
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)


def showplots(lossplot, rewardplot, expected_reward, weightplot):
    plt.plot(lossplot)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(rewardplot)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()

    plt.plot(np.sum(expected_reward))
    plt.xlabel('Episode')
    plt.ylabel('Expected Reward')
    plt.show()

if __name__=='__main__':
    get_q_values, policy, D_train, D_params, D_network = prepare_functions()
    if True:
        lossplot, rewardplot, expected_reward, weightplot = trainmodel(get_q_values, policy, D_train, D_params)
        showplots(lossplot, rewardplot, expected_reward, weightplot)
        savemodel(D_network, 'D_network.npz')
    else:
        initmodel(D_network, 'D_network.npz')
        runmodel(choose_action)

