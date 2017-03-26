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
    from lasagne.nonlinearities import rectify, sigmoid, softmax, tanh
    from lasagne.init import GlorotNormal
    network = InputLayer(shape=(None,4), input_var=input_var, name='Input')
    network = (DenseLayer(incoming=network,
                                        num_units=100,
                                        nonlinearity=tanh,
                                        W=GlorotNormal(gain=1))
                         )
    network = (DenseLayer(incoming=network,
                          num_units=100,
                          nonlinearity=tanh,
                          W=GlorotNormal(gain=1))
               )
    network = DenseLayer(incoming=network,
                                        num_units=n_actions,
                                        W=lasagne.init.GlorotNormal(),
                                        nonlinearity=rectify)
    network = lasagne.layers.ReshapeLayer(network, (-1, n_actions))
    return network


def RunEpisode(env, get_prediction, policy):

    alfa = 1
    gamma = 0.99
    expected_reward = []
    received_reward = []
    actual_reward = 0
    observations = []
    obs = env.reset()
#    Q_sdash = np.array(0,dtype='float32')
    Q_sdash = []
    for t in range(1000):
        # Assess options

        Q_s = get_prediction(obs.astype('float32').reshape(1,4))[0] # Prediction
        action = policy(obs.astype('float32').reshape(1, 4))[0]
        new_obs, reward, done, info = env.step(action)
        if not done:
            #Get the predicted future discounted rewards
            Q_sdash.append(
                    reward+gamma*get_prediction(new_obs.astype('float32').reshape(1,4))[0])
        else:
            Q_sdash.append(reward)

        obs = new_obs

        #print("{}, {}".format(Q_s, Q_sdash[-1]))
        # Book-keeping
        expected_reward.append(Q_s)
        received_reward = Q_sdash
        actual_reward += reward
        observations.append(obs)

        if done:
            break

    return observations, expected_reward, received_reward, actual_reward

def trainmodel(get_prediction, policy, D_train, D_params):

    # Initialise
    eps_per_update = 5
    Rtol = 195
    Emax = 2000
    req_number = 10
    lossplot = []
    rewardplot = []
    expect_reward = []
    weightplot = []
    running_score = []
    N = 0
    needs_more_training = True

    # Setup
    env = gym.make('CartPole-v0')
    env.reset()


    while needs_more_training:

        for _ in range(eps_per_update):
            N += 1
            if N % 100 == 0:
                print("Running {}th update".format(N))
            if N==Emax:
                needs_more_training = False
            observations, expected_reward, discounted_reward, actual_reward = RunEpisode(env, get_prediction, policy)
            # Bookkeeping
            expect_reward.append(expected_reward)
            rewardplot.append(actual_reward)
            running_score.append(actual_reward)
            # Stopping condition
            if len(running_score)>req_number:
                running_score.pop(0)
                print(np.mean(running_score))
                if np.mean(running_score)>Rtol:
                    needs_more_training=False


        observations = np.array(observations).astype('float32')
        discounted_reward = np.expand_dims(np.array(discounted_reward).astype('float32'), axis=1)

        lossplot.append(D_train(observations, discounted_reward))
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
    discounted_reward = T.matrix('actual')

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
        eps = 1e-4
        X_m = T.mean(X, keepdims=True, axis=0)
        X_var = T.var(X, keepdims=True, axis=0)
        X = (X - X_m) / (T.sqrt(X_var+eps))
        return X



    D_obj = lasagne.objectives.squared_error(prediction,
                                             discounted_reward
                                             )\
            .mean()

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-4)
    D_train = theano.function([observations, discounted_reward], D_obj, updates=D_updates, name='D_training')

    policy_action = theano.function([observations], policy(q_values), name='greedy_choice')
    return get_prediction, policy_action, D_train, D_params, D_network


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

    initial_expected_rewards = map(lambda x: x[0], expected_reward)
    plt.plot(initial_expected_rewards)
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

