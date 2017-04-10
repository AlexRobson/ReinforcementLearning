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


def normalise(X):


    eps = 1e-4
    X_m = T.mean(X, keepdims=True, axis=0)
    X_var = T.var(X, keepdims=True, axis=0)
    X = (X - X_m) / (T.sqrt(X_var+eps))
    return X


def QNetwork(input_var):

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
                                        num_units=256,
                                        nonlinearity=tanh,
                                        W=GlorotNormal(gain=1))
                         )
    network = (DenseLayer(incoming=network,
                          num_units=64,
                          nonlinearity=tanh,
                          W=lasagne.init.HeUniform())
               )
    network = DenseLayer(incoming=network,
                                        num_units=n_actions,
                                        W=lasagne.init.HeUniform(),
                                        b=lasagne.init.Constant(1),
                                        nonlinearity=None)
    network = lasagne.layers.ReshapeLayer(network, (-1, n_actions))
    return network


def RunEpisode(env, policy, eps):
    """
    This function does a run of an episode in the Open AI gym environment
    :param env: The Open AI gym environment
    :param get_prediction: A theano function that returns the value of the best Q
    :param policy: A theano function that selects the next action, based upon the given state
    :param eps: The exploration rate.
    :return: memory: This is a list of tuples encoding
     (state, action, new_states, rewards, termination) for the episode
    """

    obs = env.reset()
    memory = []
    for t in range(1000):
        action = policy(obs.astype('float32').reshape(1, 4))[0]
        if np.random.rand()<eps:
            action = np.random.random_integers(0,1, ())

        new_obs, reward, done, info = env.step(action)
        memory.append((obs, action, new_obs, reward, done))
        obs = new_obs
        if done:
            break

    return memory

def bookkeeping(episode_memory, reward_per_episode):
    """
    Helper function that updates various lists and records of parameters, states, etc.
    """

    states, actions, new_states, rewards, done = zip(*episode_memory)
    reward_per_episode.append(np.sum(rewards))



def trainmodel(get_output, policy, D_train, D_params):

    # Initialise
    eps_per_update = 2
    Rtol = 195
    Emax = 2000
    req_number = 10
    lossplot = []
    rewards_per_episode = []
    long_term_memory = []
    running_score = []
    N = 0
    needs_more_training = True

    # Setup
    env = gym.make('CartPole-v0')
    env.reset()

    eps=1
    while needs_more_training:
        for _ in range(eps_per_update):
            N += 1
            if N % 100 == 0:
                print("Running {}th update".format(N))
            if N==Emax:
                needs_more_training = False

            if eps>0.05:
                eps *= 0.99
            episode_memory = RunEpisode(env, policy, eps)
            long_term_memory.extend(episode_memory)

            # Bookkeeping and debugging
            bookkeeping(episode_memory, rewards_per_episode)
            states, actions, new_states, rewards, dones = zip(*episode_memory)
            #print(actions)

            # Stopping condition
            running_score.append(rewards_per_episode[-1])
            if len(running_score)>req_number:
                running_score.pop(0)
                print(np.mean(running_score))
                if np.mean(running_score)>Rtol:
                    needs_more_training=False

        # Updating
        lossplot.append(reflect(long_term_memory, get_output, policy, D_train))

    return lossplot, rewards_per_episode


def reflect(memory, get_q_values, policy, D_train):

    """
    This function handles the update steps. It ingests memory (a tuple of game state-actions
    transitions and then calculates two quantities, as part of the Bellman equation:
    * predictions: Predictions of the Q-values at a given input step.
    * target: The reward in the transition plus the discounted (predictions of) Q-values at the subsequent states,
    :param memory: A list of tuples: (state, action, new_state, reward, done)
    :param get_output: A theano function that does a forward-pass. Takes in states
    :param policy: A theano function that executes the policy at a given state. Takes in states
    :param get_prediction: A theano function that returns the best predicted Q-value at a given state
    :param D_train: The update function, that takes in the predictions and targets
    :return: Returns the loss that is backpropagated (via D_train)
    """

    gamma = 0.90
    batch_size = 400
    N = np.min((batch_size, len(memory)))
    batch_IDX = np.random.choice(np.arange(N), size=(N,))
    recall = np.array(memory)[batch_IDX]
    states, actions, new_states, rewards, done = zip(*recall)
    # Prediction in original states

    # These are useful helper functions for use in diagnosing
    Q_ = get_q_values(np.array(states).astype('float32'))
    Q_dash = get_q_values(np.array(new_states).astype('float32'))
    predictions = Q_[:, actions]
    target = np.array(rewards,dtype='float32')\
             +gamma*np.max(Q_dash,axis=1)
    return D_train(np.array(states,dtype='float32'), target)


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

    from lasagne.regularization import regularize_layer_params_weighted, regularize_layer_params
    from lasagne.regularization import l1, l2

    """
    This prepares the theano/lasagne functions for use in the training functions
    """
    observations = T.matrix('observations')
    srng = RandomStreams(seed=42)
    predictions = T.vector('predictions')
    predictions_ct = theano.gradient.disconnected_grad_(predictions)
    discounted_reward = T.vector('actual')
    r = T.vector('random')

    # Set up random sampling used in some policies
    rv_u = srng.uniform(size=(1,))
    r = theano.function([], rv_u)


    # Set up the network
    D_network = QNetwork(observations)
    q_values = lasagne.layers.get_output(D_network)
    probabilities = lasagne.nonlinearities.softmax(q_values)
    D_params = lasagne.layers.get_all_params(D_network, trainable=True)
    get_q_values = theano.function([observations], q_values)

    l1_penalty = regularize_layer_params(lasagne.layers.get_all_layers(D_network), l1)

    # Policies:
    # Policy1: 'greedy_choice': Greedy
    # Policy2: ' weighted_choice': chooses actions based upon probabilities
    policyname='weighted'
#    policyname='greedy'
    if policyname=='greedy':
        actions = T.argmax(q_values, axis=1)
    elif policyname=='weighted':
        actions = T.argmax(T.abs_(T.extra_ops.cumsum(probabilities,axis=1)-r()), axis=1)
    else:
        raise Exception

    policy_action = theano.function([observations], actions,  name=policyname)
    prediction = q_values[:, actions]
    get_prediction = theano.function([observations], prediction)

    D_obj = lasagne.objectives.squared_error(prediction,
                                             discounted_reward
                                             )\
            .mean() + l1_penalty

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-6)
    D_train = theano.function([observations, discounted_reward], D_obj, updates=D_updates, name='D_training')

    return get_q_values, policy_action, D_train, D_params, D_network


def savemodel(network, filename):
    np.savez(filename, *lasagne.layers.get_all_param_values(network))


def initmodel(network, filename):
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)


def showplots(lossplot, rewardplot):
    plt.plot(lossplot)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(rewardplot)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()


if __name__=='__main__':
    get_output, policy, D_train, D_params, D_network = prepare_functions()
    if True:
        lossplot, rewards_per_episode = trainmodel(get_output, policy, D_train, D_params)
        showplots(lossplot, rewards_per_episode)
        savemodel(D_network, 'D_network.npz')
    else:
        initmodel(D_network, 'D_network.npz')
        runmodel(policy)

