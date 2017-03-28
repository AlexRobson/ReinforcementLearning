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


def RunEpisode(env, get_prediction, policy, eps):

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

    states, actions, new_states, rewards, done = zip(*episode_memory)
    # Bookkeeping
    reward_per_episode.append(np.sum(rewards))



def trainmodel(get_output, get_prediction, policy, D_train, D_params):

    # Initialise
    eps_per_update = 1
    Rtol = 195
    Emax = 2000
    req_number = 10
    lossplot = []
    rewards_per_episode = []
    expect_reward = []
    weightplot = []
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
            episode_memory = RunEpisode(env, get_output, policy, eps)
            long_term_memory.extend(episode_memory)
            bookkeeping(episode_memory, rewards_per_episode)
            states, actions, new_states, rewards, dones = zip(*episode_memory)
            print(actions)
            # Stopping condition
            running_score.append(rewards_per_episode[-1])
            if len(running_score)>req_number:
                running_score.pop(0)
                print(np.mean(running_score))
                if np.mean(running_score)>Rtol:
                    needs_more_training=False

        # Updating
        lossplot.append(reflect(long_term_memory, get_output, policy, get_prediction, D_train))

    return lossplot, rewards_per_episode


def reflect(memory, get_output, policy, get_prediction, D_train):

    gamma = 0.90
    batch_size = 400
    N = np.min((batch_size, len(memory)))
    batch_IDX = np.random.choice(np.arange(N), size=(N,))
    recall = np.array(memory)[batch_IDX]
    states, actions, new_states, rewards, done = zip(*recall)
    # Prediction in original states
    # #prediction = get_prediction(states)
    #  Discounted prediction in new states

    Q_values = get_output(np.array(states).astype('float32'))
    choices = policy(np.array(states, dtype='float32'))
    predictions = get_prediction(np.array(states).astype('float32'))
    target = np.array(rewards,dtype='float32')\
             +gamma*get_prediction(np.array(new_states,dtype='float32'))
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

    observations = T.matrix('observations')
    srng = RandomStreams(seed=42)
    expected_reward = T.vector('expected')
    discounted_reward = T.vector('actual')

    D_network = ValueNetwork(observations)
    D_params = lasagne.layers.get_all_params(D_network, trainable=True)

    q_values = lasagne.layers.get_output(D_network)


    rv_u = srng.uniform(size=(1,))
    random_sampler = theano.function([], rv_u)



    policy = partial(T.argmax, axis=1)

    get_output = theano.function([observations], q_values)
    prediction = T.max(q_values, axis=1)
    get_q_values = theano.function([observations], q_values)
    get_prediction = theano.function([observations], prediction)

    # The expected_reward for action is the sum of all rewards subsequent to that action
    # The actual_reward for the action is the total reward of the episode
    def normalise(X):
        aps = 1e-4
        X_m = T.mean(X, keepdims=True, axis=0)
        X_var = T.var(X, keepdims=True, axis=0)
        X = (X - X_m) / (T.sqrt(X_var+eps))
        return X

    D_obj = lasagne.objectives.squared_error(prediction,
                                             discounted_reward
                                             )\
            .mean()

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-5)
#    D_updates = lasagne.updates.rmsprop(D_obj, D_params, learning_rate=2e-4)
    D_train = theano.function([observations, discounted_reward], D_obj, updates=D_updates, name='D_training')

    policy_action = theano.function([observations], T.argmax(q_values, axis=1),  name='greedy_choice')
    return get_output, get_prediction, policy_action, D_train, D_params, D_network


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

#    initial_expected_rewards = map(lambda x: x[0], expected_reward)
#    plt.plot(initial_expected_rewards)
#    plt.xlabel('Episode')
#    plt.ylabel('Expected Reward')
#    plt.show()

if __name__=='__main__':
    get_output, get_q_values, policy, D_train, D_params, D_network = prepare_functions()
    if True:
        lossplot, rewards_per_episode = trainmodel(get_output, get_q_values, policy, D_train, D_params)
        showplots(lossplot, rewards_per_episode)
        savemodel(D_network, 'D_network.npz')
    else:
        initmodel(D_network, 'D_network.npz')
        runmodel(policy)

