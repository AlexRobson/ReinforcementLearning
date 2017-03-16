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
                                        num_units=200,
                                        nonlinearity=rectify,
                                        W=GlorotNormal(gain=1))
                         )
    network = DenseLayer(incoming=network,
                                        num_units=1,
                                        W=GlorotNormal(),
                                        nonlinearity=sigmoid)
    return network


def RunEpisode(env, choose_action, creward, memory):

    obs = env.reset()
    for t in range(1000):
        memory['obs'].append(obs)
        action = choose_action(obs.astype('float32').reshape(1, 4), random_sampler())
        memory['act'].append(action[0][0])
        obs, reward, done, info = env.step(action[0][0])
        creward += reward
        if done:
            break

    return creward, memory

def trainmodel(choose_action, random_sampler, D_train, D_params):

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
	rescale = 1
	while needs_more_training:
		N += 1
		if N % 100 == 0:
			print("Running {}th update".format(N))

		memory = {}
		memory['obs'] = []
		memory['act'] = []
		creward = 0.

		for _ in range(eps_per_update):
			creward, memory = RunEpisode(env, choose_action, creward, memory)
			rewardplot.append(creward)
			running_score.append(creward)
			if len(running_score)>req_number:
				running_score.pop(0)
				print(np.mean(running_score))
				if np.mean(running_score)>Rtol:
					needs_more_training=False

			if creward / eps_per_update > bestreward:
				bestreward = creward / eps_per_update
				rescale = 1
			else:
				scale = 1

		rescale = 1-(creward / eps_per_update) / 200.
		lossplot.append(
			D_train(np.array(memory['obs']).astype('float32'),
			        np.array(memory['act']).astype('int8'),
					np.float32(rescale)))

		weightplot.append(np.median(D_params[1].get_value()))


def runmodel(choose_action, random_sampler, number_of_episodes=1, monitor=False):

    env = gym.make('CartPole-v0')
    if monitor:
        env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
    env.reset()
    for i_ep in range(number_of_episodes):
		obs = env.reset()
		for t in range(1000):
			env.render()
			action = choose_action(obs.astype('float32').reshape(1, 4), random_sampler())
			obs, reward, done, info = env.step(action[0][0])
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break


def prepare_functions():

    observations = T.matrix('observations')
    reward_var = T.vector('reward')
    random_var = T.vector('random')
    action_var = T.vector('actions')
    srng = RandomStreams(seed=42)
    Rgoal = T.vector('goal')
    objective_scale = T.scalar('scale')

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

    D_obj = -lasagne.objectives.binary_crossentropy(P_act, action_var).mean()*objective_scale

    D_updates = lasagne.updates.adam(D_obj, D_params,learning_rate=2e-4)
    D_train = theano.function([observations, action_var, objective_scale], D_obj, updates=D_updates, name='D_training')

    rv_u = srng.uniform(size=(1,))
    random_sampler = theano.function([], rv_u)
    D_out = T.switch(T.lt(lasagne.layers.get_output(D_network), random_var), int(0) ,int(1))

    choose_action = theano.function([observations, random_var], D_out, name='weighted_choice')

    return choose_action, random_sampler, D_train, D_params, D_network


def savemodel(network, filename):
    np.savez(filename, *lasagne.layers.get_all_param_values(network))

def initmodel(network, filename):
	with np.load(filename) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	lasagne.layers.set_all_param_values(network, param_values)


if __name__=='__main__':
    choose_action, random_sampler, D_train, D_params, D_network = prepare_functions()
    if True:
        trainmodel(choose_action, random_sampler, D_train, D_params)
        savemodel(D_network, 'D_network.npz')
    else:
        initmodel(D_network, 'D_network.npz')
        runmodel(choose_action, random_sampler)

