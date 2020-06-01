# coding: UTF-8

"""
This file is the implementation for the QLearning Algorith
We use approximate function and RBF is used here.
It is specified to solve the CartPole issue
"""


import gym
import numpy as np
import sys
import time
import pandas as pd
from collections import namedtuple,defaultdict

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler as Scaler 
from sklearn.linear_model import SGDRegressor as SGD 
from sklearn.kernel_approximation import RBFSampler as RBF 

from Plot2DResult import *


class QLearning():
    def __init__(self, actions, discount=1.0, alpha=0.5, epsilon=0.1):
        self.actions = actions
        self.nA = len(actions)
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        
        # data normalization
        self.scaler = Scaler()
        self.scaler.fit(state_example)
    
    # Create the RBF estimator
    def Estimator(self, state_example, state):
        # feature extractor
        self.featurizer = FeatureUnion([
            ("rbf1",RBF(gamma=5.0, n_components=100)),
            ("rbf2",RBF(gamma=1.0, n_components=100)),
            ])
        self.featurizer.fit(self.scaler.transform(state_example))

        # action model
        self.action_model = []
        for actionnumber in range(self.nA):
            model = SGD(learning_rate = "constant")
            model.partial_fit = ([self.__featurize_state(state)],[0])
            self.action_model.append(model)

    def __featurize_state(self, state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform()

        


    def choose_action(self, observation):
        self.check_state_exist(observation)
        # select action
        if np.random.uniform() < self.epsilon:
            # choose the best action
            state_action = self.Q_table.loc[observation,:]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose a random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_,done):
        self.check_state_exist(s_)
        q_predict = self.Q_table.loc[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.discount * self.Q_table.loc[s_,:].max()
        # update
        self.Q_table.loc[s, a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.Q_table.index:
            # append the new state into the table
            self.Q_table = self.Q_table.append(
                    pd.Series(
                        [0] * len(self.actions),
                        index = self.Q_table.columns,
                        name = state,
                        )
                    )
        





def get_bins_states(state,bins):
    """
    This function provie the method to digitize the state space into the discrete
    space. It will be used to store in the Q table.
    """
    s1_, s2_, s3_, s4_ = state
    CartPole_Cart_position_idx = np.digitize(s1_, bins[0])
    CartPole_Pole_angle_idx = np.digitize(s2_, bins[1])
    CartPole_Cart_velocity_idx = np.digitize(s3_, bins[2])
    CartPole_angle_rate_idx = np.digitize(s4_, bins[3])

    state_ =[CartPole_Cart_position_idx, CartPole_Pole_angle_idx,
             CartPole_Cart_velocity_idx, CartPole_angle_rate_idx]
    
    state = map(lambda s: int(s),state_)
    return tuple(state)
   
def cut_state_to_bins(n_bins=10):
    """
    This function define the basic bins for the Cart Pole problem.
    """
    CartPole_Cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1]
    CartPole_Pole_angle_bins = pd.cut([-2, 2], bins=n_bins, retbins=True)[1]
    CartPole_Cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1]
    CartPole_angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins, retbins=True)[1]
    return [CartPole_Cart_position_bins, CartPole_Pole_angle_bins,
            CartPole_Cart_velocity_bins, CartPole_angle_rate_bins]

def convert_state(state, n_bins=10):
    bins = cut_state_to_bins(n_bins)
    converd_state = get_bins_states(state, bins)
    return str(converd_state)

def update(RL, env, num_episodes):
    # Track the statistics of the result
    record = namedtuple("Record", ["episode_lengths","episode_rewards"])
    
    rec = record(episode_lengths=np.zeros(num_episodes),
                          episode_rewards=np.zeros(num_episodes))
    
    for i_episode in range(num_episodes):
        if 0 == (i_episode +1) % 2:
            print("This the episode {}/{}".format(i_episode, num_episodes), end = "\r")
        observation_ = env.reset()
        step =0
        while True:
            #env.render()
            observation = convert_state(observation_)
            action = RL.choose_action(observation)
            observation_next_, reward, done, info = env.step(action)
            observation_next = convert_state(observation_next_)
            RL.learn(observation, action, reward, observation_next, done)
            observation = observation_next
            
            # update the record
            step += 1
            rec.episode_lengths[i_episode] = step 
            rec.episode_rewards[i_episode] += reward

            if done:
                break

    print("Finished")
    env.close()
    return rec


if __name__ == "__main__":
    num_episodes = 2000
    env = gym.make("CartPole-v0")
    actions = [i for i in range(env.action_space.n)] 
    RL = QLearning(actions, discount=1.0, alpha=0.5, epsilon=0.1)
    rec = update(RL, env, num_episodes=num_episodes)
    episode_lengths = rec.episode_lengths
    fig = plot_episode_stats(episode_lengths, 
                       xlabel = "Episode",
                       ylabel = "Episode Length",
                       title = "Episode length over Time"
            )
    fig.savefig("./log/QLearning_CartPole_EpisodeLength.jpg")
    
    
    smoohing_window = 10

    reward_smooths = pd.Series(rec.episode_rewards).rolling(smoohing_window,\
                                                           min_periods = smoohing_window).mean()
    fig = plot_episode_stats(reward_smooths, 
                       xlabel = "Episode",
                       ylabel = "Episode Reward",
                       title = "Episode reward over time"
            )
    fig.savefig("./log/QLearning_CartPole_EpisodeReward.jpg")




