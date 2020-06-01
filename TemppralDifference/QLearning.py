# coding: UTF-8

"""
This file is the implementation for the QLearning Algorithm
It is specified to solve the CartPole issue
"""


import gym
import numpy as np
import sys
import time
import pandas as pd
from collections import defaultdict, nametuple


class QLearning():
    def __init__(self, actions, num_episodes, discount=1.0, alpha=0.5, epsilon=0.1):
        self.action = env
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.shape[0]
        self.num_episodes = num_episodes
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = pd.DataFrame()
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        # Track the statistics of the result
        self.record = nametuple("Record", ["episode_lengths","episode_rewards"])
        self.rec = record(episode_lengths=np.zeros(num_episodes),
                          episode_rewards=np.zeros(num_episodes))

    def __epsilon_greedy_policy(epsilon=0.1, nA):
        
        def policy(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(self.Q[state])
            A[best_action] += (1.0 - epsilon)
            return A
        
        return policy
    
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            # choose best action
            self.Q[stat]


    def __next_action(self, prob):
        return np.random.choice(np.arange(len(prob)), p=prob)

    def learn(self):
        policy = self.__epsilon_greedy_policy(self.epsilon, self.nA)
        sumlist = []

        for i_episode in range(self.num_episodes):
            # Print the progress
            if 0 == (i_episode +1) % 10:
                print("This the episode {}/{}".format(i_episode, self.num_episodes), end = "\r")

            step = 0

            state_ = self.env.reset()


        



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
             CartPole_Cart_velocity_idx, CartPole_angle_rate_bins]
    
    state = map(lambda s: int(s),state_)
    return tuple(state)
   
def cut_state_to_bins(n_bins):i
    """
    This function define the basic bins for the Cart Pole problem.
    """
    CartPole_Cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1]
    CartPole_Pole_angle_bins = pd.cut([-2, 2], bins=n_bins, retbins=True)[1]
    CartPole_Cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1]
    CartPole_angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins, retbins=True)[1]
    return [CartPole_Cart_position_bins, CartPole_Pole_angle_bins,
            CartPole_Cart_velocity_bins, CartPole_angle_rate_bins]


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    bins = cut_state_to_bins()

