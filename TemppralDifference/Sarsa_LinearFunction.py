# coding: UTF-8

"""
This file is the implementation for the Sarsa Algorithm
It will use RBF as the feature and the linear model as the approximation function
It is specified to solve the CartPole issue
"""

import time
import gym
import sys
import matplotlib
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.kernel_approximation import RBFSampler as RBF

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Plot2DResult import *

class Estimator():
    """
    Value Function approximator. 
    """
    def __init__(self, env):        
        self.env = env 
        
        # sampleing envrionment state in order to featurize it. 
        observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])
        
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        self.scaler = Scaler()
        self.scaler.fit(observation_examples)
                
        # Used to convert a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = FeatureUnion([
                ("rbf1", RBF(gamma=5.0, n_components=100)),
                ("rbf2", RBF(gamma=2.0, n_components=100)),
                ("rbf3", RBF(gamma=1.0, n_components=100)),
                ("rbf4", RBF(gamma=0.5, n_components=100))
                ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        # action model for SGD regressor
        self.action_models = []
        self.nA = self.env.action_space.n
        
        for na in range(self.nA):
            model = SGD(learning_rate="constant")
            model.partial_fit([self.__featurize_state(self.env.reset())], [0])
            self.action_models.append(model)
        
    def __featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.__featurize_state(s)
        if not a:
            return np.array([model.predict([features])[0] for model in self.action_models])
        else:
            return self.action_models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        cur_features = self.__featurize_state(s)
        self.action_models[a].partial_fit([cur_features], [y])


class Sarsa():
    def __init__(self, env, estimator, actions, discount=1.0, alpha=0.5, epsilon=0.1):
        self.actions = actions
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env
        self.estimator = estimator 
    
    def choose_action(self, observation):
        # select action
        if np.random.uniform() > self.epsilon:
            # choose the best action
            state_action = self.estimator.predict(observation)
            action = np.argmax(state_action)
        else:
            # choose a random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_,a_):
        q_predict = self.estimator.predict(s_)[a_]
        q_target = r + self.discount * q_predict
       
        # update
        self.estimator.update(s, a , q_target)


def update(RL, env, num_episodes):
    # Track the statistics of the result
    record = namedtuple("Record", ["episode_lengths","episode_rewards"])
    
    rec = record(episode_lengths=np.zeros(num_episodes),
                          episode_rewards=np.zeros(num_episodes))
    
    for i_episode in range(num_episodes):
        if 0 == (i_episode +1) % 2:
            print("This the episode {}/{}".format(i_episode, num_episodes), end = "\r")
        observation = env.reset()
        step =0
        while True:
            #env.render()
            action = RL.choose_action(observation)
            observation_next, reward, done, info = env.step(action)
            next_action = RL.choose_action(observation_next)

            RL.learn(observation, action, reward, observation_next, next_action)
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
    start_time = time.time()
    num_episodes = 500
    env = gym.make("MountainCar-v0")
    actions = [i for i in range(env.action_space.n)] 
    estimator = Estimator(env)
    
    RL = Sarsa(env, estimator, actions, discount=1.0, alpha=0.5, epsilon=0.1)
    rec = update(RL, env, num_episodes=num_episodes)
    
    # Plot episode length over time
    episode_lengths = rec.episode_lengths
    fig = plot_episode_stats(episode_lengths, 
                       xlabel = "Episode",
                       ylabel = "Episode Length",
                       title = "Episode length over Time"
            )
    fig.savefig("./log/FA_Sarsa_MountainCar_EpisodeLength.jpg")
    
    
    # Plot Episode reward over time
    smoohing_window = 10

    reward_smooths = pd.Series(rec.episode_rewards).rolling(smoohing_window,\
                                                           min_periods = smoohing_window).mean()
    fig = plot_episode_stats(reward_smooths, 
                       xlabel = "Episode",
                       ylabel = "Episode Reward",
                       title = "Episode reward over time"
            )
    fig.savefig("./log/FA_Sarsa_Mountain_EpisodeReward.jpg")
 
    # Plot Episode per time step
    def plot_episode_stats(stats):

        # Plot time steps and episode number
        fig3 = plt.figure(figsize=(15,7.5))
        plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
        plt.xlabel("Time Steps")
        plt.ylabel("Episode")
        plt.title("Episode per time step")
        return fig3
    fig = plot_episode_stats(rec)
    fig.savefig("./log/FA_Sarsa_Mountain_EpisodePerTimeStep.jpg")

    def plot_cost_to_go_mountain_car(env, estimator, niter, num_tiles=20):
        x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
        y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
        X, Y = np.meshgrid(x, y)
        Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
        fig = plt.figure(figsize=(15,7.5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=0, vmax=160)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Value')
        ax.set_zlim(0, 160)
        ax.set_facecolor("white")
        ax.set_title("Cost To Go Function (iter:{})".format(niter))
        fig.colorbar(surf)
        return fig
    fig = plot_cost_to_go_mountain_car(env, estimator, num_episodes)
    fig.savefig("./log/FA_Sarsa_Mountain_CostMap.jpg")


    end_time= time.time()
    print("This alogrithm cost time is :",end_time-start_time)
