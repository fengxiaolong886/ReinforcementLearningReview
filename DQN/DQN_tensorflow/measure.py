import matplotlib.pyplot as plt
import numpy as np
from constants import *
import os
import pandas as pd


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_and_save(content,path,name):
    check_path(path)    
    savename=os.path.join(path,name)
    plt.plot(np.arange(len(content)),content)
    plt.ylabel("Cost")
    plt.xlabel("steps")
    plt.savefig(savename)
    plt.show()



def plot_episode_stats1(rec, xlabel, ylabel,title):
    fig = plt.figure(figsize=(20, 10), facecolor = "white")
    ax = fig.add_subplot(111)
    ax.plot(rec)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig

def plot_episode_stats2(stats):
    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(20, 10))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    return fig3

def Plot_the_result_for_rec(rec, path, name_pefix):
    check_path(path)

    # Plot episode length over time
    episode_lengths = rec.episode_lengths
    fig = plot_episode_stats1(episode_lengths,
                       xlabel = "Episode",
                       ylabel = "Episode Length",
                       title = "Episode length over Time"
            )
    fig.show()
    jpgname=name_pefix + "_EpisodeLength.jpg"
    savename = os.path.join(path,jpgname)
    fig.savefig(savename)

    # Plot Episode reward over time
    smoohing_window = 10
    reward_smooths = pd.Series(rec.episode_rewards).rolling(smoohing_window,
            min_periods = smoohing_window).mean()
    fig = plot_episode_stats1(reward_smooths,
                       xlabel = "Episode",
                       ylabel = "Episode Reward",
                       title = "Episode reward over time"
            )
    fig.show()
    jpgname=name_pefix + "_EpisodeReward.jpg"
    savename = os.path.join(path,jpgname)
    fig.savefig(savename)

    # Plot Episode per time step
    fig = plot_episode_stats2(rec)
    fig.show()
    jpgname=name_pefix + "_EpisodePerTimeStep.jpg"
    savename = os.path.join(path,jpgname)
    fig.savefig(savename)

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

