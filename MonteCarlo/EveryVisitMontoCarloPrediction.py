# coding: utf-8

import numpy as np
import gym
import sys
from collections import defaultdict
from Plot3D import plot_3D

def mc_everyvisit_prediction(policy, env, num_episodes,
                             episode_endtime= 10, discount = 1.0):
    """
    This is the implementation for Monto Carlo First visit prediction
    algorithm.

    The algorithm is shown as below:
    
    Input: a policy pi to be evaluated.
    Initialize:
        V(s). arbitrarily for all s.
        Returns(s) <--- all empty list, for all s.

    Loop forever (for each episode):
        Generate all episode follwing pi: S_0, A_0, R_0, S_1, A_1, R_1.... R_T
        G <--- 0
        Loop for each step of episode, t = T-1, T-2, ....,0
        G  <---$\gamma * G + R_{t+1}$
        For every  S_t appear in S_0, S_1, ... S_{t-1}
            Append G to Return(S_t)
            V(S_t) <-- average(Returns(S_t))
    """
    r_sum = defaultdict(float)
    r_count = defaultdict(float)
    r_V = defaultdict(float)

    for each_episode in range(num_episodes):
        # Show the progress
        print("Episode {}/{}".format(each_episode,num_episodes),end = "\r")
        sys.stdout.flush()
        
        # Initial the episode list
        episode = []
        state = env.reset()

        # Generate an episode according the policy.
        for _ in range(episode_endtime):
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state


        # Calculate the first visit MC value
        for visit_pos, data in enumerate(episode):
            state_visit = data[0]
            G = sum([x[2] * np.power(discount, i) for i, x in enumerate(episode)])
            
            # Calculate the averate return 
            r_sum[state_visit] += G
            r_count[state_visit] += 1.0
            r_V[state_visit] = r_sum[state_visit] / r_count[state_visit]

    return r_V

def simple_policy(state):
    """
    Define a simple policy which only the player's score greater
    than 18 to take hold, otherwise take action hit.

    """
    player_score, _, _ = state
    return 0 if player_score >= 18 else 1

def process_data_for_Blackjackproblem(V,ace=True):
    """
    process data for the Blackjack problem and prepare it to plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())
    
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)
    
    if ace:
        Z = np.apply_along_axis(lambda _ : V[(_[0], _[1], True)], 2, np.dstack([X,Y]))
    else:
        Z = np.apply_along_axis(lambda _ : V[(_[0], _[1], False)], 2, np.dstack([X,Y]))

    return X, Y, Z

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    v1= mc_everyvisit_prediction(simple_policy, env, num_episodes=1000000)
    print(v1)
    X, Y, Z = process_data_for_Blackjackproblem(v1, ace=True)
    fig = plot_3D(X, Y, Z, xlabel="Player sum", ylabel="Dealer sum", zlabel="Value", title="Usable Ace")
    fig.savefig("./log/EveryVisit_Usable_Ace_1M.jpg")
    X, Y, Z = process_data_for_Blackjackproblem(v1, ace= False)
    fig = plot_3D(X, Y, Z, xlabel="Player sum", ylabel="Dealer sum", zlabel="Value", title="No Usable Ace")
    fig.savefig("./log/EveryVisit_No_Usable_Ace_1M.jpg")
