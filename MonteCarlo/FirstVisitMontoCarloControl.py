# coding: utf-8

import numpy as np
import gym
import sys
from collections import defaultdict
from Plot3D import plot_3D

def mc_firstvisit_control_epsilon_greedy(env, num_episodes,epsilon=0.1,
                             episode_endtime= 10, discount = 1.0):
    """
    This function implement the On-policy first policy Monto Carlo Control 
    algorithm.

    The algorithm is shown as below:
    
    -------------------------------------------------
    Alogrithm parameter : small epsilon >0
    Initialize:
        pi <-- an arbitary epsilon -greedy policy   
        Q(s,a ) (arbitary) for all s and a
        Return(s,a) <-- empty list , for all s and a

    Repeat forever(for each episode):
        Generate all episode follwing pi: S_0, A_0, R_0, S_1, A_1, R_1.... R_T
        G <--- 0
        Loop for each step of episode, t = T-1, T-2, ....,0
            G  <--- $\gamma * G + R_{t+1}$
            Unless S_t, A_t  appear in S_0,A_0,S_1,A_1.....:
                Append G to Return(S_t,A_t)
                $A^* \leftarrow argmax_{a} Q(S_t, A_t)$
                For all a:
                    $$
                    \pi(a|s) \leftarrow
                    \begin{cases}
                    1-\epsilon + \frac{\epsilon}{A(S_t)}  \ \ \  if  \  a = A^*
                    \\
                    \frac{\epsilon}{|A(S_t)|}
                    \end{cases}
                    $$
    -------------------------------------------------
    """
    
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    r_sum = defaultdict(float)
    r_count = defaultdict(float)
    
    #define policy
    policy = epsilon_greedy_policy(Q, epsilon, nA)

    for each_episode in range(num_episodes):
        # Show the progress
        print("Episode {}/{}".format(each_episode,num_episodes),end = "\r")
        sys.stdout.flush()
        
        # Initial the episode list
        episode = []
        state = env.reset()

        # Generate an episode according the policy.
        for _ in range(episode_endtime):
            action_prob = policy(state)
            action = np.random.choice(np.arange(action_prob.shape[0]), p=action_prob)
            
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state


        # Calculate the first visit MC value
        for visit_pos, (state, action, reward) in enumerate(episode):
            state_action_visit = (state, action)

            G = sum([x[2] * np.power(discount, i) for i, x in enumerate(episode[visit_pos:])])
            
            # Calculate the averate return 
            r_sum[state_action_visit] += G
            r_count[state_action_visit] += 1.0
            Q[state][action] = r_sum[state_action_visit] / r_count[state_action_visit]
    
    return Q

def simple_policy(state):
    """
    Define a simple policy which only the player's score greater
    than 18 to take hold, otherwise take action hit.

    """
    player_score, _, _ = state
    return 0 if player_score >= 18 else 1

def epsilon_greedy_policy(q, epsilon, nA):
    '''
    Return a function which is epsiolon greedy function
    '''
    def policy_(state):
        A_ = np.ones(nA, dtype = float)
        A = A_ * epsilon / nA
        best = np.argmax(q[state])
        A[best] += 1 - epsilon
        return A

    return policy_


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

    Q = mc_firstvisit_control_epsilon_greedy(env, num_episodes=1000000)
    
    # calcuate the value depend on the Q.
    v1= defaultdict(float)
    for state, action in Q.items():
        v1[state] = np.max(action)
    
    X, Y, Z = process_data_for_Blackjackproblem(v1, ace=True)
    fig = plot_3D(X, Y, Z, xlabel="Player sum", ylabel="Dealer sum", zlabel="Value", title="Usable Ace")
    fig.savefig("./log/First_MC_Control_Usable_Ace_1M.jpg")
    X, Y, Z = process_data_for_Blackjackproblem(v1, ace= False)
    fig = plot_3D(X, Y, Z, xlabel="Player sum", ylabel="Dealer sum", zlabel="Value", title="No Usable Ace")
    fig.savefig("./log/First_MC_Control_No_Usable_Ace_1M.jpg")
