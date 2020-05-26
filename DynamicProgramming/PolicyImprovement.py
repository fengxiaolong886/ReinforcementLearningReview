import numpy as np
import gym
from PolicyEvaluation import *

def policy_iteration(env,policy,discount_factor=1.0,endstate=15):
    '''
    Policy Iteration alogrithm is shown as below:
    
    1. Initalization
        V(s) and pi(s) arbitarily for all s
        
    2. Policy Evaluation
        Loop:
            delta <-- 0
            Loop for each s:
                v <-- V(s)
                V(s) <-- sum_a { pi(a|s) } sum_s'_r {p(s',r |s,a)[ r+ gamma * V(s')] }
                delta <-- max(delta, np.abs(v - V(s)))
    until delta < theta (a small postive number determining the accuracy of estimation)
    
    3.Policy Improvement:
        policy_stable <-- true
        For each s in S:
            old-action <-- pi(s)
            pi(s)  <-- argmax_a { sum_s'_r (p(s',r | s,a )[r + gamma * V(s')])  }
            if old-action != pi(s) ,then policy_stable <-- false
         If policy-stable, then stop and return V=v(pi_*) and pi = pi_*;
            else go to 2

    PS. endstate shown the end state of the enviroment, the 15 is the end state 
    for the ForzenLake-v0 enviroment.
    '''

    while True:
        # Evaluate current Policy
        V=policy_eval(env,policy,discount_factor)
       
        policy_stable = True
        # Policy Improvement
        for s in range(env.nS):
            # select action w.r.t the highest probability
            old_action = np.argmax(policy[s])
            
            # find the optimal action under the current policy and state
            action_value = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_value[a] += prob * (reward + discount_factor * V[next_state])
                    if done and next_state != endstate:
                        action_value[a] = float( "-inf" )

            # update the current policy as greedy.
            best_action = np.argmax(action_value)

            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_action]
        
        # exist until stable
        if policy_stable:
            return policy, V

if __name__=="__main__":
    env=gym.make("FrozenLake-v0")
    random_policy = np.ones([env.nS, env.nA])/env.nA
    finalpolicy, Value = policy_iteration(env, random_policy)
    print("Reshaped the final policy (0 = up, 1 = right, 2= down, 3 =left ):\n")
    print(np.reshape(np.argmax(finalpolicy, axis = 1), [4,4]))
    print("This is the final value:\n")
    print(Value.reshape([4,4]))
