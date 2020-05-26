import numpy as np
import gym

def policy_eval(enviroment,policy,discount_factor=1.0,theta=0.1):
    """
    This function provide the policy evaluation function.
    The alogrithm is shown as below:
    
    Iterative Policy Evaluation, for estimating V =v_pi
    Algorithm parameter: a small threshold theta > 0 determining accuracy of estimation
    Initilize V(s), for all s, arbitarily except V(terminal)=0
    
    Loop:
        delta <-- 0
        Loop for each s:
            v <-- V(s)
            V(s) <-- sum_a { pi(a|s) } sum_s'_r {p(s',r |s,a)[ r+ gamma * V(s')] }
            delta <-- max(delta, np.abs(v - V(s)))
    until delta < theta
    """
   
   # Set enviroment
    env = enviroment
   
   # Value function Initalization
    V = np.zeros(env.nS)
   
   # Start iteration
    for _ in range(500):
        delta = 0
        # Loop in state space 
        for s in range(env.nS):
            v=0
            # Loop in action space
            for a,action_prob in enumerate(policy[s]):
                # Loops in next_state and reward
                for prob,next_state,reward,done in env.P[s][a]:
                    v += action_prob * prob * ( reward + discount_factor * V[next_state])
            #chosen the max error delta
            delta=max(delta,np.abs(v-V[s]))
            V[s] =v

        if delta < theta:
            break
    return np.array(V)

def generate_policy(env,input_policy):
    policy=np.zeros([env.nS,env.nA])
    for _ , x in enumerate(input_policy):
        policy[_][x] = 1
    return policy


if __name__=="__main__":
    env=gym.make("FrozenLake-v0")
    input_policy=[2,1,2,3,2,0,2,0,1,2,2,0,0,1,1,0]
    policy=generate_policy(env,input_policy)
    Value=policy_eval(env,policy)
    print("This is the final value:\n")
    print(Value.reshape([4,4]))
