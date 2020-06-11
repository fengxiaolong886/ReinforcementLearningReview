import json
import os
import time
import dqn
import gym
from constants import *
from gym import wrappers
from measure import *
from collections import namedtuple

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    input_shape=[1,4]
    env.rendering = SHOW
    action_space=env.action_space.n
    obs_space=env.observation_space.shape
    explorationRate = INITIAL_EPSILON
    stepCounter = 0
    Agent = dqn.DeepQ(obs_space,action_space, MEMORY_SIZE, GAMMA, LEARNING_RATE,
            USE_TARGET_NETWORK)
    reward_history=[]
    record = namedtuple("Record", ["episode_lengths","episode_rewards"])
    
    rec = record(episode_lengths=np.zeros(MAX_EPOCHS),
                          episode_rewards=np.zeros(MAX_EPOCHS))

    for epoch in range(MAX_EPOCHS):
        obs = env.reset()
        observation = obs[np.newaxis,:]
        cumulated_reward = 0
        step =0
        if (epoch % TEST_INTERVAL_EPOCHS != 0 or stepCounter \
                 < LEARN_START_STEP) and TRAIN is True :  # explore
            EXPLORE = True
        else:
            EXPLORE = False
            print ("Evaluate Model")
        for t in range(1000):
            step += 1
            if EXPLORE is True: 
                 action = Agent.feedforward(observation, explorationRate)
                 obs_new, reward, done, info = env.step(action)
                 newObservation = obs_new[np.newaxis,:]
                 stepCounter += 1
                 Agent.addMemory(observation, action, reward, newObservation, done)
                 observation = newObservation
                 if stepCounter == LEARN_START_STEP:
                     print("Starting learning")
                 if Agent.getMemorySize() >= LEARN_START_STEP:
                     Agent.learnOnMiniBatch(BATCH_SIZE)
                     if explorationRate > FINAL_EPSILON and stepCounter > LEARN_START_STEP:
                         explorationRate -= (INITIAL_EPSILON - \
                            FINAL_EPSILON) / MAX_EXPLORE_STEPS
            else:
                 action = Agent.feedforward(observation,0)
                 obs_new, reward, done, info = env.step(action)
                 newObservation = obs_new[np.newaxis,:]
                 observation = newObservation
            if done:
                break
            cumulated_reward += reward
        reward_history.append(cumulated_reward)
        
        rec.episode_lengths[epoch] = step 
        rec.episode_rewards[epoch] += cumulated_reward
        
        print("This is epoch:",epoch," and the cumulated_reward is :",cumulated_reward)
    figname=ENV_NAME+str(MAX_EPOCHS)+".jpg"
    plot_and_save(reward_history,MONITOR_DIR,figname)
    
    save_prefix ="DQN_tensorflow_" + ENV_NAME
    Plot_the_result_for_rec(rec, MONITOR_DIR, save_prefix)
