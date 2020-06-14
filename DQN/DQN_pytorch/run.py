import json
import os
import time
import dqn
import gym
from constants import *
from gym import wrappers
from measure import *
from collections import namedtuple
import tensorflow as tf

if __name__ == '__main__':
    tf.set_random_seed(12)
    env = gym.make(ENV_NAME)
    env = env.unwrapped

    action_space=env.action_space.n
    obs_space=env.observation_space.shape[0]
    explorationRate = INITIAL_EPSILON
    stepCounter = 0
    Agent = dqn.DeepQ(obs_space,action_space, MEMORY_SIZE, GAMMA, LEARNING_RATE,
            USE_TARGET_NETWORK)
    reward_history=[]
    record = namedtuple("Record", ["episode_lengths","episode_rewards"])
    
    rec = record(episode_lengths=np.zeros(MAX_EPOCHS),
                          episode_rewards=np.zeros(MAX_EPOCHS))
    total_action0 =0
    total_action1 =0

    for epoch in range(MAX_EPOCHS):
        obs = env.reset()
        observation = obs[np.newaxis,:]
        cumulated_reward = 0
        step =0
        if (epoch % TEST_INTERVAL_EPOCHS != 0 or stepCounter \
                 < LEARN_START_STEP) and TRAIN is True :  # explore
            EXPLORE = True
        else:
            EXPLORE = True
#            print ("Evaluate Model")
        done = False
        action_0 = 1
        action_1 =1
    
        while True:
            if cumulated_reward >100:
                env.render()
            step += 1
            if EXPLORE is True: 
                 action = Agent.feedforward(observation, explorationRate)
                 if action == 0:
                     action_0 += 1
                     total_action0 += 1
                 else:
                     action_1 += 1
                     total_action1 += 1

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
        if cumulated_reward >50 or epoch % 1000 ==0: 
            size = Agent.getMemorySize()
            totalactionrate = float(total_action0/total_action1)
            print("This is epoch: {}, reward: {}, action: {}/{}, total action: {}/{},rate:{}".format(
                epoch, cumulated_reward, action_0, action_1,total_action0, total_action1, totalactionrate))
    figname=ENV_NAME+str(MAX_EPOCHS)+".jpg"
    plot_and_save(reward_history,MONITOR_DIR,figname)
    
    save_prefix ="DQN_tensorflow_" + ENV_NAME
    Plot_the_result_for_rec(rec, MONITOR_DIR, save_prefix)
