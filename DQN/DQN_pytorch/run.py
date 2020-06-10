import json
import os
import time
import dqn
import gym
from constants import *
from gym import wrappers
from measure import *

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    input_shape=[1,4]
    env.rendering = SHOW
    action_space=env.action_space.n
    obs_space=env.observation_space.shape[0]
    print(action_space,obs_space)
    explorationRate = INITIAL_EPSILON
    stepCounter = 0
    Agent = dqn.DeepQ(obs_space,action_space, MEMORY_SIZE, GAMMA, LEARNING_RATE,
            USE_TARGET_NETWORK)
    reward_history=[]
    for epoch in range(MAX_EPOCHS):
        obs = env.reset()
        observation=obs.reshape(input_shape)
        cumulated_reward = 0
        if (epoch % TEST_INTERVAL_EPOCHS != 0 or stepCounter \
                 < LEARN_START_STEP) and TRAIN is True :  # explore
            EXPLORE = True
        else:
            EXPLORE = False
            print ("Evaluate Model")
        for t in range(1000):
             if EXPLORE is True: 
                 action = Agent.feedforward(observation, explorationRate)
                 obs_new, reward, done, info = env.step(action)
                 newObservation=obs_new.reshape(input_shape)
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
                 newObservation = obs_new.reshape(input_shape)
                 observation = newObservation
             cumulated_reward+=reward
        reward_history.append(cumulated_reward)
        print("This is epoch:",epoch," and the cumulated_reward is :",cumulated_reward)
    figname=ENV_NAME+str(MAX_EPOCHS)+".jpg"
    plot_and_save(reward_history,MONITOR_DIR,figname) 
