import time
import random
import numpy as np
from constants import *
import pickle
import memory
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal,Categorical
from torch.utils.data.sampler import BatchSampler,SubsetRandomSampler


class Net(nn.Module):
    def __init__(self,num_state,num_action):
        super(Net,self).__init__()
        self.fc1=nn.Linear(num_state,24)
        self.fc2=nn.Linear(24,24)
        self.out=nn.Linear(24,num_action)

    def forward(self,x):
        x=self.fc1(x)
        x=F.tanh(x)
        x=self.fc2(x)
        x=F.tanh(x)
        action_value=self.out(x)
        return action_value


class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, obs_space, action_space,memorySize, discountFactor, \
            learningRate,useTargetNetwork):
        super(DeepQ,self).__init__()
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.useTargetNetwork = useTargetNetwork
        self.count_steps = 0
        self.obs_space=obs_space
        self.action_space=action_space
        self.model=Net(self.obs_space,self.action_space)
        self.targetModel=Net(self.obs_space,self.action_space)
        self.optimizer=optim.Adam(self.model.parameters(),self.learningRate)
        self.loss_func=nn.MSELoss()


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.targetModel.load_state_dict(self.model.state_dict())

    # predict Q values for all the actions
    def getQValues(self, state):
        if self.useTargetNetwork:
            predicted = self.model.forward(state)
        else:
            predicted = self.targetModel.forward(state)
        return predicted[0]

    def getMaxIndex(self, qValues):
        qValues = qValues.detach().numpy()
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.action_space)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)


    def getMemorySize(self):
        return self.memory.getCurrentSize()


    def learnOnMiniBatch(self, miniBatchSize,):
        #t0 = time.time()
        self.count_steps += 1

        state_batch,action_batch,reward_batch,newState_batch,isFinal_batch\
        = self.memory.getMiniBatch(miniBatchSize)
       
        state_batch = torch.from_numpy(np.array(state_batch)).float()
        reward_batch = np.array(reward_batch)
        newState_batch = torch.from_numpy(np.array(newState_batch)).float()

        All_action_qValues_batch = self.model(state_batch)
        qValues_batch_wrt_a = All_action_qValues_batch[0][action_batch]

        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if self.useTargetNetwork:
            qValuesNewState_batch = self.targetModel(newState_batch)
        else :
            qValuesNewState_batch = self.model(newState_batch)

        Y_sample_batch = reward_batch + self.discountFactor * np.max(qValuesNewState_batch.detach().numpy(), axis=1)
        

        Y_sample_batch = torch.tensor(Y_sample_batch)
        loss=self.loss_func(Y_sample_batch, qValues_batch_wrt_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.useTargetNetwork and self.count_steps % 1000 == 0:
            self.updateTargetNetwork()

        #print time.time() - t0, time.time() -t1
    def saveModel(self, path):
        if self.useTargetNetwork:
            self.targetModel.save(path)
        else:
            self.model.save(path)

    def loadWeights(self, path):
        self.model.load_weights(path)
        if self.useTargetNetwork:
            self.targetModel.load_weights(path)


    def feedforward(self,observation,explorationRate):
        observation=torch.unsqueeze(torch.FloatTensor(observation),0)
        qValues = self.getQValues(observation)
        action = self.selectAction(qValues, explorationRate)
        return action

