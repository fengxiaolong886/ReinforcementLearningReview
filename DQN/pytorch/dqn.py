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
        self.fc1=nn.Linear(num_state,50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2=nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(30,num_action)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        action_prob=self.out(x)
        return action_prob


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
        print('update target network')

    # predict Q values for all the actions
    def getQValues(self, state):
        if self.useTargetNetwork:
            predicted = self.model.forward(state)
        else:
            predicted = self.targetModel.forward(state)
        return predicted[0]

    def getMaxIndex(self, qValues):
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

        qValues_batch = self.model(np.array(state_batch))
    
        isFinal_batch = np.array(isFinal_batch) + 0

        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if self.useTargetNetwork:
            qValuesNewState_batch = self.targetModel(np.array(newState_batch))
        else :
            qValuesNewState_batch = self.model(np.array(newState_batch))

        Y_sample_batch = reward_batch + (1 - isFinal_batch) *\
                self.discountFactor * np.max(qValuesNewState_batch, axis=1)

#        X_batch = np.array(state_batch)
#        Y_batch = np.array(qValues_batch)
        
        loss=self.loss_func(qValues_batch,qValuesNewState_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
#        for i,action in enumerate(action_batch):
#            Y_batch[i][action] = Y_sample_batch[i]
        #t1 = time.time()
        #self.model.fit(X_batch, Y_batch, batch_size = miniBatchSize)
#        self.model.train_on_batch(X_batch, Y_batch)
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

