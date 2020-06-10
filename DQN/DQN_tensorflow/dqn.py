import time
import random
import numpy as np
import tensorflow as tf
import memory
from constants import *
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
            learningRate,useTargetNetwork=True):
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.useTargetNetwork = useTargetNetwork
        self.count_steps = 0
        self.obs_space=obs_space
        self.action_space=action_space
        self.initNetworks()
        
        target_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network")
        evaluate_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")

        with tf.variable_scope("hard_replacement"):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_parameters, evaluate_parameters)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())        

    def initNetworks(self):
        # define the placeholder
        self.s = tf.placeholder(tf.float32, [None, self.obs_space[0]], name="s")
        self.s_ = tf.placeholder(tf.float32, [None, self.obs_space[0]], name="s_")
        self.r = tf.placeholder(tf.float32, [None, ], name="r")
        self.a = tf.placeholder(tf.int32, [None, ], name="a")

        w_initlizer, b_initlizer = tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)

        # build the evaluate network
        with tf.variable_scope("eval_net"):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, 
                    kernel_initializer=w_initlizer, bias_initializer=b_initlizer, name="e1")
            self.q_eval = tf.layers.dense(e1, self.action_space, tf.nn.relu,
                    kernel_initializer=w_initlizer, bias_initializer=b_initlizer, name="q")\

        # build the target network
        with tf.variable_scope("target_network"):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, 
                    kernel_initializer=w_initlizer, bias_initializer=b_initlizer, name="t1")
            self.q_next = tf.layers.dense(t1, self.action_space, tf.nn.relu,
                    kernel_initializer=w_initlizer, bias_initializer=b_initlizer, name="q_next")
        
        with tf.variable_scope("q_target"):
            q_target = self.r + self.discountFactor * tf.reduce_max(self.q_next, axis=1, name="Qmax_s_")
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope("q_evaluate"):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0],dtype=tf.int32),self.a],axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a), name="loss")

        with tf.variable_scope("train"):
            self._train_op = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
        
    def updateTargetNetwork(self):
        self.sess.run(self.target_replace_op)
        print('update target network')

    def getQValues(self,state):
#        state = state[np.newaxis,:]
        predicted = self.sess.run(self.q_next,feeddict={self.s_, state})

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

        
        _, cost = self.sess.run([self._train_op, self.loss],
                                feeddict-{
                                    self.s : state_batch,
                                    self.a : action_batch,
                                    self.r : reward_batch,
                                    self.s_: newState_batch,
                                    })

        qValues_batch = self.model.predict(np.array(state_baddtch),batch_size=miniBatchSize)


        if self.useTargetNetwork and self.count_steps % 1000 == 0:
            self.updateTargetNetwork()

    def saveModel(self, path):
        pass

    def loadWeights(self, path):
        pass

    def feedforward(self,observation,explorationRate):
        observation = observation[np.newaxis,:]
        print("-------------------",type(observation))
        qValues = self.getQValues(observation)
        action = self.selectAction(qValues, explorationRate)
        return action
