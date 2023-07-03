from __future__ import division

import time
import numpy as np




class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError

class BernoulliBandit(Bandit):

    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            np.random.seed(int(time.time()))

            #Likelihood monotonically increasing
            #self.probas = [np.random.random(np.random.seed(np.random.randint(100000)))  for x in range(self.n)] #Likelihood
            self.probas = np.random.random(self.n) #Likelihood
            self.probas.sort()
            
            #Reward
            self.arms_reward=[1 for x in range(self.n)]

            #Cost - monotonically increasing
            # self.arms_cost= [np.random.random(np.random.seed(np.random.randint(100000))) for x in range(self.n)]
            self.arms_cost= np.random.random(self.n)
            self.arms_cost.sort()
            
            #self.arms_cost = np.subtract(self.arms_reward, self.arms_cost)
            
            #Compute the utility of each arm    
            self.util = np.subtract(self.probas, self.arms_cost) 
            
        else:
            self.probas = probas

        self.best_util = max(self.util)
        
        self.random_values = []
        
    def gen_random_values(self, nums):
        #generates randoms to use for all compared algorithms. 
        # self.random_values = np.random.uniform(-1,1, nums)
        self.random_values = np.random.random(nums)
        # self.random_values = np.clip(np.random.uniform(-1, 1, nums), -1, 1)
        
    def generate_reward(self, i):
        '''
         i (int): index of the machine.
        time: timestep 
        '''
        # The player selected the i-th machine.
        # if self.random_values[time] <= self.util[i]:
        # max = 1 - self.arms_cost[i]
        # min = 0 - self.arms_cost[i]
        if np.random.random() <= self.probas[i]:
            return 1
        else:
            return 0