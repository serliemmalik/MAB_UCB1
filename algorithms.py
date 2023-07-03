from __future__ import division

import csv
import os

import numpy as np
import pandas as pd
import time
from scipy.stats import beta
import scipy.stats as stats

from bandits import BernoulliBandit


class Algorithm(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = np.array([0] * self.bandit.n, dtype=np.uint32)   # Number of times each machine is chosen.
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret as appended
        
        self.results = [ ]
        
        self.time = 0  #time step 
        
        self.counts_header = [0] * self.bandit.n
        self.meanss = [0] * self.bandit.n
        self.meanss_util = [0] * self.bandit.n
        self.conf_bounds = [0] * self.bandit.n
        self.meanss_rad = [0] * self.bandit.n
        self.UCB_utilss = [0] * self.bandit.n
        self.LCB_utilss = [0] * self.bandit.n

        for j in range(self.bandit.n):
          self.counts_header[j] = 'Count {}'.format(j)
          self.meanss[j] = 'Mean {}'.format(j)
          self.meanss_rad[j] = 'Mean Rad {}'.format(j)
          self.conf_bounds[j] = 'Conf Bound {}'.format(j)
          self.UCB_utilss[j] = 'UCB Utility {}'.format(j)
          self.LCB_utilss[j] = 'LCB Utility {}'.format(j)
          
        self.file_header = [self.counts_header, self.meanss, self.conf_bounds, 'Regret', 'Random', 'Picked Arm', self.UCB_utilss]
    
    def update_line(self, line):
        self.results.append(line)
    
    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_util - self.bandit.util[i]
        self.regrets.append(self.regret)

    def run(self, num_steps, path):
        assert self.bandit is not None

        # line = [0, [0] * self.bandit.n, [0] * self.bandit.n, [0] * self.bandit.n,0,0,0,0,[0] * self.bandit.n]

        # self.lines = np.array([line] * num_steps, dtype=object)
        df = pd.DataFrame(columns = self.file_header)
        for _ in range(num_steps):
            i = self.run_one_step()
            self.time = _

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)

            coun = np.array([0] * self.bandit.n)
            for x in range(self.bandit.n):
              coun[x] = self.counts[x]

            # line = [_ , coun, self.estimated_probas, self.confidence_interval, self.regret, self.bandit.random_values[self.time], self.reward_gained, i, self.get_upper_bound]
            # line = [_ , coun, self.estimated_probas, self.confidence_interval, self.regret, self.reward_gained, self.get_upper_bound]
            line = [_ , coun, self.estimated_probas, self.confidence_interval, self.regret, self.get_upper_bound]
            #print(line)
            # df.loc[len(df)] = line
            self.update_line(line)

class UCB1(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c, init_a, init_b):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(UCB1, self).__init__(bandit)
        
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        
        self.num_of_chosen_arm = [0] * self.bandit.n

        self.mean = [0] * self.bandit.n

        self.estimated_likelihood = [0] * self.bandit.n
        self.estimated_util= [0] * self.bandit.n
        
        self.rad  = [0] * self.bandit.n
        
        
        self.t = 0 # time_step
        self.r =  [0] * self.bandit.n # reward        
        self.re =  0 # reward        
        
    @property
    def estimated_probas(self):
        return np.subtract(self.mean, self.bandit.arms_cost)
    
    @property
    def confidence_interval(self):
        return self.rad
        
    @property
    def get_upper_bound(self):
        return self.estimated_util
    
    @property
    def get_lower_bound(self):
        return self.lower_bound
        
    @property
    def id(self):
        return 1 # UCB
        
    @property
    def reward_gained(self):
        return self.r  #reward
        
    def run_one_step(self):
        self.t += 1
        self.picked_arm_index = -1 #highest arm index
        
        # number of times arms are chosen
        self.num_of_chosen_arm = np.add(self._as, self._bs)
        
        # estimated mean of the likelihood
        self.mean = np.divide(self._as, self.num_of_chosen_arm)
        
        # confidence interval 
        self.rad = np.sqrt(np.divide(self.c * np.log(self.t), self.num_of_chosen_arm))        
           
        # upper bounded likelihood
        self.estimated_likelihood = np.add(self.mean, self.rad)
        
        # upper bounded expected utility
        self.estimated_util = np.subtract(self.estimated_likelihood, self.bandit.arms_cost)
       
        # select arm with the highest upper bounded expexted utility 
        self.picked_arm_index = np.argmax(self.estimated_util) # highest arm index
        
        # generate reward between 0 or 1
        self.re = self.bandit.generate_reward(self.picked_arm_index)
        
        # update selected arms
        self._as[self.picked_arm_index] += self.re
        self._bs[self.picked_arm_index] += (1 - self.re)
       
        # self.mean[self.picked_arm_index] += 1. / (self.counts[self.picked_arm_index] + 1) * (self.re - self.mean[self.picked_arm_index])
        

        return self.picked_arm_index


class UCB1_M(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c, init_a, init_b):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(UCB1_M, self).__init__(bandit)
        
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        
        self.num_of_chosen_arm = [0] * self.bandit.n

        self.mean = [0] * self.bandit.n

        self.estimated_likelihood = [0] * self.bandit.n
        self.mean_estimated_likelihood = [0] * self.bandit.n
        self.estimated_util= [0] * self.bandit.n
        self.lower_estimated_likelihood = [0] * self.bandit.n
        self.lower_bound = [0] * self.bandit.n

        self.estimated_likelihood_rad = [0] * self.bandit.n
        self.mean_estimated_util = [0] * self.bandit.n
        

        self.t = 0 # time_step
        self.r =  [0] * self.bandit.n # reward        
        self.re =  0 # reward        
        
        self.rad = [0] * self.bandit.n
        # self.rad = np.sqrt(np.divide(self.c * np.log(1), np.add(self.counts,1))) 
        
    @property
    def estimated_probas(self):
        return np.subtract(self.mean, self.bandit.arms_cost)
        
    @property
    def confidence_interval(self):
        return self.rad
        
    @property
    def get_upper_bound(self):
        return self.estimated_util
    
    @property
    def get_lower_bound(self):
        return self.lower_bound

    @property
    def id(self):
        return 2 # UCB_M
        
    @property
    def reward_gained(self):
        return self.r  #reward
       
    def run_one_step(self):
         self.t += 1
         self.picked_arm_index = -1 #highest arm index
         
         # number of times arms are chosen
         self.num_of_chosen_arm = np.add(self._as, self._bs)
         
         # estimated mean of the likelihood
         self.mean = np.divide(self._as, self.num_of_chosen_arm)
        
         # confidence interval 
         self.rad = np.sqrt(np.divide(self.c * np.log(self.t), self.num_of_chosen_arm))  
         
         # upper bounded likelihood
         self.estimated_likelihood = np.add(self.mean, self.rad)
         
         # upper bounded expected utility
         self.estimated_util= np.subtract(self.estimated_likelihood, self.bandit.arms_cost)
         
         # expected utility with no confidence interval
         self.mean_estimated_util = np.subtract(self.mean, self.bandit.arms_cost)
                 
         for i in range(self.bandit.n):
         
           if i > 0:
             if self.mean[i - 1] < self.mean[i] and self.estimated_likelihood[i - 1] > self.estimated_likelihood[i]:
                self.estimated_likelihood[i - 1] = self.estimated_likelihood[i]
                # self.mean[i - 1] = self.mean[i]
                
             elif self.mean[i - 1] > self.mean[i] and self.estimated_likelihood[i - 1] > self.estimated_likelihood[i]:
                if (self.estimated_likelihood[i - 1] - self.mean[i - 1]) > (self.estimated_likelihood[i] - self.mean[i]):
                   self.estimated_likelihood[i - 1] = self.estimated_likelihood[i] 
                   # self.mean[i - 1] = self.mean[i] 
                   
                elif (self.estimated_likelihood[i - 1] - self.mean[i - 1]) < (self.estimated_likelihood[i] - self.mean[i]):
                   # This could be substituted as rad
                   self.estimated_likelihood[i] = self.estimated_likelihood[i - 1] 
                   # self.mean[i] = self.mean[i - 1]
                   
                else:
                   self.estimated_likelihood[i - 1] = self.estimated_likelihood[i]
                   # self.mean[i - 1] = self.mean[i]
                  
                
             elif self.mean[i - 1] == self.mean[i] and self.estimated_likelihood[i - 1] > self.estimated_likelihood[i]:
                self.estimated_likelihood[i- 1] = self.estimated_likelihood[i] 
                # self.mean[i - 1] = self.mean[i]
         
         # Ensure arms are 
         self.estimated_likelihood = np.flip(np.minimum.accumulate(np.flip(self.estimated_likelihood)))   
         # self.mean = np.flip(np.minimum.accumulate(np.flip(self.mean)))
         # self.rad = np.subtract(self.estimated_likelihood, self.mean)
         
         # re-computed upper bounded expected utility
         self.estimated_util= np.subtract(self.estimated_likelihood, self.bandit.arms_cost)
         
         self.mean_estimated_util = np.subtract(self.mean, self.bandit.arms_cost)
         self.lower_bound = np.subtract(self.lower_estimated_likelihood, self.rad)
         
         
         self.picked_arm_index = np.argmax(self.estimated_util)
         
        
         self.re = self.bandit.generate_reward(self.picked_arm_index)
         
         # Update arm         
         self._as[self.picked_arm_index] += self.re
         self._bs[self.picked_arm_index] += (1 - self.re)
         
         # self.mean[self.picked_arm_index] += 1. / (self.counts[self.picked_arm_index] + 1) * (self.re - self.mean[self.picked_arm_index])
                  
         return self.picked_arm_index 

#This is the UCB-VM version
class UCB1_MV(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c, init_a, init_b):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(UCB1_MV, self).__init__(bandit)
        
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        
        self.num_of_chosen_arm = [0] * self.bandit.n

        self.mean = [0] * self.bandit.n

        self.estimated_likelihood = [0] * self.bandit.n
        self.estimated_util= [0] * self.bandit.n
        
        self.t = 0 # time_step
        self.r =  [0] * self.bandit.n # reward        
        self.re =  0 # reward        
        

    @property
    def estimated_probas(self):
        return np.subtract(self.mean, self.bandit.arms_cost)
    
    @property
    def confidence_interval(self):
        return self.rad
        
    @property
    def get_upper_bound(self):
        return self.estimated_util

    @property
    def id(self):
        return 3 #"UCB_MV"
        
    @property
    def reward_gained(self):
        return self.r  #reward
       
    def run_one_step(self):
         self.t += 1
         self.picked_arm_index = -1 #highest arm index
         
         # number of times arms are chosen
         self.num_of_chosen_arm = np.add(self._as, self._bs)
         
         # estimated mean of the likelihood
         self.mean = np.divide(self._as, self.num_of_chosen_arm)
        
         # confidence interval 
         self.rad = np.sqrt(np.divide(self.c * np.log(self.t), self.num_of_chosen_arm))    
         # Updated confidence bound -        
         self.rad = np.flip(np.minimum.accumulate(np.flip(self.rad))) 
         
         # upper bounded likelihood
         self.estimated_likelihood = np.add(self.mean, self.rad)
         self.estimated_util= np.subtract(self.estimated_likelihood, self.bandit.arms_cost)

            
         self.picked_arm_index = np.argmax(self.estimated_util) #highest arm index
         
         # Generate reward
         self.re= self.bandit.generate_reward(self.picked_arm_index)

         # Update arm
         self._as[self.picked_arm_index] += self.re
         self._bs[self.picked_arm_index] += (1 - self.re)
         
         # self.mean[self.picked_arm_index] += 1. / (self.counts[self.picked_arm_index] + 1) * (self.re - self.mean[self.picked_arm_index])


         return self.picked_arm_index
