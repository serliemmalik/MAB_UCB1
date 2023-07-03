import matplotlib  # noqa
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
import numpy as np

import pdb

import multiprocessing
import time

import csv
import os
import pandas as pd

from bandits import BernoulliBandit
from algorithms import Algorithm, BUCB1, BUCB1_M, BUCB1_MV


def plot_results(algorithms, Algorithm_names, figname):
    """
    Plot the results by multi-armed bandit algorithms.

    Args:
        algorithms (list<Algorithm>): All of them should have been fitted.
        Algorithm_names (list<str)
        figname (str)
    """
    assert len(algorithms) == len(Algorithm_names)
    assert all(map(lambda s: isinstance(s, Algorithm), algorithms))
    assert all(map(lambda s: len(s.regrets) > 0, algorithms))

    b = algorithms[0].bandit

    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(bottom=0.4, wspace=0.4)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(133)
    ax3 = fig.add_subplot(132)

    # Sub.fig. 1: Regrets in time.
    for i,s in enumerate(algorithms):
        ax1.plot(range(len(s.regrets)), s.regrets, label=Algorithm_names[i])
        ax2.plot(range(b.n), s.estimated_probas, 'x', markeredgewidth=2, label=Algorithm_names[i])
        #ax2.fill_between(range(b.n), s.get_upper_bound, s.get_lower_bound, alpha=0.1)
        ax3.plot(range(b.n), np.array(s.counts) / float(len(algorithms[0].regrets)), ls='solid', lw=2)

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(bbox_to_anchor=(0.8, -0.25),loc='upper left', borderaxespad=0.)#loc=9, bbox_to_anchor=(1.82, -0.25), ncol=10)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by algorithms.
    
    ax2.plot(range(b.n), b.util, 'k--', markersize=12, label='True Utility')    
    #for i,s in enumerate(algorithms):
    #    ax2.plot(range(b.n), [s.esti /mated_probas[x] for x in range(b.n)], 'x', markeredgewidth=2, label=Algorithm_names[i])
    #    ax2.fill_between(range(b.n), [(s.estimated_probas[x] - s.confidence_interval[x]) for x in range(b.n)], [(s.estimated_probas[x] + s.confidence_interval[x]) for x in range(b.n)], alpha=0.1)
    ax2.set_xlabel('Actions') #sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.legend(bbox_to_anchor=(0.0, -0.25),loc='upper left', borderaxespad=0.)
    ax2.grid('k', ls='--', alpha=0.3)
    
    # Sub.fig. 3: Action counts
    #for s in algorithms:
    #    ax3.plot(range(b.n), np.array(s.counts) / float(len(algorithms[0].regrets)), ls='solid', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)
    
    plt.savefig(figname)
    plt.close('all')
    
def createHeader():

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
    
    return self.file_header
       
        
def aggregate(details):
      
   counts1 = details[0]
   means1 = details[1]
   conf_bounds1 = details[2]
   regrets1 = details[3]
   rewards1 = details[4]
   utils1 = details[5]
   
   for i in range(time_steps):
     counts1[i] /= 100
     regrets1[i] /= 100
     rewards[i] /= 100
     for j in range(num_of_arms):
       means1[i][j] /= 100
       conf_bounds1[i][j] /= 100
       utils1[i][j] /= 100
       
     line[i] = [counts1[i], mean1[i], conf_bounds1[i], regrets1[i], rewards1[i], utils1[i]]
     
   return line
     
      
    
def experiment(i, K, path):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of slot machiens.
        N (int): number of time steps to try.
        i (int): number of trial time.
    """
    
    b = BernoulliBandit(K)
    print ("Randomly generated Bernoulli bandit has utilities:\n", b.util)
    print ("The best machine has index: {} and Util: {}".format(
        max(range(K), key=lambda i: b.util[i]), max(b.util)))

    num_of_steps = [10000]
    
    # standard_dev = [2, 3]
    
    for step in range(len(num_of_steps)):
    
     # std_dir = '{}/{} steps/'.format(path, num_of_steps[step])
     # os.mkdir(std_dir)
     
     # for std in standard_dev:
      
     # result_dir = '{}/{} steps'.format(path, num_of_steps[step])
     result_dir = '{}'.format(path)
     # os.mkdir(result_dir)     
     # print("step: ", step)
     
     counts_header = [0] * K
     meanss = [0] * K
     conf_bounds = [0] * K
     UCB_utilss = [0] * K

     for j in range(K):
          counts_header[j] = 'Count {}'.format(j)
          meanss[j] = 'Mean {}'.format(j)
          conf_bounds[j] = 'Conf Bound {}'.format(j)
          UCB_utilss[j] = 'UCB Utility {}'.format(j)
          
     # file_header = ['Step', counts_header, meanss, conf_bounds, 'Regret', 'Reward', UCB_utilss]
     file_header = ['Step', counts_header, meanss, conf_bounds, 'Regret', UCB_utilss]
     
     st1 = [0] * num_of_steps[step]
     st2 = [0] * num_of_steps[step]
     st3 = [0] * num_of_steps[step]
     
     counts1 = [0] * num_of_steps[step]
     counts2 = [0] * num_of_steps[step]
     counts3 = [0] * num_of_steps[step]
     
     means1 = [0] * num_of_steps[step]
     means2 = [0] * num_of_steps[step]
     means3 = [0] * num_of_steps[step]
     
     conf_bounds1 = [0] * num_of_steps[step]
     conf_bounds2 = [0] * num_of_steps[step]
     conf_bounds3 = [0] * num_of_steps[step]
     
     regrets1 = [0] * num_of_steps[step]
     regrets2 = [0] * num_of_steps[step]
     regrets3 = [0] * num_of_steps[step]
     
     # rewards1 = [0] * num_of_steps[step]
     # rewards2 = [0] * num_of_steps[step]
     # rewards3 = [0] * num_of_steps[step]
     
     utils1 = [0] * num_of_steps[step]
     utils2 = [0] * num_of_steps[step]
     utils3 = [0] * num_of_steps[step]
     
     num_of_repeated_times = 1
           
     regrets_array = [[0] * 4 ]  * num_of_repeated_times
     for x in range(num_of_repeated_times):
      
        test_algorithms = [
            BUCB1(b, 2,1,1),
            BUCB1_M(b,2,1,1),
            BUCB1_MV(b,2,1,1)
        ]
        names = [
            'BUCB1',
            'BUCB1-M',
            'BUCB1-MV'
        ]              
        b.gen_random_values(num_of_steps[step]) 
        reg = [0] * len(test_algorithms)  #store regret for the three algorithms
        keep_track = 0
        graph_result = "{}/{}results_K{}_N{}".format(result_dir, x, K, num_of_steps[step])
        # print("Exp:", x)
        
        alg_num = 0
        for s in test_algorithms:
          s.run(num_of_steps[step], graph_result)
          reg[keep_track] = s.regret
          keep_track += 1      
          
          # line = [_ , coun, self.estimated_probas, self.confidence_interval, self.regret,  self.reward_gained, self.get_upper_bound]
            
          # print(s.results) 
          for i in range(num_of_steps[step]):
           if x == 0:
            if alg_num == 0:
              st1[i] = s.results[i][0]
              counts1[i] = [float(x) for x in s.results[i][1]] 
              means1[i] = [float(x) for x in s.results[i][2]]              
              conf_bounds1[i] = [float(x) for x in s.results[i][3]]   
              regrets1[i] = s.results[i][4]
              # rewards1[i] = s.results[i][5]  
              utils1[i] = [float(x) for x in s.results[i][5]]    
              
            elif alg_num == 1:
              st2[i] = s.results[i][0]
              counts2[i] = [float(x) for x in s.results[i][1]]  
              means2[i] = [float(x) for x in s.results[i][2]] 
              conf_bounds2[i] = [float(x) for x in s.results[i][3]]   
              regrets2[i] = s.results[i][4]
              # rewards2[i] = s.results[i][5]
              utils2[i] = [float(x) for x in s.results[i][5]]
              
            else:
             st3[i] = s.results[i][0] 
             counts3[i] = [float(x) for x in s.results[i][1]] 
             means3[i] = [float(x) for x in s.results[i][2]]
             conf_bounds3[i] = [float(x) for x in s.results[i][3]] 
             regrets3[i] = s.results[i][4] 
             # rewards3[i] = s.results[i][5]
             utils3[i] = [float(x) for x in s.results[i][5]] 
             
           else:              
            if alg_num == 0:
             st1[i] += s.results[i][0] 
             regrets1[i] += s.results[i][4] 
             # rewards1[i] += s.results[i][5]
                            
            elif alg_num == 1:
              st2[i] += s.results[i][0]    
              regrets2[i] += s.results[i][4]  
              # rewards2[i] += s.results[i][5]              
            else:
             st3[i] += s.results[i][0] 
             regrets3[i] += s.results[i][4] 
             # rewards3[i] += s.results[i][5]
             
            for j in range(K):
               if alg_num == 0:
                 counts1[i][j] += s.results[i][1][j]
                 means1[i][j] += s.results[i][2][j] 
                 conf_bounds1[i][j] += s.results[i][3][j] 
                 utils1[i][j] += s.results[i][5][j] 
                 
               elif alg_num == 1:
                 counts2[i][j] += s.results[i][1][j] 
                 means2[i][j] += s.results[i][2][j]
                 conf_bounds2[i][j] += s.results[i][3][j]
                 utils2[i][j] += s.results[i][5][j] 
               
               else:
                 counts3[i][j] += s.results[i][1][j] 
                 means3[i][j] += s.results[i][2][j]
                 conf_bounds3[i][j] += s.results[i][3][j]
                 utils3[i][j] += s.results[i][5][j] 
          
          alg_num += 1    
     
        # plot_results(test_algorithms, names, graph_result)
        
        regrets_array[x] = [reg[0], reg[1], reg[2]]
        
     for i in range(num_of_steps[step]): 
       st1[i] = st1[i] / num_of_repeated_times
       st2[i] = st2[i] / num_of_repeated_times
       st3[i] = st3[i] / num_of_repeated_times
       
       regrets1[i] = regrets1[i] / num_of_repeated_times
       regrets2[i] = regrets2[i] / num_of_repeated_times
       regrets3[i] = regrets3[i] / num_of_repeated_times
       
       # rewards1[i] = rewards1[i] / 100
       # rewards2[i] = rewards2[i] / 100
       # rewards3[i] = rewards3[i] / 100
       
       for j in range(K):
         counts1[i][j] = counts1[i][j] / num_of_repeated_times
         means1[i][j] = means1[i][j] / num_of_repeated_times
         conf_bounds1[i][j] = conf_bounds1[i][j] / num_of_repeated_times
         utils1[i][j] = utils1[i][j] / num_of_repeated_times
         
         counts2[i][j] = counts2[i][j] / num_of_repeated_times
         means2[i][j] = means2[i][j] / num_of_repeated_times
         conf_bounds2[i][j] = conf_bounds2[i][j] / num_of_repeated_times
         utils2[i][j] = utils2[i][j] / num_of_repeated_times
         
         counts3[i][j] = counts3[i][j] / num_of_repeated_times
         means3[i][j] = means3[i][j] / num_of_repeated_times
         conf_bounds3[i][j] = conf_bounds3[i][j] / num_of_repeated_times
         utils3[i][j] = utils3[i][j] / num_of_repeated_times
         
         
     column_names = ['BUCB1','BUCB1-M','BUCB1-MV']
     
     df1 = pd.DataFrame(columns = file_header)
     df2 = pd.DataFrame(columns = file_header)
     df3 = pd.DataFrame(columns = file_header)
     for i in range(num_of_steps[step]):
     
       line1 = [st1[i], counts1[i], means1[i], conf_bounds1[i], regrets1[i], utils1[i]]
       df1.loc[len(df1)] = line1
       line2 = [st2[i], counts2[i], means2[i], conf_bounds2[i], regrets2[i], utils2[i]]
       df2.loc[len(df2)] = line2
       line3 = [st3[i], counts3[i], means3[i], conf_bounds3[i], regrets3[i], utils3[i]]
       df3.loc[len(df3)] = line3
       
     df1.to_csv('{} BUCB'.format(path), header = file_header)
     df2.to_csv('{} BUCB-M'.format(path), header = file_header)
     df3.to_csv('{} BUCB-MV'.format(path), header = file_header)
     
     
    data = {'Likelihood': b.probas, 'Arms cost': b.arms_cost, 'Arms Reward': b.arms_reward, 'Utility': b.util}
    dff = pd.DataFrame(data)
    dff.to_csv('{} values'.format(path))         
                
def run(number_of_arms, append_file_name, exp_number):

   num_of_arms = [number_of_arms]
   num_of_times_to_run_each_step = 5
   
   g_drive_path = 'gdrive/MyDrive/Colab Notebooks'
   experiment_path = 'Experiment Results {}'.format(append_file_name)
   # os.mkdir(experiment_path)
   
   for i in range(len(num_of_arms)):
     # os.mkdir('{}/{} arms'.format(experiment_path, num_of_arms[i]))
     
     for k in range(num_of_times_to_run_each_step): 
          path = '{}/{} arms/{}'.format(experiment_path, num_of_arms[i], k + exp_number)
          # os.mkdir(path)
          experiment(k, num_of_arms[i], path)          

num_of_arms = 100
process1 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',0])
process2 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',5])

process3 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',10])
process4 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',15])

process5 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',20])
process6 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',25])

process7 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',30])
process8 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',35])

process9 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',40])
process10 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',45])

process11 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',50])
process12 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',55])

process13 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',60])
process14 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',65])

process15 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',70])
process16 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',75])

process17 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',80])
process18 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',85])

process19 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',90])
process20 = multiprocessing.Process(target=run, args=[num_of_arms,'BUCB',95])


if __name__ == '__main__':
  process1.start()
  time.sleep(10)
  process2.start()
  time.sleep(10)
  process3.start()
  time.sleep(10)
  process4.start()
  time.sleep(10)
  process5.start()
  time.sleep(10)
  
  process6.start()
  time.sleep(10)
  process7.start()
  time.sleep(10)
  process8.start()
  time.sleep(10)
  process9.start()
  time.sleep(10)
  
  process10.start()  
  time.sleep(10)
  process11.start()
  time.sleep(10)
  process12.start()
  time.sleep(10)
  process13.start()
  time.sleep(10)
  process14.start()
  
  process15.start()
  time.sleep(10)
  process16.start()
  time.sleep(10)
  process17.start()
  time.sleep(10)
  process18.start()
  time.sleep(10)
  process19.start()
  time.sleep(10)
  process20.start()
  
