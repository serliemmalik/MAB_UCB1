import pandas as pd
import time
import multiprocessing

import math
import numpy as np
from scipy.stats import beta


def compute_file(file_name, num_of_arms, timesteps):
    nums = [0] * timesteps
    counts = [0] * timesteps
    means = [0] * timesteps
    conf_bounds = [0] * timesteps
    regrets = [0] * timesteps
    rewards = [0] * timesteps
    utils = [0] * timesteps
    
    data = pd.read_csv(file_name)
    row = data.iloc[0:timesteps]
    
    for i,j in row.iterrows():
       nums[i] = j[1]  # It comes out as integer already
       counts[i] = j[2] # Space-only separated
       means[i] = j[3] #Comma and space separed
       conf_bounds[i] = j[4] #Space-inly separated
       regrets[i] = j[5] # It comes out as float already
       # rewards[i] = j[6] # It comes out as float already
       utils[i] = j[6] # Space-only separated
    counts = convert_commas(counts, num_of_arms, timesteps)
    means = convert_commas(means, num_of_arms, timesteps)
    conf_bounds = convert_commas(conf_bounds, num_of_arms, timesteps)
    utils = convert_commas(utils, num_of_arms, timesteps)
    
    return nums, counts, means, conf_bounds, regrets, utils
    
def compute_file_values(file_name, num_of_arms):
    arms = [0] * num_of_arms
    likelihoods = [0] * num_of_arms
    costs = [0] * num_of_arms
    rewards = [0] * num_of_arms
    utils = [0] * num_of_arms
    
    data = pd.read_csv(file_name)
    row = data.iloc[0:num_of_arms]
    
    for i,j in row.iterrows():
       arms[i] = j[0]  # It comes out as integer already
       likelihoods[i] = j[1] # Space-only separated
       costs[i] = j[2] #Comma and space separed
       rewards[i] = j[3] #Space-inly separated
       utils[i] = j[4] # It comes out as float already
       
    # arms = convert_commas(arms, num_of_arms, timesteps)
    # likelihoods = convert_commas(likelihoods, num_of_arms, timesteps)
    # costs = convert_commas(costs, num_of_arms, timesteps)
    # rewards = convert_commas(rewards, num_of_arms, timesteps)
    
    return arms, likelihoods, costs, rewards, utils
    
def compute_file1(file_name, num_of_arms, timesteps):
    nums = [0] * timesteps
    counts = [0] * timesteps
    means = [0] * timesteps
    conf_bounds = [0] * timesteps
    regrets = [0] * timesteps
    rewards = [0] * timesteps
    utils = [0] * timesteps
    
    data = pd.read_csv(file_name)
    row = data.iloc[0:timesteps]
    
    for i,j in row.iterrows():
       nums[i] = j[0]  # It comes out as integer already
       counts[i] = j[1] # Space-only separated
       means[i] = j[2] #Comma and space separed
       conf_bounds[i] = j[3] #Space-inly separated
       regrets[i] = j[4] # It comes out as float already
       rewards[i] = j[6] # It comes out as float already
       utils[i] = j[7] # Space-only separated
    counts = convert_spaces(counts, num_of_arms, timesteps)
    means = convert_spaces(means, num_of_arms, timesteps)
    conf_bounds = convert_spaces(conf_bounds, num_of_arms, timesteps)
    utils = convert_spaces(utils, num_of_arms, timesteps)
    
    return nums, counts, means, conf_bounds, regrets, rewards, utils
    
def convert_spaces(counts_list, num_of_arms, time_steps):
   
   new_count_list = [0] * num_of_arms
   desired_array = [0] * time_steps
   for i in range(time_steps):
     new_count_list = counts_list[i]
     new_count_list = new_count_list[1:len(new_count_list) - 1]
     
     desired_array[i] = [float(numeric_string) for numeric_string in new_count_list.split()]
     
   return desired_array   

def convert_commas(counts_list, num_of_arms, time_steps):

   new_count_list = [0] * num_of_arms
   desired_array = [0] * time_steps
   for i in range(time_steps):
     new_count_list = counts_list[i]
     new_count_list = new_count_list[1:len(new_count_list) - 1]
     
     desired_array[i] = [float(numeric_string) for numeric_string in new_count_list.split(', ')]
     
   return desired_array
    
def compute(file_name):
    
    counts = [0] * 10000
    data = pd.read_csv(file_name)
    row = data.iloc[0:]
    
    for i,j in row.iterrows():
       counts[i] = j[4]
       
       #print(j[0])
    return counts 

def summarise(num_of_arms, begins, alg, isUCB):  
  
  time_steps = 10000
  # for exp_no in range(begins,ends): #Exp Number
  # filename = 'Experiment Results UCB/{} arms/0 {}'.format(num_of_arms, begins, alg)
  filename = 'Experiment Results BUCB/{} arms/0 {}'.format(num_of_arms, alg)
  # print(compute(filename))
  
  # arr  = compute_file(filename, num_of_arms, time_steps)
  # if(isUCB == 1): 
  arr  = compute_file(filename, num_of_arms, time_steps)
  # else:
  # arr  = compute_file1(filename, num_of_arms, time_steps)
  
  timestepsL = arr[0] # single array
  counts = arr[1] # Multi dimesional array
  means = arr[2] # Multi dimesional array
  conf_bounds = arr[3] # Multi dimesional array
  regrets = arr[4] # single array
  # rewards = arr[5]# single array
  utils = arr[5] # Multi dimesional array
 
  for in_trial in range(1,100): # no of repeated trials per experiment
    # filename1 = 'Experiment Results UCB/{} arms/{} {}'.format(num_of_arms, begins, in_trial, alg)
    filename1 = 'Experiment Results BUCB/{} arms/{} {}'.format(num_of_arms, in_trial, alg)
    
    # if(isUCB == 1): 
    arr1  = compute_file(filename1, num_of_arms, time_steps)
    # else:
    # arr1  = compute_file1(filename1, num_of_arms, time_steps)
    
    timestepsL1 = arr1[0] # single array
    counts1 = arr1[1] # Multi dimesional array
    means1 = arr1[2] # Multi dimesional array
    conf_bounds1 = arr1[3] # Multi dimesional array
    regrets1 = arr1[4] # single array
    # rewards1 = arr1[5] # single array
    utils1 = arr1[5] # Multi dimesional array
    
    for i in range(time_steps):
      timestepsL[i] += timestepsL1[i]
      regrets[i] += regrets1[i]
      # rewards[i] += rewards1[i]
      
      for j in range(num_of_arms):
      
        counts[i][j] += counts1[i][j]
        means[i][j] += means1[i][j]
        conf_bounds[i][j] += conf_bounds1[i][j]
        utils[i][j] += utils1[i][j]
        
  for i in range(time_steps):
      timestepsL[i] = timestepsL[i] / 100
      regrets[i] = regrets[i] / 100
      # rewards[i] = rewards[i] / 100
      
      for j in range(num_of_arms):
        counts[i][j] = counts[i][j] / 100
        means[i][j] = means[i][j] / 100
        conf_bounds[i][j] = conf_bounds[i][j] / 100
        utils[i][j] = utils[i][j] / 100
       
  return timestepsL, counts, means, conf_bounds, regrets, utils
  
def summarise_values(num_of_arms):  
  
  
  filename = 'Experiment Results BUCB/{} arms/0 values'.format(num_of_arms)
  # print(compute(filename))
  
  # arr  = compute_file(filename, num_of_arms, time_steps)
  # if(isUCB == 1): 
  arr  = compute_file_values(filename, num_of_arms)
  # else:
  # arr  = compute_file1(filename, num_of_arms, time_steps)
  
  arm = arr[0] # single array
  likelihood = arr[1] # Multi dimesional array
  cost = arr[2] # Multi dimesional array
  reward = arr[3] # Multi dimesional array
  util = arr[4] # single array
 
  for in_trial in range(1,100): # no of repeated trials per experiment
    # filename1 = 'Experiment Results UCB/{} arms/{} {}'.format(num_of_arms, begins, in_trial, alg)
    filename1 = 'Experiment Results BUCB/{} arms/{} values'.format(num_of_arms, in_trial)
    
    # if(isUCB == 1): 
    arr1  = compute_file_values(filename1, num_of_arms)
    # else:
    # arr1  = compute_file1(filename1, num_of_arms, time_steps)
    
    arm1 = arr1[0] # single array
    likelihood1 = arr1[1] # Multi dimesional array
    cost1 = arr1[2] # Multi dimesional array
    reward1 = arr1[3] # Multi dimesional array
    util1 = arr1[4] # single array
    
    arm = np.add(arm, arm1)
    likelihood = np.add(likelihood, likelihood1)
    cost = np.add(cost, cost1)
    reward = np.add(reward, reward1)
    util = np.add(util, util1)
    
    # for i in range(num_of_arms):
    # for j in range(num_of_arms):
    # arm[i][j] += arm1[i][j]
    # likelihood[i][j] += likelihood1[i][j] 
    # cost[i][j]  += cost1[i][j] 
    # reward[i][j]  += reward1[i][j] 
    # util[i][j]  += util1[i][j] 
      
  arm = np.divide(arm, 100)
  likelihood = np.divide(likelihood, 100)
  cost = np.divide(cost, 100)
  reward = np.divide(reward, 100)
  util = np.divide(util, 100)      
  # for i in range(num_of_arms):
  # for j in range(num_of_arms):
  # arm[i][j]  = arm1[i][j]  / 100
  # likelihood[i][j]  = likelihood1[i][j]  / 100
  # cost[i][j]  = cost1[i][j]  / 100
  # reward[i][j]  = reward1[i][j]  / 100
  # util[i][j]  = util1[i][j]  / 100
      
       
  return arm, likelihood, cost, reward, util
  
def compute_std(num_of_arms, begins, alg, isUCB): 

  time_steps = 10000
  # for exp_no in range(begins,ends): #Exp Number
  # filename = 'Experiment Results UCB/{} arms/0 {}'.format(num_of_arms, begins, alg)
  filename = 'Experiment Results BUCB/{} arms/0 {}'.format(num_of_arms, alg)
  # print(compute(filename))
  
  average_file = summarise(num_of_arms, begins, alg, isUCB)
  
  # arr  = compute_file(filename, num_of_arms, time_steps)
  # if(isUCB == 1): 
  arr  = compute_file(filename, num_of_arms, time_steps)
  # else:
  # arr  = compute_file1(filename, num_of_arms, time_steps)
  
  timestepsL = np.subtract(arr[0], average_file[0]) # single array
  counts = np.subtract(arr[1], average_file[1]) # Multi dimesional array
  means = np.subtract(arr[2], average_file[2]) # Multi dimesional array
  conf_bounds = np.subtract(arr[3], average_file[3]) # Multi dimesional array
  regrets = np.subtract(arr[4], average_file[4]) # single array
  # rewards = np.subtract(arr[5], average_file[5]) # single array
  utils = np.subtract(arr[5], average_file[5]) # Multi dimesional array
  
  for i in range(time_steps):
      timestepsL[i] = np.power(abs(timestepsL[i]), 2)
      regrets[i] =  np.power(abs(regrets[i]), 2)
      # rewards[i] =  np.power(abs(rewards[i]), 2)
      
      for j in range(num_of_arms):
      
        counts[i][j] =  np.power(abs(counts[i][j]),2)
        means[i][j] =  np.power(abs(means[i][j]), 2)
        conf_bounds[i][j] =  np.power(abs(conf_bounds[i][j]), 2)
        utils[i][j] =  np.power(abs(utils[i][j]), 2)
 
  for in_trial in range(1,100): # no of repeated trials per experiment
    # filename1 = 'Experiment Results UCB/{} arms/{} {}'.format(num_of_arms, begins, in_trial, alg)
    filename1 = 'Experiment Results BUCB/{} arms/{} {}'.format(num_of_arms, in_trial, alg)
    
    # if(isUCB == 1): 
    arr1  = compute_file(filename1, num_of_arms, time_steps)
    # else:
    # arr1  = compute_file1(filename1, num_of_arms, time_steps)
    
    timestepsL1 = np.subtract(arr1[0], average_file[0]) # single array
    counts1 = np.subtract(arr1[1], average_file[1]) # Multi dimesional array
    means1 = np.subtract(arr1[2], average_file[2]) # Multi dimesional array
    conf_bounds1 = np.subtract(arr1[3], average_file[3]) # Multi dimesional array
    regrets1 = np.subtract(arr1[4], average_file[4]) # single array
    # rewards1 = np.subtract(arr1[5], average_file[5]) # single array
    utils1 = np.subtract(arr1[5], average_file[5]) # Multi dimesional array
    
    for i in range(time_steps):
      timestepsL[i] += np.power(abs(timestepsL1[i]), 2)
      regrets[i] +=  np.power(abs(regrets1[i]), 2)
      # rewards[i] +=  np.power(abs(rewards1[i]), 2)
      
      for j in range(num_of_arms):
      
        counts[i][j] +=  np.power(abs(counts1[i][j]),2)
        means[i][j] +=  np.power(abs(means1[i][j]), 2)
        conf_bounds[i][j] +=  np.power(abs(conf_bounds1[i][j]), 2)
        utils[i][j] +=  np.power(abs(utils1[i][j]), 2)
        
  for i in range(time_steps):
      timestepsL[i] = np.sqrt(timestepsL[i] / 100)
      regrets[i] = np.sqrt(regrets[i] / 100)
      # rewards[i] = np.sqrt(rewards[i] / 100)
      
      for j in range(num_of_arms):
        counts[i][j] = np.sqrt(counts[i][j] / 100)
        means[i][j] = np.sqrt(means[i][j] / 100)
        conf_bounds[i][j] = np.sqrt(conf_bounds[i][j] / 100)
        utils[i][j] = np.sqrt(utils[i][j] / 100)
       
  return timestepsL, counts, means, conf_bounds, regrets, utils
  
def compute_std_values(num_of_arms): 


  filename = 'Experiment Results BUCB/{} arms/0 values'.format(num_of_arms)
  # print(compute(filename))
  
  average_file = summarise_values(num_of_arms)
  
  # arr  = compute_file(filename, num_of_arms, time_steps)
  # if(isUCB == 1): 
  arr  = compute_file_values(filename, num_of_arms)
  # else:
  # arr  = compute_file1(filename, num_of_arms, time_steps)
  
  arm = np.subtract(arr[0], average_file[0])  # single array
  likelihood = np.subtract(arr[1], average_file[1]) # Multi dimesional array
  cost = np.subtract(arr[2], average_file[2]) # Multi dimesional array
  reward = np.subtract(arr[3], average_file[3]) # Multi dimesional array
  util = np.subtract(arr[4], average_file[4]) # single array
  
  arm = np.power(abs(arm), 2)
  likelihood =  np.power(abs(likelihood), 2)
  cost =  np.power(abs(cost), 2)
  reward =  np.power(abs(reward), 2)
  util =  np.power(abs(util), 2)
  
  # for i in range(num_of_arms):
  # for j in range(num_of_arms):
  # arm[i][j] = np.power(abs(arm[i][j]), 2)
  # likelihood[i][j] =  np.power(abs(likelihood[i][j]), 2)
  # cost[i][j] =  np.power(abs(cost[i][j]), 2)
  # reward[i][j] =  np.power(abs(reward[i][j]), 2)
  # util[i][j] =  np.power(abs(util[i][j]), 2)
      
  for in_trial in range(1,num_of_arms): # no of repeated trials per experiment
    # filename1 = 'Experiment Results UCB/{} arms/{} {}'.format(num_of_arms, begins, in_trial, alg)
    filename1 = 'Experiment Results BUCB/{} arms/{} values'.format(num_of_arms, in_trial)
    
    # if(isUCB == 1): 
    arr1  = compute_file_values(filename1, num_of_arms)
    # else:
    # arr1  = compute_file1(filename1, num_of_arms, time_steps)
    
    arm1 = np.subtract(arr1[0], average_file[0]) # single array
    likelihood1 = np.subtract(arr1[1], average_file[1]) # Multi dimesional array
    cost1 = np.subtract(arr1[2], average_file[2]) # Multi dimesional array
    reward1 = np.subtract(arr1[3], average_file[3]) # Multi dimesional array
    util1 = np.subtract(arr1[4], average_file[4]) # single array
    
    arm += np.power(abs(arm1), 2)
    likelihood +=  np.power(abs(likelihood1), 2)
    cost +=  np.power(abs(cost1), 2)
    reward +=  np.power(abs(reward1), 2)
    util +=  np.power(abs(util1), 2)
    
    # for i in range(num_of_arms):
    # for j in range(num_of_arms):
    # arm[i][j] += np.power(abs(arm1[i][j]), 2)
    # likelihood[i][j] +=  np.power(abs(likelihood1[i][j]), 2)
    # cost[i][j] +=  np.power(abs(cost1[i][j]), 2)
    # reward[i][j] +=  np.power(abs(reward1[i][j]), 2)
    # util[i][j] +=  np.power(abs(util1[i][j]), 2)
      
      
  arm = np.sqrt(np.divide(arm, 100))
  likelihood = np.sqrt(np.divide(likelihood, 100)) 
  cost = np.sqrt(np.divide(cost, 100))
  reward = np.sqrt(np.divide(reward, 100))
  util = np.sqrt(np.divide(util, 100))      
  # for i in range(num_of_arms):
  # arm[i] = np.sqrt(np.divide(arm[i], 100))
  # likelihood[i] = np.sqrt(np.divide(likelihood[i], 100)) 
  # cost[i] = np.sqrt(np.divide(cost[i], 100))
  # reward[i] = np.sqrt(np.divide(reward[i], 100))
  # util[i] = np.sqrt(np.divide(util[i], 100))
      
       
  return arm, likelihood, cost, reward, util
  
  
def write_to_file(num_of_arms, alg, isUCB):
  counts_header = [0] * num_of_arms
  meanss = [0] * num_of_arms
  conf_bounds = [0] * num_of_arms
  UCB_utilss = [0] * num_of_arms
  
  arm = [0] * num_of_arms
  likelihood = [0] * num_of_arms
  cost = [0] * num_of_arms
  reward = [0] * num_of_arms
  util = [0] * num_of_arms
  

  for j in range(num_of_arms):
    counts_header[j] = 'Count {}'.format(j)
    meanss[j] = 'Mean {}'.format(j)
    conf_bounds[j] = 'Conf Bound {}'.format(j)
    UCB_utilss[j] = 'UCB Utility {}'.format(j)
    
    # arm[i] = 'Arm {}'.format(j)
    # likelihood[i] = 'Likelihood {}'.format(j)
    # reward[i] = 'Reward {}'.format(j)
    # cost[i] = 'Cost {}'.format(j)
    # util[i] = 'Util {}'.format(j)
    
    
          
  file_header = ['Time steps', counts_header, meanss, conf_bounds, 'Regret',  UCB_utilss]
  
  file_header_values = ['arm', 'likelihood', 'cost', 'reward', 'util']
  
  line = [0, [0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms, 0 , [0] * num_of_arms]
  line1 = [0, [0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms, 0 , [0] * num_of_arms]
  
  # values = [[0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms]
  # values1 = [[0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms, [0] * num_of_arms]
  values = [0,0,0,0,0]
  values1 = [0,0,0,0,0]

  
  begins = 0
  ends = 100
  
  # for exp in range(begins, ends):
  # data = summarise(num_of_arms, exp, alg, isUCB) 
  data = summarise(num_of_arms, 0, alg, isUCB) 
  df = pd.DataFrame(columns = file_header)
  
  data1 = compute_std(num_of_arms, 0, alg, isUCB) 
  df1 = pd.DataFrame(columns = file_header)
  
  val = summarise_values(num_of_arms)
  df2 = pd.DataFrame(columns = file_header_values)
  val1 = compute_std_values(num_of_arms)
  df3 = pd.DataFrame(columns = file_header_values)
  
  for i in range(num_of_arms):
    values = [val[0][i],val[1][i], val[2][i], val[3][i], val[4][i]]
    df2.loc[len(df2)] = values
    values1 = [val1[0][i],val1[1][i], val1[2][i], val1[3][i], val1[4][i]]
    df3.loc[len(df3)] = values
    
  for i in range(10000):
    line  = [data[0][i], data[1][i], data[2][i], data[3][i], data[4][i], data[5][i]]
    df.loc[len(df)] = line
    
    line1  = [data1[0][i], data1[1][i], data1[2][i], data1[3][i], data1[4][i], data1[5][i]]
    df1.loc[len(df1)] = line1
  print(data)
  
  df.to_csv('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, alg), header = file_header)
  df1.to_csv('Experiment Results BUCB/{} arms/{}std {}'.format(num_of_arms, num_of_arms, alg), header = file_header)
  df2.to_csv('Experiment Results BUCB/{} arms/{}average values'.format(num_of_arms,num_of_arms), header = file_header_values)
  df3.to_csv('Experiment Results BUCB/{} arms/{}std values'.format(num_of_arms,num_of_arms), header = file_header_values)

num_of_arms = 100
process1 = multiprocessing.Process(target=write_to_file, args=[num_of_arms, 'BUCB', 0])
process2 = multiprocessing.Process(target=write_to_file, args=[num_of_arms, 'BUCB-M', 0])
process3 = multiprocessing.Process(target=write_to_file, args=[num_of_arms, 'BUCB-MV', 0])  

if __name__ == '__main__':

  process1.start()
  time.sleep(10)
  process2.start()
  time.sleep(10)
  process3.start()
  
  
  # a =  compute_file('Experiment Results UCB/2 arms/0 UCB', 2, 10000)
  # print(a[0])
  # print(a[1][2][1])
  # print(np.subtract(a[0], 100))
  
  # b = summarise(2, 0, 'UCB', 0)
  
  # print(np.subtract(a[0],b[0]))
  
  # c = compute_std(2, 0, 'UCB', 0)
  # print(c)
  
  # a = compute_file_values('Experiment Results/2 arms1/0 values', 2)
  # print(a)
  