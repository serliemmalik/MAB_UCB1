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


def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(bottom=0.4, wspace=0.4)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(133)
    ax3 = fig.add_subplot(132)

    # Sub.fig. 1: Regrets in time.
    for i,s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])
        ax2.plot(range(b.n), s.estimated_probas, 'x', markeredgewidth=2, label=solver_names[i])
        
        # ax2.fill_between(range(b.n), s.get_upper_bound, s.get_lower_bound, alpha=0.1)
        # ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='solid', lw=2)

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(bbox_to_anchor=(0.8, -0.25),loc='upper left', borderaxespad=0.)#loc=9, bbox_to_anchor=(1.82, -0.25), ncol=10)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    
    ax2.plot(range(b.n), b.util, 'k--', markersize=12, label='True Utility')    
    #for i,s in enumerate(solvers):
    #    ax2.plot(range(b.n), [s.esti /mated_probas[x] for x in range(b.n)], 'x', markeredgewidth=2, label=solver_names[i])
    #    ax2.fill_between(range(b.n), [(s.estimated_probas[x] - s.confidence_interval[x]) for x in range(b.n)], [(s.estimated_probas[x] + s.confidence_interval[x]) for x in range(b.n)], alpha=0.1)
    ax2.set_xlabel('Actions') #sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.legend(bbox_to_anchor=(0.0, -0.25),loc='upper left', borderaxespad=0.)
    ax2.grid('k', ls='--', alpha=0.3)
    
    # Sub.fig. 3: Action counts
    #for s in solvers:
    #    ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='solid', lw=2)
    # ax3.set_xlabel('Actions')
    # ax3.set_ylabel('Frac. # trials')
    # ax3.grid('k', ls='--', alpha=0.3)
    
    plt.savefig(figname)
    plt.close('all')
    
def readfile(file_name, num_of_arms, timesteps):
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
    
def read_value(file_name, num_of_arms):
    arms = [0] * num_of_arms
    likelihoods = [0] * num_of_arms
    costs = [0] * num_of_arms
    rewards = [0] * num_of_arms
    utils = [0] * num_of_arms
    
    
    data = pd.read_csv(file_name)
    row = data.iloc[0:num_of_arms]
    
    for i,j in row.iterrows():
       arms[i] = j[1]  # It comes out as integer already
       likelihoods[i] = j[2] # Space-only separated
       costs[i] = j[3] #Comma and space separed
       rewards[i] = j[4] #Space-inly separated
       utils[i] = j[5] # It comes out as float already
       
    # arms = convert_commas(arms, num_of_arms, timesteps)
    # likelihoods = convert_commas(likelihoods, num_of_arms, timesteps)
    # costs = convert_commas(costs, num_of_arms, timesteps)
    # rewards = convert_commas(rewards, num_of_arms, timesteps)
    
    return arms, likelihoods, costs, rewards, utils
    
def convert_commas(counts_list, num_of_arms, time_steps):

   new_count_list = [0] * num_of_arms
   desired_array = [0] * time_steps
   for i in range(time_steps):
     new_count_list = counts_list[i]
     new_count_list = new_count_list[1:len(new_count_list) - 1]
     
     desired_array[i] = [float(numeric_string) for numeric_string in new_count_list.split(',')]
     
   return desired_array

def plot_regret(ucb, ucb_m, ucb_mv, ucb_std, ucbm_std, ucbmv_std, num_of_arms, timesteps, figname):
   # Data
   x = list(range(timesteps))  # Assuming you have 26 data points

   fig = plt.figure(figsize=(10, 8))
   fig.subplots_adjust(bottom=0.1, wspace=1)

   ax = fig.add_subplot(111)
   # Plotting
   ax.plot(x, ucb, label='BUCB')
   ax.plot(x, ucb_m,   label='BUCB-M')
   # ax.plot(x, ucb_mv,  label='BUCB-MV')
   
   # ax.fill_between(x, np.add(ucb,ucb_std), np.subtract(ucb,ucb_std), alpha = 0.1)
   
   
   # Customize the graph
   plt.xlabel('Timesteps')
   plt.ylabel('Regrets')
   plt.title('{} Arms - BUCB vs BUCB-M vs BUCB-MV'.format(num_of_arms))
   plt.legend()

   # Display the graph
   plt.savefig(figname)
   # plt.show()
   
def plot_estimated_util(num_of_arms, timesteps, algorithms_names, figname):

   ucb = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[0]),num_of_arms,timesteps)
   ucb_m = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[1]),num_of_arms, timesteps)
   ucb_mv = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[2]),num_of_arms, timesteps)
   values = read_value('Experiment Results BUCB/{} arms/{}average values'.format(num_of_arms,num_of_arms), num_of_arms)
   values_std = read_value('Experiment Results BUCB/{} arms/{}std values'.format(num_of_arms,num_of_arms), num_of_arms)
   
   # Data
   x = list(range(timesteps))  # Assuming you have 26 data points

   fig = plt.figure(figsize=(10, 8))
   fig.subplots_adjust(bottom=0.1, wspace=1)

   ax = fig.add_subplot(111)
   
   estimated_utils = [ucb[2][timesteps - 1], ucb_m[2][timesteps - 1], ucb_mv[2][timesteps - 1]]
   # Plotting
   # ax.plot(x, ucb, label='UCB')
   # # ax.fill_between(x, np.add(ucb,ucb_std), np.subtract(ucb,ucb_std), alpha = 0.1)
   # ax.plot(x, ucb_m, label='UCB-M')
   # ax.plot(x, ucb_mv, label='UCB-MV')
   ax.plot(range(num_of_arms), estimated_utils[0], '--', markeredgewidth=2, label=algorithms_names[0])
   ax.plot(range(num_of_arms), estimated_utils[1], 'x', markeredgewidth=2, label=algorithms_names[1])
   # ax.plot(range(num_of_arms), estimated_utils[2], 'x', markeredgewidth=2, label=algorithms_names[2])
   ax.plot(range(num_of_arms), values[4], 'k--', markersize=12, label='True Utility') 
   ax.fill_between(range(num_of_arms), np.add(values[4], values_std[4]), np.subtract(values[4], values_std[4]), alpha=0.1)
   

   # Customize the graph
   plt.xlabel('Arms')
   plt.ylabel('Estimated Utility')
   plt.title('{} Arms - BUCB vs BUCB-M vs BUCB-MV'.format(num_of_arms))
   plt.legend()

   # Display the graph
   plt.savefig(figname)
   
def plot_selection_counts(num_of_arms, timesteps, baseline_algorithm, algorithms_names, figname):

   ucb = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, baseline_algorithm),num_of_arms,timesteps)
   ucb_m = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[0]),num_of_arms, timesteps)
   ucb_mv = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[1]),num_of_arms, timesteps)
   values = read_value('Experiment Results BUCB/{} arms/{}average values'.format(num_of_arms,num_of_arms), num_of_arms)
   values_std = read_value('Experiment Results BUCB/{} arms/{}std values'.format(num_of_arms,num_of_arms), num_of_arms)
   
   # Data
   x = list(range(timesteps))  # Assuming you have 26 data points

   fig = plt.figure(figsize=(10, 8))
   fig.subplots_adjust(bottom=0.1, wspace=1)

   ax = fig.add_subplot()
   
   counts= [ucb_m[1][timesteps - 1], ucb_mv[1][timesteps - 1]]
   
   # Plotting
   ax.plot(range(num_of_arms), ucb[1][timesteps - 1], '--', label='BUCB')
   ax.plot(range(num_of_arms), np.array(counts[0]), 'x', label=algorithms_names[0])
   ax.plot(range(num_of_arms), np.array(counts[1]), 'x', label=algorithms_names[1])
   

   # Customize the graph
   plt.xlabel('Arms')
   plt.ylabel('Selected Times')
   plt.title('{} Arms - BUCB vs BUCB-M vs BUCB-MV'.format(num_of_arms))
   
   plt.legend()

   # Display the graph
   plt.savefig(figname)
   
   
def plot_multiple_regrets(num_of_arms, timesteps, algorithms_names):
   ucb = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[0]),num_of_arms,timesteps)
   ucb_m = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[1]),num_of_arms, timesteps)
   ucb_mv = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[2]),num_of_arms, timesteps)
   
   
   ucb_std = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[0]),num_of_arms,timesteps)
   ucbm_std = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[1]),num_of_arms, timesteps)
   ucbmv_std = readfile('Experiment Results BUCB/{} arms/{}average {}'.format(num_of_arms, num_of_arms, algorithms_names[2]),num_of_arms, timesteps)
   
   plot_regret(ucb[4], ucb_m[4], ucb_mv[4], ucb_std[4], ucbm_std[4], ucbmv_std[4], num_of_arms, timesteps, 'Experiment Results BUCB/{} arms/{} arms regret'.format(num_of_arms, num_of_arms))
   
   # print([ucb[5][timesteps - 1], ucb_m[5][timesteps - 1], ucb_mv[5][timesteps - 1]])


if __name__ == '__main__':
   
   num_of_arms = 100
   timesteps = 10000
   baseline_algorithm = 'BUCB'
   algorithms_name = ['BUCB-M', 'BUCB-MV']
   algorithms_names = ['BUCB', 'BUCB-M', 'BUCB-MV']
   
   plot_estimated_util(num_of_arms, timesteps, algorithms_names, 'Experiment Results BUCB/{} arms/{} arms estimates'.format(num_of_arms, num_of_arms))
   plot_selection_counts(num_of_arms, timesteps, baseline_algorithm, algorithms_name, 'Experiment Results BUCB/{} arms/{} arms counts'.format(num_of_arms, num_of_arms))
   plot_multiple_regrets(num_of_arms, timesteps, algorithms_names)
   # print(read_value('Experiment Results/{} arms1/{}average values'.format(num_of_arms,num_of_arms), num_of_arms))
   
     