import numpy as np
import torch
import random
from statistics import NormalDist
import pickle
import sys
import os
import time
import cProfile, pstats
from data_transforms import * 
from fi_config import *
 
def get_error_map(max_lvls_cell, ber=None):
    """
    Retrieve the correct per-storage-cell error map for the configured NVM settings according to the maximum levels-per-cell used

    :param max_lvls_cell: Across the storage settings for fault injection experiment, provide the maximum number of levels-per-cell required (max 16 for 4BPC for provided fault models)
    """
    # max_lvls_cell is the maximum number of levels encoded in an NVM cell
    nvm_data_path = os.path.dirname(__file__)
    print("Using NVM model " + nvm_model)
    # print(max_lvls_cell, "possible different states")
    f = open(nvm_data_path + '/nvm_data/' + nvm_dict[nvm_model], 'rb')
    args_lut = pickle.load(f)
    print(args_lut)

    # computes the probability of level faults (either up or down one level)
    # for each possible cell configuration in fixed point representation (up to max_lvls_cell levels)

    emap_entries = int(np.log2(max_lvls_cell))
    # emap_entires is 1 if 2 states, 2 if 4 states, 3 if 8 states, etc.
    error_map = np.zeros(emap_entries, dtype=object)
    # if 2 states, then emap_entries = 1, so error_map = [0]
    # if 4 states, then emap_entries = 2, so error_map = [[0,0],[0,0]]
    if len(args_lut) < emap_entries:
        raise SystemExit("ERROR: model does not support " + str(emap_entries) + "-bit cells")
    if ber == None:
        for i in range(emap_entries):
            # create a list so for each level we get a probability of a fault shifting a level up or down one.
            num_levels = int(2 ** (i + 1))
            error_map[i] = np.zeros((num_levels, 2))
            # always solve Gauss to be sure, otherwise # of faults is overestimated
            for j in range(num_levels - 1):
                th = get_temp_th(args_lut[i], j)
                print("Input to Threshold Function", args_lut[i], j)
                print("Threshold", th)
                dist_args = (th, *args_lut[i][j + 1])
                print("Args", dist_args)
                error_map[i][j + 1, 0] = fault_rate_gen(dist_args)
                dist_args = (th, *args_lut[i][j])
                print("Args", dist_args)
                error_map[i][j, 1] = 1. - fault_rate_gen(dist_args)
    else:
        for i in range(emap_entries):
            num_levels = int(2 ** (i + 1))
            error_map[i] = np.zeros((num_levels, 2))
            for j in range(num_levels - 1):
                error_map[i][j + 1, 0] = ber
                error_map[i][j, 1] = ber
    if Debug:
        for i, emap in enumerate(error_map):
            print("Error map for", int(2 ** (i + 1)), "levels")
            print(emap, "\n\n")
    return error_map

 

def fault_rate_gen(dist_args):
  """
  Randomly generate fault rate per experiment and storage cell config according to fault model

  :param dist_args: arguments describing the distribution of level-to-level faults (programmed level means and sdevs)
  """
  try:
    x = dist_args[0]
    mu = dist_args[1]
    sigma = dist_args[2]
    cdf = NormalDist(mu, sigma).cdf(x) 
    print("Fault rate is :", cdf)
  except:
    raise SystemExit("ERROR: model not defined; please update fi_config.py")

  return cdf
  
# Use this when std dev btwn levels are not even
def solveGauss(mu1, sdev1, mu2, sdev2):
  """
  Helper function to compute intersection of two normal distributions; used to calculate probability of level-to-level fault for specific current/voltage distributions

  :param mu1: mean of first distribution
  :param mu2: mean of second distribution
  :param sdev1: standard dev of first distribution
  :param sdev2: standard dev of second distribution
  """
  a = 1./(2*sdev1**2) - 1./(2*sdev2**2)
  b = mu2/(sdev2**2) - mu1/(sdev1**2)
  c = mu1**2/(2*sdev1**2) -  mu2**2/(2*sdev2**2) - np.log(sdev2/sdev1)
  # print(f"The intersection of ({mu1},{sdev1}) and ({mu2},{sdev2})  is {np.roots([a,b,c])}")
  return np.roots([a, b, c])

def get_temp_th(dist_args, lvl):
  """
  Helper function to compute threshold for detecting a mis-read storage cell according to input fault model and stored value

  :param dist_args: arguments describing the distribution of level-to-level faults
  :param lvl: programmed value to specific memory cell (e.g., 0 or 1 for SLC)
  """
  try:
    print("inputs for solveGauss: ", dist_args[lvl][0], dist_args[lvl][1], dist_args[lvl+1][0], dist_args[lvl+1][1])
    temp_th = solveGauss(dist_args[lvl][0], dist_args[lvl][1], dist_args[lvl+1][0], dist_args[lvl+1][1])
    for temp in temp_th:
      if temp > dist_args[lvl][0] and temp < dist_args[lvl+1][0]:
        th = temp
  except:
    raise SystemExit("ERROR: model not defined; please update fi_config.py")
  return th

def inject_faults(mlc_weights, rep_conf, error_map):
  """
  Perform fault injection on input MLC-packed data values according to storage settings and fault model

  :param mlc_weights: MLC-packed data values giving stored value per memory cell, prepared for fault injection
  :param rep_conf: storage setting dictating bits-per-cell per data value
  :param error_map: generated base fault rates according to storage configs and fault model
  """

  #print("mlc_weights")
  #print(mlc_weights)
  #print("rep_conf")
  #print(rep_conf)

  before_shape = mlc_weights.shape
    
  original_data = mlc_weights.clone()
    
  # perform fault injection
  total_num_faults = 0

  for cell in range(np.size(rep_conf)):
    max_level = rep_conf[cell] - 1
    
    cell_error_map_index = int(np.log2(rep_conf[cell])) - 1
    cell_errors = error_map[cell_error_map_index]
    
    #print("Now looking at ", cell_error_map_index)

    total_num_faults = 0

    # Loop through all possible levels for cell
    for lvl in range(rep_conf[cell]):
      lvl_cell_addresses = np.where(mlc_weights[:, cell].cpu().numpy() == lvl)[0]
      if lvl_cell_addresses != []:
        # Get error probabilities for both up and down transitions
        # the probability of min level going down and max level going up is always 0

        prob_faults_down = cell_errors[lvl][0]
        prob_faults_up = cell_errors[lvl][1]

        # Compute total number of errors for lvl
        num_lvl_faults = int((prob_faults_up+prob_faults_down) * lvl_cell_addresses.size)

        if num_lvl_faults > 0:
        
          print("Fault injected!")

          faulty_lvls_indexes = np.random.choice(lvl_cell_addresses, int(num_lvl_faults), replace=False)
          # divide the total number of faults according to up/down fault ratio
          faulty_middle = int(faulty_lvls_indexes.size * prob_faults_up /(prob_faults_up + prob_faults_down))
          
          if prob_faults_up > 0:
            mlc_weights[faulty_lvls_indexes[:faulty_middle], cell] += 1
          if prob_faults_down > 0:
            mlc_weights[faulty_lvls_indexes[faulty_middle:], cell] -= 1 
          
          total_num_faults += num_lvl_faults
  
  if (torch.sum(mlc_weights[:, cell] > max_level) != 0) or (torch.sum(mlc_weights[:, cell] < 0) != 0):
    print("WARNING: Conversion error!")

  print("Number of generated faults::", total_num_faults)

    # Calculate and print the Bit Error Rate (BER)
  num_errors = (mlc_weights != original_data).sum().item()
  total_bits = mlc_weights.numel() 
  ber = num_errors / total_bits
  print("Bit Error Rate (BER)::", ber)
    
  after_shape = mlc_weights.shape
  #print(f"Inject_faults shape before: {before_shape}, after: {after_shape}")
    
  
  return mlc_weights


