# ENMI testing on real images

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing as mp
from numpy import linalg as LA
from scipy.stats import norm
import scipy.ndimage.filters as fltrs
import math

#---------- FUNCTIONS ----------------------------------------------------------------------

# two dimensional arrays & how many bins
def enhanced_NMI(observation, candidate, bins, var_per_row, cols_per_row):
    if 256 % bins != 0:
        print('256 must be divisible by the number of bins')
        return -1
    if observation.shape != candidate.shape:
        print('Arrays must be same shape for NMI')
        return -1
    if observation.ndim > 1:
        observation = observation.ravel()
    if candidate.ndim > 1:
        candidate = candidate.ravel()
    # x (cols) is observation, y (rows) is candidate
    hist_mat = np.zeros((bins, bins))
    # 2 std devs +/- mean is 95% of gaussian
    position = 0
    for row in range(cols_per_row.size):
        sigma = var_per_row[row]**0.5
        vals_per_bin = 256 / bins
        num_vals = math.ceil(2 * sigma) * 2 + 1 # get number of values to convolve over for 95% of gaussian
        half_of_vals = int(num_vals / 2)
        delta_func = np.zeros(num_vals)
        delta_func[half_of_vals] = 1
        val_conv = fltrs.gaussian_filter(delta_func.astype(np.float), sigma)
        for col in range(cols_per_row[row]):
            col_location = int(observation[position] / vals_per_bin)
            row_location = int(candidate[position] / vals_per_bin)
            high_val = observation[position] + half_of_vals
            low_val = observation[position] - half_of_vals
            high_bin = int(high_val / vals_per_bin)
            low_bin = int(low_val / vals_per_bin)
            num_bins = high_bin - low_bin + 1
            if num_bins == 1:
                hist_mat[row_location, col_location] += 1
            else:
                for i in range(val_conv.size):
                    val = low_val + i
                    bin_index = int(val / vals_per_bin)
                    if (bin_index < 0):
                        hist_mat[row_location, 0] += val_conv[i]
                    elif (bin_index >= hist_mat.shape[1]):
                        hist_mat[row_location, hist_mat.shape[1] - 1] += val_conv[i]
                    else:
                        hist_mat[row_location, bin_index] += val_conv[i]
            position += 1
    return hist_NMI(hist_mat)

# two dimensional arrays & how many bins
def NMI(array1, array2, bins):
    if array1.shape != array2.shape:
        print('Arrays must be same shape for NMI')
        return -1
    hist_2d, x_edges, y_edges = np.histogram2d(
        array1.ravel(),
        array2.ravel(),
        range=[[0, 255], [0, 255]],
        bins=bins)
    return hist_NMI(hist_2d)

def hist_NMI(array):
    # calc probs
    prob_xy = array / float(np.sum(array))
    prob_x = np.sum(prob_xy, axis=1)
    prob_y = np.sum(prob_xy, axis=0)
    prob_x_prob_y = prob_x[:, None] * prob_y[None, :] # prob(x)*prob(y)
    # we want non-zero values
    nzrs_xy = prob_xy > 0
    nzrs_x = prob_x > 0
    nzrs_y = prob_y > 0
    entropy_x = -np.sum(prob_x[nzrs_x] * np.log(prob_x[nzrs_x]))
    entropy_y = -np.sum(prob_y[nzrs_y] * np.log(prob_y[nzrs_y]))
    entropy_xy = -np.sum(prob_xy[nzrs_xy] * np.log(prob_xy[nzrs_xy]))
    return (entropy_x + entropy_y) / entropy_xy # Sid's way

#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # FOR JOINT DIST
    bins = 32
    
    