# NMI localization testing with simulated images

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

# test against random candidate ***0 for success, 1 for failure; (NMI, ENMI)***
def test_location(size, bins, variance, cols_per_row):
    # generate vector
    sample = gen_rand_vect(size) # u*
    
    # add noise
    observation = add_noise(sample, variance, cols_per_row)
    
    # get true result of NMI
    true_value_ENMI = enhanced_NMI(observation, sample, bins, variance, cols_per_row)
    true_value_NMI = NMI(observation, sample, bins)
    
    # generate random candidate
    candid = gen_rand_vect(size) # u_hat
    
    # perform NMI
    candid_value_ENMI = enhanced_NMI(observation, candid, bins, variance, cols_per_row)
    candid_value_NMI = NMI(observation, candid, bins)
    
    return (candid_value_NMI >= true_value_NMI, candid_value_ENMI >= true_value_ENMI)

def add_noise(array, var_per_row, cols_per_row):
    noisy_array = np.zeros(array.size)
    position = 0
    for row in range(cols_per_row.size):
        noise = np.random.normal(0, var_per_row[row]**0.5, cols_per_row[row]) # (mean, var, array shape)
        # noisy_array[position:position + cols_per_row[row]] = array[position:position + cols_per_row[row]] + noise
        # position += cols_per_row[row]
        for col in range(cols_per_row[row]):
            noisy_array[position] = array[position] + noise[col]
            position += 1
    return noisy_array.astype(np.uint8)

# generate random vector (values from 64 - 192 to prevent clipping)
def gen_rand_vect(length):
    # sample = np.zeros(length)
    mean = 256 / 2
    sigma = (mean - 64) / 2 # 95% of gaussian between 64 & 192
    sample = np.random.normal(mean, sigma, length)
    # for i in range(length):
        # sample[i] = random.randint(64, 192)
        # if random.randint(0, 1) == 1:
        # if random.random() < 0.5: # change prob of distinct features (0.5 is original evaluation)
            # sample[i] = values[1]
    return sample.astype(np.uint8)

#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    #---------- INPUT ------------------------------------------------------------------------

    # parameters of camera configuration ***calculated to get simple observation shape & look similar to rover images***
    camera_height = 58.3095 # cm
    camera_angle = np.deg2rad(35.9020) # below horizon line (parallel to flat surface)
    vertical_FOV = np.deg2rad(39.2962)
    horizontal_FOV = np.deg2rad(70.5288)
    focal_length = 0.0367 # cm (of the webcam on the rover)

    # parameters for squares
    side_length = 20 # cm
    
    # FOR JOINT DIST
    bins = 32
    
    # number of cores
    cores = mp.cpu_count()
    # cores = 8
    
    #-----------------------------------------------------------------------------------------
    #----------- CALCULATING PARAMETERS ------------------------------------------------------

    # distance along floor to bottom & top of sample
    dist_floor_sampleBot = camera_height / np.tan(camera_angle + vertical_FOV / 2)
    dist_floor_sampleTop = camera_height / np.tan(camera_angle - vertical_FOV / 2)
    
    # get dimensions of sample and calculate z values
    sample_rows = int(round(dist_floor_sampleTop - dist_floor_sampleBot) / side_length)
    sample_cols_per_row = np.zeros(sample_rows)
    z_values = np.zeros(sample_rows + 1)
    
    for i in range(z_values.shape[0]):
        dist_floor = dist_floor_sampleBot + i * side_length
        dist = (camera_height**2 + dist_floor**2)**0.5 # distance from camera to edge of square (hypotenuse of triangle)
        angle_above_zaxis = camera_angle - np.arctan(camera_height / dist_floor)
        z_values[i] = dist * np.cos(angle_above_zaxis)
        if i < sample_cols_per_row.shape[0]: # all but last iteration
            sample_cols_per_row[i] = int(round(2 * dist * np.tan(horizontal_FOV / 2)) / side_length)
            if sample_cols_per_row[i] % 2 == 0:
                sample_cols_per_row[i] -= 1 # make sure we have a center row of odd # columns
    sample_cols_per_row = sample_cols_per_row.astype(int)
    num_squares = np.sum(sample_cols_per_row)
    
    print(z_values)
    
    # get weights vector (Gramian matrix values) & z bounds vector
    gramian = np.zeros(num_squares)
    # variance = np.zeros(num_squares)
    z_bounds = np.zeros(sample_rows)
    position = 0
    for row in range(sample_rows):
            gramian_for_row = 0.5 * side_length * (z_values[row]**-2 - z_values[row + 1]**-2)
            # variance[row] = var_wo_z / (z_values[row]**-2 - z_values[row + 1]**-2)
            z_bounds[row] = 1 / (z_values[row]**-2 - z_values[row + 1]**-2)
            for i in range(sample_cols_per_row[row]):
                gramian[position] = gramian_for_row #0.5 * side_length * (z_values[row]**-2 - z_values[row + 1]**-2)
                # variance[position] = var
                position += 1
    
    print(z_bounds)
    exit()
    
    #----------------------------------------------------------------------------------------------
    #-------------- PERFORMING SIMULATION ---------------------------------------------------------
    
    N_0 = np.linspace(1, 1.1, 2)
    
    # arrays to store probabilities of error
    probs_NMI = np.zeros(N_0.size)
    probs_ENMI = np.zeros(N_0.size)
    
    start_time = time.time()

    for iter in range(probs_NMI.size):
        old_time = time.time()
        
        print((iter + 1), "/", probs_NMI.size, end=' ', flush=True)
        
        var_wo_z = (2 * N_0[iter] * np.cos(camera_angle)) / (focal_length**2 * camera_height * side_length)
        var_array = var_wo_z * z_bounds
        
        # run a bunch of times
        num_of_samples = 1000
        
        # list for storing results from pools
        result_list = []
        
        # make pools for multiprocessing
        pool = mp.Pool(cores)
        
        for i in range(num_of_samples):
            # test candidate locations
            result_list.append(pool.apply_async(func=test_location, args=(num_squares, bins, var_array, sample_cols_per_row, )))
        
        pool.close()
        pool.join()
        
        sum_errors_NMI = 0
        sum_errors_ENMI = 0
        for i in range(num_of_samples):
            result = result_list[i].get()
            sum_errors_NMI += result[0]
            sum_errors_ENMI += result[1]
        
        # get probabilities of error
        probs_NMI[iter] = sum_errors_NMI / num_of_samples
        probs_ENMI[iter] = sum_errors_ENMI / num_of_samples
        
        print(" done.", time.time() - old_time, "sec")

    print("total time elapsed:", time.time() - start_time, "sec")
    
    df = pd.DataFrame()
    
    # make df
    df['N_0'] = N_0
    df['NMI'] = probs_NMI
    df['ENMI'] = probs_ENMI
    
    # save to csv
    # df.to_csv('NMI_results_001_002.csv', index=False)
    
    # plot if needed
    plt.plot(N_0, probs_NMI, N_0, probs_ENMI)
    plt.legend(['NMI', 'ENMI'])
    plt.title('Prob. Error')
    plt.show()