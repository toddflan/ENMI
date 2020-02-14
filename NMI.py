# Testing NMI for localization

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA
import sys
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as fltrs
import math

# def norm_diff(mat):
    # if mat.shape[0] != mat.shape[1]:
        # print("Error: Matrix must be square.")
        # return -10
    # diag = np.zeros(mat.shape)
    # for i in range(mat.shape[0]):
        # diag[i, i] = mat[i, i]
    # return LA.norm(np.subtract(mat, diag))

# https://math.stackexchange.com/questions/1392491/measure-of-how-much-diagonal-a-matrix-is
def samp_corr_coeff(array1, array2):
    hist_2d, x_edges, y_edges = np.histogram2d(
        array1.ravel(),
        array2.ravel(),
        bins=256)
    mat = hist_2d.T
    # if mat.shape[0] != mat.shape[1]:
        # print("Error: Matrix must be square.")
        # return -10
    d = mat.shape[0]
    j = np.full(d, 1.0)
    r = np.linspace(1, d, d)
    r_2 = np.square(r)
    n = np.sum(mat)
    sum_x = np.matmul(np.matmul(r, mat), np.transpose(j))
    sum_y = np.matmul(np.matmul(j, mat), np.transpose(r))
    sum_x_2 = np.matmul(np.matmul(r_2, mat), np.transpose(j))
    sum_y_2 = np.matmul(np.matmul(j, mat), np.transpose(r_2))
    sum_xy = np.matmul(np.matmul(r, mat), np.transpose(r))
    return (n * sum_xy - sum_x * sum_y) / (np.sqrt(n * sum_x_2 - sum_x**2) * np.sqrt(n * sum_y_2 - sum_y**2))

def gen_rand_matrix(shape, values):
    mat = np.zeros(shape)
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            if random.randint(0, 1) == 1:
                mat[row, col] = values[0]
            else:
                mat[row, col] = values[1]
    return mat

def normalized_mut_info(array1, array2):
    hist_2d, x_edges, y_edges = np.histogram2d(
        array1.ravel(),
        array2.ravel(),
        # range=[[0, 255], [0, 255]],
        bins=256)
    # show hist
    # plt.imshow(hist_2d.T, origin='lower')
    # plt.show()
    # calc probs
    prob_xy = hist_2d / float(np.sum(hist_2d))
    prob_x = np.sum(prob_xy, axis=1)
    prob_y = np.sum(prob_xy, axis=0)
    prob_x_prob_y = prob_x[:, None] * prob_y[None, :] # prob(x)*prob(y)
    # we want non-zero values
    nzrs_xy = prob_xy > 0
    nzrs_x = prob_x > 0
    nzrs_y = prob_y > 0
    # print(prob_x)
    # print(prob_y)
    # print(prob_xy)
    entropy_x = -np.sum(prob_x[nzrs_x] * np.log(prob_x[nzrs_x]))
    entropy_y = -np.sum(prob_y[nzrs_y] * np.log(prob_y[nzrs_y]))
    entropy_xy = -np.sum(prob_xy[nzrs_xy] * np.log(prob_xy[nzrs_xy]))
    # mut_info = np.sum(prob_xy[nzrs_xy] * np.log(prob_xy[nzrs_xy] / prob_x_prob_y[nzrs_xy]))
    return (entropy_x + entropy_y) / entropy_xy # Sid's way
    # return 2 * mut_info / (entropy_x + entropy_y) # other way

def NMI(array):
    prob_xy = array / float(np.sum(array))
    prob_x = np.sum(prob_xy, axis=1)
    prob_y = np.sum(prob_xy, axis=0)
    prob_x_prob_y = prob_x[:, None] * prob_y[None, :] # prob(x)*prob(y)
    # we want non-zero values
    nzrs_xy = prob_xy > 0
    nzrs_x = prob_x > 0
    nzrs_y = prob_y > 0
    # print(prob_x)
    # print(prob_y)
    # print(prob_xy)
    entropy_x = -np.sum(prob_x[nzrs_x] * np.log(prob_x[nzrs_x]))
    entropy_y = -np.sum(prob_y[nzrs_y] * np.log(prob_y[nzrs_y]))
    entropy_xy = -np.sum(prob_xy[nzrs_xy] * np.log(prob_xy[nzrs_xy]))
    # mut_info = np.sum(prob_xy[nzrs_xy] * np.log(prob_xy[nzrs_xy] / prob_x_prob_y[nzrs_xy]))
    return (entropy_x + entropy_y) / entropy_xy # Sid's way
    # return 2 * mut_info / (entropy_x + entropy_y) # other way

def add_noise(array, mean, var, norm=False):
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, array.shape)
    array_wnoise = array + gaussian
    if norm:
        cv2.normalize(array_wnoise, array_wnoise, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    else:
        # clip at 0 and 255
        array_wnoise = np.clip(array_wnoise, 0, 255)
    return array_wnoise.astype(np.uint8)

def add_unequal_noise(array):
    noisy = np.zeros(array.shape)
    for row in range(array.shape[0]):
        cols = array.shape[1]
        sigma = (array.shape[0] - row - 1) * 0.05
        gaussian = np.random.normal(0, sigma, cols)
        noisy[row, :] = array[row, :] + gaussian
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_shift(array, shift):
    shifted = array + float(shift)
    shifted = np.clip(shifted, 0, 255)
    return shifted.astype(np.uint8)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x = np.linspace(-5, 5, 101)
# y = np.linspace(-5, 5, 101)
# x, y = np.meshgrid(x, y)
# mean = 0
# var = 4
# z = (2 * np.pi * var)**-1 * np.exp(-((x - mean)**2 + (y - mean)**2) / (2* var))
# print(np.round(z, 2))
# ax.plot_surface(x, y, z)
# plt.show()
# exit()

# conv = fltrs.gaussian_filter(np.float_([0, 0, 1, 0, 0]), 5)
# print(conv)
# print(np.sum(conv))
# exit()

# other NMI stuff ******************************************

# mat = np.full((32, 32), 0)

# for i in range(mat.shape[0]):
    # mat[i, i] = 15

# mat[0, 10] = 3
# mat[0, 11] = 3
# mat[0, 12] = 3
# mat[0, 13] = 3
# mat[0, 14] = 3

# print(NMI(mat))

# plt.imshow(mat, origin='lower')
# plt.show()

# exit()

# **********************************************************


# ************ gaussian filter 2d hist *********************

sample_mat = np.array([[200, 200, 100, 100],
                       [200, 200, 100, 100],
                       [150, 150, 45, 45],
                       [150, 150, 45, 45]])

# noisy_mat = add_unequal_noise(sample_mat)

# hist_2d, x_edges, y_edges = np.histogram2d(
    # sample_mat.ravel(),
    # noisy_mat.ravel(),
    # bins=32)

bins = 32

img = cv2.imread(r'C:\Users\toddf\Desktop\data_4_5_19\01.jpg', cv2.IMREAD_GRAYSCALE)

noisy_img = add_unequal_noise(img)

plt.imshow(np.hstack((img, noisy_img)))
plt.show()

hist_2d, x_edges, y_edges = np.histogram2d(
    img.ravel(),
    noisy_img.ravel(),
    normed=True,
    bins=bins)

plt.imshow(hist_2d.T, origin='lower')
plt.show()

print('NMI:', NMI(hist_2d))
# print('NMI package:', normalized_mutual_info_score(img.ravel(), noisy_img.ravel()))

# y (rows) is img, x (cols) is noisy_img
hist_mat = np.zeros((bins, bins))

# 2 std devs +/- mean is 95% of gaussian

for i in range(img.shape[0]):
    sigma = (img.shape[0] - i - 1) * 0.1
    num_bins = math.ceil(2 * sigma / (256 / bins)) * 2 + 1
    delta_func = np.zeros(num_bins)
    delta_func[int(num_bins / 2)] = 1
    conv = fltrs.gaussian_filter(delta_func.astype(np.float), sigma)
    # print(conv)
    for j in range(img.shape[1]):
        col_location = int(noisy_img[i, j] / (256 / bins))
        row_location = int(img[i, j] / (256 / bins))
        # print(row_location)
        for k in range(conv.size):
            col_shifted = col_location + k - int(num_bins / 2)
            if (col_shifted < 0): # edge cases
                hist_mat[row_location, 0] += conv[k]
            elif (col_shifted >= hist_mat.shape[1]):
                hist_mat[row_location, hist_mat.shape[1] - 1] += conv[k]
            else:
                hist_mat[row_location, col_shifted] += conv[k]

print('Enhanced NMI:', NMI(hist_mat))

plt.imshow(hist_mat.T, origin='lower')
plt.show()

bin_array = np.linspace(1, bins, bins)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
#ax = plt.subplot(projection='3d')#fig.gca(projection='3d')
x, y = np.meshgrid(bin_array, bin_array)
ax.plot_surface(x, y, hist_2d)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(x, y, hist_mat)

plt.show()

exit()

# **********************************************************

img = cv2.imread(r'C:\Users\toddf\Desktop\data_4_5_19\01.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'C:\Users\toddf\Desktop\data_4_5_19\02.jpg', cv2.IMREAD_GRAYSCALE)

number_of_points = 101
vars = np.linspace(0, 10, number_of_points)

results_NMI = []
results_CC = []

# for variance in vars:
    # for i in range(10):
    # img_noisy = add_noise(img, 0, variance)
    
    # results_NMI.append(normalized_mut_info(img, img_noisy))
    # results_CC.append(samp_corr_coeff(img, img_noisy))
    
    # samples = 1000
    # sum_errors_NMI = 0
    # sum_errors_CC = 0
    
    # for i in range(samples):
        # sample = gen_rand_matrix((5, 5), (-1, 1))
        # sample_noisy = add_noise(sample, 0, variance)
        # alt_sample = gen_rand_matrix((5, 5), (-1, 1))
        
        # if normalized_mut_info(sample_noisy, alt_sample) > normalized_mut_info(sample_noisy, sample):
            # sum_errors_NMI += 1
        
        # if samp_corr_coeff(sample_noisy, alt_sample) > samp_corr_coeff(sample_noisy, sample):
            # sum_errors_CC += 1
    
    # results_NMI.append(sum_errors_NMI / samples)
    # results_CC.append(sum_errors_CC / samples)

# df = pd.DataFrame()

# df['var'] = vars
# df['p_error_NMI'] = results_NMI
# df['p_error_CC'] = results_CC

# df.to_csv('NMI_vs_CC_loc.csv', index = False)

# exit()

img_noisy = add_noise(img, 0, 1)

img_edit = np.zeros(img.shape)
img_edit[30:, :] = img[30:, :]

# plt.imshow(np.hstack((img, img_edit)))
# plt.show()

# plt.scatter(img.ravel(), img_edit.ravel())
# plt.show()

mat = gen_rand_matrix((5, 5), (-1, 1))

test_mat = np.array([[-1, 1, 1, -1, -1],
                    [1, 1, -1, 1, 1],
                    [-1, -1, 1, -1, -1],
                    [-1, 1, -1, -1, 1],
                    [-1, 1, 1, -1, 1]])

test_mat2 = np.array([[0, 0.8, 0.8, -0.8, -0.8],
                    [0.8, 0.8, -0.8, 0.8, 0.8],
                    [-0.8, -0.8, 0.8, -0.8, -0.8],
                    [-0.8, 0.8, -0.8, -0.8, 0.8],
                    [-0.8, 0.8, 0.8, -0.8, 0.8]])

test_mat3 = np.array([[-0.5, 0.8, 0.8, -0.8, -0.8],
                    [0.8, 0.8, -0.8, 0.8, 0.8],
                    [-0.8, -0.8, 0.8, -0.8, -0.8],
                    [-0.8, 0.8, -0.8, -0.8, 0.8],
                    [-0.8, 0.8, 0.8, -0.8, 0.8]])

test_mat4 = np.array([[-1, 1],
                    [-1, 1]])

test_mat5 = np.array([[0, 0.8],
                    [-0.8, 0.8]])

test_mat6 = np.array([[-1, 0.8],
                    [-0.8, 0.8]])

test_mat7 = test_mat + 10
test_mat7[0, 0] = 0

weights = np.full(test_mat.shape, 1.0)
weights[0, :] = 0.5

# print(weights)

# plt.scatter(mat.ravel(), test_mat2.ravel())
# plt.show()

mat_wnoise = np.zeros(mat.shape)
gaussian = np.random.normal(0, 0.4, mat.shape)
mat_wnoise = mat + gaussian

other_mat = gen_rand_matrix((5, 5), (-1, 1))
# other_mat_wnoise = np.zeros(other_mat.shape)
# gaussian = np.random.normal(0, 0.1, other_mat.shape)
# other_mat_wnoise = other_mat + gaussian

# print(mat_wnoise)

# print(test_mat.ravel())

# move img 15 pixels down
img_moved = np.zeros(img.shape)
img_moved[15:, :] = img[:-15, :]

# show images
# plt.imshow(np.hstack((img, img2)))
# plt.show()

hist_2d, x_edges, y_edges = np.histogram2d(
    test_mat.ravel(),
    test_mat7.ravel(),
    # weights=weights.ravel(),
    # normed=True,
    bins=20)

other_hist_2d, other_x_edges, other_y_edges = np.histogram2d(
    other_mat.ravel(),
    mat_wnoise.ravel(),
    # weights=weights.ravel(),
    # normed=True,
    bins=20)

# score1 = normalized_mut_info(img, img_noisy)
# score2 = normalized_mut_info(other_mat, mat_wnoise)
if len(sys.argv) > 1:
    if sys.argv[1] == 'CC':
        # score1 = samp_corr_coeff(hist_2d.T)
        score2 = samp_corr_coeff(hist_2d.T)
# print(score1)
# print(score2)

# print(hist_2d.T)

array = np.linspace(0, 100, 1001)
array2 = array + 100
array2[0] = 250
print(normalized_mut_info(img.ravel(), img2.ravel()))
print(normalized_mutual_info_score(img.ravel(), img2.ravel()))
# clipped_img = np.clip(img)
# print(normalized_mut_info(img, add_shift(img, 40)))

# plt.imshow(hist_2d.T, origin='lower')
# plt.show()

# plt.imshow(other_hist_2d.T, origin='lower')
# plt.show()

# hist_2d, x_edges, y_edges = np.histogram2d(
    # test_mat4.ravel(),
    # test_mat6.ravel(),
    # weights=weights.ravel(),
    # normed=True,
    # bins=20)

# score = normalized_mut_info(test_mat4, test_mat6)
# print(score)

# plt.imshow(hist_2d.T, origin='lower')
# plt.show()

# plt.hist(img.ravel(), bins=64)
# plt.show()

# show the image
# cv2.imshow("Image 08", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()