import os
import cv2

import matplotlib
from matplotlib.colors import hsv_to_rgb as h2r
from utils import DATA_ROOT_PATH, GENRE_LIST, N_BINS, get_file_list
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from tqdm import tqdm

def distance_to_grayscale(x, y, z):
    k = (x + y + z) / 3.0
    dist = (x-k) * (x-k) + (y-k) * (y-k) + (z-k) * (z-k)
    return sqrt(dist)

def square_distance_to_grayscale(x, y, z):
    k = (x + y + z) / 3.0
    dist = (x-k) * (x-k) + (y-k) * (y-k) + (z-k) * (z-k)
    return dist

DISTANCE_MATRIX = np.zeros((N_BINS, N_BINS, N_BINS))
for i in range(N_BINS):
    for j in range(N_BINS):
        for k in range(N_BINS):
            DISTANCE_MATRIX[i, j, k] = distance_to_grayscale(i, j, k)

SQUARE_DISTANCE_MATRIX = np.zeros((N_BINS, N_BINS, N_BINS))
for i in range(N_BINS):
    for j in range(N_BINS):
        for k in range(N_BINS):
            SQUARE_DISTANCE_MATRIX[i, j, k] = square_distance_to_grayscale(i, j, k)

def total_distance_to_grayscale(color_bins):
    total_dist = 0
    return np.sum(color_bins * DISTANCE_MATRIX)

def total_square_distance_to_grayscale(color_bins):
    total_dist = 0
    return np.sum(color_bins * SQUARE_DISTANCE_MATRIX)

def visualize_3d(title, x_grid, y_grid, z_grid, color_grid, bins):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    ax.scatter(x_grid, y_grid, z_grid, c = color_grid / 256, s = 500 * bins.reshape(-1, 1) / (np.max(bins) + 0.01))
    # ax.scatter(x_grid, y_grid, z_grid, c = color_grid / 256, s = 50)
    plt.show()
    plt.close()

def visualize_2d(title, x_grid, y_grid, z_grid, color_grid_dic, bins, projection_axis):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    color_clone = color_grid_dic[projection_axis]
    bins_clone = np.sum(bins, axis=projection_axis, keepdims=False).reshape(-1, 1)
    plt.scatter(y_grid[:256], z_grid[:256], c = color_clone / 256, s = 500 * bins_clone / (np.max(bins_clone) + 0.01))
    # plt.scatter(y_grid[:256], z_grid[:256], c = color_clone / 256, s = 500)
    plt.show()
    plt.close()

if __name__ == "__main__":
    os.chdir("D:")
    xyz = np.uint8([(i, j, k) for i in range(16) for j in range(16) for k in range(16)])
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    xyz_for_hsv = np.float32(16 * xyz + 8)
    xyz_for_hsv[:, 0] *= 180.0 / 256.0
    bgr_grid = cv2.cvtColor(xyz[:, np.newaxis, :] * 16 + 8, cv2.COLOR_BGR2RGB)[:, 0, :]
    hsv_grid = cv2.cvtColor(np.uint8(xyz_for_hsv[:, np.newaxis, :]), cv2.COLOR_HSV2RGB)[:, 0, :]
    yuv_grid = cv2.cvtColor(xyz[:, np.newaxis, :] * 16 + 8, cv2.COLOR_YUV2RGB)[:, 0, :]
    yz = np.uint8([(7, i, j) for i in range(16) for j in range(16)])
    zx = np.uint8([(i, 7, j) for i in range(16) for j in range(16)])
    xy = np.uint8([(i, j, 7) for i in range(16) for j in range(16)])
    yz_for_hsv = np.float32(16 * yz + 8)
    yz_for_hsv[:, 0] *= 180.0 / 256.0
    zx_for_hsv = np.float32(16 * zx + 8)
    zx_for_hsv[:, 0] *= 180.0 / 256.0
    xy_for_hsv = np.float32(16 * xy + 8)
    xy_for_hsv[:, 0] *= 180.0 / 256.0
    bgr_grid_2d = {}
    hsv_grid_2d = {}
    yuv_grid_2d = {}
    bgr_grid_2d[0] = cv2.cvtColor(yz[:, np.newaxis, :] * 16 + 8, cv2.COLOR_BGR2RGB)[:, 0, :]
    bgr_grid_2d[1] = cv2.cvtColor(zx[:, np.newaxis, :] * 16 + 8, cv2.COLOR_BGR2RGB)[:, 0, :]
    bgr_grid_2d[2] = cv2.cvtColor(xy[:, np.newaxis, :] * 16 + 8, cv2.COLOR_BGR2RGB)[:, 0, :]
    hsv_grid_2d[0] = cv2.cvtColor(np.uint8(yz_for_hsv[:, np.newaxis, :]), cv2.COLOR_HSV2RGB)[:, 0, :]
    hsv_grid_2d[1] = cv2.cvtColor(np.uint8(zx_for_hsv[:, np.newaxis, :]), cv2.COLOR_HSV2RGB)[:, 0, :]
    hsv_grid_2d[2] = cv2.cvtColor(np.uint8(xy_for_hsv[:, np.newaxis, :]), cv2.COLOR_HSV2RGB)[:, 0, :]
    yuv_grid_2d[0] = cv2.cvtColor(yz[:, np.newaxis, :] * 16 + 8, cv2.COLOR_YUV2RGB)[:, 0, :]
    yuv_grid_2d[1] = cv2.cvtColor(zx[:, np.newaxis, :] * 16 + 8, cv2.COLOR_YUV2RGB)[:, 0, :]
    yuv_grid_2d[2] = cv2.cvtColor(xy[:, np.newaxis, :] * 16 + 8, cv2.COLOR_YUV2RGB)[:, 0, :]

    # print(bgr_grid)
    # print(bgr_grid.shape)

    # print(hsv_grid)
    # print(hsv_grid.shape)
    
    # print(yuv_grid)
    # print(yuv_grid.shape)
    
    # QUICK TEST
    for genre in GENRE_LIST[:1]:
        data_path = os.path.join(DATA_ROOT_PATH, genre)
        file_list = get_file_list(data_path, "_BGR_16.npy")
        # file_list = get_file_list(data_path, "_BGR_ALL.npy")
        for file in file_list[:10]:
            id = os.path.split(file)[1][:-11]
            print("https://www.imdb.com/title/" + id)
            color_bin = np.load(file)
            visualize_3d(id + "_BGR", x, y, z, bgr_grid, color_bin)

    # GENRE CLASSIFICATION
    # for genre in GENRE_LIST:
    #     print("GENRE: {}".format(genre))
    #     data_path = os.path.join(DATA_ROOT_PATH, genre)
    #     file_list = get_file_list(data_path, "_HSV_16.npy")
    #     # file_list = get_file_list(data_path, "_HSV_ALL.npy")
    #     color_bin = np.zeros((N_BINS, N_BINS, N_BINS))
    #     for file in tqdm(file_list):
    #         color_bin += np.load(file)
    #     visualize_3d(genre + "_HSV", x, y, z, hsv_grid, color_bin)




    # # FILTERED BY SRY
    # for genre in GENRE_LIST:
    #     data_path = os.path.join(DATA_ROOT_PATH, genre)
    #     file_list = get_file_list(data_path, "_BGR_16.npy")
    #     sry_data = np.load(os.path.join(data_path, "SRY_DATA.npy"))
    #     color_bins_0 = np.zeros((N_BINS, N_BINS, N_BINS))
    #     color_bins_1 = np.zeros((N_BINS, N_BINS, N_BINS))
    #     color_bins_2 = np.zeros((N_BINS, N_BINS, N_BINS))
    #     score_ave = np.median(sry_data[sry_data[:, 0] >= 0, 0])
    #     rated_ave = np.median(sry_data[sry_data[:, 1] >= 0, 1])
    #     year_ave = np.median(sry_data[sry_data[:, 2] >= 0, 2])
        
    #     num_0 = 0
    #     num_1 = 0
    #     num_2 = 0
    #     for (i, file) in tqdm(enumerate(file_list)):
    #         # if sry_data[i][0] < 0:
    #         #     color_bins_0 += np.load(file)
    #         #     num_0 += 1
    #         # elif sry_data[i][0] < score_ave:
    #         #     color_bins_1 += np.load(file)
    #         #     num_1 += 1
    #         # else:
    #         #     color_bins_2 += np.load(file)
    #         #     num_2 += 1
    #         # if sry_data[i][1] < 0:
    #         #     color_bins_0 += np.load(file)
    #         # elif sry_data[i][1] < rated_ave:
    #         #     color_bins_1 += np.load(file)
    #         # else:
    #         #     color_bins_2 += np.load(file)
    #         if sry_data[i][2] < 0:
    #             color_bins_0 += np.load(file)
    #             num_0 += 1
    #         elif sry_data[i][2] < year_ave:
    #             color_bins_1 += np.load(file)
    #             num_1 += 1
    #         else:
    #             color_bins_2 += np.load(file)
    #             num_2 += 1
        
    #     # visualize_3d(genre + "0", x, y, z, bgr_grid, color_bins_0)
    #     # visualize_3d(genre + " under {}".format(float(score_ave)), x, y, z, bgr_grid, color_bins_1)
    #     # visualize_3d(genre + " over {}".format(float(score_ave)), x, y, z, bgr_grid, color_bins_2)
        
    #     # visualize_2d(genre + "1", x, y, z, hsv_grid_2d, color_bins_1, 0)
    #     # visualize_2d(genre + "1", x, y, z, hsv_grid_2d, color_bins_1, 1)
    #     # visualize_2d(genre + "1", x, y, z, hsv_grid_2d, color_bins_1, 2)
        
    #     print(genre)
    #     # print(score_ave)
    #     # print(total_distance_to_grayscale(color_bins_1 / np.sum(color_bins_1)))
    #     # print(total_distance_to_grayscale(color_bins_2 / np.sum(color_bins_2)))
    #     # print(total_square_distance_to_grayscale(color_bins_1 / np.sum(color_bins_1)))
    #     # print(total_square_distance_to_grayscale(color_bins_2 / np.sum(color_bins_2)))
    #     # print(np.std(color_bins_1 / np.sum(color_bins_1)))
    #     # print(np.std(color_bins_2 / np.sum(color_bins_2)))
        
        

    #     # COLOR USAGE FREQUENCY
    #     # color_bins_1 /= np.max(color_bins_1)
    #     # hist_x = [(i / 150000) for i in range(150001)]
    #     # hist_y = [0 for i in range(150001)]
    #     # for cb in color_bins_1.flatten():
    #     #     hist_y[int((cb * 150000))] += 1
    #     # plt.title("{} distribution (before)".format(genre))
    #     # plt.scatter(hist_x[1:100], hist_y[1:100])
    #     # plt.show()
    #     # plt.close()

    #     # color_bins_2 /= np.max(color_bins_2)
    #     # hist_x = [(i / 150000) for i in range(150001)]
    #     # hist_y = [0 for i in range(150001)]
    #     # for cb in color_bins_2.flatten():
    #     #     hist_y[int((cb * 150000))] += 1
    #     # plt.title("{} distribution (after)".format(genre))
    #     # plt.scatter(hist_x[1:100], hist_y[1:100])
    #     # plt.show()
    #     # plt.close()
        


    #     # visualize_3d(genre + "0", x, y, z, bgr_grid, color_bins_0)
    #     # visualize_3d(genre + "1", x, y, z, bgr_grid, color_bins_1)
    #     # visualize_3d(genre + "2", x, y, z, bgr_grid, color_bins_2)
    #     # visualize_2d(genre + "0", x, y, z, bgr_grid_2d, color_bins_0, 0)
    #     # visualize_2d(genre + "0", x, y, z, bgr_grid_2d, color_bins_0, 1)
    #     # visualize_2d(genre + "0", x, y, z, bgr_grid_2d, color_bins_0, 2)
    #     visualize_2d(genre + "1", x, y, z, bgr_grid_2d, color_bins_1, 0)
    #     visualize_2d(genre + "1", x, y, z, bgr_grid_2d, color_bins_1, 1)
    #     visualize_2d(genre + "1", x, y, z, bgr_grid_2d, color_bins_1, 2)
    #     visualize_2d(genre + "2", x, y, z, bgr_grid_2d, color_bins_2, 0)
    #     visualize_2d(genre + "2", x, y, z, bgr_grid_2d, color_bins_2, 1)
    #     visualize_2d(genre + "2", x, y, z, bgr_grid_2d, color_bins_2, 2)

    #     # visualize_3d(genre + "0", x, y, z, hsv_grid, color_bins_0)
    #     # visualize_3d(genre + "1", x, y, z, hsv_grid, color_bins_1)
    #     # visualize_3d(genre + "2", x, y, z, hsv_grid, color_bins_2)
    #     # visualize_2d(genre + "0", x, y, z, hsv_grid_2d, color_bins_0, 0)
    #     # visualize_2d(genre + "0", x, y, z, hsv_grid_2d, color_bins_0, 1)
    #     # visualize_2d(genre + "0", x, y, z, hsv_grid_2d, color_bins_0, 2)
    #     # visualize_2d(genre + "1", x, y, z, hsv_grid_2d, color_bins_1, 0)
    #     # visualize_2d(genre + "2", x, y, z, hsv_grid_2d, color_bins_2, 0)
    #     # visualize_2d(genre + "1", x, y, z, hsv_grid_2d, color_bins_1, 1)
    #     # visualize_2d(genre + "2", x, y, z, hsv_grid_2d, color_bins_2, 1)
    #     # visualize_2d(genre + "1", x, y, z, hsv_grid_2d, color_bins_1, 2)
    #     # visualize_2d(genre + "2", x, y, z, hsv_grid_2d, color_bins_2, 2)
        

    #     # visualize_3d(genre + "0", x, y, z, yuv_grid, color_bins_0)
    #     # visualize_3d(genre + "1", x, y, z, yuv_grid, color_bins_1)
    #     # visualize_3d(genre + "2", x, y, z, yuv_grid, color_bins_2)
    #     # visualize_2d(genre + "0", x, y, z, yuv_grid_2d, color_bins_0, 0)
    #     # visualize_2d(genre + "1", x, y, z, yuv_grid_2d, color_bins_1, 0)
    #     # visualize_2d(genre + "2", x, y, z, yuv_grid_2d, color_bins_2, 0)
    #     # visualize_2d(genre + "0", x, y, z, yuv_grid_2d, color_bins_0, 1)
    #     # visualize_2d(genre + "1", x, y, z, yuv_grid_2d, color_bins_1, 1)
    #     # visualize_2d(genre + "2", x, y, z, yuv_grid_2d, color_bins_2, 1)
    #     # visualize_2d(genre + "0", x, y, z, yuv_grid_2d, color_bins_0, 2)
    #     # visualize_2d(genre + "1", x, y, z, yuv_grid_2d, color_bins_1, 2)
    #     # visualize_2d(genre + "2", x, y, z, yuv_grid_2d, color_bins_2, 2)
        
        