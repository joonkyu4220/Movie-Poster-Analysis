import os
from utils import DATA_ROOT_PATH, get_file_list, GENRE_LIST, STD_POSTER_SHAPE, STD_RATIO, STD_TOLERANCE, N_BINS
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt



def k_largest_index_argpartition_v1(a, k):
    idx = np.argpartition(-a.ravel(),k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))

k_max = 10


if __name__ == "__main__":
    os.chdir("D:")
    for genre in GENRE_LIST:
    # for genre in ["test"]:
        print("GENRE: ", genre)
        data_path = os.path.join(DATA_ROOT_PATH, genre)
        poster_list = get_file_list(data_path, ".jpg")

        for (idx, poster) in tqdm(enumerate(poster_list[:1000])):
            bgr_bin = np.zeros((N_BINS, N_BINS, N_BINS))
            hsv_bin = np.zeros((N_BINS, N_BINS, N_BINS))
            yuv_bin = np.zeros((N_BINS, N_BINS, N_BINS))

            image = cv2.imread(poster, cv2.IMREAD_COLOR)
            if abs(image.shape[0] / image.shape[1] - STD_RATIO) > STD_TOLERANCE:
                continue
            else:
                image = cv2.resize(image, STD_POSTER_SHAPE)
            # print("")
            # print(image.shape)
            # print(image[0, 0])
            image_bgr_bin = np.array(image // (256/N_BINS), np.uint8)

            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image_hsv = np.float32(image_hsv)
            image_hsv[:, :, 0] *= 256.0 / 180.0
            image_hsv_bin = np.array(image_hsv // (256/N_BINS), np.uint8)

            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image_yuv_bin = np.array(image_yuv // (256/N_BINS), np.uint8)


            # for c in image_bgr_bin.reshape(-1, 3):
            #     bgr_bin[c[0], c[1], c[2]] += 1
            # for c in image_hsv_bin.reshape(-1, 3):
            #     hsv_bin[c[0], c[1], c[2]] += 1
            for c in image_yuv_bin.reshape(-1, 3):
                yuv_bin[c[0], c[1], c[2]] += 1

            # np.save(os.path.join(poster.replace(".jpg", "_BGR_{}.npy".format(N_BINS))), bgr_bin)
            # np.save(os.path.join(poster.replace(".jpg", "_HSV_{}.npy".format(N_BINS))), hsv_bin)
            np.save(os.path.join(poster.replace(".jpg", "_YUV_{}.npy".format(N_BINS))), yuv_bin)

            # for i in range(STD_POSTER_SHAPE[1]):
            #     for j in range(STD_POSTER_SHAPE[0]):
            #         bgr_bin[image_new[i, j, 0], image_new[i, j, 1], image_new[i, j, 2]] += 1

            # # print(bgr_bin)



            # coordinates = np.array([[x, y, z] for x in range(n_bins) for y in range(n_bins) for z in range(n_bins)])
            # x = coordinates[:, 0]
            # y = coordinates[:, 1]
            # z = coordinates[:, 2]


            
            # max_k_coordinates = k_largest_index_argpartition_v1(bgr_bin, k_max)
            # x = max_k_coordinates[:, 0]
            # y = max_k_coordinates[:, 1]
            # z = max_k_coordinates[:, 2]
            # print((np.array([z, y, x])/15.0).shape)
            # for i in range(k_max):
            #     print(x[i], y[i], z[i])
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlim3d(0, 15)
            # ax.set_ylim3d(0, 15)
            # ax.set_zlim3d(0, 15)
            # ax.scatter(x,y,z,c=np.array([z, y, x]).T/15.0,edgecolors=[0, 0, 0])
            # # pnt3d=ax.scatter(x,y,z,c=bgr_bin[z, x, y])
            # # cbar=plt.colorbar(pnt3d)
            # # cbar.set_label("Values (units)")
            # plt.show()