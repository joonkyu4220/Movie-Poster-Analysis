import os

from utils import DATA_ROOT_PATH, GENRE_LIST, N_BINS, get_file_list
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


if __name__ == "__main__":
    os.chdir("D:")

    # FACE
    # for genre in GENRE_LIST:
    #     print("GENRE: ", genre)
    #     data_path = os.path.join("Courses", "CulturoInformatics", "IMDb", genre)
    #     file_list = get_file_list(data_path, "_faces.npy")
    #     concat = []
    #     idx_list = [0]
    #     idx_sum = 0
    #     num_list = []
    #     for (idx, file) in tqdm(enumerate(file_list)):
    #         data = np.load(file)
    #         if np.sum(data[0, :] >= 1):
    #             num_list.append(data.shape[0])
    #             concat.append(data)
    #             idx_sum += data.shape[0]
    #             idx_list.append(idx_sum)
    #         else:
    #             num_list.append(0)
    #             idx_list.append(idx_sum)
    #     num_list.append(-1)
    #     concat = np.concatenate(concat, axis=0)
    #     idx_list = np.array(idx_list)
    #     num_list = np.array(num_list)
    #     idx_list = np.concatenate([num_list[:, np.newaxis], idx_list[:, np.newaxis]], axis=1)
    #     np.save(os.path.join(data_path, "_FACES_ALL.npy"), concat)
    #     np.save(os.path.join(data_path, "_FACES_IDX_INFO.npy"), idx_list)


    # COLORSPACE
    # for genre in GENRE_LIST:
        # print("GENRE: ", genre)
        # data_path = os.path.join(DATA_ROOT_PATH, genre)
        
        # RGB
        # file_list = get_file_list(data_path, "_BGR_16.npy")
        # summary = np.zeros((N_BINS, N_BINS, N_BINS))
        # for file in tqdm(file_list):
        #     data = np.load(file)
        #     summary += data
        # np.save(os.path.join(data_path, "_BGR_ALL.npy"), summary)
        
        # HSV
        # file_list = get_file_list(data_path, "_HSV_16.npy")
        # summary = np.zeros((N_BINS, N_BINS, N_BINS))
        # for file in tqdm(file_list):
        #     data = np.load(file)
        #     summary += data
        # np.save(os.path.join(data_path, "_HSV_ALL.npy"), summary)

        # YUV
        # file_list = get_file_list(data_path, "_YUV_16.npy")
        # summary = np.zeros((N_BINS, N_BINS, N_BINS))
        # for file in tqdm(file_list):
        #     data = np.load(file)
        #     summary += data
        # np.save(os.path.join(data_path, "_YUV_ALL.npy"), summary)

    



    # RATED_DIC = {"TV-Y" : 1,
    #              "TV-G" :2, "G" : 2,
    #              "TV-Y7" : 3, "TV-Y7-FV" : 3,
    #              "TV-PG" : 4, "PG" : 4, "GP" : 4, "M": 4, "M/PG": 4,
    #              "PG-13" : 5,
    #              "TV-14" : 6,
    #              "R" : 8,
    #              "TV-MA" : 9, "NC-17" : 9, "MA-17" : 9,
    #              "X" : 10}


    # for genre in GENRE_LIST:
    #     print("GENRE: ", genre)
    #     data_path = os.path.join(DATA_ROOT_PATH, genre)
    #     file_list = get_file_list(data_path, "_BGR_16.npy")
    #     sry_data = np.zeros((len(file_list), 3))
    #     for (i, file) in tqdm(enumerate(file_list)):
    #         id = os.path.split(file)[1][:-11]
    #         text_file = file[:-11] + ".txt"
    #         content = open(text_file, "r")
    #         lines = content.readlines()
    #         score = lines[2][6:-1]
    #         rated = lines[4][6:-1]
    #         year = lines[5][5:-1]
    #         if score:
    #             score = float(score)
    #         else:
    #             score = -1
    #         if rated in RATED_DIC:
    #             rated = RATED_DIC[rated]
    #         else:
    #             rated = -1
    #         if year:
    #             year = float(year)
    #         else:
    #             year = -1
    #         sry_data[i][0] = score
    #         sry_data[i][1] = rated
    #         sry_data[i][2] = year
    #     np.save(os.path.join(data_path, "SRY_DATA.npy"), sry_data)
        

    score_ranges = [(i, i+1) for i in range(10)]
    scores = []
    for genre in GENRE_LIST:
        print("GENRE: ", genre)
        num = [0 for _ in range(10)]
        data_path = os.path.join(DATA_ROOT_PATH, genre)
        file_list = get_file_list(data_path, "_BGR_16.npy")
        for file in tqdm(file_list):
            data = np.load(file)
            id = os.path.split(file)[1][:-11]
            text_file = file[:-11] + ".txt"
            content = open(text_file, "r")
            lines = content.readlines()
            score = lines[2][6:-1]
            content.close()
            if score:
                score = float(score)
                scores.append(score)
            for (l, r) in score_ranges:
                if l < score <= r:
                    num[l] += 1
                    break
        mids = [0.5 + i for i in range(10)]
        fig = plt.figure(figsize=(10, 10))
        plt.title(genre)
        plt.scatter(mids, num)
        plt.title(genre + ": " + str(np.average(scores)))
        plt.show()
        plt.close()

    # half_years = list()
    # for genre in GENRE_LIST:
    #     print("GENRE: ", genre)
    #     data_path = os.path.join(DATA_ROOT_PATH, genre)
    #     file_list = get_file_list(data_path, "_BGR_16.npy")
    #     summary = dict()
    #     num = dict()
        
    #     for file in tqdm(file_list):
    #         data = np.load(file)
    #         id = os.path.split(file)[1][:-11]
    #         text_file = file[:-11] + ".txt"
    #         content = open(text_file, "r")
    #         lines = content.readlines()
    #         # score = lines[2][6:-1]
    #         # numscore = lines[3][9:-1]
    #         # rated = lines[4][6:-1]
    #         year = lines[5][5:-1]
    #         # time = lines[6][5:-1]
    #         if ((year) and (int(year) > 1800)):
    #             year = int(year)
    #         else:
    #             continue
    #         if not(year in summary):
    #             summary[int(year)] = np.zeros_like(data)
    #             num[year] = 0
    #         summary[year] += data
    #         num[year] += 1
    #         content.close()

    #     years = sorted(summary.keys())
    #     nums = [num[year] for year in years]
    #     accumulated_nums = 0
    #     for y in years:
    #         accumulated_nums += num[y]
    #         if accumulated_nums >= sum(nums) / 2:
    #             half_years.append(y)
    #             break
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.title("{}: {}".format(genre, half_years[-1]))
    #     plt.scatter(years, nums)
    #     plt.show()
    #     plt.close()
    # print(half_years)
    # print(np.average(half_years))