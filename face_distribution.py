import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.framework.config import enable_mlir_graph_optimization
from tqdm import tqdm

from utils import GENRE_LIST, STD_HEIGHT, STD_WIDTH, DATA_ROOT_PATH, get_file_list

EMOTION_DIC = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise", 6:"neutral"}

if __name__ == "__main__":
    os.chdir("D:")

    # for genre in GENRE_LIST:
    #     print("===================================================================")
    #     print("GENRE: {}".format(genre))
    #     data_path = os.path.join(DATA_ROOT_PATH, genre)
    #     file_list = get_file_list(data_path, "_faces.npy")
    #     number_of_posters = len(file_list)
    #     number_of_people = 0
    #     emotion_scores = np.zeros((7,))
    #     emotion_with_size = np.zeros((7,))
    #     size_score = 0
    #     for (i, file) in tqdm(enumerate(file_list)):
    #         data = np.load(file)
    #         if np.sum(data) == 0:
    #             continue
    #         number_of_people += data.shape[0]
    #         emotion_scores += np.sum(data[:, :7], axis=0, keepdims=False)
    #         sizes = (data[:, 9] - data[:, 7]) * (data[:, 10] - data[:, 8])
    #         size_score += np.sum(sizes)
    #         emotion_with_size += np.sum(data[:, :7] * (sizes[:, np.newaxis]), axis=0, keepdims=False)

            

    #     emotion_scores /= np.sum(emotion_scores)
    #     emotion_with_size /= np.sum(emotion_with_size)
    #     size_score /= number_of_people
    #     # size_score /= number_of_posters
    #     number_of_people /= number_of_posters
    #     # print("NUMBER OF PEOPLE: {}".format(number_of_people))
    #     # print("EMOTION SCORES : {}".format(emotion_scores))
    #     # print("SIZE SCORE: {}".format(np.sqrt(size_score)))
    #     print("EMOTION WITH SIZE: {}".format(emotion_with_size))

    for genre in GENRE_LIST:
        print("GENRE: {}".format(genre))
        data_path = os.path.join(DATA_ROOT_PATH, genre)
        data = np.load(os.path.join(data_path, "_FACES_ALL.npy"))
        idx_info = np.load(os.path.join(data_path, "_FACES_IDX_INFO.npy"))
        emotion_scores = data[:, :7]
        coordinates = data[:, 7:]
        num_people = idx_info[:, 0]
        idx_start = idx_info[:, 1]
    
    
    #     np.save(os.path.join(data_path, "_FACES_ALL.npy"), concat)
    #     np.save(os.path.join(data_path, "_FACES_IDX_INFO.npy"), idx_list)
    
    
        # all
        x_centers = (coordinates[:, 0] + coordinates[:, 2]) / 2.0
        y_centers = (coordinates[:, 1] + coordinates[:, 3]) / 2.0
        # hist, xbins, ybins, im = plt.hist2d(x_centers, 1500-y_centers, (10, 15), [[0, STD_WIDTH],[0, STD_HEIGHT]])
        # plt.colorbar()
        # plt.title("{}, all".format(genre))
        # plt.savefig(os.path.join(data_path, "hist[10, 15]_{}, all.png".format(genre)), dpi=600)
        # plt.clf()
        hist, xbins, ybins, im = plt.hist2d(x_centers, 1500-y_centers, (20, 30), [[0, STD_WIDTH],[0, STD_HEIGHT]])
        plt.colorbar()
        plt.title("{}, all".format(genre))
        plt.savefig(os.path.join(data_path, "..", "_hist[20, 30]_{}, all.png".format(genre)), dpi=600)
        plt.clf()

    #     # by emotion
    #     for e in range(7):
    #         hist, xbins, ybins, im = plt.hist2d(x_centers, 1500-y_centers, (10, 15), [[0, STD_WIDTH],[0, STD_HEIGHT]], weights=emotion_scores[:, e])
    #         plt.colorbar()
    #         plt.title("{}, {}".format(genre, EMOTION_DIC[e]))
    #         plt.savefig(os.path.join(data_path, "hist[10, 15]_{}, all, {}.png".format(genre, EMOTION_DIC[e])), dpi=600)
    #         plt.clf()
    #         hist, xbins, ybins, im = plt.hist2d(x_centers, 1500-y_centers, (20, 30), [[0, STD_WIDTH],[0, STD_HEIGHT]], weights=emotion_scores[:, e])
    #         plt.colorbar()
    #         plt.title("{}, {}".format(genre, EMOTION_DIC[e]))
    #         plt.savefig(os.path.join(data_path, "hist[20, 30]_{}, all, {}.png".format(genre, EMOTION_DIC[e])), dpi=600)
    #         plt.clf()
            
        total = data.shape[0]
        starts = []
        i = 0
        while total > 0:
            start = idx_info[idx_info[:, 0] == i, 1]
            starts.append(start)
            total -= len(start) * i
            i += 1

        # by number of people
        x_centers_list = []
        y_centers_list = []
        for i in range(1, len(starts)):
            x_centers = []
            y_centers = []
            if len(starts[i]):
                for s in starts[i]:
                    x_centers.extend((coordinates[s:s+i, 0] + coordinates[s:s+i, 2]) / 2.0)
                    y_centers.extend((coordinates[s:s+i, 1] + coordinates[s:s+i, 3]) / 2.0)
                # hist, xbins, ybins, im = plt.hist2d(np.array(x_centers), 1500-np.array(y_centers), (10, 15), [[0, STD_WIDTH],[0, STD_HEIGHT]])
                # plt.colorbar()
                # plt.title("{}, {} people".format(genre, i))
                # plt.savefig(os.path.join(data_path, "hist[10, 15]_{}, {} people.png".format(genre, i)), dpi=600)
                # plt.clf()
                hist, xbins, ybins, im = plt.hist2d(np.array(x_centers), 1500-np.array(y_centers), (20, 30), [[0, STD_WIDTH],[0, STD_HEIGHT]])
                plt.colorbar()
                plt.title("{}, {} people".format(genre, i))
                plt.savefig(os.path.join(data_path, "..", "hist[20, 30]_{}, {} people.png".format(genre, i)), dpi=600)
                plt.clf()

    #     # by number of people, regarding size
    #     x_centers_list = []
    #     y_centers_list = []
    #     for i in range(1, len(starts)):
    #         x_centers = []
    #         y_centers = []
    #         sizes = []
    #         if len(starts[i]):
    #             for s in starts[i]:
    #                 x_centers.extend((coordinates[s:s+i, 0] + coordinates[s:s+i, 2]) / 2.0)
    #                 y_centers.extend((coordinates[s:s+i, 1] + coordinates[s:s+i, 3]) / 2.0)
    #                 sizes.extend((coordinates[s:s+i, 2] - coordinates[s:s+i, 0]) * (coordinates[s:s+i, 3] - coordinates[s:s+i, 1]))
    #             hist, xbins, ybins, im = plt.hist2d(np.array(x_centers), 1500-np.array(y_centers), (10, 15), [[0, STD_WIDTH],[0, STD_HEIGHT]], weights = sizes)
    #             plt.colorbar()
    #             plt.title("{}, {} people, weighted".format(genre, i))
    #             plt.savefig(os.path.join(data_path, "hist[10, 15]_{}, {} people, weighted.png".format(genre, i)), dpi=600)
    #             plt.clf()
    #             hist, xbins, ybins, im = plt.hist2d(np.array(x_centers), 1500-np.array(y_centers), (20, 30), [[0, STD_WIDTH],[0, STD_HEIGHT]], weights = sizes)
    #             plt.colorbar()
    #             plt.title("{}, {} people, weighted".format(genre, i))
    #             plt.savefig(os.path.join(data_path, "hist[20, 30]_{}, {} people, weighted.png".format(genre, i)), dpi=600)
    #             plt.clf()

        
    #     # by number of people, by emotion
    #     x_centers_list = []
    #     y_centers_list = []
    #     for i in range(1, len(starts)):
    #         x_centers = []
    #         y_centers = []
    #         sizes = [[] for _ in range(7)]
    #         if len(starts[i]):
    #             for s in starts[i]:
    #                 x_centers.extend((coordinates[s:s+i, 0] + coordinates[s:s+i, 2]) / 2.0)
    #                 y_centers.extend((coordinates[s:s+i, 1] + coordinates[s:s+i, 3]) / 2.0)
    #                 for e in range(7):
    #                     sizes[e].extend(emotion_scores[s:s+i, e])
    #             for e in range(7):
    #                 hist, xbins, ybins, im = plt.hist2d(np.array(x_centers), 1500-np.array(y_centers), (10, 15), [[0, STD_WIDTH],[0, STD_HEIGHT]], weights = sizes[e])
    #                 plt.colorbar()
    #                 plt.title("{}, {} people, {}".format(genre, i, EMOTION_DIC[e]))
    #                 plt.savefig(os.path.join(data_path, "hist[10, 15]_{}, {} people, {}.png".format(genre, i, EMOTION_DIC[e])), dpi=600)
    #                 plt.clf()
    #                 hist, xbins, ybins, im = plt.hist2d(np.array(x_centers), 1500-np.array(y_centers), (20, 30), [[0, STD_WIDTH],[0, STD_HEIGHT]], weights = sizes[e])
    #                 plt.colorbar()
    #                 plt.title("{}, {} people, {}".format(genre, i, EMOTION_DIC[e]))
    #                 plt.savefig(os.path.join(data_path, "hist[20, 30]_{}, {} people, {}.png".format(genre, i, EMOTION_DIC[e])), dpi=600)
    #                 plt.clf()