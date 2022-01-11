import os
import glob

STD_HEIGHT = 1500
STD_WIDTH  = 1013
# GENRE_LIST = ["comedy",         "sci-fi",           "horror",
#               "romance",        "action",           "thriller",
#               "drama",          "mystery",          "crime",
#               "animation",      "adventure",        "fantasy",
#               "comedy,romance", "action,comedy"]
GENRE_LIST = ["comedy",         "sci-fi",           "horror",
              "romance",        "action",           "thriller",
              "drama",          "mystery",          "crime",
              "animation",      "adventure",        "fantasy"]
            #   "comedy,romance", "action,comedy"]
STD_POSTER_SHAPE = (1013, 1500)
STD_RATIO = STD_POSTER_SHAPE[1] / STD_POSTER_SHAPE[0]
STD_TOLERANCE = 0.1
N_BINS = 16
DATA_ROOT_PATH = os.path.join("Courses", "CulturoInformatics", "IMDb")

def try_mkdir(path):
    if not os.path.exists(path):
        os.makedirs('./{}'.format(path))

def hmin_to_minutes(string):
    if not(string):
        return 0
    hours = 0
    minutes = 0
    if "h" in string:
        string_split = string.split("h")
        hours = int(string_split[0])
        string = string_split[1]
    if "min" in string:
        string_split = string.split("min")
        minutes = int(string_split[0])
    return 60 * hours + minutes

def mk_to_int(string):
    order = 1
    if string[-1] == "M":
        order = 1000000
        num = float(string.split("M")[0])
    elif string[-1] == "K":
        order = 1000
        num = float(string.split("K")[0])
    else:
        num = int(string)
    return int(num * order)

def get_file_list(path, ext):
    files_path = os.path.join(path, "*" + ext)
    files = sorted(glob.iglob(files_path), key = lambda x : os.path.getmtime(x), reverse = False)
    return files
    # return sorted([f for f in os.listdir(path) if f.endswith("." + ext)])