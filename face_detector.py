import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import get_file_list, GENRE_LIST, DATA_ROOT_PATH

import numpy as np

import cv2

from tqdm import tqdm

from matplotlib import pyplot as plt 

# import cvzone
# from cvzone.PoseModule import PoseDetector
# from cvzone.FaceDetectionModule import FaceDetector


# import face_recognition

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model


# config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)


# OPEN POSE==================================================================================================================================================================
# BODY_PARTS = {  "Head": 0,  "Neck": 1,      "RShoulder": 2, "RElbow": 3,    "RWrist": 4,
#                                             "LShoulder": 5, "LElbow": 6,    "LWrist": 7,
#                 "RHip": 8,  "RKnee": 9,     "RAnkle": 10,
#                 "LHip": 11, "LKnee": 12,    "LAnkle": 13,
#                 "Chest": 14,
#                 "Background": 15 }

# POSE_PAIRS = [["Head", "Neck"],     ["Neck", "RShoulder"],  ["RShoulder", "RElbow"],    ["RElbow", "RWrist"],
#                                     ["Neck", "LShoulder"],  ["LShoulder", "LElbow"],    ["LElbow", "LWrist"],
#                                     ["Neck", "Chest"],      ["Chest", "RHip"],          ["RHip", "RKnee"],      ["RKnee", "RAnkle"],
#                                                             ["Chest", "LHip"],          ["LHip", "LKnee"],      ["LKnee", "LAnkle"] ]


# PAZ==================================================================================================================================================================
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import paz.processors as pr

class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)
        
    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            box2D.class_name, box2D.score = self.classify(cropped_image)['class_name'], self.classify(cropped_image)['scores']
            # print(box2D.class_name)
            # print(box2D.coordinates)
            # print(box2D.score)
        return self.draw(image, boxes2D), boxes2D

detector = EmotionDetector()

std_poster_shape = (1013, 1500)
# std_poster_shape = (2026, 3000)
std_poster_shape_view = (1013, 1500)
std_tolerance = 0.1



if __name__ == "__main__":

    os.chdir("D:")

    std_ratio = std_poster_shape[1] / std_poster_shape[0]
    # std_face_shape = (48, 48)
    # # OPEN POSE==================================================================================================================================================================
    # detector_path = os.path.join("Courses", "CulturoInformatics", "detectors")
    # proto_file  = os.path.join(detector_path, "pose_deploy_linevec_faster_4_stages.prototxt")
    # weight_file = os.path.join(detector_path, "pose_iter_160000.caffemodel")
    # pose_detector = cv2.dnn.readNetFromCaffe(proto_file, weight_file)

    # # MEDIAPIPE==================================================================================================================================================================
    # emotion_model_path = os.path.join("Courses", "CulturoInformatics", "detectors")
    # emotion_dict= {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
    # emotion_invdict = dict((v,k) for k,v in emotion_dict.items())
    # emotion_model = load_model(os.path.join(emotion_model_path, "model_v6_23.hdf5"))
    
    # pose_condition = 0.7
    # face_condition = 0.9
    # pose_detector = PoseDetector(mode=True, smooth=False, detectionCon=pose_condition, trackCon=0.5)
    # face_detector = FaceDetector(minDetectionCon=face_condition)

    for genre in GENRE_LIST:
        print("GENRE: ", genre)
        data_path = os.path.join(DATA_ROOT_PATH, genre)
        poster_list = get_file_list(data_path, ".jpg")

        people_num = []

        for (idx, poster) in tqdm(enumerate(poster_list[:1000])):
            
            # image = cv2.imread(os.path.join(data_path, poster), cv2.IMREAD_COLOR)
            image = cv2.imread(poster, cv2.IMREAD_COLOR)
            if abs(image.shape[0] / image.shape[1] - std_ratio) > std_tolerance:
                continue
            image_pose = cv2.resize(image, std_poster_shape)
            image_face = cv2.resize(image, std_poster_shape)
            # image_face = image
            
            # PAZ==================================================================================================================================================================
            predictions, boxes = detector(image_face)
            score_list = []
            coordinates_list = []
            found = False
            people_num.append(len(boxes))
            for box in boxes:
                found = True
                score_list.append(box.score)
                coordinates_list.append(box.coordinates[np.newaxis, :])
            if found:
                score_list = np.concatenate(score_list, axis=0)
                coordinates_list = np.concatenate(coordinates_list, axis=0)
                concat = np.concatenate([score_list, coordinates_list], axis=1)
                # np.save(os.path.join(data_path, poster.replace(".jpg", ".npy")), concat)
                np.save(os.path.join(poster.replace(".jpg", "_faces.npy")), concat)
            else:
                np.save(os.path.join(poster.replace(".jpg", "_faces.npy")), np.zeros((1, 11)))
            # cv2.imshow("Output-Keypoints", cv2.resize(predictions, std_poster_shape_view))
            # cv2.imshow("Output-Keypoints", cv2.resize(predictions, (predictions.shape[1]//5, predictions.shape[0]//5)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        print(np.mean(people_num))
        plt.hist(people_num, bins = [(i+0.5) for i in range(-1, 29, 1)])
        plt.title(genre)
        # plt.show()
        plt.savefig(os.path.join(data_path, 'Number of People.png'), dpi=600)
        plt.clf()
        

        # # OPEN POSE==================================================================================================================================================================
        # image_input = cv2.dnn.blobFromImage(image, 1.0/255.0, std_poster_shape, (0, 0, 0), swapRB=False, crop=False)
        # pose_detector.setInput(image_input)
        # image_output = pose_detector.forward()
        # height, width = image_output.shape[2], image_output.shape[3]
        # print(height, width)
        # points = []
        # for i in range(len(BODY_PARTS)):
        #     prob_map = image_output[0, i, :, :]
        #     min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        #     x = (std_poster_shape[0] * point[0]) / width
        #     y = (std_poster_shape[1] * point[1]) / height
        #     if prob > 0.1:
        #         cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #         cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        #         points.append((int(x), int(y)))
        #     else:
        #         points.append(None)
        # cv2.imshow("Output-Keypoints", cv2.resize(image, std_poster_shape))
        # cv2.waitKey(0)
        # imageCopy = image
        # for pair in POSE_PAIRS:
        #     from_ = BODY_PARTS[pair[0]]
        #     to_   = BODY_PARTS[pair[1]]
        #     if points[from_] and points[to_]:
        #         cv2.line(imageCopy, points[from_], points[to_], (0, 255, 0), 2)
        # cv2.imshow("Output-Keypoints", cv2.resize(imageCopy, std_poster_shape))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # MEDIAPIPE==================================================================================================================================================================
        # # pose
        # pose = pose_detector.findPose(image_pose, draw=True)
        # lmList, bboxInfo = pose_detector.findPosition(pose, draw=True, bboxWithHands=False)
        # print(len(lmList))
        # print(lmList)
        # # print(bboxInfo)
        # if bboxInfo:
        #     center = bboxInfo["center"]
        #     cv2.circle(pose, center, 5, (255, 0, 255), cv2.FILLED)
        # cv2.imshow("Image", pose)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # face
        # face_locations = face_recognition.face_locations(image_face)
        # for fl in face_locations:
        #     top, right, bottom, left = fl
        #     face_image = image_face[top:bottom, left:right]
        #     face_image = cv2.resize(face_image, std_face_shape)
        #     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
        #     cv2.imshow("face", cv2.resize(face_image, (100, 100)))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        #     face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
        #     predicted_emotion_scores = emotion_model.predict(face_image)
        #     predicted_emotion_idx = np.argmax(predicted_emotion_scores)
        #     predicted_emotion_label = emotion_invdict[predicted_emotion_idx]
        #     print(predicted_emotion_scores)
        #     print(predicted_emotion_label)