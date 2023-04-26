import json
import cv2
import numpy as np
from utils.constants import MANIPULATIONS_DETECTION_RESULT_PATH
from utils.constants import TRUE_MANIPULATIONS_PATH
from utils.constants import PATH_TO_RESULTS
from utils.constants import PATH_TO_VIDEOS
from utils.video_paths_collector import VideoPathsCollector
import matplotlib.pyplot as plt


def calculate_confusion_matrix(result_frames: list, true_frames: list, frames_number: int):
    confusion_matrix = np.zeros((2, 2), dtype=int)
    TN, TP, FP, FN = 0, 0, 0, 0
    for i in range(0, len(result_frames)):
        if result_frames[i] > 0:
            if true_frames[i] > 0:
                TP += 1
            else:
                FP += 1
        else:
            if true_frames[i] > 0:
                FN += 1
    TN = frames_number - TP - FP - FN
    confusion_matrix[1, 1] = TN
    confusion_matrix[0, 0] = TP
    confusion_matrix[0, 1] = FP
    confusion_matrix[1, 0] = FN
    return confusion_matrix


def calculate_frame_with_accuracy(frame: int):
    accuracy = 8
    frame_with_accuracy = []

    left = frame - accuracy
    right = frame + accuracy
    if left < 0:
        left = 0
    for i in range(left, right + 1):
        frame_with_accuracy.append(i)

    return frame_with_accuracy


def zeros_appending(result_frames: list, true_frames: list):
    matched_count = 0
    matched_frames = []

    for result_frame in result_frames:
        for accuracy_frame in calculate_frame_with_accuracy(result_frame):
            if true_frames.__contains__(accuracy_frame):
                if matched_frames.__contains__(accuracy_frame):
                    result_frames.pop(result_frames.index(result_frame))
                    break
                else:
                    result_frames[result_frames.index(result_frame)] = accuracy_frame
                    matched_frames.append(accuracy_frame)
                    matched_count += 1
                    break

    missing_on_result = true_frames.copy()
    missing_on_true = result_frames.copy()

    for matched_frame in matched_frames:
        missing_on_result.remove(matched_frame)
        missing_on_true.remove(matched_frame)

    for missing in missing_on_result:
        result_frames.append(missing)
    for missing in missing_on_true:
        true_frames.append(missing)

    calculated_result_frames = sorted(result_frames)
    calculated_true_frames = sorted(true_frames)

    index = 0
    for result_frame in calculated_result_frames:
        if missing_on_result.__contains__(result_frame):
            calculated_result_frames[index] = 0
        index += 1

    index = 0
    for true_frame in calculated_true_frames:
        if missing_on_true.__contains__(true_frame):
            calculated_true_frames[index] = 0
        index += 1

    return calculated_result_frames, calculated_true_frames


def collect_frames_counts():
    video_paths_collector = VideoPathsCollector(PATH_TO_VIDEOS)
    frames_counts = {}
    for video_path in video_paths_collector.collect():
        cap = cv2.VideoCapture(video_path)
        frames_count = 0
        if cap.isOpened:
            frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_counts[get_video_name(video_path)] = frames_count
    return frames_counts


def get_video_name(path: str):
    parsed = path.split("\\")
    return parsed[len(parsed) - 1]


def draw_graph(matrix_list: list):
    TPR = []
    FPR = []
    index = 0
    for matrix in matrix_list:
        TN = matrix[1, 1]
        TP = matrix[0, 0]
        FP = matrix[0, 1]
        FN = matrix[1, 0]
        if TP == 0:
            TPR.append(0.0)
        else:
            TPR.append(TP / (TP + FN))
        if FP == 0:
            FPR.append(0.0)
        else:
            FPR.append(FP / (TN + FP))
        index += 1
    plt.plot(FPR, TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    graph_path = PATH_TO_RESULTS + '\\' + 'roc_auc' + '.png'
    plt.savefig(graph_path)
    print('FPR: ', FPR)
    print('TPR: ', TPR)


if __name__ == '__main__':
    with open(MANIPULATIONS_DETECTION_RESULT_PATH, "r") as results_file:
        all_manipulations = json.load(results_file)
        with open(TRUE_MANIPULATIONS_PATH, "r") as true_results_file:
            all_true_manipulations = json.load(true_results_file)
            error_matrix_list = []
            frames_counts = collect_frames_counts()
            for video_name in all_manipulations.keys():
                anomalies = all_manipulations[video_name]
                true_frames = all_true_manipulations[video_name]
                result_frames = []
                for anomaly in anomalies:
                    score, frame = anomaly
                    contains = False
                    for result_frame in calculate_frame_with_accuracy(frame):
                        if result_frames.__contains__(result_frame):
                            contains = True
                    if not contains:
                        result_frames.append(frame)
                result_frames, true_frames = zeros_appending(result_frames, true_frames)
                error_matrix = calculate_confusion_matrix(result_frames, true_frames, frames_counts[video_name])
                print(video_name + ': \n', error_matrix)
                error_matrix_list.append(error_matrix)
            draw_graph(error_matrix_list)
        print('prikol')
