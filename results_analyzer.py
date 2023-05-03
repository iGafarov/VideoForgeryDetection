import json
import cv2
import numpy as np
from utils.constants import *
from utils.video_paths_collector import VideoPathsCollector
import matplotlib.pyplot as plt
from sklearn.metrics import *


ACCURACY = 5
MAX_SCORE = 5.0
# def calculate_confusion_matrix(result_frames: list, true_frames: list, frames_number: int):
#     confusion_matrix = np.zeros((2, 2), dtype=int)
#     TN, TP, FP, FN = 0, 0, 0, 0
#     for i in range(0, len(result_frames)):
#         if result_frames[i] > 0:
#             if true_frames[i] > 0:
#                 TP += 1
#             else:
#                 FP += 1
#         else:
#             if true_frames[i] > 0:
#                 FN += 1
#     TN = frames_number - TP - FP - FN
#     confusion_matrix[1, 1] = TN
#     confusion_matrix[0, 0] = TP
#     confusion_matrix[0, 1] = FP
#     confusion_matrix[1, 0] = FN
#     return confusion_matrix


def calculate_frame_with_accuracy(frame: int):
    frame_with_accuracy = []

    left = frame - ACCURACY
    right = frame + ACCURACY
    if left < 0:
        left = 0
    for i in range(left, right + 1):
        frame_with_accuracy.append(i)

    return frame_with_accuracy


def zeros_appending(result_frames: list, true_frames: list, frames_number: int):
    matched_count = 0
    matched_frames = []
    result_frames_copy = result_frames.copy()

    for result_frame in result_frames:
        for accuracy_frame in calculate_frame_with_accuracy(result_frame):
            if true_frames.__contains__(accuracy_frame):
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

    raw_calculated_result_frames = sorted(result_frames)
    raw_calculated_true_frames = sorted(true_frames)

    index = 0
    for result_frame in raw_calculated_result_frames:
        if missing_on_result.__contains__(result_frame):
            raw_calculated_result_frames[index] = 0
        index += 1

    index = 0
    for true_frame in raw_calculated_true_frames:
        if missing_on_true.__contains__(true_frame):
            raw_calculated_true_frames[index] = 0
        index += 1

    y_true = np.zeros(frames_number, int)
    y_pred = np.zeros(frames_number, int)

    for i in range(1, frames_number + 1):
        if raw_calculated_result_frames.__contains__(i):
            for j in calculate_frame_with_accuracy(i):
                if result_frames_copy.__contains__(j):
                    y_pred[i - 1] = j + ACCURACY

        if raw_calculated_true_frames.__contains__(i):
            y_true[i - 1] = 1

    return y_pred, y_true


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


def draw_graphs(matrix_list: list):
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


def calculate_samples_with_percent(anomalies: list, true_frames: list, frames_number: int):
    y_true = np.zeros(frames_number)
    for i in range(0, frames_number):
        if true_frames.__contains__(i):
            y_true[i - 1] = 1.0

    y_pred = np.zeros(frames_number)
    for anomaly in anomalies:
        score, frame = anomaly
        if score > MAX_SCORE:
            score = MAX_SCORE
        y_pred[frame] = score / MAX_SCORE

    return y_pred, y_true


def draw_roc_auc(pred, true, name):
    print(name, 'y_pred: ', pred)
    print(name, 'y_true: ', true)
    fpr, tpr, threshold = roc_curve(true, pred)
    roc_auc_score(true, pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    graph_path = GRAPH_PATH + '\\rocauc\\' + name + '_roc_auc' + '.png'
    print(name, '-auc: ', roc_auc_score(y_true, y_pred))
    plt.savefig(graph_path)
    plt.close()


def draw_graph(pred, name, frames_number):
    frames = []
    for i in range(0, frames_number):
        frames.append(i)
    graph_path = GRAPH_PATH + '\\graph\\' + name + '.png'
    plt.title('Video forgery')
    plt.xlabel('Frame Number')
    plt.ylabel('Anomaly Percent')
    plot_name = graph_path
    plt.plot(frames, pred)
    plt.savefig(plot_name)
    plt.close()


if __name__ == '__main__':
    with open(MANIPULATIONS_DETECTION_RESULT_PATH, "r") as results_file:
        all_manipulations = json.load(results_file)
        with open(TRUE_MANIPULATIONS_PATH, "r") as true_results_file:
            all_true_manipulations = json.load(true_results_file)
            error_matrix_list = []
            frames_counts = collect_frames_counts()
            y_preds = []
            y_trues = []
            for video_name in all_manipulations.keys():
                anomalies = all_manipulations[video_name]
                true_frames = all_true_manipulations[video_name]
                if len(anomalies) < frames_counts[video_name] - 50:
                    result_frames = []
                    for anomaly in anomalies:
                        score, frame = anomaly
                        contains = False
                        for result_frame in calculate_frame_with_accuracy(frame):
                            if result_frames.__contains__(result_frame):
                                result_frames.remove(result_frame)
                        result_frames.append(frame)
                    y_pred, y_true = zeros_appending(result_frames, true_frames, frames_counts[video_name])
                    print('y_pred: ', y_pred)
                    print('y_true: ', y_true)
                    error_matrix = confusion_matrix(y_true, y_pred)
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                    print('FPR: ', fpr)
                    print('TPR: ', tpr)
                    plt.plot(fpr, tpr)
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    graph_path = GRAPH_PATH + '\\' + video_name + '_roc_auc' + '.png'
                    plt.savefig(graph_path)
                    plt.close()
                    print(video_name + ': \n', error_matrix)
                    error_matrix_list.append(error_matrix)
                else:
                    y_pred, y_true = calculate_samples_with_percent(anomalies, true_frames, frames_counts[video_name])
                    draw_graph(y_pred, video_name, frames_counts[video_name])
                    draw_roc_auc(y_pred, y_true, video_name)
                    y_preds.append(y_pred)
                    y_trues.append(y_true)
                    # print('y_pred: ', y_pred)
                    # print('y_true: ', y_true)
                    # draw_graph(y_pred, y_true, video_name)
            y_true_general = []
            y_pred_general = []
            for y_true in y_trues:
                for percent in y_true:
                    y_true_general.append(percent)

            for y_pred in y_preds:
                for percent in y_pred:
                    y_pred_general.append(percent)
            draw_roc_auc(y_pred_general, y_true_general, 'general')
            # draw_graph(error_matrix_list)
        print('prikol')
