import json
import cv2
from utils.constants import *
from utils.threshold_calculator import ThresholdCalculator
from utils.video_paths_collector import VideoPathsCollector
import matplotlib.pyplot as plt
from utils.results_analyzer_helper import *


EXPECTED_ALL_MANIPULATIONS_PERCENT = 80


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


def draw_graphs(result_manipulations: dict, true_manipulations: dict, frames_numbers):
    for video_name in result_manipulations.keys():
        manipulations = result_manipulations[video_name]
        true_frames = true_manipulations[video_name]
        y_pred, y_true = calculate_samples_with_percent(manipulations, true_frames, frames_numbers[video_name])
        draw_graph(y_pred, video_name, frames_numbers[video_name])
        # print('y_pred: ', y_pred)
        # print('y_true: ', y_true)


def draw_error_matrices(error_matrices: dict):
    for video_name in error_matrices.keys():
        print(video_name + ': \n', error_matrices[video_name])


if __name__ == '__main__':
    all_result_manipulations = {}
    all_true_manipulations = {}
    frames_counts = collect_frames_counts()

    with open(MANIPULATIONS_DETECTION_RESULT_PATH, "r") as results_file:
        all_result_manipulations = json.load(results_file)
    with open(TRUE_MANIPULATIONS_PATH, "r") as true_results_file:
        all_true_manipulations = json.load(true_results_file)

    all_result_manipulations = generate_manipulations_with_percents(all_result_manipulations)
    # Initialize Threshold Calculator
    threshold_calculator = ThresholdCalculator(all_true_manipulations, all_result_manipulations, frames_counts)
    # Parameter of Neyman Pearson algorithm
    # means the percentage of real Manipulations, at which we will get the threshold for the graphs
    threshold = threshold_calculator.calculate_percent_via_neyman_pearson(EXPECTED_ALL_MANIPULATIONS_PERCENT)
    print('Neyman-Pearson threshold: ', threshold)

    all_result_manipulations_with_threshold = threshold_calculator.manipulations_formation_with_threshold(threshold)

    # Draw manipulations graphs of all videos
    draw_graphs(all_result_manipulations, all_true_manipulations, frames_counts)
    # Draw error matrices of featured dictionary of manipulations
    error_matrix_dict = calculate_error_matrices(
        all_result_manipulations_with_threshold, all_true_manipulations, frames_counts)
    draw_error_matrices(error_matrix_dict)

    # with open(MANIPULATIONS_DETECTION_RESULT_PATH, "r") as results_file:
    #     all_manipulations = json.load(results_file)
    #     with open(TRUE_MANIPULATIONS_PATH, "r") as true_results_file:
    #         all_true_manipulations = json.load(true_results_file)
    #         error_matrix_list = []
    #         frames_counts = collect_frames_counts()
    #         y_preds = []
    #         y_trues = []
    #         for video_name in all_manipulations.keys():
    #             anomalies = all_manipulations[video_name]
    #             true_frames = all_true_manipulations[video_name]
    #             if len(anomalies) < frames_counts[video_name] - 50:
    #                 result_frames = []
    #                 for anomaly in anomalies:
    #                     score, frame = anomaly
    #                     contains = False
    #                     for result_frame in calculate_frame_with_accuracy(frame):
    #                         if result_frames.__contains__(result_frame):
    #                             result_frames.remove(result_frame)
    #                     result_frames.append(frame)
    #                 y_pred, y_true = zeros_appending(result_frames, true_frames, frames_counts[video_name])
    #                 print('y_pred: ', y_pred)
    #                 print('y_true: ', y_true)
    #                 error_matrix = confusion_matrix(y_true, y_pred)
    #                 fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    #                 print('FPR: ', fpr)
    #                 print('TPR: ', tpr)
    #                 plt.plot(fpr, tpr)
    #                 plt.xlabel('False Positive Rate')
    #                 plt.ylabel('True Positive Rate')
    #                 graph_path = GRAPH_PATH + '\\' + video_name + '_roc_auc' + '.png'
    #                 plt.savefig(graph_path)
    #                 plt.close()
    #                 print(video_name + ': \n', error_matrix)
    #                 error_matrix_list.append(error_matrix)
    #             else:
    #                 y_pred, y_true = calculate_samples_with_percent(anomalies, true_frames, frames_counts[video_name])
    #                 draw_graph(y_pred, video_name, frames_counts[video_name])
    #                 # draw_roc_auc(y_pred, y_true, video_name)
    #                 y_preds.append(y_pred)
    #                 y_trues.append(y_true)
    #                 # print('y_pred: ', y_pred)
    #                 # print('y_true: ', y_true)
    #                 # draw_graph(y_pred, y_true, video_name)
    #         y_true_general = []
    #         y_pred_general = []
    #         for y_true in y_trues:
    #             for percent in y_true:
    #                 y_true_general.append(percent)
    #
    #         for y_pred in y_preds:
    #             for percent in y_pred:
    #                 y_pred_general.append(percent)
    #         draw_roc_auc(y_pred_general, y_true_general, 'general')
    #         # draw_graph(error_matrix_list)
    #     print('prikol')
