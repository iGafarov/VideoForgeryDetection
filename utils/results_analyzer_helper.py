import numpy as np
from utils.constants import MAX_SCORE
from utils.constants import ACCURACY


def generate_manipulations_with_percents(result_manipulations: dict):
    to_return = {}
    for video_name in result_manipulations.keys():
        anomalies = result_manipulations[video_name]
        anomalies_with_percents = []
        for anomaly in anomalies:
            score, frame = anomaly
            if score > MAX_SCORE:
                score = MAX_SCORE

            percent = (score / MAX_SCORE) * 100
            anomalies_with_percents.append((frame, percent))
        to_return[video_name] = anomalies_with_percents
    return to_return


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

    for result_frame in result_frames:
        for accuracy_frame in calculate_frame_with_accuracy(result_frame):
            if true_frames.__contains__(accuracy_frame):
                if not matched_frames.__contains__(accuracy_frame):
                    result_frames[result_frames.index(result_frame)] = accuracy_frame
                    matched_frames.append(accuracy_frame)
                    matched_count += 1
                    break

    missing_on_result = true_frames.copy()
    missing_on_true = result_frames.copy()

    true_frames_copy = true_frames.copy()
    result_frames_copy = result_frames.copy()

    for matched_frame in matched_frames:
        missing_on_result.remove(matched_frame)
        missing_on_true.remove(matched_frame)

    for missing in missing_on_result:
        result_frames_copy.append(missing)
    for missing in missing_on_true:
        true_frames_copy.append(missing)

    raw_calculated_result_frames = sorted(result_frames_copy)
    raw_calculated_true_frames = sorted(true_frames_copy)

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
            y_pred[i - 1] = 1
        if raw_calculated_true_frames.__contains__(i):
            y_true[i - 1] = 1

    return y_pred, y_true


def calculate_samples_with_percent(manipulations: list, true_frames: list, frames_number: int):
    y_true = np.zeros(frames_number)
    for i in range(1, frames_number + 1):
        if true_frames.__contains__(i):
            y_true[i - 1] = 1.0

    y_pred = np.zeros(frames_number)
    for manipulation in manipulations:
        frame, percent = manipulation
        y_pred[frame - 1] = percent / 100

    return y_pred, y_true


def calculate_error_matrices(result_manipulations: dict, true_manipulations: dict, frames_numbers):
    error_matrices = {}
    for video_name in result_manipulations.keys():
        manipulations = result_manipulations[video_name]
        true_frames = true_manipulations[video_name]
        result_frames = []
        for manipulation in manipulations:
            frame, percent = manipulation
            contains = False
            for result_frame in calculate_frame_with_accuracy(frame):
                if result_frames.__contains__(result_frame):
                    contains = True
            if not contains:
                result_frames.append(frame)
        y_pred, y_true = zeros_appending(result_frames, true_frames, frames_numbers[video_name])
        # print('y_pred: ', y_pred)
        # print('y_true: ', y_true)
        error_matrix = calculate_confusion_matrix(y_pred, y_true, frames_numbers[video_name])
        error_matrices[video_name] = error_matrix
        # print(video_name + ': \n', error_matrix)
    return error_matrices


def calculate_total_tp(error_matrices: dict):
    total_tp = 0
    for video_name in error_matrices.keys():
        error_matrix = error_matrices[video_name]
        TP = error_matrix[0, 0]
        total_tp += TP
    return total_tp


def calculate_total_true_count(error_matrices: dict):
    total_true_count = 0
    for video_name in error_matrices.keys():
        error_matrix = error_matrices[video_name]
        TP = error_matrix[0, 0]
        FN = error_matrix[1, 0]
        total_true_count = total_true_count + TP + FN
    return total_true_count

