from utils.results_analyzer_helper import calculate_error_matrices
from utils.results_analyzer_helper import calculate_total_tp
from utils.results_analyzer_helper import calculate_total_true_count


class ThresholdCalculator:
    def __init__(self, true_manipulations, result_manipulations, frames_numbers):
        self.true_manipulations = true_manipulations
        self.result_manipulations = result_manipulations
        self.frames_numbers = frames_numbers

    def calculate_percent_via_neyman_pearson(self, expected_percent: float):
        neyman_pearson_percent = 100
        for i in range(0, 100):
            neyman_pearson_percent = neyman_pearson_percent - 1
            all_result_manipulations_with_threshold = self.manipulations_formation_with_threshold(
                neyman_pearson_percent)
            error_matrix_dict = calculate_error_matrices(
                all_result_manipulations_with_threshold, self.true_manipulations, self.frames_numbers)
            total_tp = calculate_total_tp(error_matrix_dict)
            total_true_count = calculate_total_true_count(error_matrix_dict)
            potential_percent = (total_tp / total_true_count) * 100
            if potential_percent >= expected_percent:
                break
        return neyman_pearson_percent

    def manipulations_formation_with_threshold(self, threshold_percent: float):
        all_manipulations_with_threshold = {}
        for video_name in self.result_manipulations.keys():
            manipulations = self.result_manipulations[video_name]
            manipulations_with_threshold = []
            for manipulation in manipulations:
                frame, percent = manipulation
                if percent >= threshold_percent:
                    manipulations_with_threshold.append((frame, percent))
            all_manipulations_with_threshold[video_name] = manipulations_with_threshold

        return all_manipulations_with_threshold
