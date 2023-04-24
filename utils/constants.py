from utils.path_resolver import absolute_path

PATH_TO_VIDEOS = absolute_path("utils/video_paths").__str__()
# Result files
PATH_TO_RESULTS = absolute_path("results").__str__()
MANIPULATIONS_DETECTION_RESULT_PATH = absolute_path("results/json_results.json").__str__()
TRUE_MANIPULATIONS_PATH = absolute_path("results/true_results.json").__str__()

