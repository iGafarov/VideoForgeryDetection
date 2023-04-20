class VideoPathsCollector:
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def collect(self):
        video_paths = []
        with open(self.video_paths, "r") as file_object:
            for video_path in file_object.readlines():
                video_path = video_path.strip()
                video_paths.append(video_path)
        return video_paths
