import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from utils.constants import *
from utils.video_paths_collector import VideoPathsCollector
from utils.excel_writer import ExcelWriter
import time
import json


def get_video_name(path: str):
    parsed = path.split("\\")
    return parsed[len(parsed) - 1]


if __name__ == '__main__':
    th = 0
    no_of_forgery = []
    video_name = []
    video_paths_collector = VideoPathsCollector(PATH_TO_VIDEOS)
    videos_paths = video_paths_collector.collect()
    all_manipulations = {}
    excel_writer = ExcelWriter(PATH_TO_RESULTS, 'results')
    start = time.time()
    for video_path in videos_paths:
        file = get_video_name(video_path)
        print('Process ', file)
        cap = cv2.VideoCapture(video_path)
        ret, frame1 = cap.read()
        try:
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        except BaseException:
            print('ERROR ', video_path)
            continue
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        frame_no = []
        op_flow_per_frame = []
        m = 1
        f = 1
        b = 1
        a = frame1.size
        s = np.arange(a)
        frame_number = 0
        while (1):
            s = 0
            frame_number = frame_number + 1
            ret, frame2 = cap.read()
            print('frame number: ', frame_number)

            if ret == True:
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                frame_no.append(m)
                m = m + 1
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                op_flow_1D = np.resize(mag, (1, a))
                for i in op_flow_1D[0]:
                    s  = s + i
                op_flow_per_frame.append(s)
                prvs = next
                b = b + 1
            else:
                break
        vrt_factor = []
        vrt_factor_round_2 = []
        j = 1
        awq = []
        vrt_factor.append(1)
        for i in range(1, (m)):
            awq.append(i)
        for o in (range(m - 3)):
            c = (2 * op_flow_per_frame[j]) / (op_flow_per_frame[(j - 1)] + op_flow_per_frame[(j + 1)])
            vrt_factor.append(c)
            j = j + 1
        vrt_factor.append(1)
        for i in vrt_factor:
            i = round(i, 2)
            vrt_factor_round_2.append(i)
        sum = np.sum(vrt_factor_round_2)
        mean = (sum * 1.0) / (b)

        mean = round(mean, 3)
        y = 0
        poi = []
        for i in vrt_factor_round_2:
            y = y + ((i - mean) * (i - mean))
        st = (y * 1.0) / (b)
        st = round(st, 3)
        fg = math.sqrt(2 * (22 / 7) * st)
        anamoly_score = []
        st = math.sqrt(st)
        for i in vrt_factor_round_2:
            kj = abs((i - mean))
            df = (kj * 1.0) / st
            anamoly_score.append(df)
        bv = 0
        anomalies = []
        print('anomaly score!: ', anamoly_score)
        for i in range(len(anamoly_score)):
            if anamoly_score[i] >= th:
                anomaly = (anamoly_score[i], i)
                anomalies.append(anomaly)
                bv = bv + 1
        no_of_forgery.append(bv)
        video_name.append(file)
        all_manipulations[file] = anomalies
        plt.title('Video forgery')
        plt.xlabel('Frame Number')
        plt.ylabel('Anomly Score')
        plot_name = file + '.png'
        plt.plot(frame_no, anamoly_score)
        plt.savefig(plot_name)
        plt.close()
        cv2.destroyAllWindows()
        cap.release()
    end = time.time()
    print('general_time: ', end - start)
    with open(MANIPULATIONS_DETECTION_RESULT_PATH, "w") as write_file:
        json.dump(all_manipulations, write_file)
    excel_writer.write(all_manipulations, False)
