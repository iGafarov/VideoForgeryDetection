import os

import xlsxwriter


class ExcelWriter:
    def __init__(self, output_path, name):
        self.output_path = output_path
        self.name = name

    def write(self, all_manipulations: dict, remove_last_results: bool):
        if remove_last_results:
         os.remove(self.output_path)
        full_path = self.output_path + '\\' + self.name + '.xlsx'
        workbook = xlsxwriter.Workbook(full_path)
        worksheet = workbook.add_worksheet()
        i = 0
        for video_name in all_manipulations.keys():
            anomalies = all_manipulations[video_name]
            worksheet.write(0, 0 + i, video_name)
            worksheet.write(0, 1 + i, 'frame')
            worksheet.write(0, 2 + i, 'score')
            j = 0
            for anomaly in anomalies:
                score, frame = anomaly
                worksheet.write(1 + j, 1 + i, frame)
                worksheet.write(1 + j, 2 + i, score)
                j += 1
            i += 3
        workbook.close()