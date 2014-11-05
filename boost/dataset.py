__author__ = 'jiachiliu'


import numpy as np

class CsvReader:
    """
    CsvReader will read data from csv file
    """

    def __init__(self, path):
        self.path = path

    def read(self, delimiter, converter):
        f = open(self.path)
        lines = f.readlines()
        return self.parse_lines(lines, delimiter, converter)

    @staticmethod
    def parse_lines(lines, delimiter, converter):
        data = []
        for line in lines:
            if line.strip():
                row = [s.strip() for s in line.strip().split(delimiter) if s.strip()]
                data.append(row)

        return np.array(data, converter)

def load_spambase():
    reader = CsvReader('data/spambase.data')
    data = reader.read(',', float)
    total_col = data.shape[1]
    return data[:, :total_col - 1], data[:, total_col - 1]