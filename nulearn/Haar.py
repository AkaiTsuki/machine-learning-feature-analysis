__author__ = 'jiachiliu'

import numpy as np
import random
import sys


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return "rect: %s %s %s %s" % (self.x, self.y, self.w, self.h)

    def __repr__(self):
        return self.__str__()


class Haar:
    def __init__(self):
        self.search_table = None
        self.rectangles = None

    def extract_features(self, train, inners=100):
        features = []
        total = len(train)
        count = 0
        for img in train:
            features.append(self.extract_feature(img, inners))
            count += 1
            self.print_progress(count, total)
        return np.array(features)

    def extract_feature(self, matrix, inners=100):
        """
        Extract features from the given image pixel matrix
        :param matrix: a matrix that represents the image pixels
        :param rects: number of rectangle a inner image will be split
        :return: a list of haar features
        """
        m, n = matrix.shape
        self.search_table = np.zeros((m, n))
        self.compute_search_table(matrix)

        if self.rectangles is None:
            self.rectangles = self.random_rectangles(inners)

        haar_features = []
        for rect in self.rectangles:
            haar_features += self.compute_features(rect)
        return haar_features

    def random_rectangles(self, num):
        rects = []
        for i in range(num):
            rects.append(self.random_rectangle())
        return rects

    def random_rectangle(self):
        m, n = self.search_table.shape
        x = random.randint(0, m - 6)
        y = random.randint(0, n - 6)
        w = random.randint(5, m - y - 1)
        h = random.randint(5, n - x - 1)
        return Rect(x, y, w, h)

    def compute_features(self, rect):
        features = []
        rbx = rect.x + rect.h - 1
        rby = rect.y + rect.w - 1

        half_x = (rect.x + rbx) / 2
        half_y = (rect.y + rby) / 2

        h1 = self.black(rect.x, rect.y, half_x, rby)
        h2 = self.black(half_x + 1, rect.y, rbx, rby)
        features.append(h1 - h2)

        v1 = self.black(rect.x, rect.y, rbx, half_y)
        v2 = self.black(rect.x, half_y + 1, rbx, rby)
        features.append(v1 - v2)

        return features

    def compute_search_table(self, image):
        m, n = self.search_table.shape

        self.search_table[0][0] = image[0][0]
        # compuate the first row
        for col in range(1, n):
            self.search_table[0][col] = self.search_table[0][col - 1] + image[0][col]

        # compute the first column
        for row in range(1, m):
            self.search_table[row][0] = self.search_table[row - 1][0] + image[row][0]

        # fill the table
        for r in range(1, m):
            for c in range(1, n):
                self.search_table[r][c] = self.search_table[r][c - 1] + self.search_table[r - 1][c] - \
                                          self.search_table[r - 1][c - 1] + image[r][c]

    def black(self, x1, y1, x2, y2):
        """
        Compute the number of black pixels in given rectangle
        :param x1: left top x
        :param y1: left top y
        :param x2: right bottom x
        :param y2: right bottom y
        :return: the sum of pixel value in rectangle
        """
        A = 0.0 if x1 == 0 or y1 == 0 else self.search_table[x1 - 1][y1 - 1]
        B = 0.0 if x1 == 0 else self.search_table[x1 - 1][y2]
        C = 0.0 if y1 == 0 else self.search_table[x2][y1 - 1]
        D = self.search_table[x2][y2]
        return D - B - C + A

    @staticmethod
    def print_progress(k, max_loop):
        sys.stdout.write("\rProgress: %s/%s: " % (k, max_loop))
        sys.stdout.flush()