import os
import csv

import line_profiler
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets


class ReadDigits:
    def __init__(self, path: str):
        self._path = path
        self._data = None

    #@profile
    def read_train_np(self):
        self._data = np.genfromtxt(self._path, delimiter=',')

    #@profile
    def read_train_csv(self):
        with open(self._path,'r') as dest_f:
            data_iter = csv.reader(dest_f, delimiter=',')
            cols = next(data_iter)
            data = [data for data in data_iter]
            self._data = np.asarray(data)

    #@profile
    def read_train_loop(self):
        with open(self._path, 'r') as dest_f:
            data_iter = csv.reader(dest_f, delimiter=',')
            cols = next(data_iter)
            rows = []
            for row in data_iter:
                # take each row and get a list of lists
                # then reshape to build a list of lists of lists
                # np.reshape(rows, (8, 8))
                rows.append(row)
            self._data = np.asarray(rows)

    #@profile
    def read_sklearn(self):
        self._data = datasets.load_digits()
        plt.figure(1, figsize=(3, 3))
        plt.imshow(self._data.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
        import ipdb
        ipdb.set_trace()
        plt.show()

if __name__ == '__main__':
    training_data = '/Users/dtsmith/Temp/train.csv'
    # rd = ReadDigits(training_data).read_train_np()
    # rd = ReadDigits(training_data).read_train_csv()
    # rd = ReadDigits(training_data).read_train_loop()
    rd = ReadDigits(training_data).read_sklearn()
