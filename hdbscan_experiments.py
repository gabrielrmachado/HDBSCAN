import hdbscan
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import csv
import os

class Experiment(object):

    @staticmethod
    def read_csv(dataset_name):
        df = pd.read_csv(os.path.join("datasets", dataset_name), sep=',', header = 0)
        return df.values[:, :2], df.values[:, 2]

    @staticmethod
    def run_hdbscan(dataset, min_cluster_size):
        clusterer = hdbscan.HDBSCAN(min_cluster_size)
        return clusterer.fit_predict(dataset)

if __name__ == "__main__":
    data, classes = Experiment.read_csv("Aggregation.csv")
    clusterer_labels = Experiment.run_hdbscan(data, 7)
    print(clusterer_labels)