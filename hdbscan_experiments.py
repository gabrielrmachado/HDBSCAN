from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import sklearn.metrics as metrics
import csv
import os

class Experiment(object):

    @staticmethod
    def read_csv(dataset_name):
        df = pd.read_csv(os.path.join("datasets", dataset_name), sep=',', header = 0)
        return df.values[:, :2], df.values[:, 2]

    @staticmethod
    def run_hdbscan(dataset, min_cluster_size):
        clusterer = HDBSCAN(min_cluster_size)
        return clusterer.fit_predict(dataset)

    @staticmethod
    def run_dbscan(dataset, min_cluster_size, eps):
        return DBSCAN(eps = eps, min_samples = min_cluster_size).fit_predict(dataset)

if __name__ == "__main__":
    data, classes = Experiment.read_csv("Aggregation.csv")
    h_clusterer_labels = Experiment.run_hdbscan(data, 7)
    d_clusterer_labels = Experiment.run_dbscan(data, 20, 7)
    print(metrics.fowlkes_mallows_score(classes, h_clusterer_labels))
    print(metrics.fowlkes_mallows_score(classes, d_clusterer_labels))