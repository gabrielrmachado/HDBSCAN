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
        return df.values

    @staticmethod
    def run_hdbscan(dataset, min_cluster_size):
        clusterer = hdbscan.HDBSCAN(min_cluster_size)
        clusterer_labels = clusterer.fit_predict(dataset)

        print(clusterer_labels)

if __name__ == "__main__":
    dataset = Experiment.read_csv("Aggregation.csv")
    data = dataset[:, :2]
    classes = dataset[:, 2]
    print(data)
    print(classes)
    # Experiment.run_hdbscan(7)

    # data, _ = make_blobs(1000)

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    # cluster_labels = clusterer.fit_predict(data)

    # # print(cluster_labels)
    # print(data)