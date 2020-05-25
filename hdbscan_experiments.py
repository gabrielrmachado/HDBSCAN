from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize 
import itertools
import csv
import os
import sys

class Experiment(object):

    @staticmethod
    def __read_csv(dataset_name):
        df = pd.read_csv(os.path.join("datasets", dataset_name), sep=',', header = 0)
        return df.values[:, :2], df.values[:, 2]

    @staticmethod
    def __run_hdbscan(dataset, eps, min_cluster_size):
        dataset_scaled = StandardScaler().fit_transform(dataset)
        dataset_norm = normalize(dataset_scaled)
        clusterer = HDBSCAN(cluster_selection_epsilon = eps, min_cluster_size = min_cluster_size)
        return clusterer.fit_predict(dataset_norm)

    @staticmethod
    def __run_dbscan(dataset, eps, min_samples):
        dataset_scaled = StandardScaler().fit_transform(dataset)
        dataset_norm = normalize(dataset_scaled)
        clusterer = DBSCAN(eps = eps, min_samples = min_samples).fit(dataset_norm)
        return clusterer.fit_predict(dataset_norm)

    @staticmethod
    def baseline_1(dataset_name, eps_hdbscan, minPts_hdbscan, eps_dbscan, minPts_dbscan):
        data, classes = Experiment.__read_csv(dataset_name)

        print(dataset_name.upper())
        d_clusterer_labels = Experiment.__run_dbscan(data, eps_dbscan, minPts_dbscan)
        h_clusterer_labels = Experiment.__run_hdbscan(data, eps_hdbscan, minPts_hdbscan)

        print("Number of clusters found by DBSCAN: %d" % len(set(d_clusterer_labels)))
        print("Number of clusters found by HDBSCAN: %d\n" % len(set(h_clusterer_labels)))

        print("FM Index (HDBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, h_clusterer_labels)))
        print("FM Index (DBSCAN): {0}\n\n".format(metrics.fowlkes_mallows_score(classes, d_clusterer_labels)))



if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    parameters = [
        ["Aggregation.csv", 0.042, 7, 0.042, 7],
        ["diamond9.csv", 0.03, 12, 0.03, 12],
        ["cluto-t4-8k.csv", 0.02, 25, 0.02, 25],
        ["cluto-t5-8k.csv", 0.02, 25, 0.02, 25],
        ["cluto-t7-10k.csv", 0.025, 12, 0.025, 28],
        ["cluto-t8-8k.csv", 0.0218, 14, 0.0218, 14]]

    for param in parameters:
        Experiment.baseline_1(param[0], param[1], param[2], param[3], param[4])
    