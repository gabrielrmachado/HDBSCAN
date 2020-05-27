from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
import DBSCAN_Brandao
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize 
from sklearn import preprocessing
import itertools
import csv
import os
import sys

class Experiment(object):

    @staticmethod
    def __read_csv(dataset_name):
        df = pd.read_csv(os.path.join("datasets", dataset_name), sep=',', header = 0)
        return df.values[:, :2], df.values[:, 2].astype(int)

    @staticmethod
    def __run_hdbscan(dataset, eps, min_cluster_size):
        clusterer = HDBSCAN(cluster_selection_epsilon = eps, min_cluster_size = min_cluster_size)
        return clusterer.fit_predict(dataset)

    @staticmethod
    def __run_dbscan(dataset, eps, min_samples):
        clusterer = DBSCAN(eps = eps, min_samples = min_samples)
        return clusterer.fit_predict(dataset)

    @ staticmethod
    def compute_acc(ground_truth_classes, predicted_classes):
        if len(ground_truth_classes) != len(predicted_classes):
            print("Arrays must have the same dimensions.")
        else:
            matchings = 0
            i = 0
            for elem in ground_truth_classes:
                if elem == predicted_classes[i]: 
                    matchings = matchings + 1
                i = i + 1
        return matchings / len(ground_truth_classes)

    @staticmethod
    def baseline1(dataset_name, eps_dbscan, minPts_dbscan, eps_hdbscan, minPts_hdbscan):
        data, classes = Experiment.__read_csv(dataset_name)

        # data = StandardScaler().fit_transform(data)
        print(dataset_name.upper())

        d_clusterer_labels = Experiment.__run_dbscan(data, eps_dbscan, minPts_dbscan)
        h_clusterer_labels = Experiment.__run_hdbscan(data, eps_hdbscan, minPts_hdbscan)

        print("Number of clusters found by DBSCAN: %d" % len(set(d_clusterer_labels)))
        print("Number of clusters found by HDBSCAN: %d\n" % len(set(h_clusterer_labels)))
        print("Number of ground-truth clusters: %d" % len(set(classes)))

        print("FM Index (DBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, d_clusterer_labels)))
        print("FM Index (HDBSCAN): {0}\n".format(metrics.fowlkes_mallows_score(classes, h_clusterer_labels)))
        
        print("Accuracy (DBSCAN): {0}".format(Experiment.compute_acc(classes, d_clusterer_labels)))
        print("Accuracy (HDBSCAN): {0}\n\n".format(metrics.accuracy_score(classes, h_clusterer_labels)))

    @staticmethod 
    def baseline1_brandao(dataset_name, eps_dbscan, minPts_dbscan):
        data, classes = Experiment.__read_csv(dataset_name)

        # data = StandardScaler().fit_transform(data)
        print(dataset_name.upper())

        _, d_clusterer_labels = DBSCAN_Brandao.dbFun(data, data, eps_dbscan, minPts_dbscan, "teste", classes)
        Experiment.__run_dbscan(data, eps_dbscan, minPts_dbscan)

        print("Number of clusters found by DBSCAN: %d" % len(set(d_clusterer_labels)))
        print("Number of ground-truth clusters: %d" % len(set(classes)))

        print("FM Index (DBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, d_clusterer_labels)))

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    parameters = [["cluto-t4-8k.csv", 0.02, 25, 0.02, 25]]

    # parameters = [
    #     ["Aggregation.csv", 0.042, 7, 0.042, 7],
    #     ["diamond9.csv", 0.03, 12, 0.03, 12],
    #     ["cluto-t4-8k.csv", 0.02, 25, 0.02, 25],
    #     ["cluto-t5-8k.csv", 0.02, 25, 0.02, 25],
    #     ["cluto-t7-10k.csv", 0.025, 28, 0.025, 28],
    #     ["cluto-t8-8k.csv", 0.0218, 14, 0.0218, 14]]

    for param in parameters:
        Experiment.baseline1_brandao(param[0], param[1], param[2])
    