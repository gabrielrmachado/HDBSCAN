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
from random import sample
from statistics import mean, stdev
from enum import Enum
import csv
import os
import sys

class Cluster_Algorithm(Enum):
    DBSCAN = 1
    HDBSCAN = 2

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

    @staticmethod
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
    def baseline1(dataset_name, eps_dbscan, minPts_dbscan, eps_hdbscan, minPts_hdbscan, cluster_algorithm):
        data, classes = Experiment.__read_csv(dataset_name)
        d_clusterer_labels = []
        h_clusterer_labels = []

        # data = StandardScaler().fit_transform(data)
        if cluster_algorithm == Cluster_Algorithm.DBSCAN: print("\nDBSCAN EXPERIMENTS...")
        if cluster_algorithm == Cluster_Algorithm.HDBSCAN: print("\nHDBSCAN EXPERIMENTS...")
        print(dataset_name.upper())

        print("\nGround-truth labels")
        unique, counts = np.unique(classes, return_counts=True)
        print(dict(zip(unique, counts)))

        if cluster_algorithm == Cluster_Algorithm.DBSCAN:
             d_clusterer_labels = Experiment.__run_dbscan(data, eps_dbscan, minPts_dbscan)
             unique, counts = np.unique(d_clusterer_labels, return_counts=True)

        elif cluster_algorithm == Cluster_Algorithm.HDBSCAN: 
            h_clusterer_labels = Experiment.__run_hdbscan(data, eps_hdbscan, minPts_hdbscan)
            unique, counts = np.unique(h_clusterer_labels, return_counts=True)
            
        print("Cluster labels")
        print(dict(zip(unique, counts)))

        print("\nNumber of ground-truth clusters: %d" % len(set(classes)))

        if cluster_algorithm == Cluster_Algorithm.DBSCAN: 
            print("Number of clusters found by DBSCAN: %d" % len(set(d_clusterer_labels)))
            print("FM Index (DBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, d_clusterer_labels)))
            print("Accuracy (DBSCAN): {0:.10f}".format(Experiment.compute_acc(classes, d_clusterer_labels)))
        
        elif cluster_algorithm == Cluster_Algorithm.HDBSCAN: 
            print("Number of clusters found by HDBSCAN: %d" % len(set(h_clusterer_labels)))
            print("FM Index (HDBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, h_clusterer_labels)))
            print("Accuracy (HDBSCAN): {0}".format(metrics.accuracy_score(classes, h_clusterer_labels)))

    @staticmethod 
    def baseline1_brandao(dataset_name, eps_dbscan, minPts_dbscan):
        data, classes = Experiment.__read_csv(dataset_name)

        # data = StandardScaler().fit_transform(data)
        print(dataset_name.upper())

        _, d_labels = DBSCAN_Brandao.dbFun(data, data, eps_dbscan, minPts_dbscan, "teste", classes)
        unique, counts = np.unique(d_labels, return_counts=True)
        print(dict(zip(unique, counts)))

        Experiment.__run_dbscan(data, eps_dbscan, minPts_dbscan)

        print("Number of clusters found by DBSCAN: %d" % len(set(d_labels)))
        print("Number of ground-truth clusters: %d" % len(set(classes)))

        print("FM Index (DBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, d_labels)))
        print("Accuracy (DBSCAN): {0:.10f}".format(Experiment.compute_acc(classes, d_labels)))

    @staticmethod
    def baseline2(dataset_name, reduction, eps, minPts, cluster_algorithm):
        data, classes = Experiment.__read_csv(dataset_name)
        print("\n" + dataset_name.upper())

        fm_indexes = []
        accuracies = []

        if cluster_algorithm == Cluster_Algorithm.DBSCAN: print("DBSCAN 100 EXPERIMENTS...")
        if cluster_algorithm == Cluster_Algorithm.HDBSCAN: print("HDBSCAN 100 EXPERIMENTS...")

        for i in range(100):    
            len_data = len(data)
            qtd_points = (int)(len_data * (1 - reduction))
            if i == 0: print("Number of random samples: {0}".format(qtd_points))
            rnd_idx = sample(range(len(data)), qtd_points)

            data_rnd = data[rnd_idx]
            classes_rnd = classes[rnd_idx]

            if cluster_algorithm == Cluster_Algorithm.DBSCAN: 
                _, d_clusterer_labels = DBSCAN_Brandao.dbFun(data_rnd, data_rnd, eps, minPts, "teste", classes_rnd, plot=False, print_strMat=False)

            # if cluster_algorithm == Cluster_Algorithm.DBSCAN: d_clusterer_labels = Experiment.__run_dbscan(data_rnd, eps, minPts)
            elif cluster_algorithm == Cluster_Algorithm.HDBSCAN: d_clusterer_labels = Experiment.__run_hdbscan(data_rnd, eps, minPts)

            fm_indexes.append(metrics.fowlkes_mallows_score(classes_rnd, d_clusterer_labels))
            accuracies.append(metrics.accuracy_score(classes_rnd, d_clusterer_labels))
        
        print("FM-Index: Mean = {0}\tSTD = {1}".format(mean(fm_indexes), stdev(fm_indexes)))
        print("Accuracy: Mean = {0}\tSTD = {1}\n".format(mean(accuracies), stdev(accuracies)))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    # parameters = [["cluto-t8-8k.csv", 0.0218, 14, 0.0218, 14]]

    parameters = [
        ["Aggregation.csv", 0.042, 7, 0.042, 7, 0.3959],
        ["diamond9.csv", 0.03, 12, 0.03, 12, 0.7432],
        ["cluto-t4-8k.csv", 0.02, 25, 0.02, 25, 0.8606],
        ["cluto-t5-8k.csv", 0.02, 25, 0.02, 25, 0.9122],
        ["cluto-t7-10k.csv", 0.025, 28, 0.025, 28, 0.7796],
        ["cluto-t8-8k.csv", 0.0218, 14, 0.0218, 14, 0.7556]]

    # print("\n=========================")
    # print("BASELINE 2 EXPERIMENTS...")
    # print("=========================")
    # for param in parameters:
    #     Experiment.baseline2(param[0], param[5], param[1], param[2], Cluster_Algorithm.DBSCAN)
    #     Experiment.baseline2(param[0], param[5], param[3], param[4], Cluster_Algorithm.HDBSCAN)
    
    print("=========================")
    print("BASELINE 1 EXPERIMENTS...")
    print("=========================")
    for param in parameters:
        Experiment.baseline1(param[0], param[1], param[2], param[3], param[4], Cluster_Algorithm.DBSCAN)
        Experiment.baseline1(param[0], param[1], param[2], param[3], param[4], Cluster_Algorithm.HDBSCAN)
    