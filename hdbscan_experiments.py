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
    def __write_csv(file_name, data, gt_labels, clusterer_labels):
        with open(os.path.join("output_files", file_name), mode = 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter = ';')
            writer.writerow(["Index", "X", "Y", "GT_Label", "Predicted_Label"])

            size = len(data)
            for i in range(size):
                writer.writerow([i+1, data[i][0], data[i][1], gt_labels[i], clusterer_labels[i]])
        
        print("File {0} created successfully!".format(file_name))

    @staticmethod
    def __run_hdbscan(dataset, eps, min_cluster_size, min_samples):
        clusterer = HDBSCAN(cluster_selection_epsilon = eps, min_cluster_size = min_cluster_size, min_samples = min_samples, 
        algorithm='generic')
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
    def label_mapping(predicted_labels, gt_labels):
        df = pd.DataFrame({'Predicted': predicted_labels, 'Ground_Truth': gt_labels})
        uniques = np.unique(predicted_labels, return_counts=False)

        for label in uniques:
            df_c = (df.loc[df['Predicted'] == label]).groupby('Ground_Truth').count()
            max_label = df_c.index[df_c['Predicted'].argmax()]
            # for i in range(df.shape[0]):
            #     if df.loc[i, 'Predicted'] == label: 
            #         df.loc[i, 'Predicted'] = max_label

            indices = df.index[df['Predicted'] == label].tolist()
            df.loc[indices, 'Predicted'] = max_label

        return df['Predicted'].to_numpy()

    @staticmethod
    def baseline1(dataset_name, eps_dbscan, minPts_dbscan, eps_hdbscan, minPts_hdbscan, min_samples_hdbscan, cluster_algorithm, write_csv=False):
        data, classes = Experiment.__read_csv(dataset_name)
        clusterer_labels = []

        # data = StandardScaler().fit_transform(data)
        if cluster_algorithm == Cluster_Algorithm.DBSCAN: print("\nDBSCAN EXPERIMENTS...")
        if cluster_algorithm == Cluster_Algorithm.HDBSCAN: print("\nHDBSCAN EXPERIMENTS...")
        print(dataset_name.upper())

        print("\nGround-truth labels")
        unique, counts = np.unique(classes, return_counts=True)
        print(dict(zip(unique, counts)))

        if cluster_algorithm == Cluster_Algorithm.DBSCAN:
             clusterer_labels = Experiment.__run_dbscan(data, eps_dbscan, minPts_dbscan)
             unique, counts = np.unique(clusterer_labels, return_counts=True)

        elif cluster_algorithm == Cluster_Algorithm.HDBSCAN: 
            clusterer_labels = Experiment.__run_hdbscan(data, eps_hdbscan, minPts_hdbscan, min_samples_hdbscan)
            unique, counts = np.unique(clusterer_labels, return_counts=True)
            
        print("Cluster labels")
        clusterer_labels = Experiment.label_mapping(clusterer_labels, classes)
        print(dict(zip(unique, counts)))

        print("\nNumber of ground-truth clusters: %d" % len(set(classes)))
        print("Number of clusters found by {0}: {1}".format(cluster_algorithm.name, len(set(clusterer_labels))))
        print("FM Index ({0}): {1}".format(cluster_algorithm.name, metrics.fowlkes_mallows_score(classes, clusterer_labels)))
        print("Accuracy ({0}): {1:.10f}".format(cluster_algorithm.name, metrics.accuracy_score(classes, clusterer_labels)))

        Experiment.label_mapping(clusterer_labels, classes)

        if write_csv:
            file_name = cluster_algorithm.name + "_" + dataset_name
            Experiment.__write_csv(file_name, data, classes, clusterer_labels)

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
    def baseline2(dataset_name, reduction, eps, minPts, min_samples, cluster_algorithm):
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

            # if cluster_algorithm == Cluster_Algorithm.DBSCAN: 
            #     _, d_clusterer_labels = DBSCAN_Brandao.dbFun(data_rnd, data_rnd, eps, minPts, "teste", classes_rnd, plot=False, print_strMat=False)

            if cluster_algorithm == Cluster_Algorithm.DBSCAN: clusterer_labels = Experiment.__run_dbscan(data_rnd, eps, minPts)
            elif cluster_algorithm == Cluster_Algorithm.HDBSCAN: clusterer_labels = Experiment.__run_hdbscan(data_rnd, eps, minPts, min_samples)

            clusterer_labels = Experiment.label_mapping(clusterer_labels, classes_rnd)

            fm_indexes.append(metrics.fowlkes_mallows_score(classes_rnd, clusterer_labels))
            accuracies.append(metrics.accuracy_score(classes_rnd, clusterer_labels))
        
        print("FM-Index: Mean = {0}\tSTD = {1}".format(mean(fm_indexes), stdev(fm_indexes)))
        print("Accuracy: Mean = {0}\tSTD = {1}\n".format(mean(accuracies), stdev(accuracies)))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    parameters = [["aggregation.csv", 0.042, 7, 0.042, 7, 9, 0.3959]]

    # parameters = [
    #     ["aggregation.csv", 0.042, 7, 0.042, 7, 9, 0.3959], # algorithm = 'generic
    #     ["diamond9.csv", 0.03, 12, 0.015, 12, 9, 0.7432],
    #     ["cluto-t4-8k.csv", 0.02, 25, 0.005, 23, 50, 0.8606],
    #     ["cluto-t5-8k.csv", 0.02, 25, 0.012, 25, 2, 0.9122],
    #     ["cluto-t7-10k.csv", 0.025, 28, 0.015, 28, 33, 0.7796],
    #     ["cluto-t8-8k.csv", 0.0218, 14, 0.02, 20, 9, 0.7556]]

    print("=========================")
    print("BASELINE 1 EXPERIMENTS...")
    print("=========================")
    for param in parameters:
        Experiment.baseline1(param[0], param[1], param[2], param[3], param[4], param[5], Cluster_Algorithm.DBSCAN, write_csv=True)
        Experiment.baseline1(param[0], param[1], param[2], param[3], param[4], param[5], Cluster_Algorithm.HDBSCAN, write_csv=True)

    # print("\n=========================")
    # print("BASELINE 2 EXPERIMENTS...")
    # print("=========================")
    # for param in parameters:
    #     Experiment.baseline2(param[0], param[6], param[1], param[2], param[5], Cluster_Algorithm.DBSCAN)
    #     Experiment.baseline2(param[0], param[6], param[3], param[4], param[5], Cluster_Algorithm.HDBSCAN)