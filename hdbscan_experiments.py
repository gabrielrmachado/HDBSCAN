from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize 
import csv
import os
import sys

class Experiment(object):

    @staticmethod
    def read_csv(dataset_name):
        df = pd.read_csv(os.path.join("datasets", dataset_name), sep=',', header = 0)
        return df.values[:, :2], df.values[:, 2]

    @staticmethod
    def run_hdbscan(dataset, eps, min_cluster_size):
        dataset_scaled = StandardScaler().fit_transform(dataset)
        dataset_norm = normalize(dataset_scaled)
        clusterer = HDBSCAN(cluster_selection_epsilon = eps, min_cluster_size = min_cluster_size)
        return clusterer.fit_predict(dataset_norm)

    @staticmethod
    def run_dbscan(dataset, eps, min_samples):
        dataset_scaled = StandardScaler().fit_transform(dataset)
        dataset_norm = normalize(dataset_scaled)
        clusterer = DBSCAN(eps = eps, min_samples = min_samples).fit(dataset_norm)
        return clusterer.fit_predict(dataset_norm)

    @staticmethod
    def example():
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                    random_state=0)

        X = StandardScaler().fit_transform(X)

        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))

if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)

    data, classes = Experiment.read_csv("Diamond9.csv")
    h_clusterer_labels = Experiment.run_hdbscan(data, 0.03, 12)
    d_clusterer_labels = Experiment.run_dbscan(data, 0.03, 12)
    # Experiment.example()

    print("Number of clusters found by HDBSCAN: %d" % len(set(h_clusterer_labels)))
    print(h_clusterer_labels)
    print("Number of clusters found by DBSCAN: %d" % len(set(d_clusterer_labels)))
    print(d_clusterer_labels)

    print("FM Index (HDBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, h_clusterer_labels)))
    print("FM Index (DBSCAN): {0}".format(metrics.fowlkes_mallows_score(classes, d_clusterer_labels)))