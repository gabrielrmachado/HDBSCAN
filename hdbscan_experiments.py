import hdbscan
import csv
import os

class Point:
    def __init__(self, x, y, cl):
        self.x = x
        self.y = y
        self.cl = cl

class Experiment(object):
    dataset = []

    @staticmethod
    def read_csv(dataset_name):
        file = open(os.path.join("datasets", dataset_name))
        reader = csv.reader(file)

        headers = next(reader, None)
        for col in reader:
            Experiment.dataset.append(Point(col[0], col[1], col[2]))

    @staticmethod
    def print_dataset():
        size = len(Experiment.dataset)
        if size == 0: print("No dataset read yet...")
        else:
            i = 1
            for data in Experiment.dataset:
                print("Point {0}: x: {1}, y: {2}, class: {3}".format(i, data.x, data.y, data.cl))
                i = i + 1

if __name__ == "__main__":
    Experiment.read_csv("Aggregation.csv")
    Experiment.print_dataset()

# data, _ = make_blobs(1000)

# clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
# cluster_labels = clusterer.fit_predict(data)

# print(cluster_labels)