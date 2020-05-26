import numpy as np
import csv
import sys
import operator


class KNearestNeighbors:
    def __init__ (self, k):
        self.k = int(k)


    def shuffle_data(self, X, y):
        np.random.seed(None)
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        return X[index], y[index]


    def train_test_split(self, X, y, test_size):
        X, y = self.shuffle_data(X, y)
        split_i = len(y) - int(len(y) // (1 / test_size))
        X_train, X_test = X[:split_i], X[split_i:]
        y_train, y_test = y[:split_i], y[split_i:]

        return X_train, X_test, y_train, y_test


    def euclid_distance(self, v1, v2):
        distance = 0.0
        for i in range(len(v1) - 1):
            distance += (v1[i] - v2[i]) ** 2

        return np.sqrt(distance)


    def predict(self, X, y, row):
        distance = []
        for i in range(len(X)):
            distance.append((y[i], self.euclid_distance(X[i], row)))
            distance.sort(key = operator.itemgetter(1))
            neighbors = []
            count1 = 0
            count0 = 0
            neighbors = distance[:self.k]
            for k in neighbors:
                if k[0] == 1: count1 += 1
                else: count0 += 1

        if count1 > count0: return 1
        return 0


    def calculate_accuracy(self, label, prediction):
        return np.mean(label == prediction)


def main ():
    if len(sys.argv) != 7 or '--dataset' not in sys.argv or '--k' not in sys.argv or '--split' not in sys.argv:
        print("Invalid command")
        print("Make sure command is of type: " +
        "python3 knn.py --dataset /path/to/data/filename.csv --k <k_value> --split 0.2")
        return

    filePath = sys.argv[sys.argv.index('--dataset') + 1]
    k = sys.argv[sys.argv.index('--k') + 1]
    split = float(sys.argv[sys.argv.index('--split') + 1])

    try:
        with open(filePath)	as csvfile:
            reader = csv.reader(csvfile)
            data_list = list(reader)
            data_list.pop(0)

            X = np.array(data_list, dtype = np.float64)
            y = X[:, X.shape[1] - 1]
            y = y.astype(int)
            X = X[:, :-1]

            knn = KNearestNeighbors(k)

            X_train, X_test, y_train, y_test = knn.train_test_split(X, y, split)

            y_pred = []
            for row in X_test:
                y_pred.append(knn.predict(X_train, y_train, row))

            print("Error: ", 1 - knn.calculate_accuracy(y_pred, y_test))

    except IOError as e:
        print("Couldn't open the file (%s). Check file path, value of k and name again." % e)


if __name__ == "__main__":
    main()