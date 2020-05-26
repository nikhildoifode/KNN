# Assumption:
It is assumed that in the dataset, last column is for label. Also, the positive label is denoted by 1
and negative label is denoted by 0. The values are hardcoded as 0 and 1.

# Notes:
We will be splitting the dataset based on the split size from user input. For example, if the split size
is 0.2 then training data will have 80% of the data and test data will have rest 20% of data.
We are also shuffling the data based on random seed. Hence the output will be different in each iteration.
Error value in output is calculated on the test data. It is out of 1.

# How to run:
$ python3 knn.py --dataset /path/to/data/filename.csv --k <k_value> --split <split_value (0.1-0.5)>

# Example:
$ python3 knn.py --dataset Breast_cancer_data.csv --k 3 --split 0.2