'''
SL Assignment - classification.py
CS 1410 Artificial Intelligence, Brown University
Written by whackett.

Usage-

To run classification using your perceptron implementation:
    python classification.py

To run classification using our KNN implementation:
    python classification.py -knn (or python classification.py -k)

'''
from sklearn import preprocessing
from knn import KNNClassifier
from perceptron import Perceptron
from cross_validate import cross_validate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def classification(method, train_size=0.9):
    """
    Classifies data using the model type specified in method. Default is the
    perceptron.

    Returns the model accuracy on the test data.
    """

    # Load the data. Each row represents a datapoint, consisting of all the
    # feature values and a label, which indicates the class to which the point
    # belongs. Here, each row is a patient. The features are calculated
    # from a digital image of a fine needle aspirate (FNA) of a breast mass, and
    # the label represents the patient's diagnosis, i.e. malignant or benign.
    all_data = pd.read_csv("breast_cancer_diagnostic.csv")

    # Uncomment to visualize the first 5 entries and get dataset information,
    # such as the number of entries and column names.
    # print(all_data.head)
    # all_data.info()

    # Remove the id and Unnamed:32 columns. They are not necessary for prediction.
    all_data = all_data.drop(['Unnamed: 32', 'id'], axis = 1)

    # Convert the diagnosis values M and B to numeric values, such that
    # M (malignant) = 1 and B (benign) = 0
    def convert_diagnosis(diagnosis):
        if diagnosis == "B":
            return 0
        else:
            return 1
    all_data["diagnosis"] = all_data["diagnosis"].apply(convert_diagnosis)

    # Store the features of the data
    X = np.array(all_data.iloc[:, 1:])
    # Store the labels of the data
    y = np.array(all_data["diagnosis"])

    # How much of the data do you want to use for training? The rest will be used
    # as test data for cross validation.
    train_size = 0.8
    X_train, X_test, y_train, y_test = cross_validate(X, y, train_size)

    print("-" * 30)
    if method == "KNN":
        # Set the number of neighboring points to compare each datapoint to.
        # (You will want to adjust this to optimize the accuracy of your
        # KNN Classifier. Implement optimal_k to figure out the optimal k.)
        k = 9

        # Normalize the feature data, so that values are between [0,1]. This allows
        # us to use euclidean distance as a meaningful metric across features.
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)

        # For KNN, we want the feature function to return the value of the
        # given feature.
        def feature_func(x):
            return x

        # Initialize the KNN Classifier
        classifier = KNNClassifier([feature_func], k)

        # TODO: Implement optimal_k to explore which value of k is optimal,
        # where k is the number of neighbors used in the KNN classifier.

        # optimal_k(feature_func, X_train, X_test, y_train, y_test) commented out to avoid errors
    else:
        # TODO: Implement classification using perceptrons. Explore what
        # happens when you adjust learning_rate. A learning rate of 1 is
        # most likely not going to have good results.
        learning_rate = 0.1  # adjusted learning rate
        is_classifier = True
        def feature_func(x):
            return x
        classifier = Perceptron([feature_func], learning_rate, is_classifier)

    # Fit the data on the train set
    print("Training {} Classifier".format(method))
    classifier.train(X_train, y_train)

    # Evaluate the model's accuracy (between 0 and 1) on the test set
    print("Testing {} Classifier".format(method))
    accuracy = classifier.evaluate(X_test, y_test)

    print("{} Model Accuracy: {:.2f}%".format(method, accuracy*100))
    print("-" * 30)
    if method == "KNN":
        return classifier, accuracy, 9
    else:
        return classifier, accuracy


def optimal_k(feature_func, X_train, X_test, y_train, y_test):
    """
    1) Finds the optimal value of k, where k is the number of neighbors being
    looked at during KNN.
    2) Plots the accuracy values returned by performing cross validation on
    the KNN model, with k values in the range [1, 50).
    """
    neighbors = np.array([i+1 for i in range(50)])
    accuracies = np.array([])
    for k_value in neighbors:
        k_model = KNNClassifier([feature_func], k_value)
        k_model.train(X_train, y_train)
        accuracy = k_model.evaluate(X_test, y_test)
        accuracies = np.append(accuracies, accuracy)

    optimal_k = 0
    max_accuracy = 0
    for i in range(50):
        if accuracies[i] > max_accuracy:
            max_accuracy = accuracies[i]
            optimal_k = i + 1

    # Feel free to delete the following commented stencil code.
    # It is only here to help you implement and visualize optimal_k.

    # Printing the optimal_k and max_accuracy:
    print("The optimal number of neighbors is {}".format(optimal_k))
    print("Model Accuracy with k={} is {:.2f}%".format(optimal_k, max_accuracy))
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Supervised Learning - Classification")
    parser.add_argument("-k", "--knn", help="Indicates to use KNN model. Otherwise, uses perceptron.",
        action="store_true")
    args = vars(parser.parse_args())
    if args["knn"]:
        method = "KNN"
    else:
        method = "Perceptron"
    classification(method)
