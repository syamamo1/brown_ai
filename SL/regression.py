'''
SL Assignment - regression.py
CS 1410 Artificial Intelligence, Brown University
Written by jbernar3.

Usage-

To run regression using your perceptron implementation:
    python regression.py

To run regression using our sklearn implementation:
    python regression.py --sklearn (or python regression.py -s)

'''
from sklearn import preprocessing
from sklearn_model import SKLearnModel
from perceptron import Perceptron
from cross_validate import cross_validate
from preprocess_student_data import preprocess_data
import numpy as np
import pandas as pd
import argparse

# Performs regression using the model type specified in method. Default method
# is the perceptron.
def regression(method):
    # Load the data. Each row represents a datapoint, consisting of all the
    # feature values and a label, indicating the grade which the student
    # receives. In this case, each row is a student.
    all_data = pd.read_csv("student-por.csv")

    # Call preprocess from preprocess_student_data to attain numeric 2D numpy
    # from the data.
    # X is (num_students=649, num_attributes=31)
    X, y = preprocess_data(all_data)

    # How much of the data do you want to use for training? The rest will be used
    # as test data for cross validation.
    train_size = 0.8
    X_train, X_test, y_train, y_test = cross_validate(X, y, train_size)

    print("-" * 30)
    if method == "SKLearn":
        basic_func = lambda x : x
        feature_funcs = [basic_func] * 31
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)

        # Initialize SKLearn Regression Model
        regression_model = SKLearnModel(feature_funcs)

        # Fit the data on the train set
        print("Training {} Model".format(method))
        regression_model.train(X_train, y_train)

        # Evaluate the model's Mean Squared Error and R Square score on the test set
        print("Testing {} Model".format(method))
        mse, r2 = regression_model.evaluate(X_test, y_test)

        print("Sklearn Regression Model MSE: {}".format(mse))
        print("Sklearn Regression Model R2 Score: {}".format(r2))
    else:
        basic_func = lambda x : x
        feature_funcs = [basic_func] * 31
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)
        # TODO: Implement regression using perceptrons. Explore what
        # happens when you adjust learning_rate. A learning rate of 1 is
        # most likely not going to have good results.
        learning_rate = 0.001
        is_classifier = False
        perceptron = Perceptron(feature_funcs, learning_rate, is_classifier)
        print("Training {} Model".format(method))
        perceptron.train(X_train, y_train)

        # Evaluate the model's Mean Squared Error on the test set
        print("Testing {} Model".format(method))
        mse = perceptron.evaluate(X_test, y_test)

        print("Perceptron MSE: {}".format(mse))
        return perceptron, mse
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Supervised Learning - Regression")
    parser.add_argument("-s", "--sklearn", help="Indicates to use model implemented with sklearn. Otherwise, uses perceptron.",
        action="store_true")
    args = vars(parser.parse_args())
    if args["sklearn"]:
        method = "SKLearn"
    else:
        method = "Perceptron"
    regression(method)
