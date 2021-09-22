from supervisedlearner import SupervisedLearner
import numpy as np

class Perceptron(SupervisedLearner):
    def __init__(self, feature_funcs, lr, is_c):
        """

        :param lr: the rate at which the weights are modified at each iteration.
        :param is_c: True if the perceptron is for classification problems,
                     False if the perceptron is for regression problems.

        """

        super().__init__(feature_funcs)
        self.weights = None
        self.learning_rate = lr
        self._trained = False
        self.is_classifier = is_c

    # TODO: Implement the rest of this class!

    def step_function(self, inp):
        """

        :param inp: a real number
        :return: the predicted label produced by the given input

        Assigns a label of 1.0 to the datapoint if <w,x> is a positive quantity
        otherwise assigns label 0.0. Should only be called when self.is_classifier
        is True.
        """
        if inp > 0:
            return 1.0
        else:
            return 0.0

    def train(self, X, Y):
        """

        :param X: a 2D numpy array where each row represents a datapoint
        :param Y: a 1D numpy array where i'th element is the label of the corresponding datapoint in X
        :return:

        Does not return anything; only learns and stores as instance variable self.weights a 1D numpy
        array whose i'th element is the weight on the i'th feature.
        Do not forget to include the bias in your calculation.
        """
        self.weights = [0.0 for i in range(len(X[0]) + 1)]
        if self.is_classifier is True:  #classification
            for epoch in range(1000):
                j = 0
                for data in X:
                    prediction = self.predict(data)
                    error = Y[j] - prediction
                    j += 1
                    self.weights[-1] += self.learning_rate * error
                    for k in range(len(X[0])):
                        self.weights[k] = self.weights[k] + self.learning_rate * error * data[k]
        else:  # regression
            for epoch in range(1000):
                j = 0
                for data in X:
                    prediction = self.predict(data)
                    error = Y[j] - prediction
                    j += 1
                    self.weights[-1] += self.learning_rate * error
                    for k in range(len(X[0])):
                        self.weights[k] = self.weights[k] + self.learning_rate * error * data[k]


    def predict(self, x):
        """
        :param x: a 1D numpy array representing a single datapoints
        :return:

        Given a data point x, produces the learner's estimate
        of f(x). Use self.weights and make sure to use self.step_function
        if self.is_classifier is True
        """
        activation = self.weights[-1]  # a =  bias (initially)
        for i in range(len(x)):  # iterates over all parameters in data
            activation += self.weights[i] * x[i]
        if self.is_classifier is True:
            return self.step_function(activation)
        else:
            return activation

    def evaluate(self, datapoints, labels):
        """

        :param datapoints: a 2D numpy array where each row represents a datapoint
        :param labels: a 1D numpy array where i'th element is the label of the corresponding datapoint in datapoints
        :return:

        If self.is_classifier is True, returns the fraction (between 0 and 1)
        of the given datapoints to which the method predict(.) assigns the correct label
        If self.is_classifier is False, returns the Mean Squared Error (MSE)
        between the labels and the predictions of their respective inputs (You
        do not have to calculate the R2 Score)
        """
        entries = len(labels)
        if self.is_classifier is True:  # KNN
            num_correct = 0
            i = 0
            for point in datapoints:
                if labels[i] == self.predict(point):
                    num_correct += 1
                i += 1
            return num_correct/entries
        else:
            mse_sum = 0
            j = 0
            for point in datapoints:
                mse = (labels[j] - self.predict(point)) ** 2
                mse_sum += mse
                j += 1
            return mse_sum/len(labels)
