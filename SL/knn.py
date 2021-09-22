from supervisedlearner import SupervisedLearner
import numpy as np

class KNNClassifier(SupervisedLearner):
    def __init__(self, feature_funcs, k):
        super(KNNClassifier, self).__init__(feature_funcs)
        self.k = k

    def train(self, anchor_points, anchor_labels):
        """
        :param anchor_points: a 2D numpy array, in which each row is
						      a datapoint, without its label, to be used
						      for one of the anchor points

		:param anchor_labels: a list in which the i'th element is the correct label
		                      of the i'th datapoint in anchor_points

		Does not return anything; simply stores anchor_labels and the
		_features_ of anchor_points.
		"""
        self.features = [
            self.compute_features(anchor_point) for anchor_point in anchor_points
            ]
        self.labels = anchor_labels
        self._trained = True

    def predict(self, x):
        """
        Given a single data point, x, represented as a 1D numpy array,
		predicts the class of x by taking a plurality vote among its k
		nearest neighbors in feature space. Resolves ties arbitrarily.

		The K nearest neighbors are determined based on Euclidean distance
		in _feature_ space (so be sure to compute the features of x).

		Returns the label of the class to which x is predicted to belong.
		"""
        assert self._trained
        x_feature = self.compute_features(x)

        # A list containing the Euclidean distance of x from another point y,
        # each element of which is in the form (distance, y index)
        dist_list = [(np.linalg.norm(self.features[i] - x_feature), i) for i in range(len(self.features))]
        sorted_dist_list = sorted(dist_list)

        # Get the k closest points to x and their labels
        closest = sorted_dist_list[:self.k]
        label_list = [self.labels[point[1]] for point in closest]

        # Note: max(set(x), key=x.count) returns the mode of a list x.
        return max(set(label_list), key=label_list.count)

    def evaluate(self, datapoints, labels):
        """
        :param datapoints: a 2D numpy array, in which each row is a datapoint.
		:param labels: a 1D numpy array, in which the i'th element is the
		               correct label of the i'th datapoint.

		Returns the fraction (between 0 and 1) of the given datapoints to which
		predict(.) assigns the correct label
		"""
        assert self._trained
        correct = 0

        # Count the number of correct predictions and find the model accuracy
        for datapoint, label in zip(datapoints, labels):
            predicted_label = self.predict(datapoint)
            correct += predicted_label == label

        return (correct * 1.0) / len(labels)
