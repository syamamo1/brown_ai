import math
import numpy as np
from math import sqrt
from scipy.stats import norm


class touchscreenEvaluator:
    def __init__(self):
        self.past_distributions = {}

    def calc_score_spherical(self, actual_frame, estimated_frame):
        """
        Calculates the accuracy of a frame distribution. It works by looking at the estimated_frame and applying a normal
        distribution map over the actual position, rewarding points for higher distributions around the actual point.
        :param actual_frame:
        :param estimated_frame:
        :return: The score of the corresponding board
        """
        n = actual_frame.shape[0] * actual_frame.shape[1]
        actual = np.reshape(actual_frame, n)
        estimated = np.reshape(estimated_frame, n)
        total = sum(estimated)
        if any(estimated < -0.001) or total > 1.001 or total < 0.999:
            print("Estimated frame is not a probability distribution!")
            return 0
        idx = np.nonzero(actual)[0].item() # item returns the int
        r_i = estimated[idx]
        return r_i / np.linalg.norm(estimated)

    def evaluate_touchscreen_hmm(self, touchscreenHMM, simulation):
        """
        Calculates the accuracy of the students project.
        :param touchscreenHMM: An instance of a student's touchscreenHMM
        :return: The 'percentage accuracy' from the calc_score function over the whole screen, and the 'percentage
        accuracy' of just inputting the noisy location.
        """
        score = 0
        count = 0
        frame = simulation.get_frame(actual_position=True)
        while frame:
            count += 1
            student_frame = touchscreenHMM.filter_noisy_data(frame[0])
            if np.isnan(np.sum(student_frame)) or not math.isclose(np.sum(student_frame), 1):
                print("Encountered NAN or the sum of the probability distribution is not 1. Check your frame.")
                score += 0
                break
            score += self.calc_score_spherical(frame[1], student_frame)
            frame = simulation.get_frame(actual_position=True)
        #averaging all the counts
        if count == 0: 
            return 0; #preventing nan
        return round(score / count, 3) 