import unittest
import numpy as np
from touchscreen import touchscreenHMM
from supplied_models import suppliedModel


class IOTest(unittest.TestCase):
    """
    Tests IO for hmm and touchscreen implementations. Contains basic test cases.

    Each test function instantiates a hmm and checks that all returned arrays/frames are probability distributions and sum to 1.
    """

    def _is_close(self, a, b, rel_tol=1e-07, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def _check_filtered_frame(self, model):
        sample_frame = np.zeros((20, 20))
        sample_frame[0][0] = 1.0
        model_instance = model()
        filtered_frame = model_instance.filter_noisy_data(sample_frame)
        self.assertTrue(
            self._is_close(np.sum(filtered_frame), 1),
            "Filtered frame is not a probability distribution",
        )

    def test_touchscreen(self):
        self._check_filtered_frame(touchscreenHMM)


if __name__ == "__main__":
    unittest.main()
