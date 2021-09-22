import itertools as it
import math
import pprint
import time
from fractions import Fraction
from functools import wraps
from typing import List, Dict, Callable, Tuple

import numpy as np
from prettytable import PrettyTable

from hmm_filtering import filtering


def timing(f, before_message):
    """
    A helper function to measure a function's execution time.
    :param f: a function.
    :param before_message: a string with a message.
    :return: a decorator function that wraps f.
    """

    @wraps(f)
    def wrap(*args, **kw):
        print(f"{before_message}", end="")
        t0 = time.time()
        result = f(*args, **kw)
        print(f"\t -> done, it took {time.time() - t0 : .4f}s")
        return result

    return wrap


def helper_test_filtering(evidence, hmm):
    """
    A helper function to test filtering.
    :param evidence: a sequence of given observations.
    :param hmm: a fully defined hmm.
    :return: the posterior distribution over states, i.e., P(X_t| e_1, ..., e_t).
    """
    print(f"\n Evidence of length {len(evidence)} is:")
    pprint.pprint([e for e in evidence])
    posterior_over_states = timing(filtering, "\n Filtering")(evidence, *hmm)
    print(f"\n {pretty_table_helper(hmm, posterior_over_states)}")
    return posterior_over_states


def pretty_table_helper(hmm, f: Dict[int, float]):
    """
    A helper to pretty print a table with the posterior probability, in particular probabilities as fractions.
    This is useful for manual debugging. The pretty table contains only non-zero probabilities.
    :param hmm: the hmm model.
    :param f: the posterior.
    :return: a pretty table.
    """
    posterior_table = PrettyTable()
    posterior_table.field_names = ["state", "prob. (fraction)", "prob. (decimal)"]
    # hmm[0] contain the model's states. Only in
    for state in hmm[0]:
        if f[state] > 0:
            posterior_table.add_row(
                [
                    state,
                    Fraction(f[state]).limit_denominator(),
                    f"{f[state]:.6f}",
                ]
            )
    return posterior_table


def fair_bet_casino_model():
    """
    Encodes the Fair Bet Casino,
    taken from page 391 of http://www.cs.ukzn.ac.za/~hughm/bio/docs/IntroToBioinfAlgorithms.pdf, as an HMM.
    :return: the Fair Bet Casino HMM.
    """

    states = ["F", "B"]
    observations = ["H", "T"]

    # The transition model specifies the probability distribution over coins ["F", "B"] for each coin.
    transition_model = {
        "F": {"F": 9 / 10, "B": 1 / 10},
        "B": {"F": 1 / 10, "B": 9 / 10},
    }
    # The observation model specifies the probability of observing Heads or Tails for each coin.
    observation_model = {
        "F": {"H": 1 / 2, "T": 1 / 2},
        "B": {"H": 3 / 4, "T": 1 / 4},
    }
    # Uniform prior over coins
    prior = {"F": 1 / 2, "B": 1 / 2}

    return states, observations, prior, transition_model, observation_model


def rain_umbrella_model():
    """
    Encodes the Rain umbrella example from Russell and Norvig as an HMM.
    :return: the rain umbrella example HMM.
    """

    # State R = Rain, S = Sunny (same as no rain!)
    states = ["R", "S"]
    observations = ["U", "N"]
    transition_model = {
        "R": {"R": 0.7, "S": 0.3},
        "S": {"R": 0.3, "S": 0.7},
    }
    # U = umbrella, N = no umbrella.
    observation_model = {
        "R": {"U": 0.9, "N": 0.1},
        "S": {"U": 0.2, "N": 0.8},
    }
    prior = {"R": 1 / 2, "S": 1 / 2}

    return states, observations, prior, transition_model, observation_model


def encode_touchscreen_hmm_frame(frame: np.array) -> int:
    """
    Encodes a frame as a non-negative integer.
    :param frame: a numpy array.
    :return: an non-negative integer.
    """
    assert frame.shape[0] == frame.shape[1]
    (i, j) = np.nonzero(frame)
    assert len(i) == 1 and len(j) == 1
    return i[0] * frame.shape[0] + j[0]


def decode_touchscreen_hmm_frame(size: int, state: int) -> np.array:
    """
    Decodes a non-negative integer into a frame.
    :param size: the size of the frame, a positive integer.
    :param state: a non-negative integer.
    :return: a frame, i.e., numpy array.
    """
    assert size > 0 and state >= 0
    frame = np.zeros((size, size))
    frame[state // size, state % size] = 1
    return frame


def helper_get_touchscreen_frame(size: int, i: int, j: int) -> np.array:
    """
    A helper function that returns a frame, i.e., a square numpy array of the given size with
    all entries zero except a one at the given i, j position.
    :param size: the size of the frame, a positive integer.
    :param i: a non-negative integer.
    :param j: a non-negative integer.
    :return: a numpy array.
    """
    assert size > 0 and 0 <= i <= size and 0 <= j <= size
    frame = np.zeros((size, size))
    frame[i, j] = 1
    return frame


def encode_hmm_model(encoder: Callable, hmm: Tuple):
    """
    Given an HMM model and an encoder, encodes the model in terms of the encoder.
    :return: an encoded HMM model.
    """
    # Unpack the pieces of the model.
    (
        states,
        observations,
        prior,
        transition_model,
        observation_model,
    ) = hmm

    # Encode the states, observations, and prior.
    encoded_states = [encoder(state) for state in states]
    encoded_observations = [encoder(observation) for observation in observations]
    encoded_prior = {
        encoder(state): prior_probability for state, prior_probability in prior
    }

    # Encode the transition model.
    encoded_transition_model = {
        encoder(state): {
            encoder(next_state): probability for next_state, probability in next_states
        }
        for state, next_states in transition_model
    }
    # Encode the observation model.
    encoded_observation_model = {
        encoder(state): {
            encoder(observation): probability
            for observation, probability in observations
        }
        for state, observations in observation_model
    }

    return (
        encoded_states,
        encoded_observations,
        encoded_prior,
        encoded_transition_model,
        encoded_observation_model,
    )


def touch_screen_hmm_model(
    size,
) -> Tuple[
    List[np.array],
    List[np.array],
    List[Tuple[np.array, float]],
    List[Tuple[np.array, List[Tuple[np.array, float]]]],
    List[Tuple[np.array, List[Tuple[np.array, float]]]],
]:
    """
    This function generates a simple HMM model for the touch screen problem.
    :param size: the size of the frames.
    :return: an HMM model for the touch screen problem.
    """
    states = []
    observations = []
    transition_model = []
    prior = []

    for i, j in it.product(range(0, size), range(0, size)):
        state = helper_get_touchscreen_frame(size, i, j)
        states.append(state)
        observations.append(state)
        prior.append((state, 1.0 / (size * size)))
        next_frames = []

        # Transition row above.
        for k in [-1, 0, 1]:
            if 0 <= i - 1 < size and 0 <= j + k < size:
                next_frames.append(helper_get_touchscreen_frame(size, i - 1, j + k))

        # Transition same row.
        for k in [-1, 1]:
            if 0 <= j + k < size:
                next_frames.append(helper_get_touchscreen_frame(size, i, j + k))

        # Transition row below.
        for k in [-1, 0, 1]:
            if 0 <= i + 1 < size and 0 <= j + k < size:
                next_frames.append(helper_get_touchscreen_frame(size, i + 1, j + k))

        # The transition model is uniform over all next states.
        transition_model.append(
            (
                state,
                [(frame, 1.0 / len(next_frames)) for frame in next_frames],
            )
        )

    # The observation model is identical to the transition model. This is probably something you do NOT want to do!
    observation_model = transition_model.copy()

    return states, observations, prior, transition_model, observation_model


def incremental_filtering(evidences, hmm):
    """
    An example of how to do filtering incrementally, i.e., call the filtering function with one
    piece of evidence at a time, caching the posterior and feeding it as the prior for the next piece of evidence.
    :param evidences: a sequence of observations
    :param hmm: the HMM.
    """
    # Unpack the HMM.
    (
        states,
        observations,
        prior,
        transition_model,
        observation_model,
    ) = hmm

    # Filter incrementally: at each time step t, the (t - 1)-th posterior becomes
    # the t-(th)prior in a call to filtering where the evidence is just the evidence at time t.
    print(f"\n Incremental filtering of evidence {evidences}")
    for t, one_evidence in enumerate(evidences, 1):
        print(f"\n Processing Evidence at time {t}, given by {one_evidence} \n")
        posterior = filtering(
            [one_evidence],
            states,
            observations,
            prior,
            transition_model,
            observation_model,
        )
        print(
            pretty_table_helper(
                (states, observations, prior, transition_model, observation_model),
                posterior,
            )
        )
        prior = posterior


if __name__ == "__main__":

    # Fair Bet Casino example.

    # First, a simple filtering example
    print(
        f"simple usage example, posterior = {filtering(['H'], *fair_bet_casino_model())}"
    )

    # Testing evidence H.
    fair_bet_casino_evidence = ["H"]
    fair_bet_posterior = helper_test_filtering(
        fair_bet_casino_evidence, fair_bet_casino_model()
    )
    assert math.isclose(fair_bet_posterior["F"], 2 / 5) and math.isclose(
        fair_bet_posterior["B"], 3 / 5
    )

    # Testing evidence HT.
    fair_bet_posterior = helper_test_filtering(["H", "T"], fair_bet_casino_model())
    assert math.isclose(fair_bet_posterior["F"], 42 / 71) and math.isclose(
        fair_bet_posterior["B"], 29 / 71
    )

    # Testing incrementally
    incremental_filtering(["H", "T", "H"], fair_bet_casino_model())

    # Rain Umbrella example.

    # Testing evidence U.
    rain_umbrella_evidence = ["U"]
    rain_umbrella_posterior = helper_test_filtering(
        rain_umbrella_evidence, rain_umbrella_model()
    )
    assert math.isclose(rain_umbrella_posterior["R"], 9 / 11) and math.isclose(
        rain_umbrella_posterior["S"], 2 / 11
    )

    # Testing evidence UU.
    rain_umbrella_evidence = ["U", "U"]
    rain_umbrella_posterior_posterior = helper_test_filtering(
        rain_umbrella_evidence, rain_umbrella_model()
    )
    assert math.isclose(
        rain_umbrella_posterior_posterior["R"], 621 / 703
    ) and math.isclose(rain_umbrella_posterior_posterior["S"], 82 / 703)

    # Testing incrementally
    incremental_filtering(["U", "U"], rain_umbrella_model())

    # Touch screen examples.

    # The size of the rectangular grid. There are touch_hmm_size*touch_hmm_size many cells.
    touch_screen_hmm_size = 20

    # An impossible evidence - an exception should occur when given this evidence.
    touch_screen_hmm_evidence = [0, 100]
    encoded_model = encode_hmm_model(
        encode_touchscreen_hmm_frame,
        touch_screen_hmm_model(touch_screen_hmm_size),
    )
    try:
        filtering(touch_screen_hmm_evidence, *encoded_model)
    except Exception:
        print(f"\n ----- The evidence {touch_screen_hmm_evidence} is impossible! -----")

    # Some simple evidence.
    touch_screen_hmm_evidences = [[0], [15], [55, 56, 57, 58], [399]]

    # An evidence where the sensor reports row by row, from left to right.
    touch_screen_hmm_evidence = (
        [i for i in range(0, touch_screen_hmm_size)]
        + [2 * touch_screen_hmm_size - 1 - i for i in range(0, touch_screen_hmm_size)]
        + [i for i in range(2 * touch_screen_hmm_size, 3 * touch_screen_hmm_size)]
        + [4 * touch_screen_hmm_size - 1 - i for i in range(0, touch_screen_hmm_size)]
        + [i for i in range(4 * touch_screen_hmm_size, 5 * touch_screen_hmm_size)]
    )
    touch_screen_hmm_evidences.append(touch_screen_hmm_evidence)

    # A zig-zag (down, up, down, up...) evidence between the first and second row.
    touch_screen_hmm_evidence = [
        val
        for pair in zip(
            [i for i in range(0, touch_screen_hmm_size)],
            [touch_screen_hmm_size + i for i in range(0, touch_screen_hmm_size)],
        )
        for val in pair
    ]
    touch_screen_hmm_evidences.append(touch_screen_hmm_evidence)

    # Bulk test various evidence examples.
    for some_evidence in touch_screen_hmm_evidences:
        touch_screen_hmm_posterior = helper_test_filtering(
            some_evidence,
            timing(encode_hmm_model, "\n\n ************ Encoding touch screen model",)(
                encode_touchscreen_hmm_frame,
                touch_screen_hmm_model(touch_screen_hmm_size),
            ),
        )
        decoded_posterior = [
            (decode_touchscreen_hmm_frame(touch_screen_hmm_size, state), probability)
            for state, probability in touch_screen_hmm_posterior.items()
            if probability > 0
        ]
        # Set a wider line width so that 20 by 20 frames are printed without inconvenient line breaks.
        print("\n Posterior encoded as a list of tuples (frames, probability): \n")
        np.set_printoptions(linewidth=150)
        pprint.pprint(decoded_posterior)
