import itertools as it
import math
from typing import List, Dict, Any


def filtering(
    evidence: List[Any],
    states: List[Any],
    observations: List[Any],
    prior: Dict[Any, float],
    transition_model: Dict[Any, Dict[Any, float]],
    observation_model: Dict[Any, Dict[Any, float]],
) -> Dict[Any, float]:
    """
    Computes the posterior probability over states of an HMM given evidence.
    An HMM consists of states, observations, prior, transition model, and observation model.

    :param evidence: a list with the evidence to filter on.
    :param states: states of the model given as a list.
    :param observations: a list of observations given as a list.
    :param prior: prior distribution over states given as a dictionary mapping states to priors.
    :param transition_model: specifies the probability of transitions from a given state to another given as a
    dictionary mapping states to a dictionary that maps states to floats (transition probability).
    :param observation_model: specifies the probability of observations given states given as a
    dictionary mapping states to a dictionary that maps observations to float (probability of the observations given the state).

    :return: a map with posterior probabilities over states. The map has states as keys and posteriors as values.
    Mathematically, this map contains the posterior distribution over states, i.e., P(X_t| e_1, ..., e_t).
    """

    # For ease of indexing, prepend a dummy evidence at the beginning of the evidence sequence.
    evidence = [-math.inf] + evidence

    # Preparing the f[t, state] table with time 0 containing the prior for each state.
    f = {(0, state): probability for state, probability in prior.items()}

    # For efficiency purposes, we complete the transition model.
    for state, other_states in it.product(states, states):
        if other_states not in transition_model[state]:
            transition_model[state][other_states] = 0.0

    # For efficiency purposes, we complete the observation model.
    for state, observation in it.product(states, observations):
        if observation not in observation_model[state]:
            observation_model[state][observation] = 0.0

    # For each piece of evidence, compute f.
    for t in range(0, len(evidence) - 1):
        normalizer = 0

        for state in states:
            f[t + 1, state] = observation_model[state][evidence[t + 1]] * sum(
                [
                    transition_model[state][other_state] * f[t, other_state]
                    for other_state in states
                ]
            )
            normalizer += f[t + 1, state]
        # If at this point the normalizer is 0, then the evidence is impossible for the given model!
        if math.isclose(normalizer, 0):
            raise Exception(
                "Evidence is impossible, i.e., inconsistent with the model."
            )

        # Normalize f entries to make up a valid probability distribution.
        for state in states:
            f[t + 1, state] = f[t + 1, state] / normalizer

    return {state: f[len(evidence) - 1, state] for state in states}
