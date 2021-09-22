import numpy as np

### CALL THIS FUNCTION ONLY ###
def ta_filter(curr_state_dist, sens_model, trans_model, obs):
    """
    A TA solution for filtering using simplified matrix algorithms.
    "SZ" represents the number of states you choose. For example, if your
    states are 0, 1, 2... then SZ = 3 (the number of possible states).
    "O_SZ" represents the number of observation states.

    Inputs:
        curr_state_dist:
            1d numpy array of size (SZ, )
            a prior probability distribution
        sens_model:
            2d numpy array of size (SZ, O_SZ)
            each entry sens_model[State][Observation] gives ==> P (Observation | State)
        trans_model:
            2d numpy array of size (SZ, SZ)
            each entry trans_model[S_t-1][S_t] gives ==> P (S_t | S_t-1)
        obs:
            an int used to represent an observation index, from [0 - O_SZ)
    Returns:
        posterior_state_dist:
            1d numpy array of size (SZ, )
            a posterior probability distribution after the filtering (forward) algorithm
            for the next time step
    """

    # Filtering; forward equation
    predict = _predict(curr_state_dist, trans_model)
    update = _update(obs, predict, sens_model)
    posterior_state_dist = _normalize(update)
    return posterior_state_dist


### HELPER FUNCTIONS - DO NOT CALL ###
def _normalize(distribution):
    total = np.sum(distribution)
    for i in range (distribution.shape[0]):
        distribution[i] = (distribution[i]/total)
    return distribution

def _predict(state_dist, trans_model):
    # one step prediction
    return np.matmul(np.transpose(trans_model.T), state_dist)

def _update(obs, state_dist, sens_model):
    # one step update based on an observation (type int)
    # for a predicted state distributon (psd)
    return np.matmul(np.diag(sens_model[:,obs]), state_dist)
