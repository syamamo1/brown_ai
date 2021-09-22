from adversarialsearchproblem import AdversarialSearchProblem


def minimax(asp):

    """
    Implement the minimax algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum

    Input: asp - an AdversarialSearchProblem
    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """

    max_move = None
    max_val = None
    first_state = asp.get_start_state()
    player = first_state.player_to_move()
    choices = asp.get_available_actions(first_state)  # SET of actions
    for choice in choices:
        val = min_func(asp, asp.transition(first_state, choice), player)
        if max_val is None or val > max_val:
            max_move = choice
            max_val = val
    return max_move


# returns value
def min_func(asp, state, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    else:
        choices = asp.get_available_actions(state)
        min_val = None
        for choice in choices:
            val = max_func(asp, asp.transition(state, choice), player)
            if min_val is None or val < min_val:  # check this LINE < or >
                min_val = val
        return min_val


def max_func(asp, state, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    else:
        choices = asp.get_available_actions(state)
        max_val = None
        for choice in choices:
            val = min_func(asp, asp.transition(state, choice), player)
            if max_val is None or val > max_val:
                max_val = val
        return max_val


def alpha_beta(asp):

    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input: asp - an AdversarialSearchProblem
    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """

    max_move = None
    max_val = None
    a = None  # highest value choice (so far)
    b = None  # lowest value choice (so far)
    first_state = asp.get_start_state()
    player = first_state.player_to_move()
    choices = asp.get_available_actions(first_state)  # SET of actions
    for choice in choices:
        val = min_func_ab(asp, asp.transition(first_state, choice), player, a, b)
        if max_val is None or val > max_val:
            max_move = choice
            max_val = val
        if b is not None and val >= b:  # "break out" of loop
            return choice
        if a is None or val > a:   # "break out" of loop
            a = val
    return max_move


# returns value
def min_func_ab(asp, state, player, a, b):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    else:
        choices = asp.get_available_actions(state)
        min_val = None
        for choice in choices:
            val = max_func_ab(asp, asp.transition(state, choice), player, a, b)
            if min_val is None or val < min_val:  # check this LINE < or >
                min_val = val
            if a is not None and val <= a:  # "break out" of loop
                return val
            if b is None or val < b:  # "break out" of loop
                b = val
        return min_val


def max_func_ab(asp, state, player, a, b):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[player]
    else:
        choices = asp.get_available_actions(state)
        max_val = None
        for choice in choices:
            val = min_func_ab(asp, asp.transition(state, choice), player, a, b)
            if max_val is None or val > max_val:
                max_val = val
            if b is not None and val >= b:  # "break out" of loop
                return val
            if a is None or val > a:  # "break out" of loop
                a = val
        return max_val


# now need to keep track of depth, only AB prune to certain depth
def alpha_beta_cutoff(asp, cutoff_ply, eval_func):
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Inputs:
            asp - an AdversarialSearchProblem
            cutoff_ply- an Integer that determines when to cutoff the search
                    and use eval_func.
                    For example, when cutoff_ply = 1, use eval_func to evaluate
                    states that result from your first move. When cutoff_ply = 2, use
                    eval_func to evaluate states that result from your opponent's
                    first move. When cutoff_ply = 3 use eval_func to evaluate the
                    states that result from your second move.
                    You may assume that cutoff_ply > 0.
            eval_func - a function that takes in a GameState and outputs
                    a real number indicating how good that state is for the
                    player who is using alpha_beta_cutoff to choose their action.
                    You do not need to implement this function, as it should be provided by
                    whomever is calling alpha_beta_cutoff, however you are welcome to write
                    evaluation functions to test your implemention. The eval_func we provide
        does not handle terminal states, so evaluate terminal states the
        same way you evaluated them in the previous algorithms.

    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    max_move = None
    max_val = None
    a = None  # highest value choice (so far)
    b = None  # lowest value choice (so far)
    first_state = asp.get_start_state()
    player = first_state.player_to_move()
    choices = asp.get_available_actions(first_state)  # SET of actions
    for choice in choices:
        val = min_func_abc(asp, asp.transition(first_state, choice),
                           player, a, b, cutoff_ply - 1, eval_func)
        if max_val is None or val > max_val:
            max_move = choice
            max_val = val
        if b is not None and val >= b:  # "break out" of loop
            return choice
        if a is None or val > a:  # "break out" of loop
            a = val
    return max_move


# returns value
def min_func_abc(asp, state, player, a, b, depth, eval_func):
    if depth == 0 or asp.is_terminal_state(state):
        return eval_func(state)
    else:
        choices = asp.get_available_actions(state)
        min_val = None
        for choice in choices:
            val = max_func_abc(asp, asp.transition(state, choice), player, a, b, depth - 1, eval_func)
            if min_val is None or val < min_val:  # check this LINE < or >
                min_val = val
            if a is not None and val <= a:  # "break out" of loop
                return val
            if b is None or val < b:  # "break out" of loop
                b = val
        return min_val


def max_func_abc(asp, state, player, a, b, depth, eval_func):
    if depth == 0 or asp.is_terminal_state(state):
        return eval_func(state)
    else:
        choices = asp.get_available_actions(state)
        max_val = None
        for choice in choices:
            val = min_func_abc(asp, asp.transition(state, choice), player, a, b, depth - 1, eval_func)
            if max_val is None or val > max_val:
                max_val = val
            if b is not None and val >= b:  # "break out" of loop
                return val
            if a is None or val > a:  # "break out" of loop
                a = val
        return max_val
