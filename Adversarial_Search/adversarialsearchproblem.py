from abc import ABCMeta, abstractmethod

###############################################################################
# An AdversarialSearchProblem is a representation of a game that is convenient
# for running adversarial search algorithms.
#
# A game can be put into this form by extending the AdversarialSearchProblem
# class. See tttproblem.py for an example of this.
#
# Every subclass of AdversarialSearchProblem has its game states represented
# as instances of a subclass of GameState. The only requirement that of a
# subclass of GameState is that it must implement that player_to_move(.) method,
# which returns the index (0-indexed) of the next player to move.
###############################################################################


class GameState(metaclass=ABCMeta):
    @abstractmethod
    def player_to_move(self):
        """
        Output- Returns the index of the player who will move next.
        """
        pass


class AdversarialSearchProblem(metaclass=ABCMeta):
    def get_start_state(self):
        """
        Output- Returns the state from which to start.
        """
        return self._start_state

    def set_start_state(self, state):
        """
        Changes the start state to the given state.
        Note to student: You should not need to use this.
        This is only for running games.

        Input:
                state- a GameState
        """
        self._start_state = state

    @abstractmethod
    def get_available_actions(self, state):
        """
        Input:
                state- a GameState
        Output:
                Returns the set of actions available to the player-to-move
                from the given state
        """
        pass

    @abstractmethod
    def transition(self, state, action):
        """
        Input:
                state- a Gamestate
                action- the action to take
        Ouput:
                Returns the state that results from taking the given action
                from the given state. (Assume deterministic transitions.)
        """
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)
        pass

    @abstractmethod
    def is_terminal_state(self, state):
        """
        Input:
                state- a GameState
        Output:
                Returns a boolean indicating whether or not the given
                state is terminal.
        """
        pass

    @abstractmethod
    def evaluate_state(self, state):
        """
        Input:
                state- A TERMINAL GameState
        Output:
                Returns a list of numbers such that the
                i'th entry represents how good the state
                is for player i.
        """
        assert self.is_terminal_state(state)
        pass

    @abstractmethod
    def eval_func(self, state, player_index):
        """
        An evaluation function that can be used to get an idea of how good any particular state is.
        (Useful for ab-cutoff)

        Input:
                state- Any GameState
                player_index: 0-based index of player that output should be relevant to
                (a state can be good for one player, but not the other).
        Output:
                A real number indicating how good the state is for the player.
        """
        pass


###############################################################################
# GameUI is an abstraction that allows you to interact directly with
# an AdversarialSearchProblem (through gamerunner.py). See tttpbroblem or
# connect4problem for examples.
#
# Utilizing GameUI is NOT necessary for this assignment, although you can use
# it with any ASPs you may decide to create.
###############################################################################


class GameUI(metaclass=ABCMeta):
    def update_state(self, state):
        """
        Updates the state currently being rendered.
        """
        self._state = state

    @abstractmethod
    def render(self):
        """
        Renders the GameUI instance's render (presumably this will be called continuously).
        """
        pass

    @abstractmethod
    def get_user_input_action(self):
        """
        Output- Returns an action obtained through the GameUI input itself.
        (It is expected that GameUI validates that the action is valid).
        """
        pass
