#!/usr/bin/python

import numpy as np
from tronproblem import *
from tronproblem import TronProblem
from trontypes import CellType, PowerupType
import random, math
from queue import Queue, LifoQueue, PriorityQueue # need priorityqueue for voronoi
from datetime import datetime, timedelta

# Throughout this file, ASP means adversarial search problem.

class StudentBot:
    """ Write your student bot here"""

    wall_hug = False
    longest_path = None

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()

        TODO: IMPLEMENT A TIME COUNTER (with some margin for error)
        """

        i = 0
        if StudentBot.wall_hug:
            i += 1
            return StudentBot.longest_path[i]

        cutoff_ply = 1
        return self.alpha_beta_cutoff(asp, cutoff_ply)

        # now need to keep track of depth, only AB prune to certain depth

    def alpha_beta_cutoff(self, asp, cutoff_ply):
        """
        This function should:
        - search through the asp using alpha-beta pruning
        - cut off the search after cutoff_ply moves have been made.

        Inputs:
                asp: a TronProblem
                cutoff_ply: how deep to search

        Output: an action (U, D, L, R)

        TODO: IMPLEMENT A TIME COUNTER (with some margin for error)
        TODO: give an action that doesn't kill us whenever possible!!
        """
        state = asp.get_start_state()
        player = state.player_to_move()
        board = state.board
        loc = state.player_locs[player]
        actions = asp.get_available_actions(state)

        # check if we should wall hug
        if self.not_connected(state):
            print("Not connected, now wall hug")
            StudentBot.wall_hug = True
            StudentBot.longest_path = self.find_longest_path(board, loc)
            print(StudentBot.longest_path)
            return self.find_longest_path(state)[0]

        alpha = float("-inf")  # represents hightest max value
        beta = float("inf")  # represents lowest min value
        best_action = actions.pop()
        actions.add(best_action)
        for action in actions:
            payoff = self.min_value_ab_cutoff(asp, asp.transition(state, action), player, alpha, beta, cutoff_ply - 1)
            if payoff > alpha:
                alpha = payoff
                best_action = action
                # print(best_action)
        print(best_action, alpha)
        return best_action  # TODO: give an action that doesn't kill us if possible
        # (even if we've already lost against an optimal bot, we want to stay alive)

    def min_value_ab_cutoff(self, asp, state, player, alpha, beta, cutoff_ply):
        # helper function for alpha_beta
        if asp.is_terminal_state(state):
            return (asp.evaluate_state(state)[player] * 500) - 250
        if cutoff_ply <= 0:
            return self.voronoi(state, player)
        actions = asp.get_available_actions(state)
        min_payoff = float('inf')
        for action in actions:
            min_payoff = min(min_payoff,
                             self.max_value_ab_cutoff(asp, asp.transition(state, action), player, alpha, beta,
                                                      cutoff_ply - 1))
            if min_payoff < alpha:
                return min_payoff
            beta = min(beta, min_payoff)
        return min_payoff

    def max_value_ab_cutoff(self, asp, state, player, alpha, beta, cutoff_ply):
        # helper function for alpha_beta
        if asp.is_terminal_state(state):
            return (asp.evaluate_state(state)[player] * 500) - 250
        if cutoff_ply <= 0:
            return self.voronoi(state, player)
        actions = asp.get_available_actions(state)
        max_payoff = float('-inf')
        for action in actions:
            max_payoff = max(max_payoff,
                             self.min_value_ab_cutoff(asp, asp.transition(state, action), player, alpha, beta,
                                                      cutoff_ply - 1))
            if max_payoff > beta:
                return max_payoff
            alpha = max(alpha, max_payoff)
        return max_payoff

    def voronoi(self, state, player):
        """
        Input: state, a state in a TronProblem. player, index for player in game
        Output: number represneting voronoi heuristic evaluation of state
        for the given player. Positive is good for the given player

        """
        distances = self.calc_distances(state)  # distances from each player to each square, flattened matrix
        # print(distances[1] - distances[0])
        distance_diff = distances[0] - distances[1]
        # board = state.board
        # s = ""
        # for row in board:
        #     for cell in row:
        #         s += cell
        #     s += "\n"
        # print("testing board: " + str(s))

        voronoi = 0
        for i in range(len(distance_diff)):
            if ((distances[0])[i] >= 0) and ((distances[1])[i] == -1):  # if p1 can get there but p2 can't
                voronoi += 1
            elif ((distances[1])[i] >= 0) and ((distances[0])[i] == -1):  # if p2 can get there but p1 can't
                voronoi -= 1
            elif distance_diff[i] < 0:  # if p1 gets there in less time
                voronoi += 1
            elif distance_diff[i] > 0:  # if p2 gets there in less time
                voronoi -= 1

        if player == 0:
            return voronoi  # value for p1
        else:
            return -voronoi  # value for p2

    def calc_distances(self, state):
        # calculates distances from each player to each square, to be used in voronoi function
        locs = state.player_locs
        board = np.array(state.board)
        # ignored player to move, we don't need

        players = np.arange(len(locs))
        players_distance = []

        for player in players:

            # matrix where each cell is distance from player. -1 if can't get there
            distance_matrix = np.negative(np.ones(board.shape))
            distance_matrix[locs[player]] = 0

            frontier = PriorityQueue()
            frontier.put((0, locs[player]))

            while not frontier.empty():
                distance, location = frontier.get()
                adjacent = self.adjacent_squares(location)
                for square in adjacent:
                    if not (board[square] == 'x' or board[square] == '#'):
                        if distance_matrix[square] == -1:
                            distance_matrix[square] = distance + 1
                            frontier.put((distance + 1, square))

            players_distance.append(distance_matrix.flatten())

        return players_distance

    def adjacent_squares(self, location):
        # Returns tuples of squares adjacent to current square
        i = location[0]
        j = location[1]
        return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    
    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass

    def not_connected(self, state):
        distances = self.calc_distances(state)  # distances from each player to each square, flattened matrix
        distance_diff = distances[0] - distances[1]
        notify = 0
        for i in range(len(distance_diff)):
            if ((distances[0])[i] != -1) and ((distances[1])[i] != -1):  # if p1 can get there but p2 can't
                notify += 1
        if notify == 0:
            return True
        else:
            return False

    def find_longest_path(self, board, loc):
        longest_path = None
        paths_LB = []
        finished_paths = []
        for move, location, updated_board in self.find_safe_moves(board, loc):
            paths_LB.append(([move], location, updated_board))
        while len(paths_LB) != 0:  # if there are more paths to evaluate
            this_path = paths_LB[0][0]
            this_location = paths_LB[0][1]
            this_board = paths_LB[0][2]
            paths_LB = paths_LB[1:]
            print("im here 3")
            safe_moves = self.find_safe_moves(this_board, this_location)
            if len(safe_moves) != 0:  # if it is not terminal state
                print("im here 4")
                for move, new_location, new_board in safe_moves:
                    new_path = this_path + [move]
                    paths_LB.append((new_path, new_location, new_board))
            else:  # remove path from paths and add to finished paths
                finished_paths = finished_paths + this_path

        # if no more paths to eval, return longest path
        longest_length = 0
        for path in finished_paths:
            if longest_length < len(path):
                longest_path = path
                longest_length = len(path)
        return longest_path

    def find_safe_moves(self, board, location):
        print("estoy aqui")
        total_moves = self.adjacent_squares(location)
        print("location", location)
        add_board = board
        print(add_board[location[0]][location[1]])
        add_board[location[0]][location[1]] = 'x'
        for row in add_board:
            print("added board", row)
        print("added x")
        safe = []
        if board[total_moves[0][0]][total_moves[0][1]] != 'x' and board[total_moves[0][0]][total_moves[0][1]] != '#':
            safe.append(('R', total_moves[0], add_board))
            print("R")
        if board[total_moves[1][0]][total_moves[1][1]] != 'x' and board[total_moves[0][0]][total_moves[0][1]] != '#':
            safe.append(('L', total_moves[1], add_board))
            print("L")
        if board[total_moves[2][0]][total_moves[2][1]] != 'x' and board[total_moves[0][0]][total_moves[0][1]] != '#':
            safe.append(('U', total_moves[2], add_board))
            print("U")
        if board[total_moves[3][0]][total_moves[3][1]] != 'x' and board[total_moves[0][0]][total_moves[0][1]] != '#':
            safe.append(('D', total_moves[3], add_board))
            print("D")
        print("finished safe moves")
        return safe

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"
    
    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
