# NOTE TO STUDENT: Please read the handout before continuing.
from typing import Any, Tuple

from tilegameproblem import TileGame
from dgraph import DGraph
from queue import Queue, LifoQueue, PriorityQueue
import time


### GENERAL SEARCH IMPLEMENTATIONS - NOT SPECIFIC TO THE TILEGAME PROBLEM ###

# what is stencil code
def dfs(problem):
    visited = {}  # initialize visited DICTIONARY (EMPTY)
    first = problem.get_start_state()
    frontier = LifoQueue()  # initialize frontier STACK (EMPTY)
    frontier.put(first)
    visited[first] = None
    while True:
        parent = frontier.get()
        if problem.is_goal_state(parent) is False:
            for key in problem.get_successors(parent):  # finds children of nodes in frontier
                if key not in visited:  # make sure that child is not already visited
                    visited[key] = parent  # add child and parent to visited
                    frontier.put(key)  # add new nodes to frontier --> TOP of frontier, taken first
        elif problem.is_goal_state(parent) is True:
            return find_path_dfs(parent, visited)


def find_path_dfs(end_state, visited):
    ans = []
    child = end_state
    parent = visited[child]
    while parent is not None:
        ans.append(child)
        child = parent
        parent = visited[child]
    return reversed(ans)


# IDS is almost identical to DFS except we use a Priority Queue to keep track of depth
# Use a holder to keep track of "depth" and insert tuples into frontier
# Depth goes from x,x-1,...,1 (shows how many more depths can be taken + 1)


def ids(problem):
    visited = {}  # initialize visited DICTIONARY (EMPTY)
    first_state = problem.get_start_state()  # get first state
    dfs_depth = 0  # how deep to look into graph .. to be incremented
    frontier = [(first_state, 0)]  # state, depth, parent
    visited[first_state] = (None, 0)  # parent, TOTAL COST to get to first
    while True:
        if len(frontier) != 0:
            if frontier[-1][1] > 0:
                node_info = frontier.pop()  # gets HIGHEST PRIORITY node (last inputted)
                parent = node_info[0]
                value = node_info[1]
                if problem.is_goal_state(parent) is False:
                    for child in problem.get_successors(parent):  # finds children of nodes in frontier
                        if child not in visited or (child in visited and (visited[child][1] > visited[parent][1] + 1)):
                            visited[child] = (parent, visited[parent][1] + 1)
                            child_priority = value - 1
                            frontier.append((child, child_priority, parent))  # add new nodes to frontier
                elif problem.is_goal_state(parent) is True:
                    return find_path_ids(parent, first_state, visited)
            elif frontier[-1][1] == 0:
                frontier.pop()
        elif len(frontier) == 0:
            dfs_depth = dfs_depth + 1
            visited.clear()
            visited[first_state] = (None, 0)
            frontier = [(first_state, dfs_depth)]


def find_path_ids(end_state, start_state, visited):
    ans = []
    child = end_state
    parent = visited[child][0]
    while parent is not None:
        ans.append(child)
        child = parent
        parent = visited[child][0]
    ans = ans + [start_state]
    return reversed(ans)


def astar(problem, heur):  # Adjust frontier
    visited = {}  # initialize visited DICTIONARY (EMPTY)
    first_state = problem.get_start_state()  # get first state
    frontier = PriorityQueue(100000)  # initialize frontier PRIORITY QUEUE (EMPTY)
    frontier.put((heur(first_state) + 0, 0, first_state))  # update frontier (f = g + h, total cost to state, state)
    visited[first_state] = (None, 0)  # parent, TOTAL COST to get to first
    while True:
        node_info = frontier.get()  # gets HIGHEST PRIORITY node
        parent_cost = node_info[1]
        parent = node_info[2]
        if problem.is_goal_state(parent) is False:
            for child in problem.get_successors(parent):  # finds children of nodes in frontier
                cost = problem.get_successors(parent)[child]
                if child not in visited or (child in visited and (visited[child][1] > visited[parent][1] + cost)):
                    visited[child] = (parent, parent_cost + cost)  # add child and parent, total depth to visited
                    frontier.put((heur(child) + parent_cost, parent_cost + cost, child))  # add new nodes to frontier
        elif problem.is_goal_state(parent) is True:
            return find_path_astar(parent, first_state, visited)


def find_path_astar(end_state, start_state, visited):
    ans = []
    child = end_state
    parent = visited[child][0]
    while parent is not None:
        ans.append(child)
        child = parent
        parent = visited[child][0]
    ans = ans + [start_state]
    return reversed(ans)


def id_astar(problem, heur):
    first_state = problem.get_start_state()
    my_path = [first_state]
    cutoff = heur(first_state)
    while True:  # iterate until helper finds optimal path (assuming admissible heuristic)
        dic = {first_state: 0}  # dictionary that maps node to the cost to get there (g)
        threshold = helper(my_path, dic, cutoff, problem, heur)  # path list, cost dictionary, cutoff
        if threshold == -1:
            return my_path
        if threshold == float("inf"):
            return None
        cutoff = threshold  # keep increasing cutoff


def helper(my_path, dic, cutoff, problem, heur):
    current_node = my_path[-1]
    f = dic[current_node] + heur(current_node)
    frontier = PriorityQueue(100000)
    if f > cutoff:
        return f
    if problem.is_goal_state(current_node):
        return -1
    min_cost = float("inf")
    for child in problem.get_successors(current_node):
        g_child = problem.get_successors(current_node)[child] + dic[current_node]
        if child not in dic.keys() or dic[child] < g_child:
            dic[child] = g_child
            frontier.put((dic[child] + heur(child), child))
    while frontier.empty() is False:
        node_info = frontier.get()
        node = node_info[1]
        if node not in my_path:
            my_path.append(node)
            threshold = helper(my_path, dic, cutoff, problem, heur)
            if threshold == -1:
                return -1
            if threshold < min_cost:
                min_cost = threshold
            my_path.pop(-1)
    return min_cost


### SPECIFIC TO THE TILEGAME PROBLEM ###


# manhattan distance/2 for admissible
def tilegame_heuristic(state):  # underestimate
    length = len(state)  # also equal to the length of matrix
    current_column = 0
    current_row = 0
    manhattan_distance = 0
    for row in state:
        for element in row:
            correct_column = element % length - 1  # column starts at 0
            correct_row = int((element - 1)/length)  # row starts at 0
            horizontal_distance = abs(correct_column - current_column)
            vertical_distance = abs(correct_row - current_row)
            manhattan_distance = manhattan_distance + horizontal_distance + vertical_distance
            if current_column == length - 1:  # update current column
                current_column = 0
            else:
                current_column = current_column + 1
        if current_row == length - 1:  # update current row
            current_row = 0
        else:
            current_row = current_row + 1
    return manhattan_distance/2


# manhattan distance for inadmissible
def tilegame_inadmissible_heuristic(state):  # overestimate
    length = len(state)  # also equal to the length of matrix
    current_column = 0
    current_row = 0
    manhattan_distance = 0
    for row in state:
        for element in row:
            correct_column = element % length - 1  # column starts at 0
            correct_row = int((element - 1)/length)  # row starts at 0
            horizontal_distance = abs(correct_column - current_column)
            vertical_distance = abs(correct_row - current_row)
            manhattan_distance = manhattan_distance + horizontal_distance + vertical_distance
            if current_column == length - 1:  # update current column
                current_column = 0
            else:
                current_column = current_column + 1
        if current_row == length - 1:  # update current row
            current_row = 0
        else:
            current_row = current_row + 1
    return manhattan_distance  # don't divide by 2


### YOUR SANDBOX ###

start_time = time.time()

def main():
    """
    Do whatever you want in here; this is for you.
    The examples below shows how your functions might be used.
    """

    # initialize a random 3x3 TileGame problem
    tg = TileGame(3)
    # print(TileGame.board_to_pretty_string(tg.get_start_state()))
    # compute path using dfs
    path1 = id_astar(tg, tilegame_heuristic)
    path = ids(tg)
    print(tg.get_start_state())
    # display path
    print('ids')
    # TileGame.print_pretty_path(path)
    print('astar')
    TileGame.print_pretty_path(path1)
    print((time.time() - start_time))

    # initialize a small DGraph
    small_dgraph = DGraph([[None, 1], [1, None]], {1})
    # print the path using ids
    # print(ids(small_dgraph))


if __name__ == "__main__":
    tg = TileGame(3)
    path1 = id_astar(tg, tilegame_heuristic)
    # path = ids(tg)
    # print('ids')
    # TileGame.print_pretty_path(path)
    print('astar')
    TileGame.print_pretty_path(path1)
    # print(tg.get_start_state())
    # print((time.time() - start_time))
    main()

