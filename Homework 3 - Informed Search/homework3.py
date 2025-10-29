############################################################
# CIS 521: Homework 3
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random
import copy
import queue
from queue import PriorityQueue
import math

############################################################

student_name = "Jonathon Michael Delemos"

############################################################
# Section 1: Tile Puzzle
############################################################


def create_tile_puzzle(cols, rows):
    """'Creates tile puzzle with specified number of columns and rows"""
    puzzle = list([[0 for _ in range(cols)] for _ in range(rows)])
    sum = 1
    for i in range(rows):
        for o in range(cols):
            if sum == rows * cols:
                return TilePuzzle(puzzle)
            puzzle[i][o] = sum
            sum += 1
    return TilePuzzle(puzzle)


class TilePuzzle(object):
    """Initializer for TilePuzzle object"""

    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        """'Get the 2d representation of the board"""
        return self.board

    def perform_move(self, direction):
        "Method allows for the player to move the 0 tile."
        for i in range(self.rows):
            for f in range(self.cols):
                if self.board[i][f] == 0:
                    row, col = i, f
                    break

        if direction == "down" and row + 1 < self.rows:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row + 1][col]
            self.board[row + 1][col] = temp
            return True
        if direction == "right" and col + 1 < self.cols:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row][col + 1]
            self.board[row][col + 1] = temp
            return True
        if direction == "up" and row - 1 >= 0:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row - 1][col]
            self.board[row - 1][col] = temp
            return True
        if direction == "left" and col - 1 >= 0:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row][col - 1]
            self.board[row][col - 1] = temp
            return True
        return False

    def scramble(self, num_moves):
        if num_moves == 0:
            return
        my_list = ["up", "down", "left", "right"]
        random_choice = random.choice(my_list)
        self.perform_move(random_choice)
        self.scramble(num_moves - 1)

    def is_solved(self):
        j = create_tile_puzzle(self.rows, self.cols)
        if self.board == j.board:
            return True
        else:
            return False

    def copy(self):
        empty = copy.deepcopy(self)
        return empty

    def successors(self):
        "Method allows for the player to move the 0 tile."
        combo = []
        for direction in ['up', 'down', 'left', 'right']:
            new_state = self.copy()
            if new_state.perform_move(direction):
                combo.append((direction, new_state))
        return combo

    # Required
    def iddfs_helper(self, limit, moves, visited):
        # check to see if it's solved
        if self.is_solved():
            # return the single solution
            yield moves
            return
        # base case exit
        if limit == 0:
            return None
        # if we haven't seen it before, append it
        if self not in visited:
            visited.add(self)
            # examine the successors
            for move, state in self.successors():
                # find the solved states recursively
                result = state.iddfs_helper(limit - 1, moves + [move], visited)
                # return a yielded list
                if result is not None:
                    yield from result
        return None

    def find_solutions_iddfs(self):
        depth = 0
        path = []
        not_solved = True
        while not_solved:
            result = self.iddfs_helper(depth, path, set())
            for cluster in result:
                if cluster is not None:
                    yield cluster
                    not_solved = False
            depth += 1

    def __eq__(self, other):
        return self.board == other.board

    def _as_immutable(self):
        return tuple(tuple(row) for row in self.board)

    def __hash__(self):
        return hash(self._as_immutable())

    # Required
    # Required
    def find_solution_a_star(self):
        # total distance travelled so far
        g_n = 0
        # heuristic/goal to end position
        h_n = self.manhattan()
        # comparison
        f_n = g_n + h_n
        # the goal is to keep a running manhattan distance of each tile. where
        # it starts versus where it ends up.
        tiebreaker = 0
        path = []
        state = self.copy()
        # proper way to declare priority queue
        frontier = PriorityQueue()
        # added as a tuple
        frontier.put((f_n, tiebreaker, g_n, path, state))

        # visited set that is immutable
        visited = set()

        # Pop a new puzzle state current from the frontier
        # Then compute the Manhattan distance for that state’s board
        # Then use that h(n) for A*'s decision-making
        while frontier.qsize() > 0:
            f, tie, g, path, state = frontier.get()
            if state in visited:
                continue
            visited.add(state)
            if state.is_solved():
                return path
            for move, succ_state in state.successors():
                h_n = succ_state.manhattan()
                g_n = g + 1
                f_n = g_n + h_n
                tiebr = tie + 1
                if succ_state in visited:
                    continue
                frontier.put((f_n, tiebr, g_n, path + [move], succ_state))

    # For each tile value v at (row, col)
    # Figure out where v should be in the solved board.
    # Goal row = (v-1) // cols
    # Goal col = (v-1) % cols
    # Add abs(row - goal_row) + abs(col - goal_col) to the running sum.
    # That’s your heuristic h(n).
    def manhattan(self):
        manhattan_distance = 0
        for i in range(self.rows):
            for f in range(self.cols):
                v = self.board[i][f]
                if v == 0:
                    # don't use break, continue will allow you to skip*, break
                    # will exit both loops
                    continue
                # rounds down to the nearest integer
                goal_row = (v - 1) // self.cols
                goal_col = (v - 1) % self.cols
                # i is the current tile’s row index, f is the current col index
                manhattan_distance += abs(i - goal_row) + abs(f - goal_col)
        return manhattan_distance


############################################################
# Section 2: Grid Navigation
############################################################


def find_path(start, goal, scene):
    g_n = 0
    h_n = euclidean_distance(start, goal)
    f_n = g_n + h_n
    path = []
    visited = set()
    frontier = PriorityQueue()
    frontier.put((f_n, g_n, start, [start]))

    # mistakes made - I used a true loop instead of checking the frontier.
    while not frontier.empty():
        # grabbing the values to unpack after the looping while call
        f, g, current, path = frontier.get()

        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        for move, cost in successors_path(current, scene, path):
            h_n = euclidean_distance(move, goal)
            g_n = g + cost
            f_n = g_n + h_n
            if move in visited:
                continue
            frontier.put((f_n, g_n, move, path + [move]))
    return None


def successors_path(start, scene, path):
    "Method allows for path navigation."
    combo = []
    rows = len(scene)
    cols = len(scene[0])
    x, y = start

    # orthogonal
    if x + 1 < rows and not scene[x + 1][y]:
        combo.append(((x + 1, y), 1.0))
    if x - 1 >= 0 and not scene[x - 1][y]:
        combo.append(((x - 1, y), 1.0))
    if y + 1 < cols and not scene[x][y + 1]:
        combo.append(((x, y + 1), 1.0))
    if y - 1 >= 0 and not scene[x][y - 1]:
        combo.append(((x, y - 1), 1.0))

    # diagonals
    if x - 1 >= 0 and y - 1 >= 0 and not scene[x - 1][y - 1]:
        combo.append(((x - 1, y - 1), math.sqrt(2)))
    if x - 1 >= 0 and y + 1 < cols and not scene[x - 1][y + 1]:
        combo.append(((x - 1, y + 1), math.sqrt(2)))
    if x + 1 < rows and y - 1 >= 0 and not scene[x + 1][y - 1]:
        combo.append(((x + 1, y - 1), math.sqrt(2)))
    if x + 1 < rows and y + 1 < cols and not scene[x + 1][y + 1]:
        combo.append(((x + 1, y + 1), math.sqrt(2)))
    return combo


def copy2(scene):
    empty = copy.deepcopy(scene)
    return empty


def euclidean_distance(pos, goal):
    x1, y1 = pos
    x2, y2 = goal
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def solve_distinct_disks(length, n):
    disks = []
    for i in range(1, n + 1):
        disks.append(i)
    for i in range(length - n):
        disks.append(0)
    goal = tuple([0] * (length - n) + list(range(n, 0, -1)))
    disks = tuple(disks)
    g_n = 0
    f_n = g_n + 0
    visited = set()
    cost = 0
    frontier = PriorityQueue()
    frontier.put((f_n, g_n, disks, []))

    while not frontier.empty():
        f, g, disk, path = frontier.get()
        if disk == goal:
            return path
        if disk not in visited:
            visited.add(disk)
            for pos, new_disk, cost in disk_successors(disk):
                heur = disk_heuristic(new_disk)
                g_total = g + cost
                # print(f'Printing G Total: {g_total}')
                f_total = g_total + heur
                # print(f'Printing F Total: {f_total}')
                if new_disk in visited:
                    continue
                frontier.put((f_total, g_total, new_disk, path + [pos]))
    return None


def disk_heuristic(state):
    total = 0
    length = len(state)
    for i in range(1, max(state) + 1):
        index = state.index(i)
        goal = length - i
        total += abs(index - goal)
    return total // 2


def disk_successors(state):
    results = []
    length = len(state)

    for i in range(length):
        if state[i] > 0:
            # Slide right
            if i + 1 < length and state[i + 1] == 0:
                new_state = list(state)
                new_state[i + 1], new_state[i] = new_state[i], 0
                results.append(((i, i + 1), tuple(new_state), 1))

            # Jump right
            if i + 2 < length and state[i + 1] != 0 and state[i + 2] == 0:
                new_state = list(state)
                new_state[i + 2], new_state[i] = new_state[i], 0
                results.append(((i, i + 2), tuple(new_state), 1))

            # Slide left
            if i - 1 >= 0 and state[i - 1] == 0:
                new_state = list(state)
                new_state[i - 1], new_state[i] = new_state[i], 0
                results.append(((i, i - 1), tuple(new_state), 1))

            # Jump left
            if i - 2 >= 0 and state[i - 1] != 0 and state[i - 2] == 0:
                new_state = list(state)
                new_state[i - 2], new_state[i] = new_state[i], 0
                results.append(((i, i - 2), tuple(new_state), 1))

    return results
############################################################
# Section 4: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
Approximately 10-15 hours spent.
I will be studying these algorithms
heavily.
It took a couple builds to get comfortable with A*.
The concepts are fairly straightforward,
the application is easy to mix up.
"""

feedback_question_2 = """
Optimizations in the code are the most
challenging.
Often I understand
the logic but struggle to implement solutions
cleanly. I think that will just come with
time spent practicing.
"""

feedback_question_3 = """
I loved building A*. It's such a
cool and special algorithm,
I'm very glad I got the chance
to build something simple and apply it.
Thank you!
"""
