############################################################
# CIS 521: Homework 3
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import numpy as np
import random
import copy
import queue
from queue import PriorityQueue

############################################################

student_name = "Jonathon Michael Delemos"

############################################################
# Section 1: Tile Puzzle
############################################################


def create_tile_puzzle(cols, rows):
    """'Creates tile puzzle with specified number of columns and rows"""
    puzzle = np.zeros((rows, cols))
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
        elif direction == "up" and row - 1 >= 0:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row - 1][col]
            self.board[row - 1][col] = temp
            return True
        elif direction == "left" and col - 1 >= 0:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row][col - 1]
            self.board[row][col - 1] = temp
            return True
        elif direction == "right" and col + 1 < self.cols:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row][col + 1]
            self.board[row][col + 1] = temp
            return True
        else:
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
        if np.array_equal(self.board, j.board):
            return True
        else:
            return False

    def copy(self):
        empty = copy.deepcopy(self)
        return empty

    def successors(self):
        "Method allows for the player to move the 0 tile."
        combo = []
        new_board = self.copy()
        new_board2 = self.copy()
        new_board3 = self.copy()
        new_board4 = self.copy()
        for i in range(new_board.rows):
            for f in range(new_board.cols):
                if new_board.board[i][f] == 0:
                    row, col = i, f
                    break

        if row + 1 < new_board.rows:
            temp = new_board.board[row][col]
            new_board.board[row][col] = new_board.board[row + 1][col]
            new_board.board[row + 1][col] = temp
            board = new_board.copy()
            combo.append(("down", board))
        if row - 1 >= 0:
            temp = new_board2.board[row][col]
            new_board2.board[row][col] = new_board2.board[row - 1][col]
            new_board2.board[row - 1][col] = temp
            board = new_board2.copy()
            combo.append(("up", board))
        if col - 1 >= 0:
            temp = new_board3.board[row][col]
            new_board3.board[row][col] = new_board3.board[row][col - 1]
            new_board3.board[row][col - 1] = temp
            board = new_board3.copy()
            combo.append(("left", board))
        if col + 1 < new_board4.cols:
            temp = new_board4.board[row][col]
            new_board4.board[row][col] = new_board4.board[row][col + 1]
            new_board4.board[row][col + 1] = temp
            board = new_board4.copy()
            combo.append(("right", board))
        return combo

    # Required
    def find_solutions_iddfs(self):
        depth = 0
        path = []
        visited = set()
        while True:
            print(f"Trying depth: {depth}")
            result = self.idffs_helper(depth, path, set())
            if result is not None:
                yield result
                return
            else:
                depth += 1

    def idffs_helper(self, limit, moves, visited):
        if self.is_solved():
            return moves

        if limit == 0:
            return None

        if self in visited:
            return None

        visited.add(self)
        for move, state in self.successors():

            if state in visited:
                continue

            result = state.idffs_helper(limit - 1, moves + [move], visited.copy())
            if result is not None:
                return result
        return None

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        return hash(self.board.tobytes())

    # Required
    # Required
    def find_solution_a_star(self):
        # total distance travelled so far
        g_n = 0
        # heuristic/goal to end position
        h_n = self.manhattan()
        # comparison
        f_n = g_n + h_n
        # the goal is to keep a running manhattan distance of each tile. where it starts versus where it ends up.
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

    # For each tile value v at (row, col) in the current board (except 0, the blank):
    # Figure out where v should be in the solved board. (Since tiles are numbered 1..N, you can compute its goal row and column mathematically.)
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
                    # don't use break, continue will allow you to skip*, break will exit both loops
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
    pass


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def solve_distinct_disks(length, n):
    pass


############################################################
# Section 4: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
