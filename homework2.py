############################################################
# CIS 521: Homework 2
############################################################

student_name = "Jonathon Michael Delemos"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import math
import numpy as np

############################################################
# Section 1: N-Queens
############################################################


def num_placements_all(n):
    # (total number of squares, which is 8 * 8, choosing the number of squares, 8)
    return math.comb(n * n, n)


def num_placements_one_per_row(n):
    return n**n


def n_queens_valid(board):
    result = []
    column = []
    for index, value in enumerate(board):
        for index2, value2 in result:
            if index - value == index2 - value2:
                # print("Reached False Case: Negative Diagonals")
                return False
            if index + value == index2 + value2:
                # print("Reached False case: Positive Diagonals")
                return False
        if value in column:
            # print("Reached False case: Same Columns")
            return False
        result.append((index, value))
        column.append(value)
    # print("Passed")
    return True


def n_queens_solutions(n):
    results = []
    queens_helper(0, [], set(), set(), set(), n, results)
    #print(len(results))
    return results


def queens_helper(row, placement, cols, diags1, diags2, n, results):
    # we have reached base case, this is where we have found all combinations at each row level
    if row == n:
        results.append(placement.copy())
        return
    for col in range(n):
        if (col in cols) or ((row - col) in diags1) or ((row + col) in diags2):
            continue
        placement.append(col)
        cols.add(col)
        diags1.add(row - col)
        diags2.add(row + col)
        queens_helper(row + 1, placement, cols, diags1, diags2, n, results)
        #this is where I got confused
        placement.pop()
        cols.remove(col)
        diags1.remove(row - col)
        diags2.remove(row + col)


############################################################
# Section 2: Lights Out
############################################################


class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        #base case       
        self.board[row][col] = not self.board[row][col]
        #try lower
        try:
            self.board[row+1][col] = not self.board[row+1][col]
        except: 
            pass
        #try right
        try:
            self.board[row][col+1] = not self.board[row][col+1] 
        except: 
            pass
        #try up
        try:
            self.board[row-1][col] = not self.board[row-1][col]
        except: 
            pass
        #try left
        try:
            self.board[row][col-1] = not self.board[row][col-1]
        except: 
            pass

    def scramble(self):
        pass

    def is_solved(self):
        pass

    def copy(self):
        pass

    def successors(self):
        pass

    def find_solution(self):
        pass


def create_puzzle(rows, cols):
    board = np.zeros((rows,cols), dtype = bool)
    return LightsOutPuzzle(board)


############################################################
# Section 3: Linear Disk Movement
############################################################


def solve_identical_disks(length, n):
    pass


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
