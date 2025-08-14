############################################################
# CIS 521: Homework 2
############################################################

student_name = "Type your full name here."

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

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
    pass


############################################################
# Section 2: Lights Out
############################################################


class LightsOutPuzzle(object):

    def __init__(self, board):
        pass

    def get_board(self):
        pass

    def perform_move(self, row, col):
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
    pass


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
