############################################################
# CIS 521: Homework 4
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import collections
import copy
import itertools
import random
import math

############################################################

student_name = "Jonathon Michael Delemos"

############################################################
# Section 1: Dominoes Game
############################################################


class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.rows = len(board)
        self.cols = len(board[0])
        self.board = board


    def get_board(self):
        return self.board

    def reset(self):
        pass

    def is_legal_move(self, row, col, vertical):
        pass

    def legal_moves(self, vertical):
        pass

    def perform_move(self, row, col, vertical):
        pass

    def game_over(self, vertical):
        pass

    def copy(self):
        pass

    def successors(self, vertical):
        pass

    def get_random_move(self, vertical):
        pass

    # Required
    def get_best_move(self, vertical, limit):
        pass


def create_dominoes_game(rows, cols):
    return DominoesGame(list([[False for _ in range(cols)] for _ in range(rows)]))

g = create_dominoes_game(2, 3)

print(g.get_board())
############################################################
# Section 2: Feedback
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


# b = [[False, False], [False, False]]
# g = DominoesGame(b)
# print(g.get_board())
