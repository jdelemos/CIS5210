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


def create_dominoes_game(rows, cols):
    return DominoesGame(list([[False for _ in range(cols)] for _ in range(rows)]))

class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.rows = len(board)
        self.cols = len(board[0])
        self.board = board

    def get_board(self):
        return self.board

    def reset(self):
       self.board = [[False for _ in range(self.cols)] for _ in range(self.rows)]

    def is_legal_move(self, row, col, vertical):
        if row > self.rows or col > self.cols:
            return False
        if vertical == True and row + 1 < self.rows:
            if self.board[row][col] == False and self.board[row+1][col] == False:
                return True
        if vertical == False and col + 1 < self.cols:
            if self.board[row][col] == False and self.board[row][col+1] == False:
                return True
        return False

    def legal_moves(self, vertical):
        for x in range(self.rows):
            for y in range(self.cols):
                if self.is_legal_move(x,y, True):
                    yield tuple([x,y])
        return

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical) == True:
            if vertical == True:
                self.board[row][col] = True
                self.board[row+1][col] = True
                return True
            if vertical == False: 
                self.board[row][col] = True
                self.board[row][col+1] = True
                return True
        else:
            return False
        

    def game_over(self, vertical):
        if len(list(self.legal_moves(True))) == 0:
            return True
        if len(list(self.legal_moves(False))) == 0:
            return True
        return False
               
        

    def copy(self):
        return copy.deepcopy(self)

    def successors(self, vertical):
        move_set = []
        for x, y in itertools.product(range(self.rows), range(self.cols)):
            new_state = self.copy()
            if new_state.perform_move(x,y, vertical):
                move_set.append((tuple((x,y)), new_state.get_board()))
        return move_set
    
    def get_random_move(self, vertical):
        move_set = self.successors(vertical)
        print(move_set)


    # Required
    def get_best_move(self, vertical, limit):
        pass

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


# b = [[True, False], [True, False]]
# g = DominoesGame(b)
# g.get_board()
# [[True, False], [True, False]]
# g.reset()
# print(g.get_board())

# b = [[True, False], [True, False]]
# g = DominoesGame(b)
# print((g.game_over(True)))

# g = create_dominoes_game(3, 3)
# print(list(g.legal_moves(True)))


# g = create_dominoes_game(3, 3)
# g.perform_move(1,0, False)
# g2 = g.copy()
# print(g2.successors(True))

b = [[True, False], [True, False]]
g = DominoesGame(b)
print(g.successors(True))
