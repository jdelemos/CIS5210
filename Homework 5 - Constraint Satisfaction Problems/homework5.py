############################################################
# CIS 521: Homework 5
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
# Sudoku Solver
############################################################


def sudoku_cells():
    for i in range(9):
        for j in range(9):
            yield (i, j)


def sudoku_arcs():
    pass


def read_board(path):
    # open the file, read the lines, and create a dictionary mapping each cell to its value
    with open(path, 'r') as f:
        board = {}
        for i in range(9):
            for j in range(9):
                for char in f.readline().strip():
                    if char == '*':
                        board[(i, j)] = set(range(1,10))
                    else:
                        board[(i, j)] = set(char)
    return board        

class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()


#     3 points] In this section, we will view a Sudoku puzzle not from the perspective of its grid
# layout, but more abstractly as a collection of cells. Accordingly, we will represent it internally
# as a dictionary mapping from cells, i.e. (row, column) pairs, to sets of possible values. This
# dictionary should have a fixed (9 × 9 = 81) set of pairs of keys, but the number of elements in
# each set corresponding to a key will change as the board is being manipulated.
# In the Sudoku class, write an initialization method __init__(self, board) that stores such a
# mapping for future use. Also write a method get_values(self, cell) that returns the set of
# values currently available at a particular cell.
# In addition, write a function read_board(path) that reads the board specified by the file at
# the given path and returns it as a dictionary. Sudoku puzzles will be represented textually as 9
# lines of 9 characters each, corresponding to the rows of the board, where a digit between "1"
# and "9" denotes a cell containing a fixed value, and an asterisk "*" denotes a blank cell that
# could contain any digit.

    
    

    def __init__(self, board):
        #covers case where board is already a dictionary
        if isinstance(board, dict):
            self.board = board
        #covers case where board is a list of lists
        else:
            self.board = {}
            for row in range(9):
                for col in range(9):
                    if board[row][col] == '*':
                        self.board[(row,col)] = set(set(range(1,10)))
                    else:
                        self.board[(row,col)] = set(board[row][col])
        print(self.board)
            


        
        # create a dictionary mapping each cell to a set of possible values
        # we are going to need a couple for loops here, check to see if the value is already placed
        # don't eliminate the value if it's already placed, create the board as is it's given. s

    def get_values(self, cell):
        return self.board[cell]

    def remove_inconsistent_values(self, cell1, cell2):
        pass

    def infer_ac3(self):
        pass

    def infer_improved(self):
        pass

    def infer_with_guessing(self):
        pass

############################################################
# Feedback
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


b = read_board("sudoku/medium1.txt")
print(Sudoku(b).get_values((0, 0)))

b = read_board("sudoku/medium1.txt")
Sudoku(b).get_values((0, 1))
