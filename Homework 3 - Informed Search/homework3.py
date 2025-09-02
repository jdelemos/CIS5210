############################################################
# CIS 521: Homework 3
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import numpy as np
import random 

############################################################

student_name = "Jonathon Michael Delemos"

############################################################
# Section 1: Tile Puzzle
############################################################


def create_tile_puzzle(cols, rows): 
    ''''Creates tile puzzle with specified number of columns and rows'''
    puzzle = np.zeros((rows, cols))
    sum = 1
    for i in range(rows):
        for o in range(cols):
            if sum == rows * cols:
                print(puzzle)
                return TilePuzzle(puzzle)
            puzzle[i][o] = sum
            sum+=1
    return TilePuzzle(puzzle)
    

class TilePuzzle(object):
    '''Initializer for TilePuzzle object'''
    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        ''''Get the 2d representation of the board'''
        return self.board
   
    def perform_move(self, direction):
        'Method allows for the player to move the 0 tile.'
        for i in range(self.rows):
            for f in range (self.cols):
                if self.board[i][f] == 0:
                    row, col = i,f
                    break

        if direction == 'down' and  row+1< self.rows:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row+1][col]
            self.board[row+1][col] = temp
            return True
        elif direction == 'up' and row -1 >= 0:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row-1][col]
            self.board[row-1][col] = temp
            return True
        elif direction == 'left' and col-1 >=0:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row][col-1]
            self.board[row][col-1] = temp
            return True
        elif direction == 'right' and col+1 < self.cols:
            temp = self.board[row][col]
            self.board[row][col] = self.board[row][col+1]
            self.board[row][col+1] = temp
            return True
        else:
            return False

    def scramble(self, num_moves):
        my_list = ['up','down','left','right']
        random_choice = random.choice(my_list)
        self.perform_move(random_choice)
        self.scramble(num_moves=num_moves-1)

    def is_solved(self):
        pass

    def copy(self):
        pass

    def successors(self):
        pass

    # Required
    def find_solutions_iddfs(self):
        pass

    # Required
    def find_solution_a_star(self):
        pass

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
