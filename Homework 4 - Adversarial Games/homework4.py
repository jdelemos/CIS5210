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
    return DominoesGame(list([[False for _ in range(cols)]
                        for _ in range(rows)]))


class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.rows = len(board)
        self.cols = len(board[0])
        self.board = board

    def get_board(self):
        return self.board

    def reset(self):
        self.board = [[False for _ in range(self.cols)]
                      for _ in range(self.rows)]

    def is_legal_move(self, row, col, vertical):
        if row < 0 or col < 0 or row > self.rows - 1 or col > self.cols - 1:
            return False
        if vertical and row + 1 < self.rows:
            if self.board[row][col] is False and self.board[row +
                                                            1][col] is False:
                return True
        if vertical is False and col + 1 < self.cols:
            if self.board[row][col] is False and self.board[row][col
                                                                 + 1] is False:
                return True
        return False

    def legal_moves(self, vertical):
        for x in range(self.rows):
            for y in range(self.cols):
                if self.is_legal_move(x, y, vertical):
                    yield tuple([x, y])
        return

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical) is True:
            if vertical is True:
                self.board[row][col] = True
                self.board[row + 1][col] = True
                return True
            if vertical is False:
                self.board[row][col] = True
                self.board[row][col + 1] = True
                return True
        else:
            return False

    def game_over(self, vertical):
        return (
            len(list(self.legal_moves(vertical))) == 0
        )

    def copy(self):
        return copy.deepcopy(self)

    def successors(self, vertical):
        move_set = []
        for x, y in itertools.product(range(self.rows), range(self.cols)):
            new_state = self.copy()
            if new_state.perform_move(x, y, vertical):
                move_set.append((tuple((x, y)), new_state))
        return move_set

    def get_random_move(self, vertical):
        random_set = []
        move_set = self.successors(vertical)
        for move, state in move_set:
            random_set.append(move)

        return random.choice(random_set)

    # Required
    def get_best_move(self, vertical, limit):
        num_moves = [0]
        root_vertical = vertical

        def best_move_heuristic(self, vertical):
            my_moves = len(list(self.legal_moves(vertical)))
            opp_moves = len(list(self.legal_moves(not vertical)))
            return my_moves - opp_moves

        def max_value_fun(self, vertical, limit, alpha, beta):
            # if we have reached base case, return the heuristic and increment
            # the number of total moves played.
            if limit == 0 or self.game_over(vertical):
                num_moves[0] += 1
                return best_move_heuristic(self, root_vertical)
            value = float("-inf")
            moves = list(self.successors(vertical))
            for move, state in moves:
                new_value = min_value(
                    state, not vertical, limit - 1, alpha, beta)
                if new_value > value:
                    value = new_value
                alpha = max(alpha, value)
                if beta <= value:
                    break
            return value

        def min_value(self, vertical, limit, alpha, beta):
            # if we have reached base case, return the heuristic and increment
            # the number of total moves played.
            if limit == 0 or self.game_over(vertical):
                num_moves[0] += 1
                return best_move_heuristic(self, root_vertical)
            value = float("inf")
            moves = list(self.successors(vertical))
            for move, state in moves:
                new_value = max_value_fun(
                    state, not vertical, limit - 1, alpha, beta)
                if new_value < value:
                    value = new_value
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value

        best_value = float("-inf")
        best_move = None
        alpha, beta = float("-inf"), float("inf")

        for move, new_state in self.successors(root_vertical):
            value = min_value(
                new_state,
                not root_vertical,
                limit - 1,
                alpha,
                beta)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
            if alpha >= beta:   # cutoff at root too
                break

        return (best_move, best_value, num_moves[0])


############################################################
# Section 2: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = """
Creating the framework for this program was much
easier than previous works. I spent approximately 7
hours on this assignment.
"""

feedback_question_2 = """
The most challenging part of this
assignment was the creation of the alpha beta pruning
algorithm. There wasn't a data structure to
rely on. I couldn't rely on past experience.
That was tough.
"""

feedback_question_3 = """
I really liked building the alpha beta min max
algorithm. It's a unique algorithm that
can be hard to follow, but it's had a profound
impact on my understanding of adversarial games.
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

b = [[False, False, False], [False, False, False], [False, False, False]]
g = DominoesGame(b)
print(g.get_best_move(True, 2))
