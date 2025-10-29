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
    cells = []
    for i in range(9):
        for j in range(9):
            cells.append((i, j))
    return cells


def sudoku_arcs():
    arcs = []
    for cell1 in sudoku_cells():
        for cell2 in sudoku_cells():
            if cell1 != cell2 and (cell1[0] == cell2[0] or
                                   cell1[1] == cell2[1] or (
                    cell1[0] // 3 == cell2[0] // 3 and
                    cell1[1] // 3 == cell2[1] // 3)):
                arcs.append((cell1, cell2))
    return arcs


def read_board(path):
    # open the file, read the lines, and create a
    # dictionary mapping each cell
    # to its value
    with open(path, 'r') as f:
        board = {}
        for i in range(9):
            line = f.readline().strip()
            for j, char in enumerate(line):
                if char == '*':
                    board[(i, j)] = set(range(1, 10))
                else:
                    board[(i, j)] = {int(char)}

    return board


class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()

    def __init__(self, board):
        # covers case where board is already a dictionary
        if isinstance(board, dict):
            self.board = board
        # covers case where board is a list of lists
        else:
            self.board = {}
            for row in range(9):
                for col in range(9):
                    if board[row][col] == '*':
                        self.board[(row, col)] = set((range(1, 10)))
                    else:
                        self.board[(row, col)] = set([int(board[row][col])])

        # create a dictionary mapping each cell to a set of possible values
        # we are going to need a couple for loops here, check to see
        # if the value is already placed
        # don't eliminate the value if it's already placed, create the board as
        # is it's given. s

    def get_values(self, cell):
        return self.board[cell]

    def remove_inconsistent_values(self, cell1, cell2):
        removed = False
        # if cell2 has only one value
        if len(self.board[cell2]) == 1:
            # get the only value in cell2
            value = next(iter(self.board[cell2]))
            # if that value is in cell1, remove it
            if value in self.board[cell1]:
                self.board[cell1].remove(value)
                removed = True
        return removed

    def infer_ac3(self):
        queue = collections.deque(Sudoku.ARCS)
        # while there are still arcs in the queue
        while queue:
            # takes first element from the left of the queue
            (cell1, cell2) = queue.popleft()
            # remove inconsistent values from cell1 and cell2, which propogate
            # constraints
            if self.remove_inconsistent_values(cell1, cell2):
                if len(self.board[cell1]) == 0:
                    return False
                # for all cells in sudoku cells
                for cell3 in Sudoku.CELLS:
                    # if cell3 is not cell1 or cell2 and is in the same row,
                    # column, or box as cell1
                    if cell3 != cell1 and cell3 != cell2 and (cell3[0]
                                                              == cell1[0]
                                                              or cell3[1]
                                                              == cell1[1] or
                                                              (
                            cell3[0] // 3 == cell1[0] // 3 and
                            cell3[1] // 3 == cell1[1] // 3)):
                        queue.append((cell3, cell1))
        return all(len(self.board[cell]) > 0
                   for cell in Sudoku.CELLS)

    def infer_improved(self):
        # repeatedly apply the single possibility
        # and single position strategies until no more
        # progress can be made

        while True:
            changed = False
            # get the whole board
            if not self.infer_ac3():
                return False
            for unit in self.get_all_units():
                # create a dictionary mapping each value to the cells that can
                # contain it
                counts = collections.defaultdict(list)
                # for each value in the unit, add the cell to the list of cells
                # that can contain it
                for cell in unit:
                    # now this would be like self.board[0,0]: value for example
                    for value in self.board[cell]:
                        # add the cell to the list of cells that can contain
                        # the value, like for example counts[5] = [(0,0),
                        # (0,1)]
                        counts[value].append(cell)
                # if a value can only be in one cell, set that cell to that
                # value
                for value, cells in counts.items():
                    # if the cells has only one cell and that cell has more
                    # than one value
                    if len(cells) == 1 and len(self.board[cells[0]]) > 1:
                        self.board[cells[0]] = {value}
                        changed = True
            if not changed:
                break
            if not self.infer_ac3():
                return False
            if all(len(self.board[cell]) == 1 for cell in Sudoku.CELLS):
                return all(len(self.board[cell]) > 0 for cell in Sudoku.CELLS)

    def get_all_units(self):
        units = []
        for i in range(9):
            row = [(i, j) for j in range(9)]
            col = [(j, i) for j in range(9)]
            box = [(3 * (i // 3) + r, 3 * (i % 3) + c)
                   for r in range(3) for c in range(3)]
            units.append(row)
            units.append(col)
            units.append(box)
            print(row, col, box)
        return units

    def infer_with_guessing(self):
        # base case for recursion - if all cells have only one
        # value, return
        # True
        if all(len(self.board[cell]) == 1 for cell in Sudoku.CELLS):
            return True
        # choose a cell with the fewest possibilities
        cell = min(
            (c for c in Sudoku.CELLS if len(
                self.board[c]) > 1), key=lambda c: len(
                self.board[c]))
        # try each possibility for that cell
        for value in self.board[cell]:
            # create a copy of the baord
            new_board = copy.deepcopy(self.board)
            # copy the value over to the new board
            new_board[cell] = {value}
            # create a new Sudoku object with the new board
            sudoku = Sudoku(new_board)
            # if the new board is solvable, copy the solution back to the
            # original board and return True
            if sudoku.infer_ac3() or sudoku.infer_improved():
                if sudoku.infer_with_guessing():
                    self.board = sudoku.board
                    return True
        return False

############################################################
# Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
This assignment took me approximately 10 hours.
I found the most challenging part to be
implementing the AC-3 algorithm
and ensuring that all constraints
were correctly applied across the Sudoku grid.
Understanding how to efficiently
manage the propagation of constraints
was particularly tricky.
"""

feedback_question_2 = """
It would be useful if the base algorithms were provided in
pseudocode form in
the program spec.
This would help clarify the steps involved and ensure
that I am implementing
them correctly. Additionally, more examples of
Sudoku puzzles with varying
levels of difficulty
could help in understanding how different
strategies apply to different scenarios.
"""

feedback_question_3 = """
I liked the data structure choice of
using a dictionary to represent the Sudoku board.
It made it easy to access and update the
possible values for each cell.
The use of sets to represent possible
values for each cell was also a good choice,
as it allowed for efficient checking
and updating of values.
"""


def h(xs):
    return {x for x in xs if xs.count(x) == 1}
print(sorted(h([3,1,2,3,2,4])))


def Darth_Vader():
    for i in range(9):
        if i == 0:
            continue
            print("No")

        if i % 2 == 0:
            print("I")
            continue
            
        if i - 1 == 2:
            print("Am")
            break
            
        if i < 2:
            print("Your")
            pass
            print("Father") 
            
if __name__ == "__main__":
    Darth_Vader()

