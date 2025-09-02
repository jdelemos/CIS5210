import math
import numpy as np
import random as random
import copy
from collections import deque


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
                print("Reached False Case: Negative Diagonals")
                return False
            if index + value == index2 + value2:
                print("Reached False case: Positive Diagonals")
                return False
        if value in column:
            print("Reached false case")
            return False
        result.append((index, value))
        column.append(value)
    print("Passed")
    return True


n_queens_valid([3, 2])


def n_queens_solutions(n):
    results = []
    queens_helper(0, [], set(), set(), set(), n, results)
    print(len(results))
    return results


def queens_helper(row, placement, cols, diags1, diags2, n, results):
    # we have reached
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
        #recursive DFS search
        queens_helper(row + 1, placement, cols, diags1, diags2, n, results)
        #this is where I got confused
        cols.remove(col)
        diags1.remove(row - col)
        diags2.remove(row + col)


n_queens_solutions(8)

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

        #lower row case
        if self.rows > row + 1: 
            self.board[row+1][col] = not self.board[row+1][col]
        #upper row case
        if row -1 >= 0: 
            self.board[row-1][col] = not self.board[row-1][col]
        #right column case
        if self.cols > col + 1: 
             self.board[row][col+1] = not self.board[row][col+1]
        if col - 1 >= 0: 
            self.board[row][col-1] = not self.board[row][col-1]
        # #try lower
        # try:
        #     self.board[row+1][col] = not self.board[row+1][col]
        # except: 
        #     pass
        # #try right
        # try:
        #     self.board[row][col+1] = not self.board[row][col+1] 
        # except: 
        #     pass
        # #try up
        # try:
        #     self.board[row-1][col] = not self.board[row-1][col]
        # except: 
        #     pass
        # #try left
        # try:
        #     self.board[row][col-1] = not self.board[row][col-1]
        # except: 
        #     pass



    def scramble(self):
        for i in range(self.rows):
            for y in range(self.cols):
                if random.random() > .5: 
                    self.perform_move(i,y)

    def is_solved(self):
        for x in range(self.rows):
            for y in range(self.cols):
                if self.board[x][y] == True: 
                    return False 
        return True 


    def copy(self):
        new_puzzle = copy.deepcopy(self.board)
        return_puzzle = LightsOutPuzzle(new_puzzle)
        return return_puzzle

    def successors(self):
        return_list = []
        for x in range(self.rows):
            for y in range(self.cols):
                b = self.copy()
                b.perform_move(x,y)
                yield b

    def find_solution(self):
        frontier = [(self.copy(), [])]
        visited = set()
        
        #BFS search
        while frontier:
                #pop the top of the frontier, gather the path and current puzzle
                current_step, path = frontier.pop(0)
                #if current puzzle done, return path - base case
                if current_step.is_solved(): 
                    return path 
                #in this puzzle, let's iterate across each bit and perform moves, seeing what happens
                for x in range(current_step.rows):
                    for j in range(current_step.cols):
                        i = current_step.copy()
                        i.perform_move(x,j)
                        tuple_of_tuples = tuple(tuple(row) for row in i.get_board())
                        if tuple_of_tuples not in visited: 
                            visited.add(tuple_of_tuples)
                            frontier.append((i, path + [(x,j)]))
        return None  


def create_puzzle(rows, cols):
    board = np.zeros((rows,cols), dtype = bool)
    return LightsOutPuzzle(board)

def solve_identical_disks(length, n):
    disks = tuple([1] * n + [0]* (length-n) )
    #board, track moves
    frontier = [(disks, [])]
    visited = set()

    while frontier: 
        disk_config, path = frontier.pop(0)
        goal = tuple([0] * (length - n) + [1]* (n))
        #base case
        if disk_config == goal: 
            return path
        for i in range(length):
            #valid moves
            #one hop this time
            if i + 1 < length and disk_config[i] == 1 and disk_config[i +1] == 0: 
                new_board = list(disk_config)
                new_board[i],new_board[i+1] = 0,1
                new_state = tuple(new_board)
                if new_state not in visited: 
                    visited.add(new_state)
                    #adds entire path
                    frontier.append((new_state, path + [(i, i+1)]))
            #two hop this time, slide to the right
            if i + 2 < length and disk_config[i] == 1 and disk_config[i+1] == 1 and disk_config[i + 2] == 0:
                new_board = list(disk_config)
                new_board[i],new_board[i+2] = 0,1
                new_state = tuple(new_board)
                if new_state not in visited: 
                    visited.add(new_state)
                    #adds entire path
                    frontier.append((new_state, path + [(i, i+2)]))
    return None



def solve_distinct_disks(length, n):
    disks = []
    for i in range(1,n+1):
        disks.append(i)
    for i in range(length-n):
        disks.append(0)
    disks = tuple(disks)
    frontier = [((disks, []))]
    print(frontier)
    visited = set()

    while frontier: 
        current_step, path = frontier.pop(0)
        #range(start, stop, step)
        goal = tuple([0]*(length - n) + list(range(n, 0, -1)))
        if current_step == goal: 
            return path
        for i in range(length):
            if i + 1 < length and current_step[i] >0  and current_step[i+1] == 0:
                new_disk = list(current_step)
                new_disk[i+1] = new_disk[i]
                new_disk[i] = 0
                new_state = tuple(new_disk)
                if new_state not in visited:
                    visited.add(new_state)
                    frontier.append((new_state, path + [(i, i+1)]) )
            if i + 2 < length and current_step[i] > 0 and current_step[i+1] != 0 and current_step[i + 2] == 0:
                new_board = list(current_step)
                new_board[i+2] = new_board[i]
                new_board[i] = 0
                new_state = tuple(new_board)
                if new_state not in visited: 
                    visited.add(new_state)
                    #adds entire path
                    frontier.append((new_state, path + [(i, i+2)]))
            if i -1 >= 0 and current_step[i] >0  and current_step[i-1] == 0:
                new_disk = list(current_step)
                new_disk[i-1] = new_disk[i]
                new_disk[i] = 0
                new_state = tuple(new_disk)
                if new_state not in visited:
                    visited.add(new_state)
                    frontier.append((new_state, path + [(i, i-1)]) )
            if i - 2 >= 0 and current_step[i] > 0 and current_step[i-1] != 0 and current_step[i - 2] == 0:
                new_board = list(current_step)
                new_board[i-2] = new_board[i]
                new_board[i] = 0
                new_state = tuple(new_board)
                if new_state not in visited: 
                    visited.add(new_state)
                    #adds entire path
                    frontier.append((new_state, path + [(i, i-2)]))    
    
    return None

p = create_puzzle(4,4)
p.perform_move(1,1)
print(p.get_board())

p.perform_move(1,1)
print(p.get_board())
print(p.is_solved())
p.scramble()
print(p.get_board())
print(p.successors())
print(p.find_solution())



# Give me hints about how to modify this basic dfs algorithm.

def dfs(graph, start_node, visited, target):
    visited.add(start_node)
    if start_node == target:
        return [start_node]
    for neighbor in visited: 
        result = dfs(graph, neighbor, visited, target)
        if result is True: 
            return [start_node] + neighbor
    return None 
