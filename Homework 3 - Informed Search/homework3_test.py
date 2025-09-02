import queue
from queue import PriorityQueue
import numpy as np
import random 

# for reference, this is an algorithm for greedy first search

def greedy_fist(graph, heuristic, start, goal)  -> None:
    """greedy algorithm that chooses the least expensive heuristic based off it's goals"""
    start = 'start'
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    came_from[start] = None



    while not frontier.empty(): 
        current = frontier.get()

        if current == goal: 
            break 

        for next in graph(current):
            if next not in came_from: 
                priority = heuristic(goal,next)
                frontier.put(next,priority)
                came_from[next] = current


def create_tile_puzzle(cols, rows): 
    ''''Creates tile puzzle with specified number of columns and rows'''
    puzzle = np.zeros((rows, cols), dtype = int)
    sum = 1
    for i in range(rows):
        for o in range(cols):
            if sum == rows * cols:
                print(puzzle.tolist())
                return TilePuzzle(puzzle.tolist())
            puzzle[i][o] = sum
            sum+=1
    return TilePuzzle(puzzle.tolist())
    



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
        if num_moves == 0:
            return
        my_list = ['up','down','left','right']
        random_choice = random.choice(my_list)
        self.perform_move(random_choice)
        self.scramble(num_moves-1)

    def is_solved(self):
        j = create_tile_puzzle(self.rows,self.cols)
        if self.board == j.board:
            return True
        else:
            return False

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


p = create_tile_puzzle(3,3)
print(p.is_solved())

p.perform_move('up')
print(p.get_board())
p.perform_move('left')
print(p.get_board())
p.perform_move('down')
print(p.get_board())
print(p.perform_move('down'))
print(p.get_board())
print(p.scramble(4))
print(p.get_board())
print(p.is_solved())