import math


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
        queens_helper(row + 1, placement, cols, diags1, diags2, n, results)
        cols.remove(col)
        diags1.remove(row - col)
        diags2.remove(row + col)


n_queens_solutions(8)
