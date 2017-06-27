from utils import *


def grid_values(grid):
    """Convert grid string into {<box>: <value>} dict with '.' value for empties.

    Args:
        grid: Sudoku grid in string form, 81 characters long
    Returns:
        Sudoku grid in dictionary form:
        - keys: Box labels, e.g. 'A1'
        - values: Value in corresponding box, e.g. '8', or '.' if it is empty.
    """
    values_dict = {}
    for idx,item in enumerate(grid):
        row_coord = idx // 9
        col_coord = idx % 9
        if item != '.':
            values_dict[rows[row_coord] + cols[col_coord]] = item
        else:
            values_dict[rows[row_coord] + cols[col_coord]] = '123456789'
    return values_dict


def grid_values_old(grid):
    """Convert grid string into {<box>: <value>} dict with '.' value for empties.

    Args:
        grid: Sudoku grid in string form, 81 characters long
    Returns:
        Sudoku grid in dictionary form:
        - keys: Box labels, e.g. 'A1'
        - values: Value in corresponding box, e.g. '8', or '.' if it is empty.
    """
    values_dict = {}
    for idx,item in enumerate(grid):
        row_coord = idx // 9
        col_coord = idx % 9
        values_dict[rows[row_coord] + cols[col_coord]] = item

    return values_dict


def eliminate(values):
    """Eliminate values from peers of each box with a single value.

    Go through all the boxes, and whenever there is a box with a single value,
    eliminate this value from the set of values of all its peers.

    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    """
    for box in values:
        if len(values[box]) == 1:
            peer_vals = peers[box]
            for e in peer_vals:
                values[e] = values[e].replace(values[box], '')
    return values

def only_choice(values):
    """Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values


def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        eliminate(values)
        # Your code here: Use the Only Choice Strategy
        only_choice(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Using depth-first search and propagation, create a search tree and solve the sudoku."""
    # First, reduce the puzzle using the previous function
    grid_result = reduce_puzzle(values)
    if not grid_result:
        return False
    # Choose one of the unfilled squares with the fewest possibilities
    lengths = {len(grid_result[box]): box for box in grid_result}
    if all([t == 1 for t in lengths.keys()]):  # we're done
        return grid_result

    # remove any entries for boxes that are solved to prevent infinite recursion
    lengths.pop(1, None)
    start_point = lengths[min(lengths.keys())]
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for i in range(len(grid_result[start_point])):
        # copy the original
        new = {k: v for k, v in grid_result.items()}
        # write in new value
        new[start_point] = grid_result[start_point][i]
        # recursively call search
        res = search(new)
        if res:
            return res
    # search failed
    return False



if __name__ == "__main__":
    grid = grid_values('..3.2.6..9..3.5..1..18..4....81.29..7.......8..67.82....26..5..8..2.3..9..5.1.3..')
    display(grid)
    #grid2 = eliminate(grid)
    #grid3 = only_choice(grid2)
    result = search(grid)
    if result != False:
        display(result)
