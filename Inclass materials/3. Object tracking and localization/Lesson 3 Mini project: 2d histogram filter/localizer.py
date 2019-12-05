import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []

    #
    # TODO - implement this in part 2
    #
    
    for r, row in enumerate(grid):
        new_r=[]
        for c, col in enumerate(row):
            val = 0
            if col==color:
                val = beliefs[r][c]*p_hit
            else:
                val = beliefs[r][c]*p_miss
            new_r.append(val)
        new_beliefs.append(new_r)
    return normalize(new_beliefs)

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_j = (j + dy + width) % width
            new_i = (i + dx + height) % height
#             pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell
    return blur(new_G, blurring)