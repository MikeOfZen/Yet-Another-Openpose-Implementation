import sys
sys.path.append("..")

import numpy as np
import numba

@numba.jit(nopython=True)
def mark_islands(input) ->(np.ndarray,dict):
    """This creates an array marking seperate truth islands in a truth array
    :param input:2D array bool array
    :return (islands array-of marked islands,head-island connections dict)"""
    assert len(input.shape)==2 ,"Must be 2D array"
    dim0 = input.shape[0]
    dim1 = input.shape[1]

    island = np.zeros(input.shape, dtype=np.uint8)
    island_num = 1
    head={}

    for x in range(dim0):
        for y in range(dim1):
            if input[y, x]:
                iy = y + 1 #to handle array edge case, island idxs are shifter by one
                ix = x + 1
                above = island[iy - 1, ix]
                left = island[iy, ix - 1]
                if not above and not left:
                    island[iy , ix]=island_num
                    head[island_num] = 0 #stays None if island is isoalted, or gets changed to master island later
                    island_num += 1
                elif above and not left:
                    island[iy, ix] = above
                elif not above and left:
                    island[iy, ix] = left
                elif above and left:
                    #make above child of left
                    island[iy, ix] = left
                    if above != left:
                        head[above]=left
    return island[1:,1:],head
