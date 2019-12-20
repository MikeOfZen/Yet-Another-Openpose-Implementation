import sys
sys.path.append("..")

import numpy as np
import numba


@numba.njit
def mark_islands(input) ->(np.ndarray,dict):
    """This creates an array marking seperate truth islands in a truth array
    :param input:2D array bool array
    :return (islands array-of marked islands,island_hierarchy-islands connections dict)"""
    assert len(input.shape)==2 ,"Must be 2D array"
    dim0 = input.shape[0]
    dim1 = input.shape[1]
    shape=(dim0+1,dim1+1)
    islands = np.zeros(shape, dtype=np.uint16)
    island_num = 1
    island_hierarchy=numba.typed.Dict.empty(key_type=numba.types.uint16,value_type=numba.types.uint16)

    for y in range(dim0):
        for x in range(dim1):
            if input[y, x]:
                iy = y + 1 #to handle array edge case, islands idxs are shifter by one
                ix = x + 1
                above = islands[iy - 1, ix]
                left = islands[iy, ix - 1]
                if not above and not left:
                    islands[iy , ix]=island_num
                    island_hierarchy[island_num] = island_num #stays None if islands is isoalted, or gets changed to master islands later
                    island_num += 1
                elif above and not left:
                    islands[iy, ix] = above
                elif not above and left:
                    islands[iy, ix] = left
                elif above and left:
                    #make above child of left
                    islands[iy, ix] = above
                    if above != left:
                        island_hierarchy[left]=above
    return islands[1:,1:],island_hierarchy

@numba.njit
def sort_island_hierarchy(island_hierarchy):
    """converts the hierarchical island dict to a map to the top level island"""
    compact = numba.typed.Dict.empty(key_type=numba.types.uint16,value_type=numba.types.uint16)
    for child, parent in island_hierarchy.items():
        last_parent = child
        while parent != last_parent:
            last_parent = parent
            parent = island_hierarchy[parent]
        compact[child] = last_parent
    return compact

@numba.njit
def islands_max(heatmap, islands, island_hierarchy):
    """This returns the maximum value from values for each island from islands"""
    dim0=islands.shape[0]
    dim1 = islands.shape[1]

    #islands_max=np.zeros(LIMIT_NUM_ISLANDS, dtype=np.float32)-1
    islands_max=numba.typed.Dict.empty(key_type=numba.types.uint16,value_type=numba.types.float32)
    for island_num in set(island_hierarchy.values()):
        islands_max[island_num]=0 #init here, because thats the only way

    peaks={}
    for y in range(dim0):
        for x in range(dim1):
            if islands[y,x]:
                island_num=islands[y,x]
                top_island_num=island_hierarchy[island_num]
                if heatmap[y, x] > islands_max[top_island_num]: #
                    islands_max[top_island_num]=heatmap[y, x]
                    peaks[top_island_num]=(y,x)

    #translate to list view and remove island numbers
    peaks_l=[]
    islands_max_l=[]
    for k in peaks:
        peaks_l.append(peaks[k])
        islands_max_l.append(islands_max[k])

    return peaks_l,islands_max_l

@numba.njit
def find_peaks(heatmap,threshold):
    """This takes a 2D heatmap, and returns all peaks on discontinuous regions (islands) for which the heatmap is above the threshold"""
    truth_islands=heatmap>threshold #get which parts are above the threshold
    segemented_islands,island_hierarchy=mark_islands(truth_islands) #segement and label the discontinous regions, returns a island hierarchy dict.
    if not len(island_hierarchy): #in case nothing found
        return None,None
    sorted_island_hierarchy=sort_island_hierarchy(island_hierarchy) #flatten the island hierarchy to point to the top island label
    peaks,island_max=islands_max(heatmap,segemented_islands,sorted_island_hierarchy) #get the maximum peak location (and value) for each island
    return peaks,island_max


spec = [
        ('fields', numba.float32[:, :, :]),
        ('sum', numba.float32[:]),
        ('num_fields', numba.uint16),
        ]

@numba.jitclass(spec)
class LineVectorIntegral:
    def __init__(self, fields):
        self.fields = fields
        self.num_fields = self.fields.shape[-1]
        self.sum = np.zeros(self.num_fields, dtype=np.float32)

    def integrate(self, y, x):
        self.sum += self.fields[y, x, :]

    def integrate_line_high(self, y0, x0, y1, x1):
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = 2 * dx - dy
        x = x0
        for y in range(y0, y1 + 1):
            self.integrate(y, x)
            if D > 0:
                x = x + xi
                D = D - 2 * dy
            D = D + 2 * dx

    def integrate_line_low(self, y0, x0, y1, x1):
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = 2 * dy - dx
        y = y0
        for x in range(x0, x1 + 1):
            self.integrate(y, x)
            if D > 0:
                y = y + yi
                D = D - 2 * dx
            D = D + 2 * dy

    def integrate_line(self, start, end):
        y0 = start[0]
        x0 = start[1]
        y1 = end[0]
        x1 = end[1]
        self.sum[:] = 0.0
        if abs(y1 - y0) < abs(x1 - x0):
            if x0 > x1:
                self.integrate_line_low(y1, x1, y0, x0)
            else:
                self.integrate_line_low(y0, x0, y1, x1)
        else:
            if y0 > y1:
                self.integrate_line_high(y1, x1, y0, x0)
            else:
                self.integrate_line_high(y0, x0, y1, x1)
        return self.sum
