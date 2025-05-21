import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree

def group_grad_to_grid(coordinates, gradient, a, gradient_threshold):
    """
    Groups coordinates into a 3D grid of spacing `a` and computes min gradient per cell.
    Parameters:
    coordinates : 3N array of (x, y, z) coordinates.
    gradient : 1N array of gradient values associated with each point.
    a : Grid spacing.
    Returns Grid dictionary with density per cell.
    """
    grid = defaultdict(list)
    # Assign points to grid cells
    for i in range(len(coordinates)):
        cell_index = tuple((coordinates[i] // a).astype(int))
        grid[cell_index].append([i, gradient[i]]) #so that each grid cell has the necessary grad information

    # Compute minimum density per cell and use it to select which ones are worth pursuing further
    # i.e. the minimum density in the cell is below the threshold
    possible_minima = []
    for _, values in grid.items():
        min_gradient = min(values, key=lambda x: x[1])
        if min_gradient[1] < 100:
            if min_gradient[1] < gradient_threshold:
                possible_minima.append(min_gradient[0])
    return possible_minima

def find_CP_with_gradient(matrix, threshold = 0.05, radius = 0.15):
    """
    Find critical points using the gradient information in the matrix.

    Parameters:
        matrix: A 2D numpy array where columns represent x, y, z, density, and gradient.
        threshold: A small value to identify points where the gradient is near zero.
        radius: Search radius for neighboring points.

    Returns:
        List of critical points with their coordinates and density values.
    """
    coordinates = matrix[:, :3]
    density = matrix[:, 3]
    gradient = matrix[:, 4]
    # gradient threshold relative to the median of the gradient
    # This is to avoid being too sensitive to noise in the gradient
    gradient_threshold = np.percentile(gradient, threshold)
    print(" Gradient threshold: ", gradient_threshold)
    possible_minima = group_grad_to_grid(coordinates, gradient, 1, gradient_threshold)

    critical_points = []
    for i in possible_minima:
        # Identify local density
        tree = KDTree(coordinates)
        neighbors_idx = tree.query_ball_point(coordinates[i], r=radius)
        local_grad = gradient[neighbors_idx]
        if gradient[i] == min(local_grad):
            critical_points.append([coordinates[i], density[i], gradient[i]])
    print( " Number of critical points found: ", len(critical_points))
    return critical_points

def get_unique_dimeric_CPs(CP_dimer, CP_monomer_1, CP_monomer_2):
    UC = []
    C = np.array([i[0] for i in CP_dimer])
    C_1 = np.array([i[0] for i in CP_monomer_1])
    C_2 = np.array([i[0] for i in CP_monomer_2])
    for c in C:
        dist_1 = np.linalg.norm(c - C_1, axis=1)
        dist_2 = np.linalg.norm(c - C_2, axis=1)
        if sum(dist_1 < 0.1) + sum(dist_2 < 0.1) == 0:
            UC.append(c)
    return np.array(UC)

