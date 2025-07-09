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
    Returns indices of possible minima points.
    """
    # Vectorized grid cell assignment
    cell_indices = (coordinates // a).astype(int)
    unique_cells, inverse_indices = np.unique(cell_indices, axis=0, return_inverse=True)
    
    # Pre-filter gradients to avoid processing unnecessary points
    mask = gradient < gradient_threshold
    
    # Find minimum gradient per cell using reduceat
    sorted_indices = np.argsort(inverse_indices)
    sorted_gradients = gradient[sorted_indices]
    sorted_mask = mask[sorted_indices]
    
    # Get cell boundaries
    cell_boundaries = np.bincount(inverse_indices)
    cell_boundaries = np.cumsum(cell_boundaries[:-1])
    
    # Find minimum gradients per cell
    min_gradients = np.minimum.reduceat(sorted_gradients, np.r_[0, cell_boundaries])
    
    # Split sorted mask into cell groups and check if any point in each cell satisfies the mask
    cell_masks = np.split(sorted_mask, cell_boundaries)
    mask_any = np.array([np.any(m) for m in cell_masks])
    
    # Select cells with gradients below threshold
    valid_cells = (min_gradients < gradient_threshold) & mask_any
    
    # Get indices of points with minimum gradient in valid cells
    possible_minima = []
    for cell_idx in np.where(valid_cells)[0]:
        start = 0 if cell_idx == 0 else cell_boundaries[cell_idx - 1]
        end = cell_boundaries[cell_idx] if cell_idx < len(cell_boundaries) else len(sorted_gradients)
        cell_min_idx = start + np.argmin(sorted_gradients[start:end])
        possible_minima.append(sorted_indices[cell_min_idx])
    
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
    # Create KDTree and query neighbours for minima search in proximity 
    tree = KDTree(coordinates)
    neighbors_idx = [tree.query_ball_point(coordinates[i], r=radius) for i in possible_minima]
    
    # Process each point efficiently
    for idx, point_idx in enumerate(possible_minima):
        local_grad = gradient[neighbors_idx[idx]]
        if gradient[point_idx] <= np.min(local_grad):  # Using <= instead of == for numerical stability
            critical_points.append([coordinates[point_idx], density[point_idx], gradient[point_idx]])
            
    print(" Number of critical points found: ", len(critical_points))
    return critical_points

def get_unique_dimeric_CPs(CP_dimer, CP_monomer_1, CP_monomer_2):
    """Detemined which CPs are due to the intermolecular interaction is INTERMOLECULAR is not set"""

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

def write_CPs_xyz(CPs, filename):
    """Write critical points to an XYZ format with labels CP1, CP2, ... """
    with open(f"{filename}_CPs.xyz", "w") as f:
        f.write(str(len(CPs))+"\n")
        f.write("Critical points for the complex NCI clustering of interactions\n")
        for i, cp in enumerate(CPs, 1):
            coords = cp[0] if isinstance(cp[0], (list, np.ndarray)) else cp
            f.write(f"CP{i}   {coords[0]:.6f}   {coords[1]:.6f}   {coords[2]:.6f}\n")


def find_CP_Atom_matches(CPs, mol1, mol2):
    """Given the CPs' positions, find the atom-atom interactions they correspond to
        Returns a list of lists: [CP_position, mol1_atom_idx, mol2_atom_idx] """
    
    def read_xyz(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()[2:]
            coords = []
            for line in lines:
                parts = line.split()
                if len(parts) < 4:
                    continue
                coords.append([float(x) for x in parts[1:4]])
            return np.array(coords, dtype=float)
    
    cp_coords = read_xyz(CPs)
    mol1_coords = read_xyz(mol1)
    mol2_coords = read_xyz(mol2)

    matches = []
    for cp in cp_coords:
        dists1 = np.linalg.norm(mol1_coords - cp, axis=1)
        idx1 = np.argmin(dists1)
        dists2 = np.linalg.norm(mol2_coords - cp, axis=1)
        idx2 = np.argmin(dists2)
        matches.append([int(idx1), int(idx2)])

    return matches
