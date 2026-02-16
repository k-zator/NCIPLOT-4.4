from pathlib import Path
import numpy as np # type: ignore
from scipy.spatial import KDTree # type: ignore
from spatial.charge_aggregate import create_dimer_descriptors, aggregate_system_descriptors
bohr_to_angstrom = 0.52917721067

def group_grad_to_grid(coordinates, gradient, a, gradient_threshold):
    """
    Groups coordinates into a 3D grid of spacing 'a' and computes min gradient per cell.
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

def find_CP_with_gradient(matrix, threshold = 0.05, radius = 0.15, ispromol=True, mol=None):
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

    critical_points = filter_close_CPs(critical_points, min_distance=0.6)
    print(" Number of critical points found: ", len(critical_points))

    # Print CPs, their densities, gradients and neighboring atoms
    if ispromol:
        mol_coords = []; mol_names = []
        for m in mol:
            c, n = read_xyz(m)
            mol_coords.append(c)
            mol_names.append(n)
    else:  # WFN case - read from .xyz files generated from .wfn
        mol_coords = []; mol_names = []
        for m in mol:
            c, n = read_wfn(m)
            mol_coords.append(c)
            mol_names.append(n)
    print(" Densities at critical points: ")
    for i, cp in enumerate(critical_points):
        #iterate over molecules to find nearest atoms
        min_dist1 = float('inf')
        idx = -1
        for mol_idx, (mol_c, mol_n) in enumerate(zip(mol_coords, mol_names)):
            dists = np.linalg.norm(mol_c - cp[0]*bohr_to_angstrom, axis=1)
            local_min_idx = np.argmin(dists)
            #if dist is lower than before, find the name using the index
            if dists[local_min_idx] < min_dist1:
                name1 = mol_n[local_min_idx]
                min_dist1 = dists[local_min_idx]
                idx = mol_idx
        #find the second closest atom from another molecule
        min_dist2 = float('inf')
        for mol_idx, (mol_c, mol_n) in enumerate(zip(mol_coords, mol_names)):
            if mol_idx == idx:
                # i.e. for intramolecular interactions, but not the default use case
                # make sure we are not picking the same atom as for min_dist1
                dists = np.linalg.norm(mol_c - cp[0]*bohr_to_angstrom, axis=1)
#               # exclude the shortest distance found before
                dists[np.argmin(dists)] = float('inf')
                local_min_idx = np.argmin(dists)
                if dists[local_min_idx] < min_dist2:
                    name2 = mol_n[local_min_idx]
                    min_dist2 = dists[local_min_idx]
            else:
                # the default expected intermolecular case
                dists = np.linalg.norm(mol_c - cp[0]*bohr_to_angstrom, axis=1)
                local_min_idx = np.argmin(dists)
                if dists[local_min_idx] < min_dist2:
                    name2 = mol_n[local_min_idx]
                    min_dist2 = dists[local_min_idx]
        print(f"CP{i+1} Neighbours: {name1, name2}, Distances: {min_dist1:.4f}, {min_dist2:.4f}, Density: {cp[1]/100:.4f}")

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
            coords = np.array(coords) * bohr_to_angstrom
            f.write(f"CP{i}   {coords[0]:.6f}   {coords[1]:.6f}   {coords[2]:.6f}\n")

def filter_close_CPs(CPs, min_distance=0.6):
    """Remove CPs that are closer than min_distance to each other."""
    coords = np.array([cp[0] if isinstance(cp[0], (list, np.ndarray)) else cp for cp in CPs])
    keep = []
    for i, c in enumerate(coords):
        if all(np.linalg.norm(c - coords[j]) >= min_distance for j in keep):
            keep.append(i)
    return [CPs[i] for i in keep]

def read_xyz(filename, elem=False):
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]
        coords = []
        names = []
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 4:
                continue
            coords.append([float(x) for x in parts[1:4]])
            if elem:
                names.append(parts[0])
            else:
                names.append(parts[0]+str(i+1))
        return np.array(coords, dtype=float), names
    
def read_wfn(filename, elem=False):
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]
        coords = []
        names = []
        for line in lines:
            parts = line.split()
            if len(parts) > 6:
                if parts[2] == '(CENTRE':
                    coords.append([float(x)*bohr_to_angstrom for x in parts[4:7]])
                    if elem:
                        names.append(parts[1])
                    else:
                        names.append(parts[0] + parts[1])
        return np.array(coords, dtype=float), names

def find_CP_Atom_matches(CPs, mol1, mol2, ispromol):
    """Given the CPs' positions, find the atom-atom interactions they correspond to
        Returns a list of lists: [CP_position, mol1_atom_idx, mol2_atom_idx] """
        
    if ispromol:
        cp_coords = read_xyz(CPs)
        mol1_coords, _ = read_xyz(mol1)
        mol2_coords, _ = read_xyz(mol2)
    else:  # WFN case - read from .xyz files generated from .wfn
        cp_coords = read_xyz(CPs)
        mol1_coords = read_wfn(mol1)
        mol2_coords = read_wfn(mol2)

    matches = []
    for cp in cp_coords:
        dists1 = np.linalg.norm(mol1_coords - cp, axis=1)
        idx1 = np.argmin(dists1)
        dists2 = np.linalg.norm(mol2_coords - cp, axis=1)
        idx2 = np.argmin(dists2)
        matches.append([int(idx1), int(idx2)])

    return matches

def read_charges(mol1, mol2, ispromol):
    """
    Read (coords, elements, charges) for a system. Intended for xTB output where charge files contain one charge per atom.
    Geometry (coords/elements) is taken from `mol1` and `mol2`.
    Calculate the Coulomb energy descriptors for the system.

    Returns
    -------
    X ndarray of aggregated descriptor values for the system.
    """
    #parent directory of the molecule files, the charge files should be in the same directory
    folder_path = Path(mol1).parent
    # basename without extension to find the corresponding charge file
    charges1_path = folder_path / (Path(mol1).stem + "_charges.dat")
    charges2_path = folder_path / (Path(mol2).stem + "_charges.dat")

    # Parse geometry and elements for Coulomb energy and elemental encoding
    if ispromol:
        mol1_coords, elem1 = read_xyz(mol1, elem=True)
        mol2_coords, elem2 = read_xyz(mol2, elem=True)
    else:  # WFN case - read from .xyz files generated from .wfn
        mol1_coords, elem1 = read_wfn(mol1, elem=True)
        mol2_coords, elem2 = read_wfn(mol2, elem=True)

    def get_charge(charges_path, num_atoms):
        # Parse charges supporting the common xTB "charges" format, where the files only contain the single charge row
        charges: list[float] = []
        with charges_path.open("r") as f:
            for line in f:
                s = line.strip().split()
                if not s:
                    continue
                if len(s) == 1:  # only charge
                    charges.append(float(s[0]))

        charges_arr = np.asarray(charges, dtype=float)

        if charges_arr.shape[0] != num_atoms:
            raise ValueError(
                f"Charge count mismatch: got {charges_arr.shape[0]} charges but {num_atoms} atoms "
                f"from charge file: {charges_path})")
        return charges_arr

    charges1 = get_charge(charges1_path, mol1_coords.shape[0])
    charges2 = get_charge(charges2_path, mol2_coords.shape[0])
            
    pair_desc, _ = create_dimer_descriptors(
                mol1_coords, elem1, charges1,
                mol2_coords, elem2, charges2,
                include_bessel=True,
                n_bessel=6,
                r_cut=10,
                intermolecular_only=True,
                features="6key")
            
    X = aggregate_system_descriptors(pair_desc, aggregation="smst")

    return X