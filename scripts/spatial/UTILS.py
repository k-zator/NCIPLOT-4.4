import numpy as np
from multiprocessing import Pool

def process_cube(filename):
    """Read and format cubes"""
    # Load the density and gradient data
    header, denspts, densarray, _  = read_cube(f"{filename}-dens.cube")
    _, _, gradarray, _  = read_cube(f"{filename}-grad.cube")
    # Mrhos - matrix of density and gradient values
    nx, ny, nz = densarray.shape
    Mrhos = np.zeros((nx, ny, nz, 5))
    Mrhos[:, :, :, :3] = denspts
    Mrhos[:, :, :, 3] = densarray
    Mrhos[:, :, :, 4] = gradarray
    Mrhos = Mrhos.reshape(densarray.size, 5)
    # X_iso = Mrhos[Mrhos[:,4] <= s+1e-6] # just the isosurface
    grid = (nx, ny, nz)

    # and specifically monomeric densities
    # _, _, densarray1, _  = read_cube(f"{filename}-dens1.cube")
    # _, _, densarray2, _  = read_cube(f"{filename}-dens2.cube")
    return Mrhos, densarray, header, grid #, densarray1, densarray2

def read_cube(filename, verbose=False):
    """Reads in a cube file and returns grid information, atomic position, and a 3D array of cube values.
    Parameters
    ----------
    filename : str
        Cube file name.
    """
    if verbose:
        print("  Reading cube file: {}".format(filename))
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 2:
                gridinfo1 = line
                n_at = int(gridinfo1.split()[0])
                o1, o2, o3 = (
                    float(gridinfo1.split()[1]),
                    float(gridinfo1.split()[2]),
                    float(gridinfo1.split()[3]),
                )
            elif i == 3:
                gridinfo2 = line
                npx = int(gridinfo2.split()[0])
                incrx = float(gridinfo2.split()[1])
            elif i == 4:
                gridinfo3 = line
                npy = int(gridinfo3.split()[0])
                incry = float(gridinfo3.split()[2])
            elif i == 5:
                gridinfo4 = line
                npz = int(gridinfo4.split()[0])
                incrz = float(gridinfo4.split()[3])
            elif i > 5:
                break

    pts = np.zeros((npx, npy, npz, 3))
    idx = np.indices((npx, npy, npz))
    pts[:, :, :, 0] = o1 + idx[0] * incrx
    pts[:, :, :, 1] = o2 + idx[1] * incry
    pts[:, :, :, 2] = o3 + idx[2] * incrz
    coordinates = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i in range(6, 6 + n_at):
                coord = line
                coordinates.append(
                    coord.split()[0]
                    + ","
                    + coord.split()[2]
                    + ","
                    + coord.split()[3]
                    + ","
                    + coord.split()[4]
                )
            elif i > (6 + n_at):
                break
    if len(coordinates) == n_at:
        pass
    else:
        raise ValueError("There is a problem with the coordinates of the cube file!")

    lines = open(filename).readlines()
    cubeval = []
    for i in lines[n_at + 6 :]:
        for j in i.split():
            cubeval.append(j)
    cube_shaped = np.reshape(cubeval, (npx, npy, npz))
    carray = cube_shaped.astype(np.float64)
    header = []
    with open(filename, "r") as g:
        for i, line in enumerate(g):
            if i in range(0, 6 + n_at):
                header.append(line)

    atcoords = np.array([[int(coord.split(",")[0]), float(coord.split(",")[1]), float(coord.split(",")[2]), float(coord.split(",")[3])] for coord in coordinates])

    return header, pts, carray, atcoords

def write_cube(filename, cl, X, labels, header, grid, verbose=False, parallel=False):
    """Write cube files with highly optimized performance."""
    import numpy as np
    from multiprocessing import Pool
    import os
    # Pre-calculate array dimensions 
    header_str = ''.join(header)
    
    # Vectorized data preparation
    mask = (labels == cl)[:, np.newaxis]
    values = np.where(mask, X[:, [4, 3]], 100)

    def write_file(suffix, data):
        """Write a single cube file"""
        output_file = f"{filename}-cl{cl}-{suffix}.cube"
        
        # Reshape the flattened data back to 3D grid
        data_3d = data.reshape(grid)
        
        with open(output_file, 'w') as f:
            # Write header
            f.write(header_str)
            
            # Write data respecting grid dimensions
            values_per_line = 6
            for x in range(grid[0]):
                for y in range(grid[1]):
                    row = data_3d[x, y, :]  # Get full z-row
                    # Write row in chunks of 6
                    for i in range(0, len(row), values_per_line):
                        chunk = row[i:i + values_per_line]
                        line = "".join(f"{val:13.5E}" for val in chunk)
                        f.write(line + "\n")

    if parallel and len(values) > 1000000:  # Only use multiprocessing for large datasets
        with Pool() as pool:
            pool.starmap(write_file, 
                        [('grad', values[:, 0].flatten()), 
                         ('dens', values[:, 1].flatten())])
    else:
        write_file('grad', values[:, 0].flatten())
        write_file('dens', values[:, 1].flatten())

    if verbose:
        print(f"  Wrote cube files {filename}-cl{cl}-[grad/dens].cube")


def write_cube_select(filename, cl, X, labels, header, grid, verbose=False):
    """ Write simplified cube file corresponding to only a segment of space for each cluster."""

    orig_nx, orig_ny, orig_nz = grid
    coords = X[:, :3].reshape(orig_nx, orig_ny, orig_nz, 3)
    dens = X[:, 3].reshape(grid)
    grad = X[:, 4].reshape(grid)
    labels = labels.reshape(grid)

    # Find the bounds of our selection in grid coordinates
    mask = (labels == cl)
    x_indices, y_indices, z_indices = np.where(mask)
    
    # Get min/max indices for each dimension
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    
    # New grid size
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    nz = z_max - z_min + 1

    # Calculate grid spacing from original header
    spacing = float(header[3].split()[1])
    mins = coords[x_min, y_min, z_min]

    # Create a new header
    new_header = header[0]
    new_header += header[1]
    new_header += f"   1   {mins[0]:10.6f}  {mins[1]:10.6f}  {mins[2]:10.6f}\n"
    new_header += f"   {nx}  {spacing:10.6f}    0.000000    0.000000\n"
    new_header += f"   {ny}    0.000000  {spacing:10.6f}    0.000000\n"
    new_header += f"   {nz}    0.000000    0.000000  {spacing:10.6f}\n"
    new_header += f"   0   0.0 {mins[0]:10.6f} {mins[1]:10.6f} {mins[2]:10.6f}\n"

    # Fill the new grid with 100s
    selected_dens = np.full((nx, ny, nz), 100.0)
    selected_grad = np.full((nx, ny, nz), 100.0)

    # Place the selected values in the correct positions
    for xi, yi, zi in zip(x_indices, y_indices, z_indices):
        selected_dens[xi - x_min, yi - y_min, zi - z_min] = dens[xi, yi, zi]
        selected_grad[xi - x_min, yi - y_min, zi - z_min] = grad[xi, yi, zi]

    def write_file(suffix, values):
        output_file = f"{filename}-cl{cl}-{suffix}.cube"
        with open(output_file, 'w') as f:
            f.writelines(new_header)
            # Write in cube format: 6 values per line
            flat = values.flatten()
            for i in range(0, len(flat), 6):
                line = "".join(f"{val:12.6f}" for val in flat[i:i+6])
                f.write(line + "\n")

    write_file("dens", selected_dens)
    write_file("grad", selected_grad)
    if verbose:
        print(f"  Wrote cube files {filename}-cl{cl}-[grad/dens].cube")

def write_vmd(filename, labels, isovalue, verbose=False, c3=False):
    """ Write vmd script file for each cluster.
    
    Parameters
    ----------
    filename : str
         Common string in cube files name.
    labels : np.array
         One dimensional array with integers that label the data in X_iso into different clusters.
    """
    if verbose:
        print("  Writing vmd file {}...                 ".format(filename + ".vmd"), end="", flush=True)
    with open(filename + "_divided.vmd", "w") as f:
        f.write("#!/usr/local/bin/vmd \n")
        f.write("# Display settings \n")
        f.write("display projection   Orthographic \n")
        f.write("display nearclip set 0.000000 \n")

    with open(filename + "_divided.vmd", "a") as f:
        f.write("# load new molecule \n")
        f.write(
            "mol new "
            + filename
            + "-dens.cube type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all \n"
        )
        f.write("# \n")
        f.write("# representation of the atoms \n")
        f.write("mol delrep 0 top \n")
        f.write("mol representation CPK 1.000000 0.300000 118.000000 131.000000 \n")
        f.write("mol color Name \n")
        f.write("mol selection {all} \n")
        f.write("mol material Opaque \n")
        f.write("mol addrep top \n")


    for i_label, cl in enumerate(set(labels)):
        if i_label > 32:
            i_label = i_label - 32
        with open(filename + "_divided.vmd", "a") as f:
            f.write("# load new molecule \n")
            f.write(
                "mol new "
                + filename
                + "-cl"
                + str(cl)
                + "-dens.cube type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all \n"
            )
            f.write(
                "mol addfile "
                + filename
                + "-cl"
                + str(cl)
                + "-grad.cube type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all \n"
            )
            f.write("# \n")
            f.write("# add representation of the surface \n")
            f.write("mol representation Isosurface {:.5f} 1 0 0 1 1 \n".format(isovalue))
            if c3 == True:
                f.write("mol color ColorID {} \n".format(i_label))
            else:
                f.write("mol color Volume 0 \n")
            f.write("mol selection {all} \n")
            f.write("mol material Opaque \n")
            f.write("mol addrep top \n")
            f.write("mol selupdate 2 top 0 \n")
            f.write("mol colupdate 2 top 0 \n")
            f.write("mol scaleminmax top 1 -7.0000 7.0000 \n")
            f.write("mol smoothrep top 2 0 \n")
            f.write("mol drawframes top 2 {now} \n")
            f.write("color scale method BGR \n")
            f.write("set colorcmds {{color Name {C} gray}} \n")
    if verbose:
        print("done")
 