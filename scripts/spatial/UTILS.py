import numpy as np
from multiprocessing import Pool


def apply_cube_boundary_sentinels(densarray, gradarray, dens_sentinel=100.0, grad_sentinel=101.0):
    """Exclude the outermost cube shell from downstream CP and integration logic."""
    densarray = densarray.copy()
    gradarray = gradarray.copy()

    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = 0
        high[axis] = -1
        densarray[tuple(low)] = dens_sentinel
        densarray[tuple(high)] = dens_sentinel
        gradarray[tuple(low)] = grad_sentinel
        gradarray[tuple(high)] = grad_sentinel

    return densarray, gradarray

def process_cube(filename):
    """Read and format cubes"""
    # Load the density and gradient data
    header, denspts, densarray, _  = read_cube(f"{filename}-dens.cube")
    _, _, gradarray, _  = read_cube(f"{filename}-grad.cube")
    densarray, gradarray = apply_cube_boundary_sentinels(densarray, gradarray)
    # Mrhos - matrix of density and gradient values
    nx, ny, nz = densarray.shape
    Mrhos = np.zeros((nx, ny, nz, 5))
    Mrhos[:, :, :, :3] = denspts
    Mrhos[:, :, :, 3] = densarray*100 # convert to same units as in main nci program
    Mrhos[:, :, :, 4] = gradarray
    Mrhos = Mrhos.reshape(densarray.size, 5)

    # X_iso = Mrhos[Mrhos[:,4] <= s+1e-6] # just the isosurface
    grid = (nx, ny, nz)
    # and voxel for integration normalisation
    dvol = (float(header[3].split()[1]))*(float(header[4].split()[2]))*(float(header[5].split()[3]))

    # and specifically monomeric densities
    # _, _, densarray1, _  = read_cube(f"{filename}-dens1.cube")
    # _, _, densarray2, _  = read_cube(f"{filename}-dens2.cube")
    return Mrhos, densarray*100, header, grid, dvol #, densarray1, densarray2

def read_cube(filename, verbose=False):
    """Reads in a cube file and returns grid information, atomic position, and a 3D array of cube values.
    Parameters
    ----------
    filename : str
        Cube file name.
    """
    if verbose:
        print("  Reading cube file: {}".format(filename))
    with open(filename, "r") as handle:
        lines = handle.readlines()

    if len(lines) < 6:
        raise ValueError("Cube file is missing header lines")

    origin_info = lines[2].split()
    n_at = int(origin_info[0])
    o1, o2, o3 = (float(origin_info[1]), float(origin_info[2]), float(origin_info[3]))

    x_info = lines[3].split()
    y_info = lines[4].split()
    z_info = lines[5].split()
    npx = int(x_info[0])
    npy = int(y_info[0])
    npz = int(z_info[0])
    incrx = float(x_info[1])
    incry = float(y_info[2])
    incrz = float(z_info[3])

    atom_lines = lines[6 : 6 + n_at]
    if len(atom_lines) != n_at:
        raise ValueError("There is a problem with the coordinates of the cube file!")

    header = lines[: 6 + n_at]
    values = np.fromstring("".join(lines[6 + n_at :]), sep=" ", dtype=np.float64)
    expected_values = npx * npy * npz
    if values.size != expected_values:
        raise ValueError("Cube file does not contain the expected number of values")
    carray = values.reshape((npx, npy, npz))

    pts = np.empty((npx, npy, npz, 3), dtype=np.float64)
    pts[:, :, :, 0] = o1 + np.arange(npx, dtype=np.float64)[:, None, None] * incrx
    pts[:, :, :, 1] = o2 + np.arange(npy, dtype=np.float64)[None, :, None] * incry
    pts[:, :, :, 2] = o3 + np.arange(npz, dtype=np.float64)[None, None, :] * incrz

    atcoords = np.array(
        [
            [int(parts[0]), float(parts[2]), float(parts[3]), float(parts[4])]
            for parts in (line.split() for line in atom_lines)
        ]
    )

    return header, pts, carray, atcoords


def _write_cube_values(handle, flat_values, fmt):
    flat_values = np.asarray(flat_values, dtype=np.float64).reshape(-1)
    full_row_count, remainder = divmod(flat_values.size, 6)

    if full_row_count:
        np.savetxt(
            handle,
            flat_values[: full_row_count * 6].reshape(full_row_count, 6),
            fmt=fmt,
            delimiter="",
        )

    if remainder:
        handle.write("".join(fmt % value for value in flat_values[full_row_count * 6 :]) + "\n")


def _write_cube_file(output_file, header_str, flat_values, fmt):
    with open(output_file, "w") as handle:
        handle.write(header_str)
        _write_cube_values(handle, flat_values, fmt)

def write_cube(filename, cl, X, labels, header, grid, verbose=False, parallel=False):
    """Write cube files with highly optimized performance."""
    header_str = "".join(header)

    mask = (labels == cl)[:, np.newaxis]
    values = np.where(mask, X[:, [4, 3]], 100)

    file_specs = [
        (f"{filename}-cl{cl+1}-grad.cube", values[:, 0], "%13.5E"),
        (f"{filename}-cl{cl+1}-dens.cube", values[:, 1] / 100.0, "%13.5E"),
    ]

    if parallel and len(values) > 1000000:
        with Pool() as pool:
            pool.starmap(
                _write_cube_file,
                [(output_file, header_str, flat_values, fmt) for output_file, flat_values, fmt in file_specs],
            )
    else:
        for output_file, flat_values, fmt in file_specs:
            _write_cube_file(output_file, header_str, flat_values, fmt)

    if verbose:
        print(f"  Wrote cube files {filename}-cl{cl+1}-[grad/dens].cube")

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

    local_mask = mask[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    dens_block = dens[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    grad_block = grad[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]

    selected_dens = np.where(local_mask, dens_block, 100.0)
    selected_grad = np.where(local_mask, grad_block, 100.0)

    header_str = new_header
    _write_cube_file(f"{filename}-cl{cl+1}-dens.cube", header_str, selected_dens / 100.0, "%13.5E")
    _write_cube_file(f"{filename}-cl{cl+1}-grad.cube", header_str, selected_grad, "%13.5E")
    if verbose:
        print(f"  Wrote cube files {filename}-cl{cl+1}-[grad/dens].cube")

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
 