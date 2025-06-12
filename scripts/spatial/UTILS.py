import numpy as np
from multiprocessing import Pool
import mmap

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

def write_cube(filename, cl, X, labels, header, grid, verbose=False, parallel=True):
    """Write cube files with highly optimized performance."""
    import numpy as np
    from multiprocessing import Pool
    import os

    # Pre-calculate array dimensions 
    header_str = ''.join(header)
    
    # Vectorized data preparation
    mask = (labels == cl)[:, np.newaxis]
    values = np.where(mask, X[:, [4, 3]], 101)
    values = values.reshape(-1, 2, grid[2])

    def format_data(data):
        """Format data into rows of 6 values with scientific notation"""
        formatted_lines = []
        line = []
        for val in data:
            line.append(f"{val:13.5E}")
            if len(line) == 6:
                formatted_lines.append(" ".join(line))
                line = []
        if line:  # Handle any remaining values
            formatted_lines.append(" ".join(line))
        return "\n".join(formatted_lines) + "\n"

    def write_file(suffix, data):
        """Write a single cube file"""
        output_file = f"{filename}-cl{cl}-{suffix}.cube"
        
        with open(output_file, 'w') as f:
            # Write header
            f.write(header_str)
            
            # Write data in chunks
            chunk_size = 6000  # Process 1000 lines at a time
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                formatted = format_data(chunk)
                f.write(formatted)

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