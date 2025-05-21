
import numpy as np

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

def write_cube(filename, cl, X, labels, header, grid, verbose=False):
    """ Write cube file for each cluster.
    Parameters
    ----------
    filename : str
         Common string in cube files name.
    X : np.array
         Array with columns corresponding to space coordinates, sign(l2)*dens and rdg; for all data.
    labels : np.array
         One dimensional array with integers that label the data in X_iso into different clusters.
    header : list of str
         Original cube file header.
    """
    grad = np.array([g if labels[ig] == cl else 101 for ig, g in enumerate(X[:,4])])
    dens = np.array([g if labels[ig] == cl else 101 for ig, g in enumerate(X[:,3])])

    if verbose:
        print("  Writing cube file {}...      ".format(filename + "-cl" + str(cl) + "-grad.cube"),
            end="", flush=True)
    with open(f"{filename}" + "-cl" + str(cl) + "-grad.cube", "w") as f_out:
        for line in header:
            f_out.write(line)

        grad_values = grad.reshape(-1, grid[2])  # Reshape to extract C dimension
        for row in grad_values: # Print in rows of up to 6 values 
            for i in range(0, len(row), 6):
                f_out.write("".join("{:13.5E}".format(item) for item in row[i:i+6]))
                f_out.write("\n")
    if verbose:
        print("  Writing cube file {}...      ".format(filename + "-cl" + str(cl) + "-dens.cube"),
            end="", flush=True)
    with open(f"{filename}" + "-cl" + str(cl) + "-dens.cube", "w") as f_out:
        for line in header:
            f_out.write(line)
        dens_values = dens.reshape(-1, grid[2])  # Reshape to extract C dimension
        for row in dens_values:
            for i in range(0, len(row), 6):
                f_out.write("".join("{:13.5E}".format(item) for item in row[i:i+6]))
                f_out.write("\n")
    return