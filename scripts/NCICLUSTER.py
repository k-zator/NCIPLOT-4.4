#! /usr/bin/env python3

import sys
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from spatial.UTILS import process_cube, write_cube_select, write_vmd
from spatial.DIVIDE import find_CP_with_gradient, write_CPs_xyz
from spatial.INTEGRATE import integrate_NCI_cluster
from spatial.OPT_DICT import options_dict

"""
Find critical points (CPs) using the gradient information for a complex.
Then, using CPs, find the division of space into clusters.
Parameters: 
    path to dens and grad files,
    threshold: A small value to identify points where the gradient is near zero.
    radius: Search radius for neighboring points for maxima determination.
"""

def _parse_cli(argv):
    if not argv:
        print("Usage: NCICLUSTER.py input_names [OPTIONS]")
        return None, ["--help"]

    options = []
    if argv[0] != "--help":
        input_name = argv[0]
    else:
        input_name = None
        options.append("--help")

    if len(argv) > 1:
        options += argv[1:]

    return input_name, options


def main(argv=None):
    input_name, options = _parse_cli(sys.argv[1:] if argv is None else argv)
    opt_dict = options_dict(options)
    s = opt_dict["isovalue"]
    threshold = 0.1
    radius = 0.75

    if input_name is None:
        return 0

    print("")
    print("----------------------------------------------------------------------")
    print("                             NCICLUSTER                               ")
    print("----------------------------------------------------------------------")
    print(" Start -- {} \n".format(time.ctime()))

    files = []
    with open(input_name, "r") as f:
        for line in f:
            files.append(line[:-1])
    filename = files[0]

    Mrhos, densarray, header, grid, dvol = process_cube(filename)
    CPs = find_CP_with_gradient(Mrhos, threshold, radius, ispromol=opt_dict["ispromol"], mol=opt_dict["mol"])
    write_CPs_xyz(CPs, filename)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit([i[0] for i in CPs])
    _, indices = nn.kneighbors(Mrhos[:, :3])
    labels = indices.flatten()

    print("")
    print("----------------------------------------------------------------------")
    print("      RANGE CLUSTERED INTEGRATION DATA over the volumes of rho^n      ")

    integrals = []
    for i in np.unique(labels):
        cluster_grad = np.reshape(Mrhos[:, 4], grid)
        integrals.append(
            integrate_NCI_cluster(
                cluster_grad,
                densarray,
                grid,
                dvol,
                labels,
                i,
                rhoparam=s,
                rhorange=opt_dict["range"],
            )
        )
        #write_cube_select(filename, i, Mrhos, labels, header, grid)
        print("----------------------------------------------------------------------")
        print(" Cluster {}".format(i + 1))
        for r in range(len(opt_dict["range"])):
            print("----------------------------------------------------------------------")
            print(" Interval        :       {:.8f}       {:.8f}  ".format(opt_dict["range"][r][0], opt_dict["range"][r][1]))
            print("----------------------------------------------------------------------")
            print(" n=1.0           :        {:.8f}".format(integrals[-1][r][0]))
            print(" n=1.5           :        {:.8f}".format(integrals[-1][r][1]))
            print(" n=2.0           :        {:.8f}".format(integrals[-1][r][2]))
            print(" n=2.5           :        {:.8f}".format(integrals[-1][r][3]))
            print(" n=3.0           :        {:.8f}".format(integrals[-1][r][4]))
            print(" n=4/3           :        {:.8f}".format(integrals[-1][r][5]))
            print(" n=5/3           :        {:.8f}".format(integrals[-1][r][6]))
            print(" Volume          :        {:.8f}".format(integrals[-1][r][7]))

    if len(integrals) > 0:
        total_integrals = np.sum(np.array(integrals), axis=0)
        print("----------------------------------------------------------------------")
        print(" Summed-across-clusters integrals by range")
        for r in range(len(opt_dict["range"])):
            print("----------------------------------------------------------------------")
            print(" Interval        :       {:.8f}       {:.8f}  ".format(opt_dict["range"][r][0], opt_dict["range"][r][1]))
            print("----------------------------------------------------------------------")
            print(" n=1.0           :        {:.8f}".format(total_integrals[r][0]))
            print(" n=1.5           :        {:.8f}".format(total_integrals[r][1]))
            print(" n=2.0           :        {:.8f}".format(total_integrals[r][2]))
            print(" n=2.5           :        {:.8f}".format(total_integrals[r][3]))
            print(" n=3.0           :        {:.8f}".format(total_integrals[r][4]))
            print(" n=4/3           :        {:.8f}".format(total_integrals[r][5]))
            print(" n=5/3           :        {:.8f}".format(total_integrals[r][6]))
            print(" Volume          :        {:.8f}".format(total_integrals[r][7]))
    print("----------------------------------------------------------------------")

    #write_vmd(filename, labels, opt_dict["isovalue"], verbose=opt_dict["verbose"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())