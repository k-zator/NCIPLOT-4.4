#! /usr/bin/env python3

import sys 
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from spatial.UTILS import process_cube, write_cube_select, write_vmd
from spatial.DIVIDE import find_CP_with_gradient, write_CPs_xyz, find_CP_Atom_matches
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

# Set options from command line
options = []
if sys.argv[1]!="--help":
    input_name = sys.argv[1]
else:
    options.append(sys.argv[1])

if len(sys.argv)>2:
    options += sys.argv[2:]

opt_dict = options_dict(options)
s = opt_dict["isovalue"]
threshold = 0.1 # what percentile of the s values could be minima
radius = 0.75 # radius for search of local minima definition

print("")
print("----------------------------------------------------------------------")
print("                             NCICLUSTER                               ")
print("----------------------------------------------------------------------")
print(" Start -- {} \n".format(time.ctime()))

# Read input file
files = []
with open(input_name, "r") as f:
    for line in f:
        files.append(line[:-1])
filename = files[0]

# Find critical points and distinguish the dimeric ones
# produces Mrhos: Matrix of rho, signed
Mrhos, densarray, header, grid, dvol = process_cube(filename)
CPs = find_CP_with_gradient(Mrhos, threshold, radius, ispromol=opt_dict["ispromol"], mol=opt_dict["mol"])
write_CPs_xyz(CPs, filename)

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit([i[0] for i in CPs])
# Find the nearest cluster center for each point in XYZ
_, indices = nn.kneighbors(Mrhos[:,:3])
labels = indices.flatten() # cluster labels cf. CPs

print("")
print("----------------------------------------------------------------------")
print("      RANGE CLUSTERED INTEGRATION DATA over the volumes of rho^n      ")


integrals = []
for i in np.unique(labels): # Now, select each cluster at a time
    #cluster_grad = [g if labels[ig] == i else 101 for ig, g in enumerate(Mrhos[:,4])]
    cluster_grad = np.reshape(Mrhos[:,4], grid)
    integrals.append(integrate_NCI_cluster(cluster_grad, densarray, grid, dvol, labels, i, rhoparam=s, rhorange=opt_dict["range"]))
    write_cube_select(filename, i, Mrhos, labels, header, grid)
    print("----------------------------------------------------------------------")   
    print(" Cluster {}".format(i+1)) # ADD ATOM_ATOM_CPS HERE
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
print("----------------------------------------------------------------------")

write_vmd(filename, labels, opt_dict["isovalue"], verbose=opt_dict["verbose"])