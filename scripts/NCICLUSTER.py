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
l_large = opt_dict["outer"]
l_small = opt_dict["inner"]
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
CPs = find_CP_with_gradient(Mrhos, threshold, radius)
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
    integrals.append(integrate_NCI_cluster(cluster_grad, densarray, grid, dvol, labels, i, rhoparam=s, l_large=l_large, l_small=l_small))
    write_cube_select(filename, i, Mrhos, labels, header, grid)
    print("----------------------------------------------------------------------")   
    print(" Cluster {}".format(i)) # ADD ATOM_ATOM_CPS HERE
    print("----------------------------------------------------------------------")
    print(" Interval        :       -{:.8f}       -{:.8f}  ".format(opt_dict["outer"], opt_dict["inner"]))
    print("----------------------------------------------------------------------")
    print(" n=1.0           :        {:.8f}".format(integrals[-1][0][0]))
    print(" n=1.5           :        {:.8f}".format(integrals[-1][0][1]))
    print(" n=2.0           :        {:.8f}".format(integrals[-1][0][2]))
    print(" n=2.5           :        {:.8f}".format(integrals[-1][0][3]))
    print(" n=3.0           :        {:.8f}".format(integrals[-1][0][4]))
    print(" n=4/3           :        {:.8f}".format(integrals[-1][0][5]))
    print(" n=5/3           :        {:.8f}".format(integrals[-1][0][6]))
    print(" Volume          :        {:.8f}".format(integrals[-1][0][7]))
    print("----------------------------------------------------------------------")
    print(" Interval        :       -{:.8f}       {:.8f}  ".format(opt_dict["inner"], opt_dict["inner"]))
    print("----------------------------------------------------------------------")
    print(" n=1.0           :        {:.8f}".format(integrals[-1][1][0]))
    print(" n=1.5           :        {:.8f}".format(integrals[-1][1][1]))
    print(" n=2.0           :        {:.8f}".format(integrals[-1][1][2]))
    print(" n=2.5           :        {:.8f}".format(integrals[-1][1][3]))
    print(" n=3.0           :        {:.8f}".format(integrals[-1][1][4]))
    print(" n=4/3           :        {:.8f}".format(integrals[-1][1][5]))
    print(" n=5/3           :        {:.8f}".format(integrals[-1][1][6]))
    print(" Volume          :        {:.8f}".format(integrals[-1][1][7]))
    print("----------------------------------------------------------------------")
    print(" Interval        :       {:.8f}       {:.8f}  ".format(opt_dict["inner"], opt_dict["outer"]))
    print("----------------------------------------------------------------------")
    print(" n=1.0           :        {:.8f}".format(integrals[-1][2][0]))
    print(" n=1.5           :        {:.8f}".format(integrals[-1][2][1]))
    print(" n=2.0           :        {:.8f}".format(integrals[-1][2][2]))
    print(" n=2.5           :        {:.8f}".format(integrals[-1][2][3]))
    print(" n=3.0           :        {:.8f}".format(integrals[-1][2][4]))
    print(" n=4/3           :        {:.8f}".format(integrals[-1][2][5]))
    print(" n=5/3           :        {:.8f}".format(integrals[-1][2][6]))
    print(" Volume          :        {:.8f}".format(integrals[-1][2][7]))
print("----------------------------------------------------------------------")

write_vmd(filename, labels, opt_dict["isovalue"], verbose=opt_dict["verbose"])