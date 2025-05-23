#! /usr/bin/env python3

import sys 
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from spatial.UTILS import process_cube, write_cube
from spatial.DIVIDE import find_CP_with_gradient
from spatial.INTEGRATE import integrate_NCI
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

# Print output
print(" # ----------------- NCICLUSTER ------------------------")
print(" # -----------------------------------------------------")
print(" Start -- {} \n".format(time.ctime()))

# Read input file
files = []
with open(input_name, "r") as f:
    for line in f:
        files.append(line[:-1])
filename = files[0] # unless there are indeed multiple - but how?

# Find critical points and distinguish the dimeric ones
# Mrhos, densarray, header, grid, densarray1, densarray2 = process_cube(filename)
Mrhos, densarray, header, grid = process_cube(filename)
CPs_both = find_CP_with_gradient(Mrhos, threshold, radius)
#Mrhos1 = Mrhos.copy()
#Mrhos1[:, 3] = densarray1.reshape(densarray1.size,)
#CPs_1 = find_CP_with_gradient(Mrhos1, threshold, radius)
#Mrhos2 = Mrhos.copy()
#Mrhos2[:, 3] = densarray2.reshape(densarray2.size,)
#CPs_2 = find_CP_with_gradient(Mrhos2, threshold, radius)
#CPs = get_unique_dimeric_CPs(CPs_both, CPs_1, CPs_2)
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit([i[0] for i in CPs_both])
# Find the nearest cluster center for each point in XYZ
_, indices = nn.kneighbors(Mrhos[:,:3])
labels = indices.flatten() # cluster labels cf. CPs
print(" NCICLUSTER found {} clusters".format(np.unique(labels)))
print(" # -----------------------------------------------------")
print("                RANGE INTEGRATION DATA      ")
# Now, select each cluster at a time
integrals = []
for i in np.unique(labels):
    cluster_grad = [g if labels[ig] == i else 101 for ig, g in enumerate(Mrhos[:,4])]
    cluster_grad = np.reshape(cluster_grad, grid)
    integrals.append(integrate_NCI(cluster_grad, densarray, grid, rhoparam=s, l_large=l_large, l_small=l_small))
    write_cube(filename, i, Mrhos, labels, header, grid)
    print (" Cluster {}".format(i))
    print (" Interval        :       -{}       -{}  ".format(opt_dict["outer"], opt_dict["inner"]))
    print (" n=1.0           : {:.8f}".format(integrals[-1][0][0]))
    print (" n=1.5           : {:.8f}".format(integrals[-1][0][1]))
    print (" n=2.0           : {:.8f}".format(integrals[-1][0][2]))
    print (" n=2.5           : {:.8f}".format(integrals[-1][0][3]))
    print (" n=3.0           : {:.8f}".format(integrals[-1][0][4]))
    print (" n=4/3           : {:.8f}".format(integrals[-1][0][5]))
    print (" n=5/3           : {:.8f}".format(integrals[-1][0][6]))
    print (" Volume          : {:.8f}".format(integrals[-1][0][7]))
    print (" Interval        :       -{}       {}  ".format(opt_dict["inner"], opt_dict["inner"]))
    print (" n=1.0           : {:.8f}".format(integrals[-1][1][0]))
    print (" n=1.5           : {:.8f}".format(integrals[-1][1][1]))
    print (" n=2.0           : {:.8f}".format(integrals[-1][1][2]))
    print (" n=2.5           : {:.8f}".format(integrals[-1][1][3]))
    print (" n=3.0           : {:.8f}".format(integrals[-1][1][4]))
    print (" n=4/3           : {:.8f}".format(integrals[-1][1][5]))
    print (" n=5/3           : {:.8f}".format(integrals[-1][1][6]))
    print (" Volume          : {:.8f}".format(integrals[-1][1][7]))
    print (" Interval        :       {}       {}  ".format(opt_dict["inner"], opt_dict["outer"]))
    print (" n=1.0           : {:.8f}".format(integrals[-1][2][0]))
    print (" n=1.5           : {:.8f}".format(integrals[-1][2][1]))
    print (" n=2.0           : {:.8f}".format(integrals[-1][2][2]))
    print (" n=2.5           : {:.8f}".format(integrals[-1][2][3]))
    print (" n=3.0           : {:.8f}".format(integrals[-1][2][4]))
    print (" n=4/3           : {:.8f}".format(integrals[-1][2][5]))
    print (" n=5/3           : {:.8f}".format(integrals[-1][2][6]))
    print (" Volume          : {:.8f}".format(integrals[-1][2][7]))
    print(" # -----------------------------------------------------")

print(" # -----------------------------------------------------")
print(" End -- {} \n".format(time.ctime()))