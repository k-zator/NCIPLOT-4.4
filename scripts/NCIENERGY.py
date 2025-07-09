#! /usr/bin/env python3

import sys 
import time
import numpy as np
from spatial.OPT_DICT import options_energy_calc
from spatial.calculate_NCI_energy import calculate_energy_cluster, calculate_energy_single

"""
Calculate binding energy from the NCIPLOT analysis. Crucially, only calculate it when the correct parameters are set.
Otherwise, print a message saying it only works for the correct set.
Implemented for the NCICLUSTER results so that a single function could be used for any complex, 
including ones where sigma hole interactions are present. 
Parameters: path to nci_output_file. (afaik, it produces the values in the file in previous step so this is sort of recursive)
"""


# Set options from command line
options = []
if sys.argv[1]!="--help":
    input_name = sys.argv[1]
else:
    options.append(sys.argv[1])

if len(sys.argv)>2:
    options += sys.argv[2:]

opt_dict = options_energy_calc(options)#
gamma    = opt_dict["gamma"]
l_large  = opt_dict["outer"]
l_small  = opt_dict["inner"]
isovalue = opt_dict["isovalue"]
intermol = opt_dict["intermol"]
ispromol = opt_dict["ispromol"]
cluster  = opt_dict["cluster"]
if cluster:
    mol1 = opt_dict["mol1"]
    mol2 = opt_dict["mol2"]
    
# Read input file
files = []
with open(input_name, "r") as f:
    for line in f:
        files.append(line[:-1])
filename = files[0]

# The equation is rather parameter-dependent hence the all the parameters need to be set correctly
# to produce a reasonable energy estimate using the given equations
if isovalue == 1.0 and l_large == 0.2 and l_small == 0.02 and intermol == True:
    if (ispromol and gamma == 0.85) or (not ispromol and gamma == 0.75):
        pass
    else:
        print(" NCIENERGY mode runs only with the default parameters:")
        print(" DENS_CUTOFF 0.85 (0.75 for DFT)")
        print(" RDG_CUTOFF 1.00 0.30")
        print(" INTEGRATE")
        print(" RANGE")
        print(" -0.20 -0.02")
        print(" -0.02  0.02")
        print("  0.02  0.20")

    
    # obtain the contents of the nci_output file
    with open("nci_output_xyz_monomer.txt") as f: #need to know the NAME of output!!!
        contents = f.readlines()
    print("----------------------------------------------------------------------")
    print("                             NCIENERGY                                ")
    print("----------------------------------------------------------------------")
    if ispromol:
        print(" Calculating energy using the promolecular equation")
        if cluster:
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol, mol1, mol2, filename)
            for cluster_id, (e_sum, e_polar, e_vdw) in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f" Cluster {cluster_id} energies / kJ/mol")
                print(" E_sum   :        {:.8f}".format(e_sum))
                print(" E_polar :        {:.8f}".format(e_polar))
                print(" E_vdw   :        {:.8f}".format(e_vdw))
                print("----------------------------------------------------------------------")
            print("And summed-across-clusters integrals:")
            print(" E_sum   :        {:.8f}".format(E_sum.sum()))
            print(" E_polar :        {:.8f}".format(E_polar.sum()))
            print(" E_vdw   :        {:.8f}".format(E_vdw.sum()))
        else:
            print(" If your system contains sigma hole interactions, " \
            "consider using the clustering mode for better energy accuracy")
            print("----------------------------------------------------------------------")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol)
            print(f" NCI energies  / kJ/mol")
            print(" E_sum   :        {:.8f}".format(E_sum))
            print(" E_polar :        {:.8f}".format(E_polar))
            print(" E_vdw   :        {:.8f}".format(E_vdw))
            print("----------------------------------------------------------------------")

    else: # using WFN (vs. WFX?)
        print(" Calculating energy using the DFT equation")
        if cluster:
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol, mol1, mol2, filename)
            for cluster_id, (e_sum, e_polar, e_vdw) in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f" Cluster {cluster_id} energies  / kJ/mol")
                print(" E_sum   :        {:.8f}".format(e_sum))
                print(" E_polar :        {:.8f}".format(e_polar))
                print(" E_vdw   :        {:.8f}".format(e_vdw))
                print("----------------------------------------------------------------------")

        else:           
            print(" If your system contains sigma hole interactions, " \
            "consider using the clustering mode for better energy accuracy")
            print("----------------------------------------------------------------------")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol)
            print(f" NCI energies  / kJ/mol")
            print(" E_sum   :        {:.8f}".format(E_sum))
            print(" E_polar :        {:.8f}".format(E_polar))
            print(" E_vdw   :        {:.8f}".format(E_vdw))
            print("----------------------------------------------------------------------")

else:
    print(" NCIENERGY mode runs only with the default parameters:")
    print(" DENS_CUTOFF 0.85 (0.75 for DFT)")
    print(" RDG_CUTOFF 1.00 0.30")
    print(" INTEGRATE")
    print(" RANGE")
    print(" -0.20 -0.02")
    print(" -0.02  0.02")
    print("  0.02  0.20")
