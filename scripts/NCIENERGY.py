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

opt_dict = options_energy_calc(options)
gamma    = opt_dict["gamma"]
l_large  = opt_dict["outer"]
l_small  = opt_dict["inner"]
isovalue = opt_dict["isovalue"]
intermol = opt_dict["intermol"]
ispromol = opt_dict["ispromol"]
cluster  = opt_dict["clustering"]

# The equation is rather parameter-dependent hence the all the parameters need to be set correctly
# to produce a reasonable energy estimate using the given equations
if isovalue == 1.0 and gamma == 0.85 and l_large == 0.2 and l_small == 0.02 and intermol == True:
    
    # obtain the contents of the nci_output file
    with open("nci_output.txt") as f:
        contents = f.readlines()

    if ispromol:
        print("Calculating energy using the promolecular equation")
        if cluster:
            print("Warning, this does not yet work with the integration...")
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol)
            for cluster_id, e_sum, e_polar, e_vdw in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f"Cluster {cluster_id} energies")
                print(" E_sum   :        {:.8f}".format(e_sum))
                print(" E_polar :        {:.8f}".format(e_polar))
                print(" E_vdw   :        {:.8f}".format(e_vdw))

        else:
            print("If your system contains sigma hole interactions, " \
            "consider using the clustering mode for better energy accuracy")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol)
            print(f"NCI volume energies")
            print(" E_sum   :        {:.8f}".format(E_sum))
            print(" E_polar :        {:.8f}".format(E_polar))
            print(" E_vdw   :        {:.8f}".format(E_vdw))

    else: # using WFN (vs. WFX?)
        print("Calculating energy using the DFT equation")
        if cluster:
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol)
            for cluster_id, e_sum, e_polar, e_vdw in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f"Cluster {cluster_id} energies")
                print(" E_sum   :        {:.8f}".format(e_sum))
                print(" E_polar :        {:.8f}".format(e_polar))
                print(" E_vdw   :        {:.8f}".format(e_vdw))

        else:           
            print("If your system contains sigma hole interactions, " \
            "consider using the clustering mode for better energy accuracy")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol)
            print(f"NCI volume energies")
            print(" E_sum   :        {:.8f}".format(E_sum))
            print(" E_polar :        {:.8f}".format(E_polar))
            print(" E_vdw   :        {:.8f}".format(E_vdw))

else:
    print(" NCIENERGY mode runs only with the default parameters:")
    print(" INTERCUT 0.85 0.75")
    print(" CUTOFFS 0.50 1.00")
    print(" INTEGRATE")
    print(" RANGE")
    print(" -0.20 -0.02")
    print(" -0.02  0.02")
    print("  0.02  0.20")
