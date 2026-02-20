#! /usr/bin/env python3

import sys 
from spatial.OPT_DICT import options_energy_calc
from spatial.calculate_NCI_energy import (
    calculate_energy_cluster,
    calculate_energy_single,
    calculate_charge_correction,
    calculate_charge_correction_cluster,
)

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
oname    = opt_dict["oname"]
gamma    = opt_dict["gamma"]
l_large  = opt_dict["outer"]
l_small  = opt_dict["inner"]
isovalue = opt_dict["isovalue"]
intermol = opt_dict["intermol"]
ispromol = opt_dict["ispromol"]
cluster  = opt_dict["cluster"]
supra    = opt_dict["supra"]
mol1 = opt_dict["mol1"]
mol2 = opt_dict["mol2"]
total_charges = opt_dict["total_charges"]
use_charges = opt_dict["use_charges"]
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
        print(" RDG_CUTOFF 1.00 0.30")
        print(" INTEGRATE")
        print(" INTERMOL_CUTOFF 0.85 (0.75 for DFT)")
        print(" RANGE 3")
        print(" -0.20 -0.02")
        print(" -0.02  0.02")
        print("  0.02  0.20")

    # obtain the contents of the nci_output file
    try:
        with open(f"nci_{oname}.out") as f: # need to know the NAME of output which it techincally has no way of finding out!!!
            contents = f.readlines()
    except FileNotFoundError:
        print(f"Error: nci_{oname}.out not found. Please ensure that your NCIPLOT output file is named 'nci_{oname}.out'.")
        sys.exit(1)
    print("----------------------------------------------------------------------")
    print("                             NCIENERGY                                ")
    print("----------------------------------------------------------------------")
    if ispromol:
        print(" Calculating energy using the promolecular equations")
        use_charge_model = use_charges
        if cluster:
            if use_charge_model:
                E_charge_correction, E_charge_clusters = calculate_charge_correction_cluster(
                    mol1,
                    mol2,
                    ispromol,
                    total_charges,
                    filename + "_CPs.xyz",
                    cutoff=1.0,
                    supra=supra,
                    return_shapley=True,
                )
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol, supra, mol1, mol2, filename)
            for cluster_id, (e_sum, e_polar, e_vdw) in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f" Cluster {cluster_id} energies / kJ/mol")
                if use_charge_model and cluster_id < len(E_charge_clusters):
                    print(" E_sum    :        {:.2f}".format(e_sum + E_charge_clusters[cluster_id]))
                else:
                    print(" E_sum    :        {:.2f}".format(e_sum))
                print(" E_polar  :        {:.2f}".format(e_polar))
                print(" E_vdw    :        {:.2f}".format(e_vdw))
                if use_charge_model and cluster_id < len(E_charge_clusters):
                    print(" E_charge :        {:.2f}".format(E_charge_clusters[cluster_id]))
                elif use_charge_model:
                    print(" E_charge :        n/a")
                print("----------------------------------------------------------------------")
            if use_charge_model and len(E_charge_clusters) != len(E_sum):
                print(" Warning: cluster count mismatch between NCI integration and charge attribution vectors.")
            print(" Summed-across-clusters integrals / kJ/mol")
            if use_charge_model:
                print(" E_sum    :        {:.2f}".format(E_sum.sum()+ E_charge_correction))
            else:
                print(" E_sum    :        {:.2f}".format(E_sum.sum()))
            print(" E_polar  :        {:.2f}".format(E_polar.sum()))
            print(" E_vdw    :        {:.2f}".format(E_vdw.sum()))
            if use_charge_model:
                print(" E_charge :        {:.2f}".format(E_charge_correction))
            print("----------------------------------------------------------------------")
        else:
            if use_charge_model:
                E_charge_correction = calculate_charge_correction(mol1, mol2, ispromol, total_charges, supra=supra)
            print(" If your system contains sigma hole interactions, " \
            "consider using the clustering mode for better energy accuracy")
            print("----------------------------------------------------------------------")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol, supra)
            print(f" NCI energies / kJ/mol")
            if use_charge_model:
                print(" E_sum    :        {:.2f}".format(E_sum + E_charge_correction))
            else:
                print(" E_sum    :        {:.2f}".format(E_sum))
            print(" E_polar  :        {:.2f}".format(E_polar))
            print(" E_vdw    :        {:.2f}".format(E_vdw))
            if use_charge_model:
                print(" E_charge :        {:.2f}".format(E_charge_correction))
            print("----------------------------------------------------------------------")

    else: # using WFN (vs. WFX?)
        print(" Calculating energy using the DFT equations")
        if cluster:
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol, supra, mol1, mol2, filename)
            for cluster_id, (e_sum, e_polar, e_vdw) in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f" Cluster {cluster_id} energies / kJ/mol")
                print(" E_sum   :        {:.2f}".format(e_sum))
                print(" E_polar :        {:.2f}".format(e_polar))
                print(" E_vdw   :        {:.2f}".format(e_vdw))
                print("----------------------------------------------------------------------")
            print(" Summed-across-clusters integrals / kJ/mol")
            print(" E_sum   :        {:.2f}".format(E_sum.sum()))
            print(" E_polar :        {:.2f}".format(E_polar.sum()))
            print(" E_vdw   :        {:.2f}".format(E_vdw.sum()))
            print("----------------------------------------------------------------------")
        else:           
            print(" If your system contains sigma hole interactions, " \
            "consider using the clustering mode for better energy accuracy")
            print("----------------------------------------------------------------------")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol, supra)
            print(f" NCI energies / kJ/mol")
            print(" E_sum   :        {:.2f}".format(E_sum))
            print(" E_polar :        {:.2f}".format(E_polar))
            print(" E_vdw   :        {:.2f}".format(E_vdw))
            print("----------------------------------------------------------------------")

else:
    print(" NCIENERGY mode runs only with the default parameters:")
    print(" RDG_CUTOFF 1.00 0.30")
    print(" INTEGRATE")
    print(" INTERMOL_CUTOFF 0.85 (0.75 for DFT)")
    print(" RANGE")
    print(" -0.20 -0.02")
    print(" -0.02  0.02")
    print("  0.02  0.20")
