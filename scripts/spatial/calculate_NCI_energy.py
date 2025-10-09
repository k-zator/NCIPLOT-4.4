#!/bin/python
import sys
import numpy as np
from spatial.sigma_hole_detection import find_sigma_bond
from spatial.DIVIDE import find_CP_Atom_matches

"""Critically also need to find the place to add the sigma hole detection to use a different energy equation 
    BUT this also requires finding the whole geometry of the complex"""

def calculate_energy_single(output, ispromol, supra):
    """Calculate the NCI energy given the NCIPLOT output contents for single integration"""

    sis = output.index("               RANGE INTEGRATION DATA                                 \n")
    polar = np.array([float(i.split(":")[1].split()[0]) for i in output[sis+8:sis+15]])
    vdw = np.array([float(i.split(":")[1].split()[0]) for i in output[sis+33:sis+40]])
    rep = np.array([float(i.split(":")[1].split()[0]) for i in output[sis+58:sis+65]])
    NCI_index_dict = {"Strong": polar[[0,5,1,6,2,3,4]] , "Weak": vdw[[0,5,1,6,2,3,4]] , "Repulsion": rep[[0,5,1,6,2,3,4]]}

    if ispromol:
        if supra:
            E_polar = -1381.3065*np.array(NCI_index_dict["Strong"][3])
            E_vdw = 424.2966*np.array(NCI_index_dict["Weak"][1]) -109.60437*np.array(NCI_index_dict["Weak"][0] + NCI_index_dict["Weak"][6]**(1/3))
        else:
            E_polar  = -np.array(27.424896*np.power(NCI_index_dict["Strong"][1], 0.333) + 2759.675*(NCI_index_dict["Strong"][4]))
            E_vdw = -np.array(-79.92235*np.power(NCI_index_dict["Weak"][3], 0.333) + 50.483402*np.power(NCI_index_dict["Weak"][0],0.5))
    else: #WFN
        if supra:
            print(" Supramolecular mode is currently incompatible with the WFN mode, " \
            "       check back soon for the updates or use the promolecular mode")
            sys.exit(1)
        else:
            E_polar = -np.array(3399.1965*NCI_index_dict["Strong"][2])
            E_vdw = -np.array(-811.5827*NCI_index_dict["Weak"][0]**2 + 115.258*np.power(NCI_index_dict["Weak"][5], 0.333) + 3399.1965*NCI_index_dict["Weak"][2])
    
    return E_polar + E_vdw, E_polar, E_vdw



def calculate_energy_cluster(output, ispromol, supra, mol1, mol2, filename):
    """Calculate the NCI energy given the NCIPLOT output contents for clustering intergration"""

    def energy(NCI_index_dict, ispromol, supra, sigma_hole=False):
        "Calculate energy of a single cluster given its NCI indices"
        if ispromol:
            if sigma_hole:
                E_polar = -1064.0465*np.array(NCI_index_dict["Strong"][3])
                E_vdw = -1064.0465*np.array(NCI_index_dict["Weak"][3]) - 5.8970227
            else:
                if supra:
                    E_polar = -1381.3065*np.array(NCI_index_dict["Strong"][3])
                    E_vdw = 424.2966*np.array(NCI_index_dict["Weak"][1]) -109.60437*np.array(NCI_index_dict["Weak"][0] + NCI_index_dict["Weak"][6]**(1/3))
                else:
                    E_polar = -np.array(27.424896*np.power(NCI_index_dict["Strong"][1], 0.333) + 2759.675*(NCI_index_dict["Strong"][4]))
                    E_vdw = -np.array(-79.92235*np.power(NCI_index_dict["Weak"][3], 0.333) + 50.483402*np.power(NCI_index_dict["Weak"][0],0.5))
        
        else: #WFN
            E_polar = -np.array(3399.1965*NCI_index_dict["Strong"][2])
            E_vdw = -np.array(-811.5827*NCI_index_dict["Weak"][0]**2 + 115.258*np.power(NCI_index_dict["Weak"][5], 0.333) + 3399.1965*NCI_index_dict["Weak"][2])
        return E_polar + E_vdw, E_polar, E_vdw

    no_clusters = [int(s.split()[5]) for s in output if " Number of critical points found:" in s][0]
    cluster_idx = output.index("      RANGE CLUSTERED INTEGRATION DATA over the volumes of rho^n      \n") + 1

    # SIGMA HOLE CHECK 
    CPs = filename + "_CPs.xyz"
    print(" For the time being, can only use XYZ input files for sigma hole detection. WFN alternative soon to come")
    CP_Atoms = find_CP_Atom_matches(CPs, mol1, mol2)
    sigma_bonds = find_sigma_bond(mol1, mol2)  # will be [] if None
    # now, are there any Atom Pairings that match? Those should be sigma_holes
    sigma_hole_mask = [0]*no_clusters
    for i in range(no_clusters):
        cluster_atoms = CP_Atoms[i]
        if cluster_atoms in sigma_bonds:
            sigma_hole_mask[i] = 1

    E_sum = []; E_polar = []; E_vdw = []; polar_i = 0; nonpolar_i = 0
    for i in range(no_clusters):
        polar = np.array([float(i.split(":")[1].split()[0]) for i in output[cluster_idx+5:cluster_idx+12]])
        vdw = np.array([float(i.split(":")[1].split()[0]) for i in output[cluster_idx+16:cluster_idx+23]])
        rep = np.array([float(i.split(":")[1].split()[0]) for i in output[cluster_idx+27:cluster_idx+34]])
        NCI_index_dict = {"Strong": polar[[0,5,1,6,2,3,4]] , "Weak": vdw[[0,5,1,6,2,3,4]] , "Repulsion": rep[[0,5,1,6,2,3,4]]}
        cluster_idx += 35 # the integration section length per cluster
        polar_i += polar[0]
        nonpolar_i += vdw[0]
        if sigma_hole_mask[i]:
            print(f" Note, cluster {i} contains a sigma hole. Energy is calculated with a dedicated sigma hole equation")
        e_sum, e_polar, e_vdw = energy(NCI_index_dict, ispromol, supra, sigma_hole=sigma_hole_mask[i])
        E_sum.append(e_sum)
        E_polar.append(e_polar)
        E_vdw.append(e_vdw)

    return np.array(E_sum),np.array(E_polar), np.array(E_vdw)


