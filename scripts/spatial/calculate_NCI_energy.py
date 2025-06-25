#!/bin/python
import os
import numpy as np
from spatial.sigma_hole_detection import find_sigma_bond

"""Critically also need to find the place to add the sigma hole detection to use a different energy equation 
    BUT this also requires finding the whole geometry of the complex"""

def calculate_energy_single(output, ispromol):
    """Calculate the NCI energy given the NCIPLOT output contents for single integration"""

    sis = output.index("               RANGE INTEGRATION DATA                                 \n")
    polar = np.array([float(i.split(":")[1].split()[0]) for i in output[sis+9:sis+16]])
    vdw = np.array([float(i.split(":")[1].split()[0]) for i in output[sis+39:sis+46]])
    rep = np.array([float(i.split(":")[1].split()[0]) for i in output[sis+69:sis+76]])
    NCI_index_dict = {"Strong": polar[[0,5,1,6,2,3,4]] , "Weak": vdw[[0,5,1,6,2,3,4]] , "Repulsion": rep[[0,5,1,6,2,3,4]]}

    if ispromol:
        E_polar  = -np.array(27.424896*np.power(NCI_index_dict["Strong"][1], 0.333) + 2759.675*(NCI_index_dict["Strong"][4]))
        E_vdw = -np.array(-79.92235*np.power(NCI_index_dict["Weak"][3], 0.333) + 50.483402*np.power(NCI_index_dict["Weak"][0],0.5))
    else: #WFN
        E_polar = -np.array(3399.1965*NCI_index_dict["Strong"][2])
        E_vdw = -np.array(-811.5827*NCI_index_dict["Weak"][0]**2 + 115.258*np.power(NCI_index_dict["Weak"][5], 0.333) + 3399.1965*NCI_index_dict["Weak"][2])
    
    return E_polar + E_vdw, E_polar, E_vdw



def calculate_energy_cluster(output, ispromol):
    """Calculate the NCI energy given the NCIPLOT output contents for clustering intergration"""

    def energy(NCI_index_dict, ispromol, sigma_hole=False):
        "Calculate energy of a single cluster given its NCI indices"
        if ispromol:
            if sigma_hole:
                E_polar = -1064.0465*np.array(NCI_index_dict["Strong"][3])
                E_vdw = -1064.0465*np.array(NCI_index_dict["Weak"][3]) - 5.8970227
            else:
                E_polar = -np.array(27.424896*np.power(NCI_index_dict["Strong"][1], 0.333) + 2759.675*(NCI_index_dict["Strong"][4]))
                E_vdw = -np.array(-79.92235*np.power(NCI_index_dict["Weak"][3], 0.333) + 50.483402*np.power(NCI_index_dict["Weak"][0],0.5))
        
        else: #WFN
            E_polar = -np.array(3399.1965*NCI_index_dict["Strong"][2])
            E_vdw = -np.array(3399.1965*NCI_index_dict["Weak"][2] + 115.258*np.power(NCI_index_dict["Weak"][5], 0.333) -811.5827*NCI_index_dict["Weak"][0]**2)
        return E_polar + E_vdw, E_polar, E_vdw
    

    # iterate over the cluster:
    no_clusters = [int(s.split()[5]) for i, s in output if " Number of critical points found:" in s][0]
    cluster_idx = output.index("      RANGE CLUSTERED INTEGRATION DATA over the volumes of rho^n      \n") + 1

    """There should be a sigma hole check here"""
    path1 = "mol1.xyz" # need to feed this information along somewhere (also, necessarily XYZ!)
    path2 = "mol2.xyz"
    sigma_bonds = find_sigma_bond(path1, path2) # will be [] if None
    # cluster integration will also contain the information about which atoms correpond to the clusters

    E_sum = []; E_polar = []; E_vdw = []
    for i_cluster in range(no_clusters):
        print( output[cluster_idx])
        polar = np.array([float(i.split(":")[1].split()[0]) for i in output[cluster_idx+3:cluster_idx+10]])
        vdw = np.array([float(i.split(":")[1].split()[0]) for i in output[cluster_idx+14:cluster_idx+21]])
        rep = np.array([float(i.split(":")[1].split()[0]) for i in output[cluster_idx+25:cluster_idx+32]])
        NCI_index_dict = {"Strong": polar[[0,5,1,6,2,3,4]] , "Weak": vdw[[0,5,1,6,2,3,4]] , "Repulsion": rep[[0,5,1,6,2,3,4]]}
        cluster_idx += 34 # the integration section length per cluster

        e_sum, e_polar, e_vdw = energy(NCI_index_dict, ispromol)
        E_sum.append(e_sum)
        E_polar.append(e_polar)
        E_vdw.append(e_vdw)

    return np.array(E_sum),np.array(E_polar), np.array(E_vdw)

