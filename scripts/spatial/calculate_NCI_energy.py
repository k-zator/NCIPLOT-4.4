#! /usr/bin/env python3
from pathlib import Path
import numpy as np # type: ignore
import onnxruntime as rt
from spatial.charge_aggregate import (
    calculate_charges,
    build_cluster_descriptors,
    predict_cluster_sum_delta_onnx,
    shapley_cluster_attributions_onnx,
)
from spatial.sigma_bond_detection import find_sigma_bond
from spatial.DIVIDE import find_CP_Atom_matches, read_charges

def _charge_model_path(cluster=False, supra=False):
    base_dir = Path(__file__).resolve().parent
    if cluster:
        name = "gbr_charge_cluster.onnx" if supra else "gbr_charge_small_cluster.onnx"
    else:
        name = "gbr_charge.onnx" if supra else "gbr_charge_small.onnx"
    return str(base_dir / name)


def calculate_charge_correction(mol1, mol2, ispromol, total_charges, supra=False):
    """Wrapper function to xTB calculate charges, their aggregate descriptors,
        and the charge-based energy correction term for the system"""
    # 1. Calculate charges with xTB if not already present
    calculate_charges(mol1, mol2, total_charges)
    # 2. Read calculated charge files and compute Coulomb energy descriptors
    X = read_charges(mol1, mol2, ispromol)
    # 3. Compute a charge-based energy correction term for the system via a regression model
    # Gradient-Boosting Regressor trained on NCIAtlas, S30L, L7, and CIM13 datasets
    model_path = _charge_model_path(cluster=False, supra=supra)
    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X.reshape(1, len(X)).astype(np.float32)})
    return pred_onx[0][0][0]


def calculate_charge_correction_cluster(
    mol1,
    mol2,
    ispromol,
    total_charges,
    cp_xyz,
    cutoff=1.0,
    supra=False,
    return_shapley=False,
    shapley_n_perm=256,
):
        """Charge correction for clustered mode using latent-additive cluster descriptors.
        This follows the same logic as lacr.py:
            1) build per-cluster descriptors around CP centers,
            2) sum descriptors across clusters,
            3) predict one total delta with GBR ONNX.
        """
        calculate_charges(mol1, mol2, total_charges)

        X_clusters, _ = build_cluster_descriptors(
                mol1,
                mol2,
                cp_xyz,
                cutoff=float(cutoff),
                features="6key",
                aggregation="smst",
                n_bessel=6,
                r_cut=10.0,
                append_bias=True,
                fallback_full_system=True,
                ispromol=ispromol,
        )

        model_path = _charge_model_path(cluster=True, supra=supra)
        sess = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        total_charge = predict_cluster_sum_delta_onnx(X_clusters, onnx_session=sess)

        if not return_shapley:
            return total_charge

        phi, _, _ = shapley_cluster_attributions_onnx(
            X_clusters,
            onnx_session=sess,
            n_perm=shapley_n_perm,
            random_state=42,
        )
        return total_charge, phi


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
            E_polar = -19199.29366455*NCI_index_dict["Strong"][4]
            E_vdw = -396.6232038*NCI_index_dict["Weak"][5]**(1/3) -51.94821914*NCI_index_dict["Weak"][0]
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
                # this has to be extensive and the only good enough equation is the SUPRA (incl. L7/CIM13) one
                E_polar =  -183.88803*NCI_index_dict["Strong"][0]
                E_vdw = -183.88803*NCI_index_dict["Weak"][0] + 756.96106*NCI_index_dict["Weak"][1]

        else: #WFN
            if sigma_hole:
                E_polar  = -560.6157*NCI_index_dict["Strong"][0]**(1/2) - 1521.1189*NCI_index_dict["Strong"][2]**(1/2)
                E_vdw = - 289.4128*NCI_index_dict["Weak"][0]
            else:
                E_polar = -776.39044*NCI_index_dict["Strong"][0]
                E_vdw =  -6138.2095*NCI_index_dict["Weak"][6] -92.60362*NCI_index_dict["Weak"][0]
    
        return E_polar + E_vdw, E_polar, E_vdw

    no_clusters = [int(s.split()[5]) for s in output if " Number of critical points found:" in s][0]
    cluster_idx = output.index("      RANGE CLUSTERED INTEGRATION DATA over the volumes of rho^n      \n") + 1

    # SIGMA HOLE CHECK 
    CPs = filename + "_CPs.xyz"
    CP_Atoms = find_CP_Atom_matches(CPs, mol1, mol2, ispromol)
    sigma_bonds = find_sigma_bond(mol1, mol2, ispromol)  # will be [] if None
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


