#!/bin/python
import numpy as np #type: ignore
from pathlib import Path
import subprocess
from itertools import combinations
from scipy.special import spherical_jn #type: ignore

def calculate_charges(mol1, mol2, total_charges):
    """
    Run xTB to compute per-atom charges for an XYZ geometry.

    This function:
      1) (Optionally) infers total charge (best-effort) OR uses provided `total_charge` (recommended).
      2) runs:  xtb input.xyz -charge Q   (fallback: xtb input.xyz --chrg Q)
      3) writes the resulting charges file to `charges_out` (or next to xyz as <stem>.charges)

    Notes
    -----
    * Inferring charge from XYZ alone is generally unreliable. Pass `total_charge=...` explicitly.
    * xTB writes several files in the working directory; this runs in a temp dir by default.

    Returns
    -------
    Path to the written charges file.
    """
    parent_path = Path(mol1).parent
    # first assumption, xyz files and not WFN are present
    # (otherwise, we'll need to create .xyz)
    [total_charge_1, total_charge_2] = total_charges

    # Run primary command requested by you: `xtb input.xyz -charge Q`
    def _run(mol, charge):
        cmd = ["xtb", str(mol), "-charge", str(int(charge)), "--chrg", str(int(charge))]
        subprocess.run(
            cmd,
            cwd=str(parent_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        # and move the generated charges file to the correct location
        cmd_move = ["mv", str(parent_path / "charges"), str(parent_path / (Path(mol).stem + "_charges.dat"))]
        subprocess.run(cmd_move, cwd=str(parent_path), check=False)

    # if the files don't already exist, run the commands to generate them
    if not (parent_path / (Path(mol1).stem + "_charges.dat")).exists():
        _run(mol1, total_charge_1)
        _run(mol2, total_charge_2)
        print(f"Generated charges files for {mol1} set charges {total_charge_1}.")
        print(f"Generated charges files for {mol2} set charges {total_charge_2}.")

def compute_bessel_expansion(distances, n_max=6, r_cut=10.0):
    """Compute spherical Bessel function expansion for distances."""
    n_orders = np.arange(1, n_max + 1)
    scaled_r = distances * np.pi / r_cut
    
    bessel_vals = np.zeros((len(distances), n_max))
    for i, n in enumerate(n_orders):
        bessel_vals[:, i] = spherical_jn(n, scaled_r)
    
    return bessel_vals


def create_dimer_descriptors(coords_A, elements_A, charges_A, coords_B, elements_B, charges_B,
                             include_bessel=True, n_bessel=6, r_cut=10.0, intermolecular_only=True, features="full"):
    """Create pairwise interaction descriptors for a dimer complex. Yes, this will create n_A x n_B x 14 features"""

    element_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Se': 34, 'Cl': 17, 'Br': 35, 'I': 53}
    pair_descriptors = []
    pair_info = []
    n_A = len(coords_A)
    n_B = len(coords_B)
    
    # INTERMOLECULAR interactions (A-B cross terms)
    for i in range(n_A):
        for j in range(n_B):
            r_vec = coords_A[i] - coords_B[j]
            distance = np.linalg.norm(r_vec)
            
            if distance < 1e-6:
                continue
            
            q_i, q_j = charges_A[i], charges_B[j]
            elem_i, elem_j = elements_A[i], elements_B[j]
            
            # Basic features
            inv_distance = 1.0 / distance
            charge_product = q_i * q_j
            charge_weighted_dist = charge_product * inv_distance
            
            z_i = element_map.get(elem_i, 0)
            z_j = element_map.get(elem_j, 0)
            elem_encoding = [min(z_i, z_j), max(z_i, z_j)]
            
            if features == "full":
                basic_features = [
                    inv_distance,
                    charge_product, 
                    abs(charge_product), 
                    charge_weighted_dist,
                    q_i + q_j, 
                    abs(q_i - q_j), 
                    *elem_encoding
                ]
            elif features == "10key": #i.e. 4 charge + 6 Bessel
                basic_features = [
                    charge_weighted_dist,
                    abs(charge_product),
                    abs(q_i - q_j),
                ]
            elif features == "6key": #i.e. 6 charge only
                basic_features = [
                    abs(charge_product),
                    charge_weighted_dist,
                    q_i + q_j,
                    abs(q_i - q_j),
                    *elem_encoding
                ]
            elif features == "2key":
                basic_features = [
                    charge_weighted_dist,
                    charge_product * np.exp(-distance / 3.0)
                ]
            
            if include_bessel and features not in ["6key", "2key"]:
                bessel_feats = compute_bessel_expansion(np.array([distance]), n_max=n_bessel, r_cut=r_cut)[0]
                basic_features.extend(bessel_feats)
            
            pair_descriptors.append(basic_features)
            pair_info.append({
                'atoms': (i, j),
                'molecules': ('A', 'B'),
                'elements': (elem_i, elem_j),
                'distance': distance,
                'charges': (q_i, q_j),
                'type': 'intermolecular'
            })
    
    # INTRAMOLECULAR interactions (optional, usually less important for binding)
    if not intermolecular_only:
        # A-A intramolecular
        for i, j in combinations(range(n_A), 2):
            r_vec = coords_A[i] - coords_A[j]
            distance = np.linalg.norm(r_vec)
            if distance < 1e-6:
                continue
            
            q_i, q_j = charges_A[i], charges_A[j]
            elem_i, elem_j = elements_A[i], elements_A[j]
            
            inv_distance = 1.0 / distance
            charge_product = q_i * q_j
            charge_weighted_dist = charge_product * inv_distance
            
            z_i = element_map.get(elem_i, 0)
            z_j = element_map.get(elem_j, 0)
            elem_encoding = [min(z_i, z_j), max(z_i, z_j)]
            
            basic_features = [
                distance, inv_distance, distance**2,
                charge_product, abs(charge_product), charge_weighted_dist,
                q_i + q_j, abs(q_i - q_j), *elem_encoding
            ]
            
            if include_bessel and features not in ["6key", "2key"]:
                bessel_feats = compute_bessel_expansion(np.array([distance]), n_max=n_bessel, r_cut=r_cut)[0]
                basic_features.extend(bessel_feats)
            
            pair_descriptors.append(basic_features)
            pair_info.append({
                'atoms': (i, j), 'molecules': ('A', 'A'), 
                'elements': (elem_i, elem_j), 'distance': distance,
                'charges': (q_i, q_j), 'type': 'intramolecular_A'
            })
        
        # B-B intramolecular (same logic)
        for i, j in combinations(range(n_B), 2):
            r_vec = coords_B[i] - coords_B[j]
            distance = np.linalg.norm(r_vec)
            if distance < 1e-6:
                continue
            
            q_i, q_j = charges_B[i], charges_B[j]
            elem_i, elem_j = elements_B[i], elements_B[j]
            
            inv_distance = 1.0 / distance
            charge_product = q_i * q_j
            charge_weighted_dist = charge_product * inv_distance
            
            z_i = element_map.get(elem_i, 0)
            z_j = element_map.get(elem_j, 0)
            elem_encoding = [min(z_i, z_j), max(z_i, z_j)]
            
            basic_features = [
                distance, inv_distance, distance**2,
                charge_product, abs(charge_product), charge_weighted_dist,
                q_i + q_j, abs(q_i - q_j), *elem_encoding
            ]
            
            if include_bessel:
                bessel_feats = compute_bessel_expansion(np.array([distance]), n_max=n_bessel, r_cut=r_cut)[0]
                basic_features.extend(bessel_feats)
            
            pair_descriptors.append(basic_features)
            pair_info.append({
                'atoms': (i, j), 'molecules': ('B', 'B'),
                'elements': (elem_i, elem_j), 'distance': distance,
                'charges': (q_i, q_j), 'type': 'intramolecular_B'
            })
    
    return np.array(pair_descriptors), pair_info

def aggregate_system_descriptors(pair_descriptors, aggregation='both', top_k=20, r_scale=3.0):
    """Aggregate pairwise descriptors into fixed-size system-level descriptor."""
    if aggregation == 'sum':
        return np.sum(pair_descriptors, axis=0)
    elif aggregation == 'max':
        return np.max(pair_descriptors, axis=0)
    elif aggregation == 'mean':
        return np.mean(pair_descriptors, axis=0)
    elif aggregation == 'both':
        return np.concatenate([np.sum(pair_descriptors, axis=0),np.mean(pair_descriptors, axis=0)])
    elif aggregation == 'focused':
        dist = pair_descriptors[:, 0]
        qprod_abs = np.abs(pair_descriptors[:, 3])
        weights = qprod_abs * np.exp(-dist / max(1e-6, r_scale))
        wsum = weights.sum() + 1e-12

        mean_feat = pair_descriptors.mean(axis=0)
        wmean     = (weights[:, None] * pair_descriptors).sum(axis=0) / wsum

        k = min(top_k, pair_descriptors.shape[0])
        topk_idx = np.argsort(-weights)[:k]
        topk_mean = pair_descriptors[topk_idx].mean(axis=0)

        return np.concatenate([mean_feat, wmean, topk_mean])
    elif aggregation == 'smst':
        sum_feat = np.sum(pair_descriptors, axis=0)
        max_feat = np.max(pair_descriptors, axis=0)
        std_feat = np.std(pair_descriptors, axis=0)
        return np.concatenate([sum_feat, max_feat, std_feat])
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
