#! /usr/bin/env python3
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
        cmd = ["xtb", str(mol), "-charge", str(int(charge))]
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


def read_xyz_geometry(xyz_path):
    """Read element labels and Cartesian coordinates from an XYZ file."""
    elements = []
    coords = []
    with open(xyz_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4:
                continue
            try:
                x, y, z = float(p[1]), float(p[2]), float(p[3])
            except ValueError:
                continue
            elements.append(p[0])
            coords.append([x, y, z])
    return elements, np.asarray(coords, dtype=float)


def read_wfn_geometry(wfn_path):
    """Read element labels and Cartesian coordinates from a .wfn-like text file."""
    bohr_to_angstrom = 0.52917721067
    elements = []
    coords = []
    with open(wfn_path, "r") as f:
        lines = f.readlines()[2:]
        for line in lines:
            p = line.split()
            if len(p) > 6 and p[2] == "(CENTRE":
                try:
                    x = float(p[4]) * bohr_to_angstrom
                    y = float(p[5]) * bohr_to_angstrom
                    z = float(p[6]) * bohr_to_angstrom
                except ValueError:
                    continue
                elements.append(p[1])
                coords.append([x, y, z])
    return elements, np.asarray(coords, dtype=float)


def read_xtb_charges(charge_file):
    """Read xTB `charges`/`*_charges.dat` values (one charge per line)."""
    charges = []
    with open(charge_file, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 0:
                continue
            try:
                charges.append(float(p[0]))
            except ValueError:
                continue
    return np.asarray(charges, dtype=float)


def read_cp_centers(cp_xyz):
    """Read CP coordinates from XYZ-like file (uses last 3 numeric columns)."""
    if cp_xyz is None:
        return np.zeros((0, 3), dtype=float)
    cp_path = Path(cp_xyz)
    if not cp_path.exists():
        return np.zeros((0, 3), dtype=float)

    centers = []
    with open(cp_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4:
                continue
            try:
                centers.append([float(p[-3]), float(p[-2]), float(p[-1])])
            except ValueError:
                continue
    return np.asarray(centers, dtype=float)


def infer_xtb_charge_path(mol_xyz):
    """Infer xTB charge path produced by `calculate_charges`: <stem>_charges.dat."""
    xyz_path = Path(mol_xyz)
    return xyz_path.with_name(f"{xyz_path.stem}_charges.dat")


def mask_near_center(coords, center, cutoff):
    distances = np.linalg.norm(coords - center[None, :], axis=1)
    return distances <= cutoff


def _cluster_feature_from_masks(
    coords_A, elements_A, charges_A,
    coords_B, elements_B, charges_B,
    mask_A, mask_B,
    features="6key", aggregation="smst", n_bessel=6, r_cut=10.0,
    append_bias=True,
):
    if mask_A.sum() == 0 or mask_B.sum() == 0:
        return None

    pair_desc, _ = create_dimer_descriptors(
        coords_A[mask_A], [elements_A[i] for i in np.where(mask_A)[0]], charges_A[mask_A],
        coords_B[mask_B], [elements_B[i] for i in np.where(mask_B)[0]], charges_B[mask_B],
        include_bessel=True,
        n_bessel=n_bessel,
        r_cut=r_cut,
        intermolecular_only=True,
        features=features,
    )

    if len(pair_desc) == 0:
        return None

    x = aggregate_system_descriptors(pair_desc, aggregation=aggregation)
    x = np.asarray(x, dtype=float)
    if append_bias:
        x = np.concatenate([x, np.array([1.0], dtype=float)])
    return x


def build_cluster_descriptors(
    mol1_xyz,
    mol2_xyz,
    cp_xyz,
    charge_file_1=None,
    charge_file_2=None,
    cutoff=7.0,
    features="6key",
    aggregation="smst",
    n_bessel=6,
    r_cut=10.0,
    append_bias=True,
    fallback_full_system=True,
    ispromol=True,
):
    """
    Build latent-additive cluster descriptors X_clusters (K, D) for one dimer.

    Clusters are CP-centered neighborhoods. If no valid CP neighborhoods are found,
    optionally fallback to one whole-system descriptor.
    """
    if ispromol:
        elements_A, coords_A = read_xyz_geometry(mol1_xyz)
        elements_B, coords_B = read_xyz_geometry(mol2_xyz)
    else:
        elements_A, coords_A = read_wfn_geometry(mol1_xyz)
        elements_B, coords_B = read_wfn_geometry(mol2_xyz)

    if charge_file_1 is None:
        charge_file_1 = infer_xtb_charge_path(mol1_xyz)
    if charge_file_2 is None:
        charge_file_2 = infer_xtb_charge_path(mol2_xyz)

    charges_A = read_xtb_charges(charge_file_1)
    charges_B = read_xtb_charges(charge_file_2)

    if len(charges_A) != len(coords_A):
        raise ValueError(f"Charge/atom mismatch for mol1: {len(charges_A)} vs {len(coords_A)}")
    if len(charges_B) != len(coords_B):
        raise ValueError(f"Charge/atom mismatch for mol2: {len(charges_B)} vs {len(coords_B)}")

    centers = read_cp_centers(cp_xyz)
    X_clusters = []
    centers_used = []

    for center in centers:
        mA = mask_near_center(coords_A, center, cutoff)
        mB = mask_near_center(coords_B, center, cutoff)
        x = _cluster_feature_from_masks(
            coords_A, elements_A, charges_A,
            coords_B, elements_B, charges_B,
            mA, mB,
            features=features,
            aggregation=aggregation,
            n_bessel=n_bessel,
            r_cut=r_cut,
            append_bias=append_bias,
        )
        if x is not None:
            X_clusters.append(x)
            centers_used.append(center)

    if len(X_clusters) == 0 and fallback_full_system:
        mA_all = np.ones(len(coords_A), dtype=bool)
        mB_all = np.ones(len(coords_B), dtype=bool)
        x_all = _cluster_feature_from_masks(
            coords_A, elements_A, charges_A,
            coords_B, elements_B, charges_B,
            mA_all, mB_all,
            features=features,
            aggregation=aggregation,
            n_bessel=n_bessel,
            r_cut=r_cut,
            append_bias=append_bias,
        )
        if x_all is not None:
            X_clusters = [x_all]
            centers_used = [np.array([np.nan, np.nan, np.nan], dtype=float)]

    if len(X_clusters) == 0:
        raise ValueError("No valid cluster descriptors could be constructed.")

    return np.vstack(X_clusters), np.asarray(centers_used, dtype=float)


def _onnx_predict_1d(onnx_session, x_1d):
    x_2d = np.asarray(x_1d, dtype=np.float32).reshape(1, -1)
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    out = onnx_session.run([output_name], {input_name: x_2d})[0]
    return float(np.asarray(out).reshape(-1)[0])


def predict_cluster_sum_delta_onnx(X_clusters, model_path=None, onnx_session=None):
    """Predict total correction from summed cluster descriptor with an ONNX model."""
    if onnx_session is None:
        if model_path is None:
            raise ValueError("Provide either `model_path` or `onnx_session`.")
        import onnxruntime as rt
        onnx_session = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    x_sum = np.asarray(X_clusters, dtype=float).sum(axis=0)
    return _onnx_predict_1d(onnx_session, x_sum)


def shapley_cluster_attributions_onnx(
    X_clusters,
    model_path=None,
    onnx_session=None,
    n_perm=256,
    random_state=42,
    baseline=None,
):
    """
    Monte-Carlo Shapley attributions for cluster inputs used in sum-model inference.

    Returns
    -------
    phi : ndarray shape (K,)
        Cluster contributions such that baseline + phi.sum() ~= total prediction.
    total : float
        Model prediction for sum(X_clusters).
    baseline : float
        Model prediction for zero-vector if not user-provided.
    """
    if onnx_session is None:
        if model_path is None:
            raise ValueError("Provide either `model_path` or `onnx_session`.")
        import onnxruntime as rt
        onnx_session = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    X_clusters = np.asarray(X_clusters, dtype=float)
    K, D = X_clusters.shape
    phi = np.zeros(K, dtype=float)
    rng = np.random.default_rng(random_state)

    if baseline is None:
        baseline = _onnx_predict_1d(onnx_session, np.zeros(D, dtype=float))

    for _ in range(n_perm):
        order = rng.permutation(K)
        state = np.zeros(D, dtype=float)
        prev = _onnx_predict_1d(onnx_session, state)
        for k in order:
            state = state + X_clusters[k]
            cur = _onnx_predict_1d(onnx_session, state)
            phi[k] += (cur - prev)
            prev = cur

    phi /= float(n_perm)
    total = _onnx_predict_1d(onnx_session, X_clusters.sum(axis=0))

    # enforce exact reconstruction against this baseline
    corr = (total - baseline) - float(phi.sum())
    phi += corr / max(1, K)
    return phi, total, baseline
