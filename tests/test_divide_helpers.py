from pathlib import Path
import numpy as np

import spatial.DIVIDE as divide


def test_group_grad_to_grid_and_filter_close_cps():
    coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [2.0, 2.0, 2.0]])
    gradient = np.array([0.5, 0.1, 0.05])
    minima = divide.group_grad_to_grid(coords, gradient, a=1.0, gradient_threshold=0.2)
    assert set(minima) == {1, 2}

    cps = [[np.array([0.0, 0.0, 0.0]), 1.0, 0.1], [np.array([0.1, 0.0, 0.0]), 1.1, 0.1], [np.array([1.0, 0.0, 0.0]), 1.2, 0.1]]
    filtered = divide.filter_close_CPs(cps, min_distance=0.5)
    assert len(filtered) == 2


def test_get_unique_dimeric_cps():
    cp_dimer = [[np.array([0.0, 0.0, 0.0])], [np.array([1.0, 1.0, 1.0])]]
    cp_m1 = [[np.array([0.0, 0.0, 0.0])]]
    cp_m2 = [[np.array([2.0, 2.0, 2.0])]]
    unique = divide.get_unique_dimeric_CPs(cp_dimer, cp_m1, cp_m2)
    assert unique.shape == (1, 3)
    assert np.allclose(unique[0], np.array([1.0, 1.0, 1.0]))


def test_write_cps_xyz_and_read_wrappers(monkeypatch, tmp_path):
    cps = [[np.array([1.0, 2.0, 3.0]), 0.0, 0.0]]
    base = tmp_path / "demo"
    divide.write_CPs_xyz(cps, str(base))
    out = (tmp_path / "demo_CPs.xyz").read_text()
    assert "CP1" in out

    monkeypatch.setattr(divide, "read_xyz_geometry", lambda _f: (["H", "O"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])))
    coords, names = divide.read_xyz("x.xyz")
    assert coords.shape == (2, 3)
    assert names == ["H1", "O2"]

    monkeypatch.setattr(divide, "read_wfn_geometry", lambda _f: (["Cl"], np.array([[0.0, 0.0, 0.0]])))
    coords_w, names_w = divide.read_wfn("x.wfn")
    assert coords_w.shape == (1, 3)
    assert names_w == ["1Cl"]


def test_find_cp_atom_matches_and_read_charges(monkeypatch):
    monkeypatch.setattr(divide, "read_xyz", lambda f, elem=False: (np.array([[0.0, 0.0, 0.0]]) if "CP" in f else np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), ["X"]))
    matches = divide.find_CP_Atom_matches("CP.xyz", "m1.xyz", "m2.xyz", ispromol=True)
    assert matches == [[0, 0]]

    monkeypatch.setattr(divide, "infer_xtb_charge_path", lambda mol: f"{mol}.charges")
    monkeypatch.setattr(divide, "read_xyz", lambda f, elem=False: (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), ["H", "O"]))
    monkeypatch.setattr(divide, "read_xtb_charges", lambda p: np.array([0.1, -0.1]))
    monkeypatch.setattr(divide, "create_dimer_descriptors", lambda *args, **kwargs: (np.array([[1.0, 2.0, 3.0]]), []))
    monkeypatch.setattr(divide, "aggregate_system_descriptors", lambda pair_desc, aggregation="smst": np.array([pair_desc.sum()]))

    x = divide.read_charges("mol1.xyz", "mol2.xyz", ispromol=True)
    assert x.shape == (1,)
    assert float(x[0]) == 6.0


def test_find_cp_with_gradient_returns_points(monkeypatch):
    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 100.0, 0.1],
            [1.0, 0.0, 0.0, 80.0, 0.2],
        ]
    )
    monkeypatch.setattr(divide, "group_grad_to_grid", lambda *args, **kwargs: [0])
    monkeypatch.setattr(divide, "filter_close_CPs", lambda cps, min_distance=0.6: cps)
    monkeypatch.setattr(divide, "read_xyz", lambda m: (np.array([[0.0, 0.0, 0.0]]), ["H1"]))

    cps = divide.find_CP_with_gradient(matrix, threshold=0.1, radius=0.5, ispromol=True, mol=["m1.xyz", "m2.xyz"])
    assert len(cps) == 1
    assert np.allclose(cps[0][0], np.array([0.0, 0.0, 0.0]))
