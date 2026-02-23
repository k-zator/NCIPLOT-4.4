from pathlib import Path
import numpy as np

import spatial.sigma_bond_detection as sbd


def test_obtain_coordinates_xyz(tmp_path):
    f1 = tmp_path / "a.xyz"
    f2 = tmp_path / "b.xyz"
    f1.write_text("2\ncomment\nCl 0.0 0.0 0.0\nC 1.0 0.0 0.0\n")
    f2.write_text("1\ncomment\nO 0.0 3.0 0.0\n")

    c1, c2, e1, e2 = sbd.obtain_coordinates(str(f1), str(f2), ispromol=True)
    assert c1.shape == (2, 3)
    assert c2.shape == (1, 3)
    assert e1.tolist() == ["Cl", "C"]
    assert e2.tolist() == ["O"]


def test_obtain_coordinates_wfn_like(tmp_path):
    f1 = tmp_path / "a.wfn"
    f2 = tmp_path / "b.wfn"
    f1.write_text("h1\nh2\nCl X (CENTRE 1.0 2.0 3.0 0.0\n")
    f2.write_text("h1\nh2\nBr X (CENTRE 2.0 3.0 4.0 0.0\n")

    c1, c2, e1, e2 = sbd.obtain_coordinates(str(f1), str(f2), ispromol=False)
    assert c1.shape == (1, 3)
    assert c2.shape == (1, 3)
    assert e1.tolist() == ["Cl"]
    assert e2.tolist() == ["Br"]


def test_obtain_connectivity_and_distances():
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [5.0, 0.0, 0.0]])
    bonds = sbd.obtain_coonectivity(coords, bond_threshold=2.0)
    assert bonds == [[0, 1]]

    d = sbd.calculate_distance(coords[:2], coords[2:])
    assert d == np.linalg.norm(coords[1] - coords[2])

    md, idx = sbd.calculate_min_dist_to_atom(coords[0], coords[1:])
    assert idx == 0
    assert md == np.linalg.norm(coords[0] - coords[1])


def test_detect_sigma_relevant_atoms_and_find_sigma_bond(monkeypatch):
    elem_1 = np.array(["Cl", "C"])
    bonds_1 = [[0, 1]]
    coord_1 = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    coord_2 = np.array([[2.0, 0.0, 0.0]])

    results = sbd.detect_sigma_relevant_atoms(elem_1, bonds_1, coord_1, coord_2, halogen="Cl")
    assert len(results) == 1
    assert results[0][0] == 0
    assert results[0][1] == 0

    monkeypatch.setattr(
        sbd,
        "obtain_coordinates",
        lambda *_args, **_kwargs: (
            np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.array([[2.0, 0.0, 0.0]]),
            np.array(["Cl", "C"]),
            np.array(["O"]),
        ),
    )

    sigma_bonds = sbd.find_sigma_bond("a", "b", ispromol=True)
    assert sigma_bonds == [[0, 0]]
