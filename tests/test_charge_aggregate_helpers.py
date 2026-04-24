from pathlib import Path
import sys
import types
import numpy as np
import pytest

import spatial.charge_aggregate as agg


class _DummySession:
    def get_inputs(self):
        return [type("X", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("Y", (), {"name": "output"})()]

    def run(self, _outs, feed):
        x = np.array(next(iter(feed.values())), dtype=float)
        return [np.array([[x.sum()]], dtype=np.float32)]


def test_compute_bessel_expansion_shape():
    d = np.array([1.0, 2.0])
    out = agg.compute_bessel_expansion(d, n_max=4, r_cut=8.0)
    assert out.shape == (2, 4)


def test_create_dimer_descriptors_and_aggregate():
    coords_a = np.array([[0.0, 0.0, 0.0]])
    coords_b = np.array([[1.0, 0.0, 0.0]])
    desc, info = agg.create_dimer_descriptors(
        coords_a,
        ["H"],
        np.array([0.1]),
        coords_b,
        ["O"],
        np.array([-0.2]),
        include_bessel=False,
        features="6key",
    )
    assert desc.shape[0] == 1
    assert info[0]["type"] == "intermolecular"

    smst = agg.aggregate_system_descriptors(desc, aggregation="smst")
    assert smst.shape[0] == desc.shape[1] * 3

    with pytest.raises(ValueError, match="Unknown aggregation"):
        agg.aggregate_system_descriptors(desc, aggregation="bad")

    desc10, _ = agg.create_dimer_descriptors(
        coords_a, ["H"], np.array([0.1]), coords_b, ["O"], np.array([-0.2]), include_bessel=True, features="10key"
    )
    assert desc10.shape[1] == 3 + 6

    desc2, _ = agg.create_dimer_descriptors(
        coords_a, ["H"], np.array([0.1]), coords_b, ["O"], np.array([-0.2]), include_bessel=False, features="2key"
    )
    assert desc2.shape[1] == 2

    focused = agg.aggregate_system_descriptors(np.array([[1.0, 2.0, 3.0, 0.1], [2.0, 3.0, 4.0, 0.2]]), aggregation="focused")
    assert focused.size > 0


def test_geometry_and_charge_readers(tmp_path):
    xyz = tmp_path / "a.xyz"
    xyz.write_text("2\ncomment\nH 0.0 0.0 0.0\nO 1.0 0.0 0.0\n")
    elems, coords = agg.read_xyz_geometry(xyz)
    assert elems == ["H", "O"]
    assert coords.shape == (2, 3)

    wfn = tmp_path / "a.wfn"
    wfn.write_text("line1\nline2\n Cl X (CENTRE 1.0 2.0 3.0 0.0\n")
    elems_w, coords_w = agg.read_wfn_geometry(wfn)
    assert elems_w == ["Cl"]
    assert coords_w.shape == (1, 3)

    ch = tmp_path / "a.charges"
    ch.write_text("0.1\n-0.2\n")
    charges = agg.read_xtb_charges(ch)
    assert np.allclose(charges, np.array([0.1, -0.2]))


def test_cp_helpers_and_mask(tmp_path):
    cp = tmp_path / "cp.xyz"
    cp.write_text("2\ncomment\nCP1 0.0 1.0 2.0\nCP2 1.0 2.0 3.0\n")
    centers = agg.read_cp_centers(cp)
    assert centers.shape == (2, 3)
    assert agg.read_cp_centers(None).shape == (0, 3)

    inferred = agg.infer_xtb_charge_path("/tmp/mol.xyz")
    assert str(inferred).endswith("mol_charges.dat")

    mask = agg.mask_near_center(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]), 1.0)
    assert mask.tolist() == [True, False]


def test_cluster_feature_and_build_descriptors(monkeypatch):
    monkeypatch.setattr(agg, "create_dimer_descriptors", lambda *args, **kwargs: (np.array([[1.0, 2.0, 3.0]]), []))
    monkeypatch.setattr(agg, "aggregate_system_descriptors", lambda pair_desc, aggregation="smst": np.array([pair_desc.sum()]))

    x = agg._cluster_feature_from_masks(
        np.array([[0.0, 0.0, 0.0]]), ["H"], np.array([0.1]),
        np.array([[1.0, 0.0, 0.0]]), ["O"], np.array([-0.1]),
        np.array([True]), np.array([True]), append_bias=True,
    )
    assert x.shape == (2,)
    assert x[-1] == 1.0

    none_x = agg._cluster_feature_from_masks(
        np.array([[0.0, 0.0, 0.0]]), ["H"], np.array([0.1]),
        np.array([[1.0, 0.0, 0.0]]), ["O"], np.array([-0.1]),
        np.array([False]), np.array([True]), append_bias=True,
    )
    assert none_x is None

    monkeypatch.setattr(agg, "read_xyz_geometry", lambda p: (["H"], np.array([[0.0, 0.0, 0.0]])))
    monkeypatch.setattr(agg, "read_xtb_charges", lambda p: np.array([0.1]))
    monkeypatch.setattr(agg, "read_cp_centers", lambda p: np.array([[0.0, 0.0, 0.0]]))
    monkeypatch.setattr(agg, "_cluster_feature_from_masks", lambda *args, **kwargs: np.array([2.0, 1.0]))

    x_clusters, centers = agg.build_cluster_descriptors("m1.xyz", "m2.xyz", "cp.xyz")
    assert x_clusters.shape == (1, 2)
    assert centers.shape == (1, 3)

    monkeypatch.setattr(agg, "read_xtb_charges", lambda p: np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="Charge/atom mismatch"):
        agg.build_cluster_descriptors("m1.xyz", "m2.xyz", "cp.xyz")


def test_onnx_wrappers_and_shapley():
    sess = _DummySession()
    pred = agg._onnx_predict_1d(sess, np.array([1.0, 2.0]))
    assert pred == pytest.approx(3.0)

    total = agg.predict_cluster_sum_delta_onnx(np.array([[1.0, 2.0], [3.0, 4.0]]), onnx_session=sess)
    assert total == pytest.approx(10.0)

    phi, total2, baseline = agg.shapley_cluster_attributions_onnx(
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        onnx_session=sess,
        n_perm=8,
        random_state=0,
    )
    assert len(phi) == 2
    assert total2 == pytest.approx(3.0)
    assert baseline == pytest.approx(0.0)
    assert phi.sum() == pytest.approx(total2 - baseline)

    with pytest.raises(ValueError, match="Provide either `model_path` or `onnx_session`"):
        agg.predict_cluster_sum_delta_onnx(np.array([[1.0, 2.0]]), onnx_session=None, model_path=None)

    with pytest.raises(ValueError, match="Provide either `model_path` or `onnx_session`"):
        agg.shapley_cluster_attributions_onnx(np.array([[1.0, 2.0]]), onnx_session=None, model_path=None)


def test_create_dimer_descriptors_full_skips_zero_distance_pair():
    coords_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords_b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    desc, info = agg.create_dimer_descriptors(
        coords_a,
        ["H", "C"],
        np.array([0.2, -0.1]),
        coords_b,
        ["O", "N"],
        np.array([-0.2, 0.3]),
        include_bessel=False,
        intermolecular_only=True,
        features="full",
    )

    # 2x2 total pairs minus one zero-distance pair.
    assert len(info) == 3
    assert desc.shape == (3, 8)
    assert all(d["distance"] > 0.0 for d in info)


def test_aggregate_system_descriptors_basic_modes():
    pair_desc = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert np.allclose(agg.aggregate_system_descriptors(pair_desc, aggregation="sum"), np.array([4.0, 6.0]))
    assert np.allclose(agg.aggregate_system_descriptors(pair_desc, aggregation="max"), np.array([3.0, 4.0]))
    assert np.allclose(agg.aggregate_system_descriptors(pair_desc, aggregation="mean"), np.array([2.0, 3.0]))
    assert np.allclose(
        agg.aggregate_system_descriptors(pair_desc, aggregation="both"),
        np.array([4.0, 6.0, 2.0, 3.0]),
    )


def test_create_dimer_descriptors_includes_intramolecular_pairs_current_shape_mismatch():
    coords_a = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    coords_b = np.array([[0.0, 2.0, 0.0], [1.5, 2.0, 0.0]])

    with pytest.raises(ValueError, match="inhomogeneous shape"):
        agg.create_dimer_descriptors(
            coords_a,
            ["H", "C"],
            np.array([0.2, -0.1]),
            coords_b,
            ["O", "N"],
            np.array([-0.2, 0.3]),
            include_bessel=False,
            intermolecular_only=False,
            features="2key",
        )


def test_create_dimer_descriptors_intermolecular_only_false_skips_zero_intra_distance():
    coords_a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    coords_b = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    desc, info = agg.create_dimer_descriptors(
        coords_a,
        ["H", "H"],
        np.array([0.2, -0.1]),
        coords_b,
        ["O", "O"],
        np.array([-0.2, 0.3]),
        include_bessel=False,
        intermolecular_only=False,
        features="full",
    )

    # Intramolecular A-A/B-B are skipped due to zero distance, so only 4 A-B pairs remain.
    assert len(info) == 4
    assert desc.shape == (4, 8)


def test_create_dimer_descriptors_intermolecular_with_bessel_hits_intramol_bessel_paths():
    coords_a = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    coords_b = np.array([[0.0, 2.0, 0.0], [1.5, 2.0, 0.0]])

    with pytest.raises(ValueError, match="inhomogeneous shape"):
        agg.create_dimer_descriptors(
            coords_a,
            ["H", "C"],
            np.array([0.2, -0.1]),
            coords_b,
            ["O", "N"],
            np.array([-0.2, 0.3]),
            include_bessel=True,
            intermolecular_only=False,
            features="full",
        )


def test_reader_helpers_skip_invalid_rows_and_missing_paths(tmp_path):
    xyz = tmp_path / "bad.xyz"
    xyz.write_text("2\ncomment\nH 0.0 0.0 0.0\nC x y z\n")
    elems, coords = agg.read_xyz_geometry(xyz)
    assert elems == ["H"]
    assert coords.shape == (1, 3)

    wfn = tmp_path / "bad.wfn"
    wfn.write_text("line1\nline2\n Cl X (CENTRE 1.0 bad 3.0 0.0\n Br X (CENTRE 2.0 3.0 4.0 0.0\n")
    elems_w, coords_w = agg.read_wfn_geometry(wfn)
    assert elems_w == ["Br"]
    assert coords_w.shape == (1, 3)

    ch = tmp_path / "bad.charges"
    ch.write_text("\nnot_a_number\n0.1\n")
    charges = agg.read_xtb_charges(ch)
    assert np.allclose(charges, np.array([0.1]))

    missing = tmp_path / "missing_cp.xyz"
    assert agg.read_cp_centers(missing).shape == (0, 3)

    cp_bad = tmp_path / "cp_bad.xyz"
    cp_bad.write_text("CP1 a b c\nCP2 1.0 2.0 3.0\n")
    centers = agg.read_cp_centers(cp_bad)
    assert centers.shape == (1, 3)


def test_cluster_feature_returns_none_when_pair_descriptors_empty(monkeypatch):
    monkeypatch.setattr(agg, "create_dimer_descriptors", lambda *args, **kwargs: (np.array([]), []))

    x = agg._cluster_feature_from_masks(
        np.array([[0.0, 0.0, 0.0]]), ["H"], np.array([0.1]),
        np.array([[1.0, 0.0, 0.0]]), ["O"], np.array([-0.1]),
        np.array([True]), np.array([True]), append_bias=True,
    )
    assert x is None


def test_build_cluster_descriptors_wfn_fallback_full_system(monkeypatch):
    monkeypatch.setattr(agg, "read_wfn_geometry", lambda p: (["H"], np.array([[0.0, 0.0, 0.0]])))
    monkeypatch.setattr(agg, "read_xtb_charges", lambda p: np.array([0.1]))
    monkeypatch.setattr(agg, "read_cp_centers", lambda p: np.zeros((0, 3)))
    monkeypatch.setattr(agg, "_cluster_feature_from_masks", lambda *args, **kwargs: np.array([9.0, 1.0]))

    x_clusters, centers = agg.build_cluster_descriptors(
        "m1.wfn",
        "m2.wfn",
        "cp.xyz",
        fallback_full_system=True,
        ispromol=False,
    )

    assert x_clusters.shape == (1, 2)
    assert centers.shape == (1, 3)
    assert np.isnan(centers[0]).all()


def test_build_cluster_descriptors_charge_mismatch_for_mol2(monkeypatch):
    def _read_xyz_geometry(path):
        if "m1" in str(path):
            return ["H"], np.array([[0.0, 0.0, 0.0]])
        return ["H", "O"], np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    monkeypatch.setattr(agg, "read_xyz_geometry", _read_xyz_geometry)
    monkeypatch.setattr(agg, "read_xtb_charges", lambda p: np.array([0.1]))
    monkeypatch.setattr(agg, "read_cp_centers", lambda p: np.array([[0.0, 0.0, 0.0]]))
    monkeypatch.setattr(agg, "_cluster_feature_from_masks", lambda *args, **kwargs: np.array([2.0, 1.0]))

    with pytest.raises(ValueError, match="Charge/atom mismatch for mol2"):
        agg.build_cluster_descriptors("m1.xyz", "m2.xyz", "cp.xyz")


def test_build_cluster_descriptors_raises_without_fallback(monkeypatch):
    monkeypatch.setattr(agg, "read_xyz_geometry", lambda p: (["H"], np.array([[0.0, 0.0, 0.0]])))
    monkeypatch.setattr(agg, "read_xtb_charges", lambda p: np.array([0.1]))
    monkeypatch.setattr(agg, "read_cp_centers", lambda p: np.zeros((0, 3)))

    with pytest.raises(ValueError, match="No valid cluster descriptors could be constructed"):
        agg.build_cluster_descriptors("m1.xyz", "m2.xyz", "cp.xyz", fallback_full_system=False)


def test_predict_cluster_sum_delta_onnx_model_path_uses_cpu_provider(monkeypatch, tmp_path):
    calls = []

    class _TrackingSession(_DummySession):
        def __init__(self, model_path, providers=None):
            calls.append((model_path, providers))

    fake_rt = types.SimpleNamespace(InferenceSession=_TrackingSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_rt)

    model_path = tmp_path / "dummy_model.onnx"
    pred = agg.predict_cluster_sum_delta_onnx(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        model_path=model_path,
        onnx_session=None,
    )

    assert pred == pytest.approx(10.0)
    assert len(calls) == 1
    assert calls[0][0] == str(model_path)
    assert calls[0][1] == ["CPUExecutionProvider"]


def test_shapley_cluster_attributions_onnx_model_path_with_provided_baseline(monkeypatch, tmp_path):
    calls = []

    class _TrackingSession(_DummySession):
        def __init__(self, model_path, providers=None):
            calls.append((model_path, providers))

    fake_rt = types.SimpleNamespace(InferenceSession=_TrackingSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_rt)

    model_path = tmp_path / "dummy_model.onnx"
    phi, total, baseline = agg.shapley_cluster_attributions_onnx(
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        model_path=model_path,
        onnx_session=None,
        n_perm=8,
        random_state=0,
        baseline=1.5,
    )

    assert len(calls) == 1
    assert calls[0][0] == str(model_path)
    assert calls[0][1] == ["CPUExecutionProvider"]
    assert total == pytest.approx(3.0)
    assert baseline == pytest.approx(1.5)
    assert phi.sum() == pytest.approx(total - baseline)


def test_calculate_charges_invokes_subprocess_when_missing(monkeypatch, tmp_path):
    m1 = tmp_path / "m1.xyz"
    m2 = tmp_path / "m2.xyz"
    m1.write_text("1\ncomment\nH 0 0 0\n")
    m2.write_text("1\ncomment\nH 0 0 1\n")

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        class _R:
            pass
        return _R()

    monkeypatch.setattr(agg.subprocess, "run", _fake_run)
    agg.calculate_charges(str(m1), str(m2), [0, 0])

    assert any(cmd[0] == "xtb" for cmd in calls)
    assert any(cmd[0] == "mv" for cmd in calls)
