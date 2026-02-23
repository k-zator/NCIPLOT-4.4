import numpy as np
import pytest
from pathlib import Path
import types

import spatial.calculate_NCI_energy as nci_energy


TEST_FILES = Path(__file__).resolve().parent / "test_files"


def _line(value: float) -> str:
    return f" value          :        {value:.8f}\n"


def _build_single_output(polar_vals, weak_vals, rep_vals):
    out = [" filler\n"] * 80
    out[5] = "               RANGE INTEGRATION DATA                                 \n"
    for i, v in enumerate(polar_vals):
        out[13 + i] = _line(v)
    for i, v in enumerate(weak_vals):
        out[38 + i] = _line(v)
    for i, v in enumerate(rep_vals):
        out[63 + i] = _line(v)
    return out


def _build_cluster_output(cluster_count, per_cluster_values):
    out = [" header\n"] * (4 + 35 * cluster_count)
    out[0] = f" Number of critical points found: {cluster_count}\n"
    out[1] = "      RANGE CLUSTERED INTEGRATION DATA over the volumes of rho^n      \n"

    cursor = 2
    for values in per_cluster_values:
        polar_vals, weak_vals, rep_vals = values
        for i, v in enumerate(polar_vals):
            out[cursor + 5 + i] = _line(v)
        for i, v in enumerate(weak_vals):
            out[cursor + 16 + i] = _line(v)
        for i, v in enumerate(rep_vals):
            out[cursor + 27 + i] = _line(v)
        cursor += 35

    return out


def test_calculate_energy_single_promol_non_supra_equation():
    ones = [1.0] * 7
    output = _build_single_output(ones, ones, ones)

    e_sum, e_polar, e_vdw = nci_energy.calculate_energy_single(output, ispromol=True, supra=False)

    assert e_polar == pytest.approx(-2787.099896, abs=1e-3)
    assert e_vdw == pytest.approx(29.438948, abs=1e-3)
    assert e_sum == pytest.approx(-2757.660948, abs=1e-3)


def test_calculate_energy_cluster_applies_sigma_hole_branch(monkeypatch):
    monkeypatch.setattr(nci_energy, "find_CP_Atom_matches", lambda *args, **kwargs: [[0, 1], [1, 2]])
    monkeypatch.setattr(nci_energy, "find_sigma_bond", lambda *args, **kwargs: [[0, 1]])

    ones = [1.0] * 7
    output = _build_cluster_output(
        cluster_count=2,
        per_cluster_values=[(ones, ones, ones), (ones, ones, ones)],
    )

    e_sum, e_polar, e_vdw = nci_energy.calculate_energy_cluster(
        output,
        ispromol=True,
        supra=False,
        mol1="mol1.xyz",
        mol2="mol2.xyz",
        filename="demo",
    )

    assert e_sum.shape == (2,)
    # Cluster 0 uses sigma-hole branch
    assert e_sum[0] == pytest.approx(-2133.9900227, abs=1e-3)
    # Cluster 1 uses normal branch
    assert e_sum[1] == pytest.approx(389.185, abs=1e-3)


def test_calculate_energy_single_parses_real_output_file():
    output_path = TEST_FILES / "nci_water--water.out"
    output = output_path.read_text().splitlines(keepends=True)

    e_sum, e_polar, e_vdw = nci_energy.calculate_energy_single(output, ispromol=True, supra=False)

    assert np.isfinite(e_sum)
    assert np.isfinite(e_polar)
    assert np.isfinite(e_vdw)
    assert e_sum == pytest.approx(e_polar + e_vdw, abs=1e-9)


class _DummySession:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def get_outputs(self):
        return [types.SimpleNamespace(name="output")]

    def run(self, _outs, feed):
        arr = np.asarray(next(iter(feed.values())))
        return [np.array([[[float(arr.sum())]]], dtype=np.float32)]


def test_charge_model_path_variants():
    p1 = nci_energy._charge_model_path(cluster=False, supra=False)
    p2 = nci_energy._charge_model_path(cluster=False, supra=True)
    p3 = nci_energy._charge_model_path(cluster=True, supra=False)
    p4 = nci_energy._charge_model_path(cluster=True, supra=True)
    assert p1.endswith("gbr_charge_small.onnx")
    assert p2.endswith("gbr_charge.onnx")
    assert p3.endswith("gbr_charge_small_cluster.onnx")
    assert p4.endswith("gbr_charge_cluster.onnx")


def test_calculate_charge_correction_with_mocked_onnx(monkeypatch):
    monkeypatch.setattr(nci_energy, "calculate_charges", lambda *args, **kwargs: None)
    monkeypatch.setattr(nci_energy, "read_charges", lambda *args, **kwargs: np.array([1.0, 2.0]))
    monkeypatch.setattr(nci_energy.rt, "InferenceSession", _DummySession)

    delta = nci_energy.calculate_charge_correction("m1.xyz", "m2.xyz", True, [0, 0], supra=False)
    assert delta == pytest.approx(3.0)


def test_calculate_charge_correction_cluster_with_shapley(monkeypatch):
    monkeypatch.setattr(nci_energy, "calculate_charges", lambda *args, **kwargs: None)
    monkeypatch.setattr(nci_energy, "build_cluster_descriptors", lambda *args, **kwargs: (np.array([[1.0, 1.0], [2.0, 0.0]]), np.array([[0, 0, 0], [1, 1, 1]])))
    monkeypatch.setattr(nci_energy.rt, "InferenceSession", _DummySession)
    monkeypatch.setattr(nci_energy, "predict_cluster_sum_delta_onnx", lambda X_clusters, onnx_session=None: float(np.asarray(X_clusters).sum()))
    monkeypatch.setattr(nci_energy, "shapley_cluster_attributions_onnx", lambda *args, **kwargs: (np.array([1.0, 1.0]), 4.0, 2.0))

    total, per_cluster = nci_energy.calculate_charge_correction_cluster(
        "m1.xyz",
        "m2.xyz",
        True,
        [0, 0],
        "cp.xyz",
        return_shapley=True,
    )
    assert total == pytest.approx(4.0)
    assert len(per_cluster) == 2
    assert per_cluster.sum() == pytest.approx(total)


@pytest.mark.parametrize(
    "ispromol,supra",
    [
        (True, True),
        (False, True),
        (False, False),
    ],
)
def test_calculate_energy_single_other_branches(ispromol, supra):
    ones = [1.0] * 7
    output = _build_single_output(ones, ones, ones)
    e_sum, e_polar, e_vdw = nci_energy.calculate_energy_single(output, ispromol=ispromol, supra=supra)
    assert np.isfinite(e_sum)
    assert np.isfinite(e_polar)
    assert np.isfinite(e_vdw)


def test_calculate_energy_cluster_wfn_branch(monkeypatch):
    monkeypatch.setattr(nci_energy, "find_CP_Atom_matches", lambda *args, **kwargs: [[0, 1]])
    monkeypatch.setattr(nci_energy, "find_sigma_bond", lambda *args, **kwargs: [[0, 1]])
    ones = [1.0] * 7
    output = _build_cluster_output(cluster_count=1, per_cluster_values=[(ones, ones, ones)])
    e_sum, e_polar, e_vdw = nci_energy.calculate_energy_cluster(
        output,
        ispromol=False,
        supra=False,
        mol1="mol1.wfn",
        mol2="mol2.wfn",
        filename="demo",
    )
    assert e_sum.shape == (1,)
    assert np.isfinite(e_sum[0])
