from pathlib import Path

import numpy as np
import pytest

import NCIENERGY
import NCICLUSTER


TEST_FILES = Path(__file__).resolve().parent / "test_files"


def _default_energy_opts():
    return {
        "oname": "output",
        "gamma": 0.85,
        "outer": 0.2,
        "inner": 0.02,
        "isovalue": 1.0,
        "intermol": True,
        "ispromol": True,
        "cluster": False,
        "supra": False,
        "mol1": None,
        "mol2": None,
        "total_charges": [0, 0],
        "use_charges": False,
    }


def test_ncienergy_main_no_args_returns_zero_with_help_options_stub(monkeypatch):
    monkeypatch.setattr(NCIENERGY, "options_energy_calc", lambda _opts: _default_energy_opts())
    rc = NCIENERGY.main([])
    assert rc == 0


def test_ncicluster_main_no_args_returns_zero_with_options_stub(monkeypatch):
    monkeypatch.setattr(
        NCICLUSTER,
        "options_dict",
        lambda _opts: {
            "isovalue": 1.0,
            "ispromol": True,
            "range": np.array([[-0.2, -0.02]]),
            "verbose": False,
            "mol": ["mol1.xyz", "mol2.xyz"],
        },
    )
    rc = NCICLUSTER.main([])
    assert rc == 0


def test_ncienergy_strict_guard_exits_early(monkeypatch, capsys):
    monkeypatch.setattr(
        NCIENERGY,
        "options_energy_calc",
        lambda _opts: {**_default_energy_opts(), "gamma": 0.99},
    )

    rc = NCIENERGY.main(["input_names"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "NCIENERGY mode runs only with the default parameters" in out


def test_ncienergy_main_single_mode_success(monkeypatch, tmp_path, capsys):
    input_names = tmp_path / "input_names"
    input_names.write_text("water--water\n")

    monkeypatch.chdir(TEST_FILES)

    rc = NCIENERGY.main([
        str(input_names),
        "--oname",
        "water--water",
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "NCI energies / kJ/mol" in out
    assert "E_sum" in out


def test_ncienergy_main_errors_if_output_missing(monkeypatch, tmp_path, capsys):
    input_names = tmp_path / "input_names"
    input_names.write_text("water--water\n")

    monkeypatch.chdir(TEST_FILES)
    rc = NCIENERGY.main([
        str(input_names),
        "--oname",
        "definitely_missing_output",
    ])
    out = capsys.readouterr().out

    assert rc == 1
    assert "not found" in out


def test_ncienergy_main_dft_single_branch(monkeypatch, tmp_path, capsys):
    input_names = tmp_path / "input_names"
    input_names.write_text("demo\n")
    (tmp_path / "nci_output.out").write_text("dummy\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        NCIENERGY,
        "options_energy_calc",
        lambda _opts: {
            **_default_energy_opts(),
            "ispromol": False,
            "gamma": 0.75,
        },
    )
    monkeypatch.setattr(NCIENERGY, "calculate_energy_single", lambda *_args, **_kwargs: (4.0, 1.0, 3.0))

    rc = NCIENERGY.main([str(input_names)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Calculating energy using the DFT equations" in out


def test_ncienergy_main_cluster_with_charge_model(monkeypatch, tmp_path, capsys):
    input_names = tmp_path / "input_names"
    input_names.write_text("demo\n")
    (tmp_path / "nci_output.out").write_text("dummy\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        NCIENERGY,
        "options_energy_calc",
        lambda _opts: {
            **_default_energy_opts(),
            "cluster": True,
            "use_charges": True,
            "mol1": "m1.xyz",
            "mol2": "m2.xyz",
        },
    )
    monkeypatch.setattr(NCIENERGY, "calculate_charge_correction_cluster", lambda *_args, **_kwargs: (1.5, np.array([0.5, 1.0])))
    monkeypatch.setattr(NCIENERGY, "calculate_energy_cluster", lambda *_args, **_kwargs: (np.array([1.0, 2.0]), np.array([0.2, 0.3]), np.array([0.8, 1.7])))

    rc = NCIENERGY.main([str(input_names)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Summed-across-clusters integrals" in out
    assert "E_charge" in out


def test_ncienergy_main_dft_cluster_branch(monkeypatch, tmp_path, capsys):
    input_names = tmp_path / "input_names"
    input_names.write_text("demo\n")
    (tmp_path / "nci_output.out").write_text("dummy\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        NCIENERGY,
        "options_energy_calc",
        lambda _opts: {
            **_default_energy_opts(),
            "ispromol": False,
            "gamma": 0.75,
            "cluster": True,
        },
    )
    monkeypatch.setattr(NCIENERGY, "calculate_energy_cluster", lambda *_args, **_kwargs: (np.array([2.0]), np.array([0.5]), np.array([1.5])))

    rc = NCIENERGY.main([str(input_names)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Summed-across-clusters integrals" in out


def test_ncicluster_main_real_files_missing_grad_cube_raises(monkeypatch, tmp_path):
    input_names = tmp_path / "input_names"
    input_names.write_text("water--water\n")

    monkeypatch.chdir(TEST_FILES)
    with pytest.raises(FileNotFoundError):
        NCICLUSTER.main([
            str(input_names),
            "--mol",
            "water--water_1.xyz",
            "--mol",
            "water--water_2.xyz",
            "--range",
            "-0.20",
            "--range",
            "-0.02",
        ])


def test_ncicluster_main_runs_with_mocked_dependencies(monkeypatch, tmp_path):
    input_names = tmp_path / "input_names"
    input_names.write_text("demo\n")

    monkeypatch.setattr(
        NCICLUSTER,
        "options_dict",
        lambda _opts: {
            "isovalue": 1.0,
            "ispromol": True,
            "range": np.array([[-0.2, -0.02]]),
            "verbose": False,
            "mol": ["mol1.xyz", "mol2.xyz"],
        },
    )

    mrhos = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.1],
            [1.0, 0.0, 0.0, 1.0, 0.2],
            [0.0, 1.0, 0.0, 1.0, 0.3],
            [1.0, 1.0, 0.0, 1.0, 0.4],
        ]
    )
    dens = np.ones((2, 2, 1))
    header = ["h1\n", "h2\n", "h3\n", "h4\n", "h5\n", "h6\n"]
    grid = (2, 2, 1)
    dvol = 1.0

    monkeypatch.setattr(NCICLUSTER, "process_cube", lambda _filename: (mrhos, dens, header, grid, dvol))
    monkeypatch.setattr(
        NCICLUSTER,
        "find_CP_with_gradient",
        lambda *_args, **_kwargs: [
            [np.array([0.0, 0.0, 0.0]), 1.0, 0.1],
            [np.array([1.0, 1.0, 0.0]), 1.0, 0.1],
        ],
    )
    monkeypatch.setattr(NCICLUSTER, "write_CPs_xyz", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(NCICLUSTER, "write_cube_select", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(NCICLUSTER, "write_vmd", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        NCICLUSTER,
        "integrate_NCI_clusters",
        lambda *_args, **_kwargs: (
            np.unique(_args[4]),
            np.ones((len(np.unique(_args[4])), 1, 8)),
        ),
    )

    rc = NCICLUSTER.main([str(input_names)])
    assert rc == 0
