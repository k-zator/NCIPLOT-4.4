import numpy as np
import pytest

from spatial.OPT_DICT import options_dict, options_energy_calc


def test_options_dict_parses_range_pairs():
    parsed = options_dict([
        "--isovalue", "0.8",
        "--range", "-0.2",
        "--range", "-0.02",
        "--range", "-0.02",
        "--range", "0.02",
        "--mol", "mol1.xyz",
    ])

    assert parsed["isovalue"] == pytest.approx(0.8)
    assert parsed["range"].shape == (2, 2)
    assert np.allclose(parsed["range"], np.array([[-0.2, -0.02], [-0.02, 0.02]]))
    assert parsed["mol"] == ["mol1.xyz"]


def test_options_dict_rejects_odd_range_values():
    with pytest.raises(ValueError, match="must be provided in pairs"):
        options_dict(["--range", "-0.2", "--range", "-0.02", "--range", "0.1"])


def test_options_energy_calc_parses_charge_switches_and_totals():
    parsed = options_energy_calc([
        "--oname", "/tmp/my_output",
        "--clustering", "T",
        "--supra", "T",
        "--use_charges", "T",
        "--total_charges", "1,-1",
    ])

    assert parsed["oname"] == "my_output"
    assert parsed["cluster"] is True
    assert parsed["supra"] is True
    assert parsed["use_charges"] is True
    assert parsed["total_charges"] == [1, -1]


def test_options_dict_invalid_flag_raises():
    with pytest.raises(ValueError, match="not a valid option"):
        options_dict(["--bad", "x"])


def test_options_dict_invalid_verbose_value_raises():
    with pytest.raises(ValueError, match="not a valid option for -v"):
        options_dict(["-v", "maybe"])


def test_options_energy_calc_sets_false_flags_and_paths():
    parsed = options_energy_calc([
        "--intermol", "F",
        "--ispromol", "F",
        "--oname", "simple_name",
    ])
    assert parsed["intermol"] is False
    assert parsed["ispromol"] is False
    assert parsed["oname"] == "simple_name"


def test_options_dict_help_exits():
    with pytest.raises(SystemExit):
        options_dict(["--help", "x"])


def test_options_energy_calc_help_exits():
    with pytest.raises(SystemExit):
        options_energy_calc(["--help", "x"])


def test_options_energy_calc_parses_all_common_flags():
    parsed = options_energy_calc([
        "--isovalue", "0.9",
        "--oname", "/tmp/a.out",
        "--outer", "0.25",
        "--inner", "0.03",
        "--gamma", "0.75",
        "--intermol", "F",
        "--ispromol", "F",
        "--clustering", "T",
        "--supra", "T",
        "--mol1", "m1.xyz",
        "--mol2", "m2.xyz",
        "--total_charges", "1,-1",
        "--use_charges", "T",
    ])

    assert parsed["isovalue"] == pytest.approx(0.9)
    assert parsed["oname"] == "a.out"
    assert parsed["outer"] == pytest.approx(0.25)
    assert parsed["inner"] == pytest.approx(0.03)
    assert parsed["gamma"] == pytest.approx(0.75)
    assert parsed["intermol"] is False
    assert parsed["ispromol"] is False
    assert parsed["cluster"] is True
    assert parsed["supra"] is True
    assert parsed["mol1"] == "m1.xyz"
    assert parsed["mol2"] == "m2.xyz"
    assert parsed["total_charges"] == [1, -1]
    assert parsed["use_charges"] is True
