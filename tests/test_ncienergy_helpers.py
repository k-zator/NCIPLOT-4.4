import NCIENERGY


def _default_opts():
    return {
        "isovalue": 1.0,
        "outer": 0.2,
        "inner": 0.02,
        "intermol": True,
        "gamma": 0.85,
        "ispromol": True,
    }


def test_parse_cli_no_args(capsys):
    input_name, options = NCIENERGY._parse_cli([])
    out = capsys.readouterr().out
    assert input_name is None
    assert options == ["--help"]
    assert "Usage: NCIENERGY.py" in out


def test_parse_cli_help():
    input_name, options = NCIENERGY._parse_cli(["--help"])
    assert input_name is None
    assert options == ["--help"]


def test_parse_cli_normal_args():
    input_name, options = NCIENERGY._parse_cli(["input_names", "--oname", "x"])
    assert input_name == "input_names"
    assert options == ["--oname", "x"]


def test_validate_default_parameters_true():
    assert NCIENERGY._validate_default_parameters(_default_opts()) is True


def test_validate_default_parameters_rejects_core_mismatch(capsys):
    opts = _default_opts()
    opts["outer"] = 0.3
    assert NCIENERGY._validate_default_parameters(opts) is False
    out = capsys.readouterr().out
    assert "NCIENERGY mode runs only with the default parameters" in out
    assert "RANGE 3" not in out


def test_validate_default_parameters_rejects_gamma_mismatch(capsys):
    opts = _default_opts()
    opts["gamma"] = 0.99
    assert NCIENERGY._validate_default_parameters(opts) is False
    out = capsys.readouterr().out
    assert "RANGE 3" in out


def test_print_default_parameter_message_includes_range_variant(capsys):
    NCIENERGY._print_default_parameter_message(include_range_n=True)
    out = capsys.readouterr().out
    assert "RANGE 3" in out
