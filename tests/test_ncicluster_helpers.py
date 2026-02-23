import NCICLUSTER


def test_parse_cli_no_args(capsys):
    input_name, options = NCICLUSTER._parse_cli([])
    out = capsys.readouterr().out
    assert input_name is None
    assert options == ["--help"]
    assert "Usage: NCICLUSTER.py" in out


def test_parse_cli_help():
    input_name, options = NCICLUSTER._parse_cli(["--help"])
    assert input_name is None
    assert options == ["--help"]


def test_parse_cli_normal_args():
    input_name, options = NCICLUSTER._parse_cli(["input_names", "--range", "-0.2"])
    assert input_name == "input_names"
    assert options == ["--range", "-0.2"]
