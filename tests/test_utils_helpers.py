from pathlib import Path
import numpy as np

import spatial.UTILS as utils


def _write_tiny_cube(path: Path, values):
    content = [
        "CUBE FILE\n",
        "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n",
        " 1 0.000000 0.000000 0.000000\n",
        " 2 1.000000 0.000000 0.000000\n",
        " 2 0.000000 1.000000 0.000000\n",
        " 1 0.000000 0.000000 1.000000\n",
        " 1 0.0 0.0 0.0 0.0\n",
        " " + " ".join(str(v) for v in values) + "\n",
    ]
    path.write_text("".join(content))


def test_read_cube_and_process_cube(tmp_path):
    base = tmp_path / "demo"
    _write_tiny_cube(tmp_path / "demo-dens.cube", [1.0, 2.0, 3.0, 4.0])
    _write_tiny_cube(tmp_path / "demo-grad.cube", [0.1, 0.2, 0.3, 0.4])

    header, pts, carray, atcoords = utils.read_cube(tmp_path / "demo-dens.cube")
    assert len(header) >= 7
    assert pts.shape == (2, 2, 1, 3)
    assert carray.shape == (2, 2, 1)
    assert atcoords.shape[0] == 1

    mrhos, dens, header2, grid, dvol = utils.process_cube(str(base))
    assert mrhos.shape[1] == 5
    assert dens.shape == (2, 2, 1)
    assert grid == (2, 2, 1)
    assert dvol == 1.0


def test_write_cube_and_cube_select_and_vmd(tmp_path):
    base = tmp_path / "demo"
    header = [
        "CUBE FILE\n",
        "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n",
        " 1 0.000000 0.000000 0.000000\n",
        " 2 1.000000 0.000000 0.000000\n",
        " 2 0.000000 1.000000 0.000000\n",
        " 1 0.000000 0.000000 1.000000\n",
        " 1 0.0 0.0 0.0 0.0\n",
    ]
    grid = (2, 2, 1)
    X = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 0.1],
            [1.0, 0.0, 0.0, 11.0, 0.2],
            [0.0, 1.0, 0.0, 12.0, 0.3],
            [1.0, 1.0, 0.0, 13.0, 0.4],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    utils.write_cube(str(base), 0, X, labels, header, grid)
    assert (tmp_path / "demo-cl1-grad.cube").exists()
    assert (tmp_path / "demo-cl1-dens.cube").exists()

    utils.write_cube_select(str(base), 1, X, labels, header, grid)
    assert (tmp_path / "demo-cl2-grad.cube").exists()
    assert (tmp_path / "demo-cl2-dens.cube").exists()

    utils.write_vmd(str(base), labels, isovalue=1.0)
    vmd = (tmp_path / "demo_divided.vmd").read_text()
    assert "mol new" in vmd
    assert "Isosurface" in vmd
