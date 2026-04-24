import numpy as np
import pytest

from spatial.INTEGRATE import integrate_NCI_cluster, integrate_NCI_cluster_wfn


def test_integrate_nci_cluster_filters_by_label_and_reports_volume():
    grid = (2, 2, 2)
    gradarray = np.zeros(grid)
    densarray = np.full(grid, 100.0)
    dvol = 8.0

    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    result = integrate_NCI_cluster(
        gradarray=gradarray,
        densarray=densarray,
        grid=grid,
        dvol=dvol,
        labels=labels,
        cluster_id=0,
        rhoparam=1.0,
        promol=False,
        rhorange=[[-2.0, 2.0]],
    )

    # 4 voxels in cluster 0, each contributes dvol/8 = 1 to the p=0 (volume) channel
    assert result.shape == (1, 8)
    assert result[0, 7] == pytest.approx(4.0, abs=1e-9)


def test_integrate_nci_cluster_none_cluster_uses_all_points():
    grid = (2, 2, 2)
    gradarray = np.zeros(grid)
    densarray = np.full(grid, 100.0)
    dvol = 8.0
    labels = np.zeros(8, dtype=int)

    result = integrate_NCI_cluster(
        gradarray=gradarray,
        densarray=densarray,
        grid=grid,
        dvol=dvol,
        labels=labels,
        cluster_id=None,
        rhoparam=1.0,
        promol=False,
        rhorange=[[-2.0, 2.0]],
    )

    assert result[0, 7] == pytest.approx(8.0, abs=1e-9)


def test_integrate_nci_cluster_promol_path_multi_range():
    grid = (3, 3, 2)
    gradarray = np.ones(grid)
    gradarray[0:2, 0:2, 0:2] = 0.0
    densarray = np.full(grid, 100.0)
    dvol = 8.0
    labels = np.zeros(np.prod(grid), dtype=int)

    result = integrate_NCI_cluster(
        gradarray=gradarray,
        densarray=densarray,
        grid=grid,
        dvol=dvol,
        labels=labels,
        cluster_id=0,
        rhoparam=0.5,
        promol=True,
        rhorange=[[-2.0, 2.0], [2.0, 3.0]],
    )

    assert result.shape == (2, 8)
    assert result[0, 7] > 0.0
    assert result[1, 7] == pytest.approx(0.0, abs=1e-9)


def test_integrate_nci_cluster_wfn_cluster_filter_and_reversed_bounds():
    grid = (2, 2, 2)
    gradarray = np.full(grid, 0.5)
    densarray = np.array([-10.0, -10.0, 50.0, 50.0, -10.0, -10.0, 50.0, 50.0]).reshape(grid)
    dvol = 8.0
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    result = integrate_NCI_cluster_wfn(
        gradarray=gradarray,
        densarray=densarray,
        grid=grid,
        dvol=dvol,
        labels=labels,
        cluster_id=0,
        rhoparam=1.0,
        rhocut=0.6,
        rhorange=[[0.0, -0.2], [0.2, 0.4]],
    )

    assert result.shape == (2, 8)
    # Four cluster-0 vertices are in [-0.2, 0.0], each contributes dvol/8 = 1.
    assert result[0, 7] == pytest.approx(4.0, abs=1e-9)
    assert result[0, 0] == pytest.approx(0.4, abs=1e-9)
    assert result[1, 7] == pytest.approx(0.0, abs=1e-9)


def test_integrate_nci_cluster_wfn_rhocut_refinement_removes_box():
    grid = (2, 2, 2)
    gradarray = np.full(grid, 0.5)
    densarray = np.full(grid, 30.0)  # |rho|=0.3 > rhocut, with positive gradient
    dvol = 8.0
    labels = np.zeros(np.prod(grid), dtype=int)

    result = integrate_NCI_cluster_wfn(
        gradarray=gradarray,
        densarray=densarray,
        grid=grid,
        dvol=dvol,
        labels=labels,
        cluster_id=None,
        rhoparam=1.0,
        rhocut=0.2,
        rhorange=[[-1.0, 1.0]],
    )

    assert np.allclose(result, 0.0)
