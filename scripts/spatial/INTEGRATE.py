#! /usr/bin/env python3
import numpy as np

INTEGRAL_POWERS = np.array([1, 1.5, 2, 2.5, 3, 4 / 3, 5 / 3, 0], dtype=float)
VERTEX_OFFSETS = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=np.int64,
)
POINT_CHUNK_SIZE = 200_000


def _as_rhorange_array(rhorange):
    ranges = np.asarray(rhorange, dtype=float)
    if ranges.size == 0:
        return np.empty((0, 2), dtype=float)
    if ranges.ndim == 1:
        if ranges.size != 2:
            raise ValueError("rhorange must contain bound pairs")
        return ranges.reshape(1, 2)
    if ranges.shape[-1] != 2:
        raise ValueError("rhorange must contain bound pairs")
    return ranges.reshape(-1, 2)


def _resolve_cluster_ids(labels, cluster_ids):
    if cluster_ids is None:
        return np.unique(labels)

    resolved = np.asarray(cluster_ids, dtype=np.asarray(labels).dtype)
    if resolved.ndim == 0:
        resolved = resolved.reshape(1)
    return resolved


def _map_cluster_positions(point_labels, cluster_ids):
    if point_labels.size == 0 or cluster_ids.size == 0:
        return np.empty(point_labels.shape, dtype=np.intp), np.zeros(point_labels.shape, dtype=bool)

    sort_order = np.argsort(cluster_ids)
    sorted_ids = cluster_ids[sort_order]
    insertion = np.searchsorted(sorted_ids, point_labels)

    valid = insertion < sorted_ids.size
    valid[valid] &= sorted_ids[insertion[valid]] == point_labels[valid]

    positions = np.full(point_labels.shape, -1, dtype=np.intp)
    positions[valid] = sort_order[insertion[valid]]
    return positions, valid


def _accumulate_range_integrals(result, range_index, cluster_positions, abs_density, region_mask, base_weight):
    if not np.any(region_mask):
        return

    range_positions = cluster_positions[region_mask]
    range_density = abs_density[region_mask]
    cluster_count = result.shape[0]

    for power_index, power in enumerate(INTEGRAL_POWERS):
        if power == 0:
            weights = np.full(range_positions.shape, base_weight, dtype=float)
        else:
            weights = np.power(range_density, power) * base_weight
        result[:, range_index, power_index] += np.bincount(
            range_positions,
            weights=weights,
            minlength=cluster_count,
        )


def _accumulate_cluster_integrals(
    index_chunks,
    densarray,
    dvol,
    labels,
    rhorange,
    cluster_ids=None,
    range_masks_flat=None,
):
    labels = np.asarray(labels)
    cluster_ids = _resolve_cluster_ids(labels, cluster_ids)
    result = np.zeros((cluster_ids.size, len(rhorange), len(INTEGRAL_POWERS)), dtype=float)
    if cluster_ids.size == 0 or len(rhorange) == 0:
        return cluster_ids, result

    dens_flat = densarray.reshape(-1)
    base_weight = dvol / 8.0

    for linear_indices in index_chunks:
        if linear_indices.size == 0:
            continue

        cluster_positions, valid = _map_cluster_positions(labels[linear_indices], cluster_ids)
        if not np.all(valid):
            linear_indices = linear_indices[valid]
            cluster_positions = cluster_positions[valid]

        if linear_indices.size == 0:
            continue

        dens_norm = dens_flat[linear_indices] / 100.0
        abs_density = np.abs(dens_norm)

        for range_index, bounds in enumerate(rhorange):
            if range_masks_flat is None:
                lowerbound, upperbound = bounds
                region_mask = (dens_norm > lowerbound) & (dens_norm < upperbound)
            else:
                region_mask = range_masks_flat[range_index][linear_indices]
            _accumulate_range_integrals(
                result,
                range_index,
                cluster_positions,
                abs_density,
                region_mask,
                base_weight,
            )

    return cluster_ids, result


def _iter_point_linear_index_chunks(point_mask, chunk_size=POINT_CHUNK_SIZE):
    point_indices = np.flatnonzero(point_mask.reshape(-1))
    for start in range(0, point_indices.size, chunk_size):
        yield point_indices[start : start + chunk_size]


def _iter_box_vertex_linear_index_chunks(active_box, grid, chunk_size=POINT_CHUNK_SIZE):
    active_shape = active_box.shape
    if 0 in active_shape:
        return

    box_indices = np.flatnonzero(active_box.reshape(-1))
    if box_indices.size == 0:
        return

    box_plane = active_shape[1] * active_shape[2]
    box_row = active_shape[2]
    ny, nz = grid[1], grid[2]

    for start in range(0, box_indices.size, chunk_size):
        chunk = box_indices[start : start + chunk_size]
        x = chunk // box_plane
        yz = chunk % box_plane
        y = yz // box_row
        z = yz % box_row

        linear_indices = (
            (x[:, None] + VERTEX_OFFSETS[None, :, 0]) * (ny * nz)
            + (y[:, None] + VERTEX_OFFSETS[None, :, 1]) * nz
            + (z[:, None] + VERTEX_OFFSETS[None, :, 2])
        )
        yield linear_indices.reshape(-1)


def _empty_active_box(shape):
    return np.zeros(tuple(max(dim - 1, 0) for dim in shape), dtype=bool)


def _build_promol_active_box(gradarray, rhoparam):
    if min(gradarray.shape) < 2:
        return _empty_active_box(gradarray.shape)

    grad_blocks = np.lib.stride_tricks.sliding_window_view(gradarray, (2, 2, 2))
    return np.any(grad_blocks < rhoparam, axis=(-3, -2, -1))


def _build_wfn_active_box(gradarray, densarray, rhoparam, rhocut):
    if min(gradarray.shape) < 2:
        return _empty_active_box(gradarray.shape)

    active_box = gradarray[:-1, :-1, :-1] < rhoparam
    rho_blocks = np.lib.stride_tricks.sliding_window_view(densarray / 100.0, (2, 2, 2))
    grad_blocks = np.lib.stride_tricks.sliding_window_view(gradarray, (2, 2, 2))
    refined_out = np.any((np.abs(rho_blocks) > rhocut) & (grad_blocks > 0.0), axis=(-3, -2, -1))
    return active_box & ~refined_out


def _build_active_vertex_mask(active_box, grid):
    active_vertices = np.zeros(grid, dtype=bool)
    if 0 in active_box.shape:
        return active_vertices

    for dx, dy, dz in VERTEX_OFFSETS:
        active_vertices[
            dx : dx + active_box.shape[0],
            dy : dy + active_box.shape[1],
            dz : dz + active_box.shape[2],
        ] |= active_box

    return active_vertices


def integrate_NCI_clusters(
    gradarray,
    densarray,
    grid,
    dvol,
    labels,
    rhoparam=2,
    promol=True,
    rhorange=[-0.2, -0.02],
    cluster_ids=None,
):
    rhorange = _as_rhorange_array(rhorange)

    if promol:
        active_box = _build_promol_active_box(gradarray, rhoparam)
        index_chunks = _iter_box_vertex_linear_index_chunks(active_box, grid)
    else:
        index_chunks = _iter_point_linear_index_chunks(gradarray < rhoparam)

    return _accumulate_cluster_integrals(
        index_chunks,
        densarray,
        dvol,
        labels,
        rhorange,
        cluster_ids=cluster_ids,
    )


def integrate_NCI_cluster(gradarray, densarray, grid, dvol, labels, cluster_id,rhoparam=2, promol=True, rhorange=[-0.2, -0.02]):
    """
    Range integration matching original fortran logic, but with cluster pre-selection.
    
    Args:
        gradarray: Full gradient array (same as original)
        densarray: Full density array (same as original) 
        grid: Grid dimensions (nx, ny, nz)
        dvol: Volume element
        labels: Nx1 array of cluster labels for each linearized grid point
        cluster_id: Which cluster to integrate (or None for all)
        l_large, l_small, rhoparam: Same as original
        promol: Same as original
    """
    requested_clusters = None if cluster_id is None else np.array([cluster_id])
    _, integrals = integrate_NCI_clusters(
        gradarray,
        densarray,
        grid,
        dvol,
        labels,
        rhoparam=rhoparam,
        promol=promol,
        rhorange=rhorange,
        cluster_ids=requested_clusters,
    )

    if cluster_id is None:
        return integrals.sum(axis=0)
    return integrals[0]


def integrate_NCI_clusters_wfn(
    gradarray,
    densarray,
    grid,
    dvol,
    labels,
    rhoparam=1.0,
    rhocut=0.2,
    rhorange=[[-0.2, -0.02]],
    cluster_ids=None,
):
    """
    Batch WFN-style integration that computes the grid-wide active region once and
    accumulates per-cluster integrals in a single pass.
    """
    rhorange = _as_rhorange_array(rhorange)
    active_box = _build_wfn_active_box(gradarray, densarray, rhoparam, rhocut)
    active_vertices = _build_active_vertex_mask(active_box, grid)
    dens_norm = densarray / 100.0
    grad_positive = gradarray > 0.0

    range_masks_flat = []
    for bounds in rhorange:
        lowerbound, upperbound = sorted(bounds)
        point_in_range = active_vertices & (dens_norm > lowerbound) & (dens_norm < upperbound) & grad_positive
        range_masks_flat.append(point_in_range.reshape(-1))

    return _accumulate_cluster_integrals(
        _iter_box_vertex_linear_index_chunks(active_box, grid),
        densarray,
        dvol,
        labels,
        rhorange,
        cluster_ids=cluster_ids,
        range_masks_flat=range_masks_flat,
    )


def integrate_NCI_cluster_wfn(
    gradarray,
    densarray,
    grid,
    dvol,
    labels,
    cluster_id,
    rhoparam=1.0,
    rhocut=0.2,
    rhorange=[[-0.2, -0.02]],
):
    """
    Exact WFN-style reconstruction of nciplot.f90 range integration:
      - rebuild final active-box mask from written cubes
      - build per-range point mask from active boxes
      - integrate with dataGeom_points semantics
    """
    requested_clusters = None if cluster_id is None else np.array([cluster_id])
    _, integrals = integrate_NCI_clusters_wfn(
        gradarray,
        densarray,
        grid,
        dvol,
        labels,
        rhoparam=rhoparam,
        rhocut=rhocut,
        rhorange=rhorange,
        cluster_ids=requested_clusters,
    )

    if cluster_id is None:
        return integrals.sum(axis=0)
    return integrals[0]