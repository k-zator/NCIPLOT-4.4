import numpy as np

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
    integral_powers = np.array([1, 1.5, 2, 2.5, 3, 4/3, 5/3, 0])
    nx, ny, nz = grid
    
    # First, do the original voxel selection (same as before)
    valid_xx, valid_yy, valid_zz = [], [], []
    
    if promol:
        for x in range(nx-1):
            for y in range(ny-1):
                for z in range(nz-1):
                    block = gradarray[x:x+2, y:y+2, z:z+2]
                    if np.any(block < rhoparam):
                        for dx in [0,1]:
                            for dy in [0,1]:
                                for dz in [0,1]:
                                    xx, yy, zz = x+dx, y+dy, z+dz
                                    valid_xx.append(xx)
                                    valid_yy.append(yy)
                                    valid_zz.append(zz)
    else:
        mask = gradarray < rhoparam
        valid_xx, valid_yy, valid_zz = np.where(mask)
    
    valid_xx = np.array(valid_xx)
    valid_yy = np.array(valid_yy)
    valid_zz = np.array(valid_zz)
    
    # Now filter by cluster membership
    if cluster_id is not None:
        # Convert 3D indices to linear indices (same as numpy's ravel/flatten)
        linear_indices = valid_xx * (ny * nz) + valid_yy * nz + valid_zz
        
        # Keep only points that belong to the specified cluster
        cluster_mask = labels[linear_indices] == cluster_id
        valid_xx = valid_xx[cluster_mask]
        valid_yy = valid_yy[cluster_mask]
        valid_zz = valid_zz[cluster_mask]
    
    # Rest is same as original
    dens_vals = densarray[valid_xx, valid_yy, valid_zz]
    dens_norm = dens_vals / 100
    
    # Region masks
    masks = []
    for r in range(len(rhorange)):
        masks.append((dens_norm > rhorange[r][0]) & (dens_norm < rhorange[r][1]))
    
    def region_integral(region_mask):
        vals = dens_vals[region_mask]
        return np.array([(np.power(np.abs(vals/100), p) * dvol / 8).sum() for p in integral_powers])
    
    sum_rhon_vol = []
    for m in masks:
        sum_rhon_vol.append(region_integral(m))

    return np.array(sum_rhon_vol)