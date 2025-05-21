import numpy as np

def little_integral(d, power):
    dvol = 0.188973**3
    return np.power(abs(d/100),power)*dvol/8

def integrate_NCI(gradarray, densarray, grid, l_large = 0.2, l_small = 0.02, rhoparam=2):
    """Integration scheme analogous to NCIPLOT's - over space and each grid point as averaged cube"""
    sum_rhon_vol_polar = [0]*8
    sum_rhon_vol_vdw = [0]*8
    sum_rhon_vol_rep = [0]*8
    integral_powers = [1,1.5,2,2.5,3,4/3,5/3,0]
    nx, ny, nz = grid

    for x in range(0, nx-1):
        for y in range(0, ny-1):
            for z in range(0, nz-1):
                if gradarray[x,y,z] < rhoparam+0.1: # picking correct isosurface
                    for xx in range(x, x+2): # three more for loops to average over integrated point as cube
                        for yy in range(y, y+2): # that includes neighbouring points
                            for zz in range(z, z+2):
                                integrals = [little_integral(densarray[xx,yy,zz],n) for n in integral_powers]
                                if -l_large < densarray[xx,yy,zz]/100 < -l_small:
                                    sum_rhon_vol_polar = np.add(sum_rhon_vol_polar, integrals)
                                elif -l_small <= densarray[xx,yy,zz]/100 < 0:
                                    sum_rhon_vol_vdw = np.add(sum_rhon_vol_vdw, integrals)
                                elif 0 < densarray[xx,yy,zz]/100 < l_large:
                                    sum_rhon_vol_rep = np.add(sum_rhon_vol_rep, integrals)
    return np.array([sum_rhon_vol_polar, sum_rhon_vol_vdw, sum_rhon_vol_rep])