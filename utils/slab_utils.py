import numpy as np


def read_depth_from_slab2(max_depth=60, depth_range=None):
    """Reads the slab2 grid and returns depth, strike, dip, and positions of points on the slab
    such that the depth is less than 60 km."""
    longitude_correction = [-360, 0, 0]
    # depth_filename = 'INPUT_FILES/TRENCH_SLAB2_MODELS/cas_slab2_dep_02.24.18.xyz'
    depth_filename = '../../DATA/common/slab/slab2/cas_slab2_dep_02.24.18.xyz'
    depth_grid = np.loadtxt(depth_filename, delimiter=',') + longitude_correction

    if depth_range is None:
        ind_depth = np.where(depth_grid[:, 2] > - max_depth)[0]
    else:
        # 20 < depth < 40 --> -40 < depth < -20
        ind_depth = np.where(np.logical_and(depth_grid[:, 2] > -depth_range[1], depth_grid[:, 2] < -depth_range[0]))[0]

    region = [np.min(depth_grid[:, 1][ind_depth]), np.max(depth_grid[:, 1][ind_depth]),
              np.min(depth_grid[:, 0][ind_depth]),
              np.max(depth_grid[:, 0][ind_depth])]  # min/max_latitude, min/max_longitude

    return depth_grid[ind_depth], region
