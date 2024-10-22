from matplotlib.colors import LinearSegmentedColormap


def custom_blues_colormap(teal=False, reversed=False):
    # https://gka.github.io/palettes/#/9|s|00429d,96ffea,fdfdfd|ffffe0,ff005e,93003a|1|1
    # https://gka.github.io/palettes/#/9|s|005a74,3aa794,fdfdfd|ffffe0,ff005e,93003a|1|1
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
    # colors = ['#00429d', '#2e59a8', '#4771b2', '#5d8abd', '#73a2c6', '#8abccf', '#a5d5d8', '#c5eddf', '#ffffe0']
    if not teal:
        colors = ['#00429d', '#2e59a8', '#4771b3', '#5c89be', '#72a2c9', '#89bbd4', '#a3d4e0', '#c3eced', '#fdfdfd']
    else:
        colors = ['#005a74', '#1f6f7f', '#3a858b', '#559a9a', '#72aeaa', '#92c3bd', '#b3d7d0', '#d7eae6', '#fdfdfd']
    n_bins = 256
    cmap_name = 'my_blues'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors if not reversed else colors[::-1], N=n_bins)
    return cmap
