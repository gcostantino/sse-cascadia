from mpl_toolkits.basemap import Basemap


def _get_cascadia_box():
    cascadia_box = [40 - 1.5, 51.8 - 0.2, -128.3 - 2.5, -121 + 2]  # min/max_latitude, min/max_longitude
    return cascadia_box


def isodepth_label_fmt(x):
    return rf"{int(x)} km"


def init_basemap_cascadia(axis):
    cascadia_box = _get_cascadia_box()
    cascadia_map = Basemap(llcrnrlon=cascadia_box[2], llcrnrlat=cascadia_box[0],
                           urcrnrlon=cascadia_box[3], urcrnrlat=cascadia_box[1],
                           lat_0=cascadia_box[0], lon_0=cascadia_box[2],
                           resolution='i', projection='lcc', ax=axis)

    cascadia_map.drawcoastlines(linewidth=0.5)
    cascadia_map.fillcontinents(color='bisque', lake_color='lightcyan')  # burlywood
    cascadia_map.drawmapboundary(fill_color='lightcyan')
    cascadia_map.drawmapscale(-122.5, 51.05, -122.5, 51.05, 300, barstyle='fancy', zorder=10)

    cascadia_map.drawparallels([38, 40, 42, 44, 46, 48, 50], labels=[1, 0, 0, 0], linewidth=0.1)
    cascadia_map.drawmeridians([-122, -124, -126, -128, -130], labels=[0, 0, 0, 1], linewidth=0.1)
    cascadia_map.readshapefile('../../DATA/common/plates/tectonicplates-master/PB2002_boundaries', '',
                               linewidth=0.3)
    return cascadia_map
