import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from mpl_toolkits.basemap import Basemap


def _get_cascadia_box():
    cascadia_box = [40 - 1.5, 51.8 - 0.2, -128.3 - 2.5, -121 + 2]  # min/max_latitude, min/max_longitude
    return cascadia_box


def isodepth_label_fmt(x):
    return rf"{int(x)} km"


def init_basemap_cascadia(axis, projection='lcc', mapscale=True, draw_meridians=True, draw_parallels=True,
                          draw_less_meridians=False):
    cascadia_box = _get_cascadia_box()
    cascadia_map = Basemap(llcrnrlon=cascadia_box[2], llcrnrlat=cascadia_box[0],
                           urcrnrlon=cascadia_box[3], urcrnrlat=cascadia_box[1],
                           lat_0=cascadia_box[0], lon_0=cascadia_box[2],
                           resolution='i', projection=projection, ax=axis)

    cascadia_map.drawcoastlines(linewidth=0.5)
    cascadia_map.fillcontinents(color='bisque', lake_color='lightcyan')  # burlywood
    cascadia_map.drawmapboundary(fill_color='lightcyan')
    if mapscale:
        cascadia_map.drawmapscale(-122.5, 51.05, -122.5, 51.05, 300, barstyle='fancy', zorder=10)
    if draw_parallels:
        cascadia_map.drawparallels([38, 40, 42, 44, 46, 48, 50], labels=[1, 0, 0, 0], linewidth=0.1)
    if draw_meridians:
        if not draw_less_meridians:
            cascadia_map.drawmeridians([-122, -124, -126, -128, -130], labels=[0, 0, 0, 1], linewidth=0.1)
        else:
            cascadia_map.drawmeridians([-122, -126, -130], labels=[0, 0, 0, 1], linewidth=0.1)
    cascadia_map.readshapefile('../../DATA/common/plates/tectonicplates-master/PB2002_boundaries', '',
                               linewidth=0.3)
    return cascadia_map

def get_proj_cascadia():
    # Create a Lambert Conformal projection centered on Cascadia
    cascadia_box = _get_cascadia_box()
    proj = ccrs.LambertConformal(central_longitude=cascadia_box[2],
                                 central_latitude=(cascadia_box[0] + cascadia_box[1]) / 2)
    return proj

def init_cartopy_cascadia(ax, resolution='10m', mapscale=True, draw_meridians=True, draw_parallels=True):
    """
    Initialize the Cartopy map for Cascadia.

    Parameters:
      ax (GeoAxes): A Matplotlib GeoAxes to draw on.
      mapscale (bool): Whether to add a map scale (custom implementation needed).
      draw_meridians (bool): Whether to draw meridians.
      draw_parallels (bool): Whether to draw parallels.

    Returns:
      ax (GeoAxes): The configured GeoAxes.
    """
    # Get bounding box: assume _get_cascadia_box() returns [lat_min, lat_max, lon_min, lon_max]
    cascadia_box = _get_cascadia_box()  # e.g., [lat_min, lat_max, lon_min, lon_max]

    # Set extent in PlateCarree coordinates: [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent([cascadia_box[2], cascadia_box[3], cascadia_box[0], cascadia_box[1]],
                  crs=ccrs.PlateCarree())

    # Add coastlines and country borders
    ax.coastlines(resolution=resolution, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    # Add land and ocean features
    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.OCEAN, facecolor='lightcyan')

    # Optionally add gridlines for parallels and meridians
    gl = ax.gridlines(draw_labels=True, linewidth=0.1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Map scale: Cartopy doesn't provide a built-in map scale, so you may need a custom function.
    if mapscale:
        # Optionally, add a custom map scale (here we just print a note)
        ax.text(0.05, 0.05, "Scale bar here", transform=ax.transAxes, fontsize=8)

    # Add tectonic plate boundaries from shapefile.
    shp_path = '../../DATA/common/plates/tectonicplates-master/PB2002_boundaries.shp'
    reader = shpreader.Reader(shp_path)
    geometries = list(reader.geometries())
    ax.add_geometries(geometries, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=0.3)

    return ax
