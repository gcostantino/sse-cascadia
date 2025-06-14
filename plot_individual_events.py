from datetime import datetime

import cartopy.crs as ccrs
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

from FUNCTIONS.functions_slab import UTM_GEO
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.date_parsing import ymd_decimal_year_lookup
from utils.slab_plot_functions import init_basemap_cascadia, init_cartopy_cascadia
from utils.slab_utils import read_depth_from_slab2


def modified_turbo():
    cmap = matplotlib.colormaps.get_cmap('turbo')
    colors = cmap(np.linspace(0, 1, 256))
    light_grey = np.array([0.95, 0.95, 0.95, .5])
    n_blend = 10
    for i in range(n_blend):
        blend_factor = i / (n_blend - 1)  # 0 for i=0, 1 for i=n_blend-1
        colors[i] = (1 - blend_factor) * light_grey + blend_factor * colors[i]
    cmap = ListedColormap(colors)
    return cmap


def pre_utm_conversion(geometry):
    tot_data = []
    for ii in range(len(geometry[:, 0])):
        x_geo, y_geo = UTM_GEO(geometry[ii, [12, 15, 18, 12]], geometry[ii, [13, 16, 19, 13]])
        x_fill, y_fill = UTM_GEO(geometry[ii, [12, 15, 18]], geometry[ii, [13, 16, 19]])
        tot_data.append([x_geo, y_geo, x_fill, y_fill])
    return tot_data


def plot_slip_subduction(figure, axis, slip, geometry, admissible_depth, utm_conv, coords, return_cbar=False,
                         color_code_time=None):
    """Expects slip in meters."""

    cascadia_map = init_basemap_cascadia(axis, mapscale=False, draw_meridians=False, draw_parallels=False)
    if color_code_time is None:
        # cmap = custom_blues_colormap(teal=False, reversed=True)  # plt.cm.get_cmap('Blues')
        cmap = modified_turbo()  # matplotlib.colormaps.get_cmap('turbo')
        norm = Normalize(vmin=np.min(slip), vmax=np.max(slip))
        # depth = - geometry[:, 11]
    else:
        # color_code_time is the time array
        cmap = matplotlib.colormaps.get_cmap('cividis')
        norm = Normalize(vmin=np.min(color_code_time), vmax=np.max(color_code_time))

    for ii in range(len(geometry[:, 0])):
        x_geo, y_geo, x_fill, y_fill = utm_conv[ii]
        cascadia_map.plot(x_geo, y_geo, 'k', linewidth=0.2, latlon=True, alpha=.1)

    if color_code_time is None:
        for ii in range(len(geometry[:, 0])):
            x_geo, y_geo, x_fill, y_fill = utm_conv[ii]
            x_fill, y_fill = cascadia_map(x_fill, y_fill)
            p, = axis.fill(x_fill, y_fill, color=cmap(norm(slip[ii])))
    else:
        # now slip is a matrix (time, patches)
        for t in range(len(color_code_time)):
            activated_patches = slip[t] != 0.
            for ii in range(len(geometry[:, 0][activated_patches])):
                x_geo, y_geo, x_fill, y_fill = utm_conv[ii]
                x_fill, y_fill = cascadia_map(x_fill, y_fill)
                s = axis.scatter(x_fill, y_fill, color=cmap(norm(color_code_time[t])))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Set an empty array to ensure the ScalarMappable has the correct norm

    # colorbar_axes = figure.add_axes([0.44, 0.18, 0.08, 0.015])  # [left, bottom, width, height]
    colorbar_axes = axis.inset_axes([0.2, 0.15, 0.05, 0.2])  # [left, bottom, width, height]
    cbar = figure.colorbar(sm, cax=colorbar_axes, orientation='vertical')
    # cbar.ax.set_xlabel('Total slip [mm]', fontsize=12)

    levels = [20, 40, 60]
    x_dep, y_dep = admissible_depth[:, 0], admissible_depth[:, 1]
    depth = admissible_depth[:, 2]
    x_dep_map, y_dep_map = cascadia_map(x_dep, y_dep)
    isodepth = axis.tricontour(x_dep_map, y_dep_map, -depth, levels=levels, colors='k', linewidths=0.7)
    if return_cbar:
        return cascadia_map, cbar
    return cascadia_map


def plot_slip_subduction_cartopy(figure, ax, slip, geometry, admissible_depth, utm_conv, coords):
    """
    Plot slip distribution and additional features on a Cascadia map using Cartopy.

    Parameters:
      figure (Figure): The Matplotlib Figure.
      ax (GeoAxes): The Cartopy GeoAxes to draw on.
      slip (array): Slip values (in meters).
      geometry (array): Geometry data.
      admissible_depth (array): Data for isodepth contours.
      utm_conv (array): Precomputed UTM conversion results: list of tuples (x_geo, y_geo, x_fill, y_fill).
      coords: Additional coordinates (if needed).
    """
    # Initialize the Cartopy map for Cascadia.
    ax = init_cartopy_cascadia(ax, mapscale=True, draw_meridians=True, draw_parallels=True)

    # Setup colormap and normalization.
    cmap = modified_turbo()
    norm = Normalize(vmin=np.min(slip), vmax=np.max(slip))

    # Plot each event polygon.
    for ii in range(len(geometry)):
        x_geo, y_geo, x_fill, y_fill = utm_conv[ii]
        # Optionally, if you need to convert x_fill, y_fill with cascadia_map (here, not necessary because our ax is already set)
        # Plot the contour lines (polygon edges).
        ax.plot(x_geo, y_geo, 'k-', linewidth=0.2, transform=ccrs.PlateCarree(), alpha=0.1)
        # Fill the polygon.
        ax.fill(x_fill, y_fill, color=cmap(norm(slip[ii])), transform=ccrs.PlateCarree())

    # Create a ScalarMappable for the colorbar.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for colorbar

    # Add a colorbar using inset_axes if needed.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="5%", height="20%", loc='lower left', borderpad=3)
    cbar = figure.colorbar(sm, cax=cax, orientation='vertical')

    # Plot isodepth contours using tricontour.
    levels = [20, 40, 60]
    x_dep, y_dep = admissible_depth[:, 0], admissible_depth[:, 1]
    depth = admissible_depth[:, 2]
    # Transform depth coordinates if needed.
    # Here assume x_dep, y_dep are lon, lat.
    ax.tricontour(x_dep, y_dep, -depth, levels=levels, colors='k', linewidths=0.7, transform=ccrs.PlateCarree())


if __name__ == '__main__':
    n_rows, n_cols = 3, 4
    mo_rate_scale = 1e17
    refine_durations = True
    se = SlowSlipEventExtractor()
    se.load_extracted_events_unfiltered()
    time_array = se.ddh.corrected_time

    date_lookup = ymd_decimal_year_lookup(from_decimal=True)
    admissible_depth, _ = read_depth_from_slab2(max_depth=100)
    utm_converted = pre_utm_conversion(se.ma.fault_geometry)

    for thresh in se.slip_thresholds:
        mo_rate_list_patches = se.get_moment_rate_events(thresh, refine_durations)
        slip_rate_list_patches = se.get_slip_rate_patches_events(thresh, refine_durations)
        mo_rate_list_all_events = [np.sum(mo_rate_list_patches[i], axis=1) for i in range(len(mo_rate_list_patches))]
        dates_idx = se.get_event_date_idx(thresh, refine_durations)
        patches = se.get_event_patches(thresh, refine_durations)

        n_events = len(mo_rate_list_patches)
        batch_size = n_rows * n_cols
        n_plots = n_events // batch_size + 1

        for n in range(n_plots):
            fig = plt.figure(figsize=(10, 15), dpi=100)
            outer_gs = gridspec.GridSpec(3, 4, wspace=0.3, hspace=0.3)
            for i in range(n_rows):
                for j in range(n_cols):
                    idx_ev = i * n_cols + j + n * batch_size
                    if idx_ev >= len(mo_rate_list_patches):
                        break
                    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i, j],
                                                                height_ratios=[1, 5], hspace=0.3)

                    ax_mo = plt.Subplot(fig, inner_gs[0])
                    ax_slip = plt.Subplot(fig, inner_gs[1])
                    # ax_slip = GeoAxes(fig, inner_gs[1], projection=get_proj_cascadia())
                    fig.add_subplot(ax_mo)
                    fig.add_subplot(ax_slip)

                    mo_rate = mo_rate_list_all_events[idx_ev]
                    tot_slip_event = np.sum(slip_rate_list_patches[idx_ev], axis=0)
                    ev_start_idx, ev_end_idx = dates_idx[idx_ev]
                    if ev_end_idx + 1 == len(time_array):
                        ev_end_idx -= 1  # deal with last day
                        mo_rate = mo_rate[:-1]
                    start_date_dec, end_date_dec = time_array[ev_start_idx], time_array[ev_end_idx + 1]
                    start_date, end_date = date_lookup[start_date_dec], date_lookup[end_date_dec]
                    start_year, start_month, start_day = start_date
                    end_year, end_month, end_day = end_date
                    # ev_time_array = np.arange(datetime(start_year, start_month, start_day), datetime(end_year, end_month, end_day), timedelta(days=1)).astype(datetime)
                    ev_dec_time_array = time_array[ev_start_idx: ev_end_idx + 1]
                    ev_time_array = [datetime(*date_lookup[date]) for date in ev_dec_time_array]

                    ax_mo.plot(ev_dec_time_array, mo_rate / mo_rate_scale)
                    cascadia_map = plot_slip_subduction(fig, ax_slip, tot_slip_event, se.ma.fault_geometry,
                                                        admissible_depth, utm_converted, None)

                    # Adjust the top subplot's position: shrink its width.
                    pos_top = ax_mo.get_position()
                    shrink = 0.035
                    new_pos_top = [pos_top.x0 + shrink / 2, pos_top.y0, pos_top.width - shrink, pos_top.height]
                    ax_mo.set_position(new_pos_top)
                    ax_mo.ticklabel_format(useOffset=False)
                    ax_mo.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
                    ax_mo.set_title(f'event #{idx_ev + 1}\nduration:{len(mo_rate)} days')
                    if j == 0:
                        ax_mo.set_ylabel('$\dot M_0$ (Nmd$^{-1})$\n($\\times10^{17}$)')
                        cascadia_map.drawparallels([38, 40, 42, 44, 46, 48, 50], labels=[1, 0, 0, 0], linewidth=0.1)
                    else:
                        cascadia_map.drawparallels([38, 40, 42, 44, 46, 48, 50], labels=[0, 0, 0, 0], linewidth=0.1)

                    if i == n_rows - 1:
                        cascadia_map.drawmeridians([-122, -126, -130], labels=[0, 0, 0, 1], linewidth=0.1)
                    else:
                        cascadia_map.drawmeridians([-122, -126, -130], labels=[0, 0, 0, 0], linewidth=0.1)
            # plt.tight_layout()
            plt.savefig(f'figures/all_events/thresh_{thresh}/all_events_{n + 1}.pdf', bbox_inches='tight')
            plt.close(fig)
