import os
import sys

import matplotlib
from torch.utils.data import DataLoader as TorchDataLoader

from config_files.plotting_style import set_matplotlib_style

from sse_generator_v2.synthetic_sse_cascadia import cosinusoidal_rise_time
from utility_methods import read_from_slab2, detrend_nan_v2

from sse_denoiser.dataset_utils import Dataset
from sse_denoiser.stagrnn import STAGRNNDenoiser
from sse_generator_v2.artificial_noise_cascadia import generate_artificial_noise
import matplotlib.pyplot as plt
import numpy as np
from sse_generator_v2.okada import forward as okada85
from utils.slab_plot_functions import init_basemap_cascadia

np.random.seed(42)

set_matplotlib_style()


def _generate_dislocation(station_coordinates, epi_lat=47.5, epi_lon=-123.5, depth_variability=0, **params):
    """Generates one set of synthetic displacements by applying Okada model (Okada, 1985) on an event placed on
    the slab (slab2) of the Cascadia subduction, registered at a given set of stations. The slab information
    is passed in input as {depth_list}, {strike_list} and {dip_list}"""
    depth_list, strike_list, dip_list, _ = read_from_slab2()
    idx_slab = np.logical_and(depth_list[:, 0] == epi_lon, depth_list[:, 1] == epi_lat)
    hypo_depth = - depth_list[idx_slab, 2]  # - depth_list[random_idx_slab, 2]
    if hypo_depth > 14.6:
        hypo_depth = hypo_depth + depth_variability
    if hypo_depth < 0:
        raise Exception('Negative depth')
    print('depth:', hypo_depth)
    strike = strike_list[idx_slab, 2]  # strike_list[random_idx_slab, 2]
    dip = dip_list[idx_slab, 2]  # dip_list[random_idx_slab, 2]
    rake = 90  # 75 + 25 * params['uniform_vector'][i, 1]  # rake from 75 to 100 deg
    u = 150  # mm
    L = 100  # km
    W = 20  # km

    displacement = okada85((station_coordinates[:, 1] - epi_lon) * 111.3194,
                           (station_coordinates[:, 0] - epi_lat) * 111.3194, 0, 0,
                           hypo_depth + W / 2 * np.sin(np.deg2rad(dip)), L, W, u, 0, strike, dip, rake)
    return displacement, epi_lat, epi_lon, hypo_depth, strike, dip, rake, u, L, W


def generate_fast_okada_time_series(synthetic_displacement, window_length, event_center_time, duration):
    transients = np.zeros((len(station_codes), window_length, 2))

    transient_time_array = np.linspace(0, window_length, window_length)

    for station in range(len(station_codes)):
        for direction in range(2):
            transient = cosinusoidal_rise_time(transient_time_array,
                                               synthetic_displacement[direction][station],
                                               duration,
                                               event_center_time)
            transients[station, :, direction] += transient
    return transients


def _draw_table_data(data_window, time_span, station_coordinates, every_nth=17, title='', figtitle=None, vmin=None,
                     vmax=None, colorbar=False, ylabel=True, folder='fig_synth_data'):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    latsort = np.argsort(station_coordinates[:, 0])[::-1]
    cmap = matplotlib.colormaps.get_cmap("RdBu_r")
    if vmin is None:
        vmin = -5
    if vmax is None:
        vmax = 10
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    cmap.set_bad('black', 1.)
    matshow_extent = [time_span[0], time_span[1], station_coordinates[:, 0].min(),
                      station_coordinates[:, 0].max()]
    mat = ax.matshow(data_window[latsort, :, 0], cmap=cmap, norm=norm, aspect='auto',  # , aspect='auto',
                     extent=matshow_extent)
    if ylabel:
        ax.set_ylabel('Latitude [°]', labelpad=7)
    ax.set_xlabel('Time [days]')
    ax.set_title(title)
    ax.set_yticks(np.around(station_coordinates[latsort, 0], decimals=1))
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
            ax.yaxis.get_major_ticks()[n].set_visible(False)
    if colorbar:
        cbar = fig.colorbar(mat, ax=ax)
        cbar.ax.set_ylabel('E-W displacement [mm]', rotation=270, labelpad=17)

    # plt.tight_layout()
    # plt.savefig(f'{os.path.join(folder, figtitle)}.pdf', bbox_inches='tight')
    plt.show()
    # plt.close(fig)


def plot_static_disp(station_coordinates, synthetic_displacement, draw_disp_field=True, draw_gnss_network=True,
                     draw_dislocation=True,
                     dislocation_parameters=None):
    figure, axis = plt.subplots(1, 1, figsize=(14, 7), dpi=100, sharex='all', sharey='all')
    cascadia_map = init_basemap_cascadia(axis)
    if draw_disp_field:
        max_length = 5 # np.max(np.sqrt(synthetic_displacement[:, 0] ** 2 + synthetic_displacement[:, 1] ** 2))
        # concatenate unit arrow
        unit_arrow_x, unit_arrow_y = -127., 39.5
        x = np.concatenate((station_coordinates[:, 1], [unit_arrow_x]))
        y = np.concatenate((station_coordinates[:, 0], [unit_arrow_y]))
        disp_x = np.concatenate((synthetic_displacement[:, 0], [-max_length]))
        disp_y = np.concatenate((synthetic_displacement[:, 1], [0.]))

        cascadia_map.quiver(x, y, disp_x, disp_y, color='black', latlon=True, width=0.0035)

        label_pos_x, label_pos_y = cascadia_map(unit_arrow_x - 1.45, unit_arrow_y + 0.25)

        plt.annotate(f'{np.around(max_length, decimals=1)}mm', xy=(unit_arrow_x, unit_arrow_y), xycoords='data',
                     xytext=(label_pos_x, label_pos_y), color='black')
    if draw_dislocation:
        x_center = dislocation_parameters['epi_lon']
        y_center = dislocation_parameters['epi_lat']
        strike = np.deg2rad(dislocation_parameters['strike'])
        dip = np.deg2rad(dislocation_parameters['dip'])
        L = dislocation_parameters['L'] / 111
        W = dislocation_parameters['W'] / 111

        alpha = np.pi / 2 - strike
        x_fault = L / 2 * np.cos(alpha) * np.array([-1., 1., 1., -1.]) + np.sin(alpha) * np.cos(dip) * W / 2 * np.array(
            [-1., -1., 1., 1.])
        y_fault = L / 2 * np.sin(alpha) * np.array([-1., 1., 1., -1.]) + np.cos(alpha) * np.cos(dip) * W / 2 * np.array(
            [1., 1., -1., -1.])
        x_fault += x_center
        y_fault += y_center
        x_fault_map, y_fault_map = cascadia_map(x_fault, y_fault)
        # x_fault_map, y_fault_map = x_fault, y_fault
        axis.add_patch(matplotlib.patches.Polygon(np.vstack((x_fault_map, y_fault_map)).T,
                                                  closed=True, fill=True, color='black', lw=1.5,
                                                  alpha=.4))
    if draw_gnss_network:
        cascadia_map.scatter(station_coordinates[:, 1], station_coordinates[:, 0], marker='^', s=15, color='C3',
                             latlon=True, edgecolors='black', linewidth=0.7)
    return figure


if __name__ == '__main__':
    reference_period = (2007, 2023)
    n_cores = 12
    noise_windows, station_codes, station_coordinates = generate_artificial_noise(n_cores,
                                                                                  200,
                                                                                  60,
                                                                                  reference_period,
                                                                                  p=0.,
                                                                                  add_trend=True,
                                                                                  parallel=False)
    noise = noise_windows[5]  # exclude first one because of np.roll
    # noise[noise == 0.] = np.nan
    detrended_noise, trend_params = detrend_nan_v2(np.linspace(0, 60, 60), noise)
    okada_disp, epi_lat, epi_lon, hypo_depth, strike, dip, rake, slip, L, W = _generate_dislocation(station_coordinates)
    # okada disp is a tuple
    okada_ts = generate_fast_okada_time_series(okada_disp, 60, 30, 1)
    # compute stress drop
    shear_modulus = 30e9
    area = L * W * 1e6  # m^2
    equiv_radius = np.sqrt(area / np.pi)
    stress_drop = shear_modulus * slip * 1e-3 / equiv_radius
    print('stress_drop', stress_drop * 1e-6, 'MPa')
    exit(0)
    dislocation_dict = {'epi_lat': epi_lat, 'epi_lon': epi_lon, 'strike': strike, 'dip': dip, 'L': L, 'W': W}
    fig = plot_static_disp(station_coordinates, okada_ts[:, -1, :], draw_disp_field=True, dislocation_parameters=dislocation_dict)
    plt.savefig('figures/supplement/synth_disloc.pdf', bbox_inches='tight')

    # _draw_table_data(detrended_noise, [0, 60], station_coordinates, colorbar=True)
    #_draw_table_data(okada_ts, [0, 60], station_coordinates, colorbar=True, title='Okada displacement')
    synthetic_ts = okada_ts + noise

    synthetic_ts_detrended, _ = detrend_nan_v2(np.linspace(0, 60, 60), synthetic_ts)

    n_stations = station_coordinates.shape[0]
    mean = np.nanmean(synthetic_ts, axis=1, keepdims=True)
    synthetic_ts = synthetic_ts - mean  # de-meaning

    test_dataset = Dataset(synthetic_ts[np.newaxis, ...], okada_ts[np.newaxis, ...])

    test_loader = TorchDataLoader(test_dataset, batch_size=1, shuffle=False)

    residual = False
    residual2 = False
    residual3 = False
    batch_size = 1

    params = {'n_stations': n_stations,
              'window_length': 60,
              'n_directions': 2,
              'batch_size': 1,
              'learning_rate': 0.001,
              'y_test': okada_ts[np.newaxis, ...],
              'verbosity': 1,
              'station_coordinates': station_coordinates[:, :2],
              'residual': residual,
              'residual2': residual2,
              'residual3': residual3,
              'learn_static': False}

    denoiser = STAGRNNDenoiser(**params)
    denoiser.build()
    denoiser.associate_optimizer()
    denoiser.set_data_loaders(test_loader, test_loader, test_loader)

    # denoiser.summary_nograph(test_loader.dataset[:batch_size][0])

    # weight_path = 'weights/best_cascadia_02Jul2023-013725_train_denois_realgaps_v5_STAGRNN_no_CNN_old_sd.pt'
    weight_path = 'weights/best_cascadia_09Feb2024-103329_train_denois_realgaps_v7_30disloc_slip5_250_lr0.001_bs128_demean.pt'
    denoiser.load_weights(weight_path)

    pred = denoiser.inference()
    print('pred.shape', pred.shape)
    denoised_ts = pred[0]
    # detrended_denoised_ts, _ = detrend_nan_v2(np.linspace(0, 60, 60), denoised_ts)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex='all', sharey='all')
    a11, a12, a21, a22 = axes.flatten()
    a11.plot(detrended_noise[..., 0].T)
    a21.plot(okada_ts[..., 0].T)
    a12.plot((okada_ts + detrended_noise)[..., 0].T)
    a22.plot(denoised_ts[..., 0].T)
    a21.set_xlabel('Synthetic time [days]')
    a22.set_xlabel('Synthetic time [days]')
    a11.set_ylabel('E-W displacement [mm]')
    a21.set_ylabel('E-W displacement [mm]')
    a11.set_title('Artificial noise (detrended)')
    a21.set_title('Synthetic templates')
    a12.set_title('Synthetic time series (noise+templates)')
    a22.set_title('Denoised time series')
    plt.savefig('figures/supplement/synth_ts_denoised.pdf', bbox_inches='tight')

    '''plt.plot(denoised_ts[..., 0].T)
    plt.show()

    _draw_table_data(denoised_ts, [0, 60], station_coordinates, colorbar=True, title='denoised displacement')'''
