import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
from scipy.signal import detrend
from sklearn.cluster import DBSCAN

from utils import _preliminary_operations, cascadia_filtered_stations, detrend_nan_v2, tremor_catalogue


def generate_denoised_time_series():
    reference_period = (2006, 2023.4)
    window_length = 60
    n_selected_stations = 200

    with np.load('predictions/pred_SSEdenoiser_v7_30disloc_slip5_1000_noise_with_trend_demean_2006_2024.npz') as f:
        full_denoised_real_data = f['pred']

    full_denoised_real_data = np.gradient(full_denoised_real_data, axis=2)

    selected_gnss_data, selected_time_array, _, _ = _preliminary_operations(reference_period, detrend=False)

    # we take a sliding window and we average the displacement over time
    offset_cut = 20
    averaged_continuous_denoised_data = np.zeros(
        (selected_time_array.shape[0] - window_length - 2 * offset_cut, n_selected_stations, 2))

    # for i in range(full_denoised_real_data.shape[0]):
    for i in range(averaged_continuous_denoised_data.shape[0] - offset_cut):
        averaged_continuous_denoised_data[i:i + (window_length - 2 * offset_cut)] += np.transpose(
            full_denoised_real_data[i, :, offset_cut:window_length - offset_cut, :], (1, 0, 2))

    num_windows_per_sample = np.zeros((len(averaged_continuous_denoised_data)))
    num_windows_per_sample[:(window_length - 2 * offset_cut)] = np.arange(1, (window_length - 2 * offset_cut + 1))
    num_windows_per_sample[-(window_length - 2 * offset_cut):] = np.arange(1, (window_length - 2 * offset_cut + 1))[
                                                                 ::-1]
    num_windows_per_sample[
    (window_length - 2 * offset_cut):-(window_length - 2 * offset_cut)] = window_length - 2 * offset_cut

    averaged_continuous_denoised_data = averaged_continuous_denoised_data / num_windows_per_sample[
        ..., np.newaxis, np.newaxis]
    return selected_time_array, averaged_continuous_denoised_data


def _latitude_time_plot(time, data, tremors, station_coordinates, latsort, tol=0.01, window_length=60, offset=0,
                        static=False, downsample_tremors=False, draw_tremors=True, tremor_alpha=1.,
                        data_pcolormesh=False, fig_path='denoising_figures/overall_lat_time.pdf', zoom=False):
    data[np.abs(data) < tol] = np.nan
    data[data > 0] = np.nan
    vmin, vmax = -0.1, 0
    # vmin, vmax = -4, 0
    # vmin, vmax = np.nanmin(data), np.nanmax(data)
    cmap = matplotlib.cm.get_cmap("turbo_r").copy()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # figure = plt.figure(figsize=(16, 8), dpi=300)
    figure = plt.figure(figsize=(16, 8), dpi=100)
    if static:
        x, y = np.meshgrid(time[window_length // 2:-window_length // 2], station_coordinates[latsort, 0])
    else:
        if zoom:
            x, y = np.meshgrid(time, station_coordinates[latsort, 0])
        else:
            x, y = np.meshgrid(time[0 + offset:-window_length - offset], station_coordinates[latsort, 0])
    if data_pcolormesh:
        f = interp2d(x, y, data[:, latsort, 0].T, kind='cubic')
        x_up = np.linspace(time[window_length // 2], time[-window_length // 2], len(x) * 1)
        y_up = np.linspace(station_coordinates[latsort, 0][0], station_coordinates[latsort, 0][-1], len(y) * 1)
        data1 = f(x_up, y_up)
        Xn, Yn = np.meshgrid(x_up, y_up)
        plt.pcolormesh(Xn, Yn, data1, cmap=cmap, norm=norm, zorder=0)
        # plt.pcolormesh(x, y, data[:, latsort, 0].T,  cmap=cmap, norm=norm, zorder=0, antialiased=True, shading='gouraud')
    else:
        # plot in reverse order to avoid to mask the sse growth
        plt.scatter(x[:, ::-1], y[:, ::-1], c=data[:, latsort, 0].T[:, ::-1], cmap=cmap, norm=norm, s=10, alpha=0.7,
                    zorder=0, edgecolors='none')
        # plt.scatter(x, y, c=data[:, latsort, 0].T, cmap=cmap, norm=norm, s=10, alpha=0.5, zorder=0, edgecolors='none')
        cbar = plt.colorbar()
        cbar.solids.set_alpha(1)
        cbar.ax.set_ylabel('Displacement rate [mm/day]', rotation=270, labelpad=25, size=13)

    if draw_tremors:
        tremor_scatter_size = 0.2
        if downsample_tremors:
            fraction_points_per_cluster = 0.05
            # we only downsample tremors for PNSN catalogue
            idx_pnsn = tremors[:, 3] > 2009
            # ide's catalogue is kept as it is
            plt.scatter(tremors[~idx_pnsn, 3], tremors[~idx_pnsn, 0], s=tremor_scatter_size, alpha=tremor_alpha,
                        color='black', zorder=1)
            dbscan = DBSCAN(eps=0.1, min_samples=10).fit(tremors[idx_pnsn][:, (0, 3)])
            labels = dbscan.labels_

            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_points = np.where(labels == label)[0]
                n_points_per_cluster = int(fraction_points_per_cluster * len(cluster_points))
                # print('#points:', n_points_per_cluster)
                selected_indices = np.random.choice(cluster_points, size=n_points_per_cluster)
                plt.scatter(tremors[idx_pnsn][selected_indices, 3], tremors[idx_pnsn][selected_indices, 0],
                            s=tremor_scatter_size, alpha=tremor_alpha, color='black', zorder=1)
        else:
            plt.scatter(tremors[:, 3], tremors[:, 0], s=tremor_scatter_size, alpha=tremor_alpha, color='black',
                        zorder=1)
    plt.ylabel('Latitude')
    plt.xlabel('Time [years]')
    plt.show()
    '''plt.savefig(fig_path, bbox_inches='tight')
    plt.close(figure)'''


def _movingmean_same(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def _albh_figure(time, data, time_michel, data_michel, selected_gnss_data, albh_index, offset_cut, window_length=60):
    # fig = plt.figure(figsize=(21, 7), dpi=300)
    fig = plt.figure(figsize=(21, 7), dpi=100)
    albh_cumsum = np.cumsum(data[:, albh_index, 0])
    albh_index = station_codes.tolist().index('ALBH')
    # plt.scatter(selected_time_array, selected_gnss_data[albh_index, :, 0] + selected_time_array * trend_info_albh[0] + trend_info_albh[1], s=5)
    plt.scatter(time, selected_gnss_data[albh_index, :, 0], s=5)
    # plt.plot(t_michel, albh_michel, color='C3')
    plt.plot(time, _movingmean_same(selected_gnss_data[albh_index, :, 0], 30), color='black')
    plt.plot(time_michel, _movingmean_same(data_michel, 30), color='C3', label='Michel et al.')
    # plt.plot(selected_time_array[offset_cut:-(window_length + offset_cut)], albh_cumsum, color='C1')
    plt.plot(time[offset_cut:-(window_length + offset_cut)], detrend(albh_cumsum), color='C1', label='This study')
    # plt.title('ALBH, E-W')
    plt.xlabel('Time [years]')
    plt.ylabel('Displacement [mm]')
    plt.legend()
    plt.show()
    # plt.savefig('denoising_figures/albh_denoised_median.pdf', bbox_inches='tight')
    # plt.close(fig)


if __name__ == '__main__':
    time, denoised_ts = generate_denoised_time_series()
    tremors = tremor_catalogue()
    tremors = tremors[tremors[:, 0] > 39]
    np.savez('denoised_ts_slip5_1000_noise_with_trend_demean_2006_2024', time=time, data=denoised_ts, tremors=tremors)
    exit(0)
    with np.load('denoised_ts_RES2.npz') as f:
        time, data, tremors = f['time'], f['data'], f['tremors']
    n_selected_stations = 200
    station_codes, station_coordinates, full_station_codes, full_station_coordinates, station_subset = cascadia_filtered_stations(
        n_selected_stations)
    latsort = np.argsort(station_coordinates[:, 0])[::-1]
    window_length = 60
    tol = 0.02
    offset_cut = 20
    '''_latitude_time_plot(time, data, tremors, station_coordinates, latsort,
                        tol=tol, offset=offset_cut, window_length=window_length, downsample_tremors=True,
                        draw_tremors=True, data_pcolormesh=False, fig_path='denoising_figures/overall_lat_time.png')'''

    reference_period = (2007, 2023)
    selected_gnss_data, selected_time_array, _, _ = _preliminary_operations(reference_period, detrend=False)

    selected_gnss_data = selected_gnss_data[station_subset]

    original_nan_pattern = np.isnan(selected_gnss_data[:, :, 0])
    selected_gnss_data, trend_info = detrend_nan_v2(selected_time_array, selected_gnss_data)

    albh_index = station_codes.tolist().index('ALBH')
    ################################################ data from Sylvain Michel
    albh_michel = np.loadtxt(
        'data_michel/X_Dat_Cascadia_EulerPoleCorr_OffsetsCorr_detrended_CorrSeasonal_Common_Post_ForConstantino_TXT.txt',
        delimiter=',')[0]
    t_michel = np.loadtxt(
        'data_michel/timeline_Cascadia_EulerPoleCorr_OffsetsCorr_detrended_CorrSeasonal_Common_Post_ForConstantino_TXT.txt',
        delimiter=',')
    ################################################
    # trend_info_albh = trend_info[albh_index]

    _albh_figure(time, data, t_michel, albh_michel, selected_gnss_data, albh_index, offset_cut, window_length=60)
