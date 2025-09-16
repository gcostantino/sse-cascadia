import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from utils.gnss_utils import _preliminary_operations, cascadia_filtered_stations


def calculate_velocity_linreg2(time1, d, tsw):
    vnew, vnew2 = [], []
    aux = tsw // 2
    aux2 = 0
    for j in range(len(d)):
        if j >= 0 and j <= aux - 1:
            at = time1[j - aux2:j + aux]
            at = np.array(at)
            ad = d[j - aux2:j + aux]
            mask = ~np.isnan(at) & ~np.isnan(ad)
            if len(ad[mask]) == 0 or len(at[mask]) == 0:
                vnew.append(np.nan)
                aux2 = aux2 + 1
            else:
                slope, intercept, rvalue, pvalue, std_err = stats.linregress(at[mask], ad[mask])
                vnew.append(slope)
                aux2 = aux2 + 1
        elif j > len(d) - 1 - aux and j <= len(d) - 1:
            at2 = time1[j - aux:len(d) - 1]
            at2 = np.array(at2)
            ad2 = d[j - aux:len(d) - 1]
            mask = ~np.isnan(at2) & ~np.isnan(ad2)
            if len(ad2[mask]) == 0 or len(at2[mask]) == 0:
                vnew.append(np.nan)
            else:
                slope, intercept, rvalue, pvalue, std_err = stats.linregress(at2[mask], ad2[mask])
                vnew.append(slope)
        else:
            at3 = time1[j - aux:j + aux]
            at3 = np.array(at3)
            ad3 = d[j - aux:j + aux]
            mask = ~np.isnan(at3) & ~np.isnan(ad3)
            if len(ad3[mask]) == 0 or len(at3[mask]) == 0:
                vnew.append(np.nan)
            else:
                slope, intercept, rvalue, pvalue, std_err = stats.linregress(at3[mask], ad3[mask])
                vnew.append(slope)
    for jj in range(len(d)):
        if (np.isnan(d[jj])):
            vnew2.append(np.nan)
        else:
            vnew2.append(vnew[jj])
    return vnew2


def calculate_vel_movil(vn, ve, vv, tm):
    # tm must be par
    vn2 = []
    ve2 = []
    vv2 = []
    aux = tm // 2
    aux2 = 1
    for j in range(len(vn)):
        # Primer dato
        if j == 0:
            vn2.append(np.nanmean(vn[j:j + aux]))
        # Entre el primer dato y el tamaño de la media movil
        elif j >= 1 and j <= aux - 1:
            vn2.append(np.nanmean(vn[j - aux2:j + aux]))
            aux2 = aux2 + 1
        # Parte final de los datos
        elif j > len(vn) - 1 - aux and j < len(vn) - 1:
            vn2.append(np.nanmean(vn[j - aux:len(vn) - 1]))
        # Media movil parte final de la muestra
        elif j == len(vn) - 1:
            vn2.append(np.nanmean(vn[j - aux:j]))
        # Media movil normal
        else:
            vn2.append(np.nanmean(vn[j - aux:j + aux]))
    return vn2, ve2, vv2


def _smoothing_derivative(x, t, T):
    # der = calculate_velocity_linreg2(t, detrend(x), T)
    der = calculate_velocity_linreg2(t, x, T)
    der_smooth, _, _ = calculate_vel_movil(der, der, der, T)
    return der_smooth


def _detrend_nan_1d_v2(x, y):
    # find linear regression line, subtract off data to detrend
    not_nan_ind = ~np.isnan(y)
    detrend_y = np.zeros(y.shape)
    detrend_y.fill(np.nan)
    m, b = 0, 0
    if y[not_nan_ind].size > 0:
        m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], y[not_nan_ind])
        detrend_y = y - (m * x + b)
    return detrend_y, m, b


def detrend_nan_v2(x, data):
    detrended_data = np.zeros(data.shape)
    trend_params = np.zeros((data.shape[0], data.shape[2], 2))
    for observation in range(data.shape[0]):
        for direction in range(data.shape[2]):
            detrended_, m, b = _detrend_nan_1d_v2(x, data[observation, :, direction])
            detrended_data[observation, :, direction] = detrended_
            trend_params[observation, direction, :] = [m, b]
    return detrended_data, trend_params


def smoothing_derivative(data, t, T):
    # data is detrended first
    data, _ = detrend_nan_v2(t, data)
    t = np.arange(t.shape[0])
    smoothed_data = np.zeros(data.shape)
    smoothed_data.fill(np.nan)
    for ns in range(data.shape[0]):
        for nc in range(data.shape[2]):
            d = data[ns, :, nc]
            # idx_non_nan = ~np.isnan(d)
            # smoothed_data[ns, idx_non_nan, nc] = _smoothing_derivative(d[idx_non_nan], t[idx_non_nan], T)
            smoothed_data[ns, :, nc] = _smoothing_derivative(d[:], t[:], T)
    return smoothed_data


def _load_gnss_data(n_selected_stations=200, work_directory='/Users/giuseppe/Documents/DATA/GNSS'):
    data, time, _, _, = _preliminary_operations(reference_period, detrend=False, work_directory=work_directory)
    codes, coords, _, _, station_subset = cascadia_filtered_stations(n_selected_stations, work_directory=work_directory)
    selected_gnss_data = data[station_subset]
    # selected_gnss_data, trend_info = detrend_nan_v2(selected_time_array, selected_gnss_data)
    # selected_gnss_data = selected_gnss_data.transpose((1, 0, 2))
    return selected_gnss_data, time, codes, coords


def smoothing_vel(time, data, T, station_idx=None):
    if station_idx is not None:
        data = data[station_idx][None, ...]
    detrended_data, trend_info = detrend_nan_v2(time, data)
    vel_filtered_data = smoothing_derivative(detrended_data, time, T=T)
    return vel_filtered_data

    return detrended_data, trend_params


def _draw_table_data(data_window, time_span, station_coordinates, every_nth=17, title='', figtitle=None, vmin=None,
                     vmax=None, colorbar=False, ylabel=True, folder='fig_synth_data'):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    latsort = np.argsort(station_coordinates[:, 0])[::-1]
    cmap = matplotlib.colormaps.get_cmap('RdBu_r')
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


def smooth_rows(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a moving average (mean) filter to each row of a 2D array.

    Parameters:
    - data: 2D numpy array with shape (n_obs, time)
    - window_size: length of the moving average window (must be >= 1)

    Returns:
    - smoothed: 2D numpy array of the same shape as `data`
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    window = np.ones(window_size) / window_size
    smoothed = np.array([np.convolve(row, window, mode='same') for row in data])
    return smoothed


def smooth_ts(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a moving average filter along the 'time' axis for each (row, component).

    Parameters:
    - data: 3D numpy array with shape (n_obs, time, n_comp)
    - window_size: length of the moving average window

    Returns:
    - smoothed: 3D numpy array of the same shape as `data`
    """
    n_obs, n_time, n_comp = data.shape
    smoothed = np.empty_like(data)

    for comp in range(n_comp):
        # Smooth each component slice independently
        smoothed[:, :, comp] = smooth_rows(data[:, :, comp], window_size)

    return smoothed


if __name__ == '__main__':
    reference_period = (2007, 2023)
    data, time, station_codes, station_coordinates = _load_gnss_data()
    # vel = smoothing_vel(time, data, station_idx=0)
    data, _ = detrend_nan_v2(time, data)  # (200, 5827, 2)

    with np.load('../../DATA/sse-cascadia/denoised_ts/denoised_ts_slip5_1000_noise_with_trend_demean.npz') as f:
        denoised_ts, denoised_time = f['data'], f['time']

    cat = np.loadtxt('catalogues/decimal_year/sse_catalogue_thresh_0.07.txt', skiprows=1, delimiter=',')
    # Event ID, start date (decimal year), end date (decimal year), SSE centroid longitude, SSE centroid latitude, SSE centroid depth (km), Mw, duration (days)
    start_dates, end_dates = cat[:, 1], cat[:, 2]

    '''albh_ew = data[0, ..., 0]
    
    for i, sse_dates in enumerate(zip(start_dates, end_dates)):
        ev_start_year, ev_end_year = sse_dates
        start_idx, end_idx = np.where(time == ev_start_year)[0][0], np.where(time == ev_end_year)[0][0]
        sse_crop = albh_ew[start_idx: end_idx]
        time_crop = time[start_idx: end_idx]
        plt.plot(time_crop, sse_crop)
        plt.title(f'Duration: {end_idx - start_idx + 1} days')
        plt.show()'''

    # focus on the 2016 slow slip event: ID 47 [starting from zero]

    #data = np.transpose(denoised_ts, (1, 0, 2))
    #time = denoised_time

    ev_2016_start = start_dates[47]
    ev_2016_end = end_dates[47]

    # use denoised data to have a station-wise detection
    denoised_data = np.transpose(denoised_ts, (1, 0, 2))  # velocity
    dates_stations = []
    ev_2016_start_idx_denoised, ev_2016_end_idx_denoised = np.where(denoised_time == ev_2016_start)[0][0], np.where(denoised_time == ev_2016_end)[0][0]
    for sta_id in range(len(station_coordinates)):
        event_denoised_data_ev = denoised_data[sta_id, ev_2016_start_idx_denoised - 40: ev_2016_end_idx_denoised + 14, 0]
        plt.plot(event_denoised_data_ev)
        plt.show()


    ev_2016_start_idx, ev_2016_end_idx = np.where(time == ev_2016_start)[0][0], np.where(time == ev_2016_end)[0][0]
    print(ev_2016_start, ev_2016_end)


    for sta_id in range(len(station_coordinates)):
        data_sta = data[latsort][len(station_coordinates) - 1 - sta_id, ev_2016_start_idx: ev_2016_end_idx, 0]
        time_sta = time[ev_2016_start_idx: ev_2016_end_idx]
        # norm_data = (data_sta - np.min(data_sta)) / (np.max(data_sta) - np.min(data_sta))
        plt.plot(time_sta, data_sta + sta_id)
        # plt.plot(time_sta, np.cumsum(data_sta) + sta_id)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    '''data, _ = detrend_nan_v2(time, data)
    cmap = plt.get_cmap('RdBu_r')
    norm = TwoSlopeNorm(vmin=np.min(vel[0, :, 0]), vcenter=0., vmax=np.max(vel[0, :, 0]))
    plt.scatter(time, data[0, :, 0], c=vel[0, :, 0], norm=norm, cmap=cmap)
    plt.colorbar()
    plt.show()'''
    '''# extract TS in a latitude range
    station_mask = np.logical_and(station_coordinates[:, 0] >= 47, station_coordinates[:, 0] <= 49)
    data_subset = data[station_mask]
    station_coordinates = station_coordinates[station_mask]
    latsort = np.argsort(station_coordinates[:, 0])[::-1]

    # smoothed_data = smooth_ts(data_subset, window_size=7)
    vel = smoothing_vel(time, data_subset, T=30)
    print(data.shape)
    print(data_subset.shape)

    _draw_table_data(vel, (time[0], time[-1]), station_coordinates, vmin=-0.1, vmax=0.1, every_nth=1, colorbar=True)'''
