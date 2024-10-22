import os

import numpy as np
from scipy.stats import linregress


def _preliminary_operations(reference_period, detrend=True, **kwargs):
    """Returns GNSS data in reference_period. When detrending is used, the trends are not returned."""
    gnss_data, time_array = _load_gnss_data_cascadia(reference_period=reference_period, **kwargs)
    if detrend:
        gnss_data = detrend_nan(time_array, gnss_data)  # gnss data is detrended first
    reference_time_indices = get_reference_period(time_array, reference_period)
    reference_time_array = time_array[reference_time_indices]
    selected_gnss_data = gnss_data[:, reference_time_indices, :]
    return selected_gnss_data, reference_time_array, gnss_data, time_array


def _load_gnss_data_cascadia(**kwargs):
    """Loads all the GNSS data from Cascadia and returns it along with a time array.
    The data is taken by considering the time span associated to the longest available time series.
    The unit measure is converted in mm."""
    n_directions = kwargs.pop('n_directions', 2)
    work_directory = os.path.expandvars(kwargs['work_directory']) if 'work_directory' in kwargs else './geo_data'
    station_codes, station_coordinates = cascadia_coordinates()
    n_stations = len(station_codes)
    file_lines = np.zeros((len(station_codes),))
    for i, code in enumerate(station_codes):  # check the longest file to have the largest time span
        num_lines = sum(1 for line in open(os.path.join(work_directory, f'GNSS_CASCADIA/txt/{code}.txt')))
        file_lines[i] = num_lines
    target_station = station_codes[np.argmax(file_lines)]
    time_array = np.loadtxt(os.path.join(work_directory, f'GNSS_CASCADIA/txt/{target_station}.txt'))[:, 0]
    # time_array = _decyr_time_array(kwargs['reference_period'])  # this will be used soon
    gnss_data = np.zeros((n_stations, len(time_array), n_directions))
    gnss_data.fill(np.nan)
    for i, code in enumerate(station_codes):
        data = np.loadtxt(os.path.join(work_directory, f'GNSS_CASCADIA/txt/{code}.txt'))[:, :n_directions + 1]
        correspondence_indices = np.searchsorted(time_array, data[:, 0])
        gnss_data[i, correspondence_indices, :] = data[:, 1:] * 1e03
    return gnss_data, time_array


def _detrend_nan_1d(x, y):
    # find linear regression line, subtract off data to detrend
    not_nan_ind = ~np.isnan(y)
    detrend_y = np.zeros(y.shape)
    detrend_y.fill(np.nan)
    if y[not_nan_ind].size > 0:
        m, b, r_val, p_val, std_err = linregress(x[not_nan_ind], y[not_nan_ind])
        detrend_y = y - (m * x + b)
    return detrend_y


def _remove_stations(codes_to_remove, codes, coords, gnss_data):
    codes = np.array(codes)
    mask = ~np.isin(codes, codes_to_remove)
    new_coords = coords[mask]
    new_codes = codes[mask]
    new_gnss_data = gnss_data[mask]
    return new_codes, new_coords, new_gnss_data


def get_reference_period(time_array, period):
    """Returns indices in the data for the specified reference period, supposed to be a tuple of (start,end) dates."""
    reference_time_indices = np.where(np.logical_and(time_array > period[0], time_array < period[1]))[0]
    return reference_time_indices


def cascadia_coordinates(work_directory='../../DATA/GNSS/NGL_info'):
    """3D position of GNSS stations in Cascadia"""
    os.path.join(work_directory, 'NGL_stations_cascadia.txt')
    with open(os.path.join(work_directory, 'NGL_stations_cascadia.txt')) as f:
        rows = f.read().splitlines()
        station_codes = []
        station_coordinates = []
        for row in rows:
            station_code, station_lat, station_lon, station_height = row.split(' ')
            station_codes.append(station_code)
            station_coordinates.append([station_lat, station_lon, station_height])
        station_coordinates = np.array(station_coordinates, dtype=np.float_)
    return station_codes, station_coordinates


def detrend_nan(x, data):
    detrended_data = np.zeros(data.shape)
    for observation in range(data.shape[0]):
        for direction in range(data.shape[2]):
            detrended_data[observation, :, direction] = _detrend_nan_1d(x, data[observation, :, direction])
    return detrended_data


def cascadia_filtered_stations(n_selected_stations, gnss_data=None, reference_period=(2007, 2023), **kwargs):
    """Removes meaningless stations and returns the corresponding indices 'stations_subset_full',
    referring to the whole cascadia network."""
    if gnss_data is None:
        gnss_data, _, _, _ = _preliminary_operations(reference_period, detrend=False, **kwargs)
    full_station_codes, full_station_coordinates = cascadia_coordinates()
    stations_to_remove = ['WSLB', 'YBHB', 'P687', 'BELI', 'PMAR', 'TGUA', 'OYLR', 'FTS5', 'RPT5', 'RPT6', 'P791',
                          'P674', 'P656', 'TWRI', 'WIFR', 'FRID', 'PNHG', 'COUR', 'SKMA', 'CSHR', 'HGP1', 'CBLV',
                          'PNHR', 'NCS2', 'TSEP', 'BCSC', 'LNG2']
    station_codes, station_coordinates, selected_gnss_data = _remove_stations(stations_to_remove, full_station_codes,
                                                                              full_station_coordinates, gnss_data)
    original_nan_pattern = np.isnan(selected_gnss_data[:, :, 0])
    n_nans_stations = original_nan_pattern.sum(axis=1)
    stations_subset = np.sort(np.argsort(n_nans_stations)[:n_selected_stations])
    station_codes_subset, station_coordinates_subset = np.array(station_codes)[stations_subset], station_coordinates[
                                                                                                 stations_subset, :]
    stations_subset_full = np.nonzero(np.in1d(full_station_codes, station_codes_subset))[0]
    return station_codes_subset, station_coordinates_subset, station_codes, station_coordinates, stations_subset_full


def load_gnss_data_cascadia(reference_period, n_selected_stations, detrend=False, **kwargs):
    selected_gnss_data, reference_time_array, gnss_data, time_array = _preliminary_operations(reference_period,
                                                                                              detrend=detrend, **kwargs)
    station_codes, station_coordinates, full_station_codes, full_station_coordinates, station_subset = cascadia_filtered_stations(
        n_selected_stations, gnss_data=selected_gnss_data)
    selected_gnss_data = selected_gnss_data[station_subset]
    return selected_gnss_data, reference_time_array, gnss_data, time_array


def load_albh_data_cascadia(reference_period, code='ALBH', m_to_mm=1e03, **kwargs):
    """Temporary load function, to be used for future development. E-W data is loaded only."""
    work_directory = os.path.expandvars(kwargs.pop('work_directory', './geo_data'))
    td = np.loadtxt(os.path.join(work_directory, f'GNSS_CASCADIA/txt/{code}.txt'), usecols=(0, 1))
    d_std = np.loadtxt(os.path.join(work_directory, f'GNSS_CASCADIA/tenv3/{code}.txt'), skiprows=1, usecols=14)
    time_array, data = td[:, 0], td[:, 1]
    data, d_std = data * m_to_mm, d_std * m_to_mm
    reference_time_indices = get_reference_period(time_array, reference_period)
    return time_array[reference_time_indices], data[reference_time_indices], d_std[reference_time_indices]


def load_cascadia_selected_stations():
    station_codes = np.loadtxt('INPUT_FILES/stations_cascadia_200.txt', usecols=0, dtype=np.str_).tolist()
    station_coordinates = np.loadtxt('INPUT_FILES/stations_cascadia_200.txt', usecols=(1, 2))
    return station_codes, station_coordinates
