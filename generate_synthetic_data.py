import os
import sys

import joblib

from data_config_files import v7_parameters as params
from sse_generator_v2.synthetic_time_series_cascadia import synthetic_time_series


def generate_synthetic_dataset():
    if len(sys.argv) <= 1:
        raise Exception('Please provide CLI arguments.')
    n_samples = int(sys.argv[1])

    n_stations = params.n_stations
    window_length = params.window_length
    depth_range = params.depth_range
    rake_range = params.rake_range
    data_gap_proba = params.data_gap_proba
    slip_range = params.slip_range
    max_n_disloc = params.max_n_disloc
    spatial_proba = params.spatial_proba
    min_centroid_distance = params.min_centroid_distance
    max_centroid_distance = params.max_centroid_distance
    noise_has_trend = params.noise_has_trend

    dset_type = 'denois'
    base_dir = os.path.expandvars('$WORK')  # '.'

    dataset_filename = f'{dset_type}_synth_ts_cascadia_realgaps_extended_v7_{n_stations}stations_depth_{depth_range[0]}_{depth_range[1]}_20x20_{max_n_disloc}_disloc_slip_{slip_range[0]}_{slip_range[1]}_noise_with_trend.data'

    dataset_path = os.path.join(base_dir, dataset_filename)

    synth_data, rand_dur, cat, synth_disp, templ, stat_codes, stat_coord = synthetic_time_series(
        n_samples, n_stations, window_length=window_length, max_n_disloc=max_n_disloc, slip_range=slip_range,
        depth_range=depth_range, rake_range=rake_range, p=data_gap_proba, spatial_proba=spatial_proba,
        min_centroid_distance=min_centroid_distance, max_centroid_distance=max_centroid_distance,
        noise_has_trend=noise_has_trend)

    data_dict = dict()
    data_dict['synthetic_data'] = synth_data
    data_dict['random_durations'] = rand_dur
    data_dict['catalogue'] = cat
    data_dict['static_displacement'] = synth_disp
    data_dict['time_templates'] = templ
    data_dict['station_codes'] = stat_codes
    data_dict['station_coordinates'] = stat_coord

    joblib.dump(data_dict, dataset_path)


if __name__ == '__main__':
    generate_synthetic_dataset()
