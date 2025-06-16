import os

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from sse_denoiser.dataset_utils import Dataset
from sse_denoiser.stagrnn import STAGRNNDenoiser
from utils import _preliminary_operations, cascadia_filtered_stations

if __name__ == '__main__':
    reference_period = (2006, 2023.4)
    n_selected_stations = 200
    window_length = 60
    work_directory = '$WORK'
    batch_size = 32
    n_directions = 2

    selected_gnss_data, selected_time_array, _, _ = _preliminary_operations(reference_period, detrend=False,
                                                                            work_directory=work_directory)
    station_codes, station_coordinates, full_station_codes, full_station_coordinates, station_subset = cascadia_filtered_stations(
        n_selected_stations, work_directory=work_directory)
    selected_gnss_data = selected_gnss_data[station_subset]
    original_nan_pattern = np.isnan(selected_gnss_data[:, :, 0])
    # selected_gnss_data, trend_info = detrend_nan_v2(selected_time_array, selected_gnss_data)

    # selected_gnss_data[original_nan_pattern] = 0.  # NaNs are replaced with zeros

    running_test_set = []

    for i in range(selected_time_array.shape[0] - window_length):
        data_window = selected_gnss_data[:, i:i + window_length, :]
        running_test_set.append(data_window - np.nanmean(data_window, axis=1, keepdims=True))

    running_test_set = np.array(running_test_set)

    running_test_set[np.isnan(running_test_set)] = 0.

    '''import matplotlib.pyplot as plt
    for i in range(len(running_test_set)):
        f, a=plt.subplots(1, 2)
        a1, a2 = a
        a1.plot(selected_gnss_data[:, i:i + window_length, 0].T)
        a2.plot(running_test_set[i, :, :, 0].T)
        plt.show()'''

    dummy_y_test = torch.Tensor(np.zeros((running_test_set.shape[0], n_selected_stations, window_length, n_directions)))
    test_dataset = Dataset(running_test_set, dummy_y_test)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    params = {'n_stations': n_selected_stations,
              'window_length': window_length,
              'n_directions': n_directions,
              'batch_size': batch_size,
              'station_coordinates': station_coordinates[:, :2],
              'y_test': dummy_y_test,
              'learn_static': False,
              'residual': False,
              'residual2': False}

    denoiser = STAGRNNDenoiser(**params)
    denoiser.build()

    denoiser.set_data_loaders(None, None, test_loader)

    weight_path = os.path.join(os.path.expandvars(work_directory),
                               'weights/SSEdetector_char/best_cascadia_09Feb2024-103329_train_denois_realgaps_v7_30disloc_slip5_250_lr0.001_bs128_demean.pt')  # noise with trend

    denoiser.load_weights(weight_path)

    pred = denoiser.inference()
    np.savez(os.path.join(os.path.expandvars(work_directory),
                          'pred_SSEdenoiser_v7_30disloc_slip5_250_noise_with_trend_demean_2006_2024'), pred=pred)
