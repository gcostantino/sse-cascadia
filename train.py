import os
import sys

import joblib
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader

from sse_denoiser.dataset_utils import Dataset
from sse_denoiser.stagrnn import STAGRNNDenoiser

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception('Please provide CLI arguments.')
    n_samples, batch_size, learning_rate, work_directory = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), \
        sys.argv[4]

    set_callbacks = sys.argv[5].lower()
    set_callbacks = True if set_callbacks == 'true' else False

    verbosity = int(sys.argv[6]) if len(sys.argv) >= 7 else 1

    str_transfer_learning = 'false'
    if len(sys.argv) >= 8:
        str_transfer_learning = sys.argv[7].lower()
    # transfer_learning = True if str_transfer_learning == 'true' else False
    transfer_learning = False

    train_codename = 'train_denois_realgaps_v7_30disloc_slip5_250' + (
        '_transfer_learning' if transfer_learning else '') + f'_lr{learning_rate}_bs{batch_size}_demean'

    dataset_filename = os.path.expandvars(
        work_directory) + '/denois_synth_ts_cascadia_realgaps_extended_v7_200stations_depth_20_40_20x20_30_disloc_slip_5_250'

    data_dict = joblib.load(f'{dataset_filename}.data')

    data = data_dict['synthetic_data']
    durations = data_dict['random_durations']
    cat = data_dict['catalogue']
    static_displacement = data_dict['static_displacement']
    time_templates = data_dict['time_templates']
    station_codes = data_dict['station_codes']
    station_coordinates = data_dict['station_coordinates']

    y = time_templates[..., :2]

    de_mean = True

    y = y[:n_samples]
    data = data[:n_samples]
    cat = cat[:n_samples]

    if de_mean:
        # data = data - data[:, :, 0, :][:, :, np.newaxis, :]
        mean_vals = np.mean(data, axis=2, keepdims=True)  # inputs only, not target
        # Remove the mean from each component
        data = data - mean_vals

    n_stations = station_coordinates.shape[0]

    ind_val = int(n_samples * 0.8)
    ind_test = int(n_samples * 0.9)

    train_dataset = Dataset(data[:ind_val], y[:ind_val])
    val_dataset = Dataset(data[ind_val:ind_test], y[ind_val:ind_test])
    test_dataset = Dataset(data[ind_test:], y[ind_test:])

    cat_train, cat_val, cat_test = cat[:ind_val], cat[ind_val:ind_test], cat[ind_test:]
    '''templates_train, templates_val, templates_test = templates[:ind_val, :, :], templates[ind_val:ind_test, :,
                                                                                :], templates[ind_test:, :, :]'''

    y_train, y_val, y_test = y[:ind_val], y[ind_val:ind_test], y[ind_test:]

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    residual = False
    residual2 = False
    residual3 = False
    custom_loss = False
    custom_loss_coeff = 0.5
    add_transformer = True
    use_spatial_attention = True
    use_temporal_attention = True

    params = {'n_stations': n_stations,
              'window_length': 60,
              'n_directions': 2,
              'batch_size': batch_size,
              'n_epochs': 500,
              'learning_rate': learning_rate,
              'verbosity': verbosity,
              'patience': 500,
              'loss': 'mean_squared_error',
              'val_catalogue': cat_val,
              'station_coordinates': station_coordinates[:, :2],
              'y_val': y_val,
              'residual': residual,
              'learn_static': False,
              'custom_loss': custom_loss,
              'custom_loss_coeff': custom_loss_coeff,
              'residual2': residual2,
              'residual3': residual3,
              'add_transformer': add_transformer,
              'use_temporal_attention': use_temporal_attention,
              'use_spatial_attention': use_spatial_attention}

    denoiser = STAGRNNDenoiser(**params)
    denoiser.build()
    denoiser.associate_optimizer()
    denoiser.set_data_loaders(train_loader, val_loader, test_loader)

    if transfer_learning:
        weight_path = os.path.join(os.path.expandvars(work_directory),
                                   'weights/SSEdetector_char/best_cascadia_02Jul2023-013725_train_denois_realgaps_v5_STAGRNN_no_CNN_old_sd.pt')
        denoiser.load_weights(weight_path)

    denoiser.summary_nograph(train_loader.dataset[:batch_size][0])

    if set_callbacks:
        denoiser.set_callbacks(train_codename)

    denoiser.train()
