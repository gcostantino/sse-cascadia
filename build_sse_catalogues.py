import numpy as np

from FUNCTIONS.functions_slab import UTM_GEO
from sse_extraction.sse_extraction_from_slip import get_events_from_slip_model, refine_durations
from utils.geometry_utils import load_cascadia_geometry
from utils.gnss_utils import load_cascadia_selected_stations
from utils.slip_model_utils import load_slip_models

if __name__ == '__main__':

    with np.load('../../DATA/sse-cascadia/denoised_ts/denoised_ts_slip5_1000_noise_with_trend_demean.npz') as f:
        denoised_ts, time_vec = f['data'], f['time']

    TS = np.zeros((denoised_ts.shape[0], denoised_ts.shape[1], 3))
    TS[:, :, :2] = denoised_ts
    n_time_steps = TS.shape[0]

    window_length = 60
    offset = 20
    corrected_time = time_vec[0 + offset:-window_length - offset]


    slip_thresholds = [0.07, 0.3, 0.5, 0.7]


    sse_info_thresh, _ = get_events_from_slip_model(slip_thresholds, slip, signed_slip, area, corrected_time,
                                                    x_centr_lon, y_centr_lat, n_time_steps, shear_modulus=shear_modulus,
                                                    load=True, spatiotemporal=True, cut_neg_slip=True,
                                                    base_folder='../../DATA/sse-cascadia/sse_info')
    new_duration_dict = refine_durations(slip_thresholds, sse_info_thresh, mo_rate_percentage=.95)
