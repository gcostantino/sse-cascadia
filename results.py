import joblib
import matplotlib.pyplot as plt
import numpy as np

from FUNCTIONS.functions_figures import spectra_compilation_mo_rate, mo_rate_stack_asymmetry_patchwise, mo_rate_stack_asymmetry_eventwise
from FUNCTIONS.functions_inv import compute_forward_model_vectorized
from FUNCTIONS.functions_slab import UTM_GEO
from config_files.plotting_style import set_matplotlib_style
from sse_extraction.sse_extraction_from_slip import get_events_from_slip_model, refine_durations
from utils.geometry_utils import load_cascadia_geometry
from utils.gnss_utils import load_cascadia_selected_stations
from utils.slip_model_utils import load_slip_models

set_matplotlib_style()

if __name__ == '__main__':

    slip_model = load_slip_models()
    geometry, green_td = load_cascadia_geometry()

    station_codes, station_coordinates = load_cascadia_selected_stations()

    with np.load('../../DATA/sse-cascadia/denoised_ts/denoised_ts_slip5_1000_noise_with_trend_demean.npz') as f:
        denoised_ts, time_vec = f['data'], f['time']

    TS = np.zeros((denoised_ts.shape[0], denoised_ts.shape[1], 3))
    TS[:, :, :2] = denoised_ts
    n_time_steps = TS.shape[0]

    window_length = 60
    offset = 20
    corrected_time = time_vec[0 + offset:-window_length - offset]

    M_DS = slip_model[:, :(len(geometry[:, 9]))]
    M_SS = slip_model[:, (len(geometry[:, 9])):]

    slip = np.sqrt(M_DS ** 2 + M_SS ** 2)  # in mm
    signed_slip = np.sign(M_DS) * np.sqrt(M_DS ** 2 + M_SS ** 2)  # in mm
    # signed_slip = M_DS

    x_centr_lon, y_centr_lat = UTM_GEO(geometry[:, 9], geometry[:, 10])

    area = geometry[:, 21] * 1e+6  # in m^2

    slip_potency = area * slip * 1e-03  # slip converted to meters

    shear_modulus = 30 * 1e9  # GPa

    Mo = shear_modulus * slip_potency

    '''tremors = tremor_catalogue()
    # tremors = tremors[tremors[:, 0] > 39]
    tremors = tremors[(tremors[:, 0] > 40) & (tremors[:, 0] < 50)]
    n_tremors_per_day = get_n_tremors_per_day(corrected_time, tremors)'''

    total_moment_release = np.sum(Mo, axis=1)  # for all subfaults

    signed_slip_potency = area * signed_slip * 1e-03  # slip converted to meters
    signed_Mo = shear_modulus * signed_slip_potency

    ###
    n_points = 0
    modeled_TS = compute_forward_model_vectorized(slip_model, green_td, n_points, n_time_steps)

    # slip_thresholds = [0.07, 0.2, 0.5]
    slip_thresholds = [0.07, 0.3, 0.5, 0.7]

    sse_info_thresh, _ = get_events_from_slip_model(slip_thresholds, slip, signed_slip, area, corrected_time,
                                                    x_centr_lon, y_centr_lat, n_time_steps, shear_modulus=shear_modulus,
                                                    load=True, spatiotemporal=True, cut_neg_slip=True,
                                                    base_folder='../../DATA/sse-cascadia/sse_info')
    new_duration_dict = refine_durations(slip_thresholds, sse_info_thresh, mo_rate_percentage=.99)

    #spectra_compilation_mo_rate(sse_info_thresh, Mo, corrected_time, slip_thresholds, slip, area, shear_modulus, num_freq_bins=40)

    '''mo_rate_stack_asymmetry_patchwise(sse_info_thresh, slip_thresholds, n_dur_bins=5, show_fit=False, show_individual_mo=False,
                            align_start=True, rescale_zero_y=False)'''

    mo_rate_stack_asymmetry_eventwise(sse_info_thresh, slip_thresholds, new_duration_dict, n_dur_bins=5, show_fit=False,
                                      show_individual_mo=False, align_start=True, rescale_zero_y=False,
                                      refine_durations=True)


































    exit(0)
    # spectrum_denoised_data_as_fcn_latitude(TS, corrected_time, station_coordinates)

    # slip_vs_tremors_as_fctn_lat(slip, corrected_time, tremors)

    # total_slip_subduction(slip, geometry, station_coordinates)

    sse_split_scaling_laws(slip, signed_slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps,
                           shear_modulus, cut_neg_slip=True, show_fit_GR=False)
    exit(0)
    # sse_split_further_scaling_laws(slip, signed_slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps, shear_modulus, cut_neg_slip=True)

    # sse_visualization(slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps, shear_modulus, station_coordinates)

    # sse_visualization_animation(slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps, shear_modulus, station_coordinates, tremors)
    sse_visualization_animation_parallel(slip, signed_slip, area, corrected_time, x_centr_lon, y_centr_lat,
                                         n_time_steps,
                                         shear_modulus, station_coordinates, tremors, TS, modeled_TS, signed=True)

    '''sse_visualization_animation_video(slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps,
                                         shear_modulus, station_coordinates, tremors, signed=True)'''

    # sse_duration_redefinition_test(slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps, shear_modulus)

    # asymmetry_analysis_events(slip, signed_slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps, shear_modulus, min_duration=0, cut_neg_slip=True)
    impulsivity_analysis_events(slip, signed_slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps,
                                shear_modulus, min_duration=0, cut_neg_slip=True)

    # interevent_time_analysis(slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps, shear_modulus)

    # interevent_time_albh(slip, area, corrected_time, x_centr_lon, y_centr_lat, n_time_steps, shear_modulus)

    ######################################################## finalized results
