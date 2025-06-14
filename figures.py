"""
Script to produce most of the plots for the main figure of the paper.
"""
import joblib
import numpy as np

from FUNCTIONS.functions_figures import total_slip_subduction, albh_figure, moment_rate_figure_comparison
from FUNCTIONS.functions_inv import compute_forward_model_vectorized
from config_files.plotting_style import set_matplotlib_style, temporary_matplotlib_syle, get_style_attr, get_style_dict
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.geometry_utils import load_cascadia_geometry
from utils.gnss_utils import load_gnss_data_cascadia, load_albh_data_cascadia, load_cascadia_selected_stations
from utils.slip_model_utils import load_slip_models

set_matplotlib_style()

if __name__ == '__main__':
    slip_model = load_slip_models()
    geometry, green_td = load_cascadia_geometry()

    station_codes, station_coordinates = load_cascadia_selected_stations()

    with np.load('../../DATA/sse-cascadia/denoised_ts/denoised_ts_slip5_1000_noise_with_trend_demean.npz') as f:
        denoised_ts, denoised_time = f['data'], f['time']

    M_DS = slip_model[:, :(len(geometry[:, 9]))]
    M_SS = slip_model[:, (len(geometry[:, 9])):]

    slip = np.sqrt(M_DS ** 2 + M_SS ** 2)  # in mm

    # total_slip_subduction(slip, geometry, station_coordinates)
    reference_period = (2007, 2023)  # (2006, 2024.5)
    work_directory = '../../DATA/GNSS'
    '''n_selected_stations = 200
    gnss_data, gnss_time, _, _ = load_gnss_data_cascadia(reference_period, n_selected_stations,
                                                         work_directory=work_directory, detrend=False)'''

    t_albh, d_albh, std_albh = load_albh_data_cascadia(reference_period, code='ALBH', work_directory=work_directory)
    modeled_TS = compute_forward_model_vectorized(slip_model, green_td, n_points=0, n_time_steps=len(denoised_ts))

    # temporary_style = {'xtick.labelsize': get_style_attr('xtick.labelsize') + 10.}
    current_style_dict = get_style_dict()

    temporary_style = {key: current_style_dict[key] + 14. for key in current_style_dict if
                       type(current_style_dict[key]) != str}
    with temporary_matplotlib_syle(temporary_style):
        '''albh_figure(denoised_time, denoised_ts, modeled_TS, station_codes, (d_albh, std_albh), offset_cut=20,
                    window_length=60, dpi=300, station_code='ALBH')'''
        # figure for moment rates here
        refine_durations = False
        se = SlowSlipEventExtractor()
        se.load_extracted_events_unfiltered()
        time_array = se.ddh.corrected_time
        moment_rate_figure_comparison(se, time_array, refine_durations=refine_durations, dpi=300)