from FUNCTIONS.functions_figures import slip_latitude_time_plot
from config_files.plotting_style import set_matplotlib_style
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.catalogue_parsing import tremor_catalogue

set_matplotlib_style()

if __name__ == '__main__':
    refine_durations = True
    se = SlowSlipEventExtractor()
    slip_rates = se.ma.slip_rates  # (5727, 1358)
    time_array = se.ddh.corrected_time
    y_centr_lat = se.ma.y_centr_lat

    '''import numpy as np
    for slip_thresh in se.slip_thresholds:
        se.get_extracted_events_unfiltered()
        mw = se.get_magnitude_events(slip_thresh, refine_durations)
        dates_idx = se.get_event_date_idx(slip_thresh, refine_durations)

        start_date_idx, end_date_idx = np.array(dates_idx).T

        # start_date, end_date = time_array[start_date_idx], time_array[end_date_idx]
        start_date_dec, end_date_dec = time_array[start_date_idx], time_array[end_date_idx]
        michel_time = np.logical_and(start_date_dec >= 2007, end_date_dec < 2018)
        mw = mw[michel_time]
        mw = mw[~np.isinf(mw)]

        print(np.nanmin(mw))'''

    window_length, offset = se.ddh.window_length, se.ddh.offset

    tremors = tremor_catalogue()
    tremors = tremors[(tremors[:, 0] > 40) & (tremors[:, 0] < 50)]
    tremors = tremors[(tremors[:, 3] > 2007) & (tremors[:, 3] < 2023 - (window_length + 2 * offset) / 365)]

    slip_latitude_time_plot(time_array, slip_rates, tremors, y_centr_lat, offset=20, n_lat_bins=100,
                            window_length=60, scatter_size=20,
                            static=False, downsample_tremors=True, draw_tremors=True, tremor_alpha=1.,
                            data_pcolormesh=False, zoom=False, show=False, dpi=300, trenchwards_only=True,
                            save_as='pdf', raster_scatter=True, modified_cmap=True)
