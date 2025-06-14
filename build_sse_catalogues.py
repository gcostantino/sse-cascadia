import os

import numpy as np

from FUNCTIONS.functions_slab import UTM_GEO
from config_files.plotting_style import set_matplotlib_style
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from utils.date_parsing import ymd_decimal_year_lookup

set_matplotlib_style()

if __name__ == '__main__':
    # catalogue structure:
    # event_idx, start_date, end_date, x_centroid, y_centroid, z_centroid, Mw, duration
    local_dir = os.path.join('.', 'catalogues')
    os.makedirs(local_dir, exist_ok=True)

    refine_durations = True
    se = SlowSlipEventExtractor()
    time_array = se.ddh.corrected_time
    decimal_year_cat = True

    if decimal_year_cat:
        for slip_thresh in se.slip_thresholds:
            filename = os.path.join(local_dir, f'sse_catalogue_thresh_{slip_thresh}.txt')
            sse_info_thresh, new_duration_dict = se.get_extracted_events_unfiltered()
            mw = se.get_magnitude_events(slip_thresh, refine_durations)
            patches = se.get_event_patches(slip_thresh, refine_durations)
            dates_idx = se.get_event_date_idx(slip_thresh, refine_durations)
            event_areas = se.get_area_events(slip_thresh, refine_durations)
            nucleation_idx, arrest_idx, valid_mask = se.get_start_end_patch(0.07, delta_win=5)
            nuc_x, nuc_y = se.ma.x_centr_km[nucleation_idx], se.ma.y_centr_km[nucleation_idx]
            arr_x, arr_y = se.ma.x_centr_km[arrest_idx], se.ma.y_centr_km[arrest_idx]
            distances = np.sqrt((nuc_x - arr_x) ** 2 + (nuc_y - arr_y) ** 2)
            valid_indices = np.nonzero(valid_mask)[0]
            with open(filename, mode='w') as f:
                f.write(
                    'Event ID, start date (YMD), end date (YMD), SSE centroid longitude, SSE centroid latitude, SSE centroid depth (km), Mw, SSE duration (days), SSE area (km^2), distance nucleation point-arrest point (km)\n')
                for ev_idx in range(len(mw)):
                    x_centroid = np.mean(se.ma.x_centr_km[patches[ev_idx]])
                    y_centroid = np.mean(se.ma.y_centr_km[patches[ev_idx]])
                    z_centroid = np.mean(se.ma.z_centr_km[patches[ev_idx]])
                    lon_centroid, lat_centroid = UTM_GEO(x_centroid, y_centroid)
                    start_date_idx, end_date_idx = dates_idx[ev_idx]
                    # start_date, end_date = time_array[start_date_idx], time_array[end_date_idx]
                    start_date_dec, end_date_dec = time_array[start_date_idx], time_array[end_date_idx]
                    date_lookup = ymd_decimal_year_lookup(from_decimal=True)
                    start_date, end_date = date_lookup[start_date_dec], date_lookup[end_date_dec]
                    start_year, start_month, start_day = start_date
                    end_year, end_month, end_day = end_date
                    mw_event = mw[ev_idx]
                    duration = end_date_idx - start_date_idx + 1
                    area = event_areas[ev_idx]
                    distance = distances[np.where(valid_indices == ev_idx)[0][0]] if valid_mask[ev_idx] else np.nan
                    f.write(
                        f'{ev_idx + 1}, {start_year} {start_month} {start_day}, {end_year} {end_month} {end_day}, {lon_centroid}, {lat_centroid}, {-z_centroid}, {mw_event}, {duration}, {area}, {distance}\n')
    else:
        for slip_thresh in se.slip_thresholds:
            filename = os.path.join(local_dir, f'sse_catalogue_thresh_{slip_thresh}.txt')
            sse_info_thresh, new_duration_dict = se.get_extracted_events_unfiltered()
            mw = se.get_magnitude_events(slip_thresh, refine_durations)
            patches = se.get_event_patches(slip_thresh, refine_durations)
            dates_idx = se.get_event_date_idx(slip_thresh, refine_durations)
            event_areas = se.get_area_events(slip_thresh, refine_durations)
            nucleation_idx, arrest_idx, valid_mask = se.get_start_end_patch(0.07, delta_win=5)
            nuc_x, nuc_y = se.ma.x_centr_km[nucleation_idx], se.ma.y_centr_km[nucleation_idx]
            arr_x, arr_y = se.ma.x_centr_km[arrest_idx], se.ma.y_centr_km[arrest_idx]
            distances = np.sqrt((nuc_x - arr_x) ** 2 + (nuc_y - arr_y) ** 2)
            valid_indices = np.nonzero(valid_mask)[0]
            with open(filename, mode='w') as f:
                f.write(
                    '#Event ID, start date (decimal year), end date (decimal year), SSE centroid longitude, SSE centroid latitude, SSE centroid depth (km), Mw, SSE duration (days), SSE area (km^2), distance nucleation point-arrest point (km)\n')
                for ev_idx in range(len(mw)):
                    x_centroid = np.mean(se.ma.x_centr_km[patches[ev_idx]])
                    y_centroid = np.mean(se.ma.y_centr_km[patches[ev_idx]])
                    z_centroid = np.mean(se.ma.z_centr_km[patches[ev_idx]])
                    lon_centroid, lat_centroid = UTM_GEO(x_centroid, y_centroid)
                    start_date_idx, end_date_idx = dates_idx[ev_idx]
                    start_date_dec, end_date_dec = time_array[start_date_idx], time_array[end_date_idx]
                    date_lookup = ymd_decimal_year_lookup(from_decimal=True)
                    start_date = date_lookup[start_date_dec]
                    start_year, start_month, start_day = start_date
                    mw_event = mw[ev_idx]
                    duration = end_date_idx - start_date_idx + 1
                    area = event_areas[ev_idx]
                    distance = distances[np.where(valid_indices == ev_idx)[0][0]] if valid_mask[ev_idx] else np.nan
                    f.write(
                        f'{start_year} {start_month} {start_day} {lat_centroid} {lon_centroid} {-z_centroid} {mw_event} {duration} {area} {distance}\n')
