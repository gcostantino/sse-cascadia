import os

import numpy as np

from FUNCTIONS.functions_slab import UTM_GEO
from config_files.plotting_style import set_matplotlib_style
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor

set_matplotlib_style()

if __name__ == '__main__':
    # catalogue structure:
    # event_idx, start_date, end_date, x_centroid, y_centroid, z_centroid, Mw, duration
    local_dir = os.path.join('.', 'catalogues')
    os.makedirs(local_dir, exist_ok=True)

    refine_durations = True
    se = SlowSlipEventExtractor()
    time_array = se.ddh.corrected_time

    for slip_thresh in se.slip_thresholds:
        filename = os.path.join(local_dir, f'sse_catalogue_thresh_{slip_thresh}.txt')
        sse_info_thresh, new_duration_dict = se.get_extracted_events_unfiltered()
        mw = se.get_magnitude_events(slip_thresh, refine_durations)
        patches = se.get_event_patches(slip_thresh, refine_durations)
        dates_idx = se.get_event_date_idx(slip_thresh, refine_durations)
        with open(filename, mode='w') as f:
            f.write(
                'Event ID, start date (decimal year), end date (decimal year), SSE centroid longitude, SSE centroid latitude, SSE centroid depth (km), Mw, duration (days)\n')
            for ev_idx in range(len(mw)):
                x_centroid = np.mean(se.ma.x_centr_km[patches[ev_idx]])
                y_centroid = np.mean(se.ma.y_centr_km[patches[ev_idx]])
                z_centroid = np.mean(se.ma.z_centr_km[patches[ev_idx]])
                lon_centroid, lat_centroid = UTM_GEO(x_centroid, y_centroid)
                start_date_idx, end_date_idx = dates_idx[ev_idx]
                start_date, end_date = time_array[start_date_idx], time_array[end_date_idx]
                mw_event = mw[ev_idx]
                duration = end_date_idx - start_date_idx + 1
                f.write(
                    f'{ev_idx}, {start_date}, {end_date}, {lon_centroid}, {lat_centroid}, {-z_centroid}, {mw_event}, {duration}\n')
   