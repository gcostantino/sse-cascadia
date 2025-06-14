import matplotlib.pyplot as plt

from FUNCTIONS.functions_slab import GEO_UTM
from plot_individual_events import plot_slip_subduction, pre_utm_conversion
from sse_extraction.SlowSlipEventExtractor import SlowSlipEventExtractor
from matplotlib.patches import Ellipse
import numpy as np

from utils.slab_utils import read_depth_from_slab2

if __name__ == '__main__':
    n_rows, n_cols = 1, 3
    mo_rate_scale = 1e17
    refine_durations = True
    se = SlowSlipEventExtractor()
    se.load_extracted_events_unfiltered()
    time_array = se.ddh.corrected_time
    thresh = se.slip_thresholds[0]

    admissible_depth, _ = read_depth_from_slab2(max_depth=100)
    utm_converted = pre_utm_conversion(se.ma.fault_geometry)

    dates_idx = se.get_event_date_idx(thresh, refine_durations)

    ev_idx = 47  # SSE occurring in 2016
    ev_start_idx, ev_end_idx = dates_idx[ev_idx]
    ev_dec_time_array = time_array[ev_start_idx: ev_end_idx + 1]
    delta_win = 5  # days
    slip_rate_thresh = .5

    slip_rate = se.get_slip_rate_patches_events(thresh, refine_durations)[ev_idx]
    start_slip_rate, end_slip_rate = slip_rate[:delta_win], slip_rate[-delta_win:]
    start_ellipse_params = se._find_patch_center_mo_distribution(start_slip_rate, slip_rate_thresh)
    end_ellipse_params = se._find_patch_center_mo_distribution(end_slip_rate, slip_rate_thresh)
    print(start_ellipse_params)
    print(end_ellipse_params)

    fig, (ax1, ax2, ax3) = plt.subplots(n_rows, n_cols, figsize=(12, 7.2))
    cascadia_map, cbar1 = plot_slip_subduction(fig, ax1, np.sum(slip_rate, axis=0), se.ma.fault_geometry,
                                        admissible_depth, utm_converted, None, return_cbar=True)

    _, cbar2 = plot_slip_subduction(fig, ax2, np.sum(start_slip_rate, axis=0), se.ma.fault_geometry,
                             admissible_depth, utm_converted, None, return_cbar=True)

    _, cbar3 = plot_slip_subduction(fig, ax3, np.sum(end_slip_rate, axis=0), se.ma.fault_geometry,
                                        admissible_depth, utm_converted, None, return_cbar=True)

    xc_start, yc_start, a_start, b_start, theta_start = start_ellipse_params
    xc_end, yc_end, a_end, b_end, theta_end = end_ellipse_params

    theta_start_deg = theta_start * 180 / np.pi  # subtract 90 to pass from focal axes to 'north'
    theta_end_deg = theta_end * 180 / np.pi

    xc_start, yc_start = cascadia_map(xc_start, yc_start)
    xc_end, yc_end = cascadia_map(xc_end, yc_end)
    a_start, b_start = a_start * 111. * 1e3, b_start * 111. * 1e3  # fake utm
    a_end, b_end = a_end * 111. * 1e3, b_end * 111. * 1e3  # fake utm

    ell_patch_start = Ellipse((xc_start, yc_start), 2 * a_start, 2 * b_start, angle=theta_start_deg,
                              edgecolor='red', facecolor='none', linewidth=2.)
    ell_patch_end = Ellipse((xc_end, yc_end), 2 * a_end, 2 * b_end, angle=theta_end_deg,
                              edgecolor='red', facecolor='none', linewidth=2.)

    ax2.add_patch(ell_patch_start)
    ax3.add_patch(ell_patch_end)

    cbar1.ax.set_ylabel('slip [mm]')
    cbar2.ax.set_ylabel('slip [mm]')
    cbar3.ax.set_ylabel('slip [mm]')

    ax1.set_title('Total slip')
    ax2.set_title('First 5 days')
    ax3.set_title('Last 5 days')

    plt.savefig('figures/fitted_ellipses.pdf', bbox_inches='tight')