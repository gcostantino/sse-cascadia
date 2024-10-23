import os.path

import joblib
import numpy as np

from FUNCTIONS.functions_slab import UTM_GEO
from sse_extraction.sse_clustering_utils import cluster_events


def get_events_from_slip_model(slip_thresholds, slip, signed_slip, area, time_array, x_centr_lon, y_centr_lat,
                               n_time_steps, shear_modulus=30e9, load=True, spatiotemporal=False, cut_neg_slip=False,
                               base_folder='.'):
    """Expects slip in mm"""
    if not load:
        sse_info_thresh = dict()  # contains sse information for each threshold
        event_cluster_dict = dict()
        for slip_thresh in slip_thresholds:
            print('Evaluating thresh:', slip_thresh, 'mm/day')
            events = cluster_events(slip, time_array, x_centr_lon, y_centr_lat, n_time_steps, slip_thresh,
                                    spatiotemporal=spatiotemporal)
            sse_info = extract_information_from_events(events, slip, signed_slip, area, x_centr_lon, y_centr_lat,
                                                       time_array, shear_modulus=shear_modulus,
                                                       spatiotemporal=spatiotemporal, cut_neg_slip=cut_neg_slip)
            sse_info_thresh[slip_thresh] = sse_info
            event_cluster_dict[slip_thresh] = events
        joblib.dump(sse_info_thresh, os.path.join(base_folder, 'sse_info_thresh' + ('_spatiotemporal' if spatiotemporal else '') + ('_nonneg_slip' if cut_neg_slip else '')))
        joblib.dump(event_cluster_dict, os.path.join(base_folder, 'event_clusters' + ('_spatiotemporal' if spatiotemporal else '') + ('_nonneg_slip' if cut_neg_slip else '')))
    else:
        sse_info_thresh = joblib.load(os.path.join(base_folder, 'sse_info_thresh' + ('_spatiotemporal' if spatiotemporal else '') + ('_nonneg_slip' if cut_neg_slip else '')))
        event_cluster_dict = joblib.load(os.path.join(base_folder, 'event_clusters' + ('_spatiotemporal' if spatiotemporal else '') + ('_nonneg_slip' if cut_neg_slip else '')))
    return sse_info_thresh, event_cluster_dict


def extract_information_from_events(events, slip, signed_slip, area, x_centr_lon, y_centr_lat, time_array,
                                    shear_modulus=30e9, spatiotemporal=False, cut_neg_slip=False):
    """Expects area in meters^2 and slip in meters"""
    # slow slip events are now available
    event_duration_list = []
    event_moment_list = []
    event_area_list = []
    slip_event_list = []
    patch_idx_list = []
    date_list = []
    mo_rate_list = []
    slip_rate_list = []
    if spatiotemporal:
        for ev_id in events.keys():
            time_keys = list(events[ev_id])
            min_t_idx, max_t_idx = min(time_keys), max(time_keys)
            duration = max_t_idx - min_t_idx + 1  # * 365
            slip_rate_array = np.zeros((duration, slip.shape[1]))
            slip_potency_array = np.zeros((duration, slip.shape[1]))
            mo_rate_array = np.zeros((duration, slip.shape[1]))
            global_patch_idx = set()
            for t in events[ev_id].keys():  # initialize dictionaries
                patch_ev_idx = events[ev_id][t]
                global_patch_idx.update(patch_ev_idx)
                # slip_rate_dict[t] = np.zeros(slip.shape[1])
                # slip_potency_dict[t] = np.zeros(slip.shape[1])
                # mo_rate_dict[t] = np.zeros(slip.shape[1])
            global_patch_idx = list(global_patch_idx)

            for t in events[ev_id].keys():
                patch_ev_idx = events[ev_id][t]
                slip_rate_array[t - min_t_idx, patch_ev_idx] = slip[
                    t, patch_ev_idx]  # slip_rate_dict[t][patch_ev_idx] = slip[t, patch_ev_idx]
                slip_potency = slip[t, patch_ev_idx] * 1e-03 * area[patch_ev_idx]
                if cut_neg_slip:
                    neg_slip_mask = signed_slip[t, patch_ev_idx] < 0
                    slip_potency[neg_slip_mask] = 0.  # masks the negative slip to zero
                slip_potency_array[
                    t - min_t_idx, patch_ev_idx] = slip_potency  # slip_potency_dict[t][patch_ev_idx] = slip_potency
                mo_rate_array[
                    t - min_t_idx, patch_ev_idx] = shear_modulus * slip_potency  # mo_rate_dict[t][patch_ev_idx] = shear_modulus * slip_potency
                '''plt.scatter(x_centr_lon[patch_ev_idx], y_centr_lat[patch_ev_idx], c=slip[t, patch_ev_idx])
                plt.xlim([-130, -120])
                plt.ylim([40, 50])
                plt.show()'''

            total_ev_mo = np.sum(mo_rate_array)  # sum(mo_rate_dict)
            total_ev_area = area[global_patch_idx] * 1e-6  # convert to km^2
            event_moment_list.append(total_ev_mo)
            event_area_list.append(np.sum(total_ev_area))
            slip_event_list.append(np.sum(slip_rate_array))  # cumulative slip
            patch_idx_list.append(global_patch_idx)
            date_list.append([min_t_idx, max_t_idx])
            event_duration_list.append(duration)
            mo_rate_list.append(mo_rate_array)
            slip_rate_list.append(slip_rate_array)
        return event_moment_list, event_duration_list, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list
    else:
        latsort = np.argsort(y_centr_lat)[::-1]
        for bounding_box in events:
            min_t, min_lat, max_t, max_lat = bounding_box
            duration = (max_t - min_t) * 365
            event_duration_list.append(duration)
            # get back to the patches to retrieve the slip and thus the Mo
            patch_ev_idx = np.where((y_centr_lat[latsort] >= min_lat) & (y_centr_lat[latsort] <= max_lat))[0]
            start_idx, end_idx = np.where(time_array == min_t)[0][0], np.where(time_array == max_t)[0][0]
            slip_potency = slip[:, latsort][start_idx:end_idx, patch_ev_idx] * 1e-03 * area[latsort][patch_ev_idx]
            Mo = shear_modulus * slip_potency
            total_moment = np.sum(Mo)
            event_moment_list.append(total_moment)
            event_areas = area[latsort][patch_ev_idx] * 1e-6  # convert to km^2
            event_area_list.append(np.sum(event_areas))
            slip_event_list.append(slip[:, latsort][start_idx:end_idx, patch_ev_idx])
            patch_idx_list.append(patch_ev_idx)
            date_list.append([min_t, max_t])
        event_moment_list = np.array(event_moment_list)
        event_duration_list = np.array(event_duration_list)
        event_area_list = np.array(event_area_list)
        # slip_event_list = np.array(slip_event_list)
        # patch_idx_list = np.array(patch_idx_list)
        date_list = np.array(date_list)
        return event_moment_list, event_duration_list, event_area_list, slip_event_list, patch_idx_list, date_list


def refine_durations(slip_thresholds, sse_info_thresh, min_duration=5, mo_rate_percentage=.99):
    new_duration_dict = dict()
    for thresh in slip_thresholds:
        event_moment_list, event_duration_list, event_area_list, slip_event_list, patch_idx_list, date_list, mo_rate_list, slip_rate_list = \
            sse_info_thresh[thresh]
        new_duration_list = []
        for i in range(len(mo_rate_list)):
            mo_rate_fcn = np.sum(mo_rate_list[i], axis=1)
            tot_moment = np.sum(mo_rate_fcn)
            win_length = len(mo_rate_fcn)
            t1 = 0
            t2 = win_length
            loss = np.inf
            best_model = None
            initial_duration = t2 - t1 + 1
            if initial_duration > min_duration:
                for a in range(win_length // 2):
                    curr_t1 = t1 + a
                    for b in range(win_length // 2):
                        curr_t2 = t2 - b
                        partial_integral = np.sum(mo_rate_fcn[curr_t1:curr_t2])
                        curr_loss = (partial_integral - mo_rate_percentage * tot_moment) ** 2
                        if curr_loss < loss:
                            loss = curr_loss
                            best_model = [curr_t1, curr_t2]

                t1, t2 = best_model
            new_duration_list.append((t1, t2))
        new_duration_dict[thresh] = new_duration_list
    return new_duration_dict


if __name__ == '__main__':
    preferred_L = 100  # km
    preferred_sigma_m = 0.4  # mm
    preferred_sigma_d = 0.1
    slip_model = joblib.load(
        f'inversed_ts/dump_50km/slip_inversion_sigma_d_01/inversed_ts_slip5_1000_noise_with_trend_demean_corrlength_{preferred_L}km_sigmam_{preferred_sigma_m}_sigmanoise3D_{preferred_sigma_d}')
    geometry, recs = joblib.load('geometry_cascadia')
    green_td = joblib.load('green_td_cascadia')
    station_coordinates = np.loadtxt('INPUT_FILES/stations_cascadia_200.txt', usecols=(1, 2))

    with np.load('denoised_ts_slip5_1000_noise_with_trend_demean.npz') as f:
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

    area = geometry[:, 21] * 1e+6  # in m^2

    shear_modulus = 30 * 1e9  # GPa

    x_centr_lon, y_centr_lat = UTM_GEO(geometry[:, 9], geometry[:, 10])

    slip_thresholds = [0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1]

    sse_info_thresh = get_events_from_slip_model(slip_thresholds, slip, area, corrected_time, x_centr_lon, y_centr_lat,
                                                 n_time_steps, shear_modulus=shear_modulus, load=False,
                                                 spatiotemporal=True)
