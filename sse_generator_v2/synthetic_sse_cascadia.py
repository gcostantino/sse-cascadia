"""
Script used to model a synthetic slow slip event in the Cascadia subduction zone.
"""
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from utils import read_from_slab2, compute_geodesic_km_conversion_array, _find_nearest_val
from .okada import forward as okada85
import geopy.distance


def _synthetic_displacement_stations_cascadia_v4(i, depth_list, strike_list, dip_list, station_coordinates,
                                                 n_dislocations, random_idx_slab_array, int_bearing_vector, **params):
    """Same as '_synthetic_displacement_stations_cascadia_v3'. What's new:
    - when a dislocation is computed, this will be drawn in a vicinity of existing dislocations with probability p,
    otherwise its position will be randomly generated with probability 1-p."""
    displacement_all, epi_lat_all, epi_lon_all, hypo_depth_all, strike_all, dip_all, rake_all, u_all = [], [], [], [], [], [], [], []
    for n in range(n_dislocations):  # may also be zero
        p = params['uniform_vector'][i * n_dislocations + n, 3]
        if p < params['spatial_proba'] and n > 0:  # with probability p -> choose centroid +/-  variability
            epi_lat_initial = sum(epi_lat_all) / len(epi_lat_all)
            epi_lon_initial = sum(epi_lon_all) / len(epi_lon_all)
            random_distance = params['min_centroid_distance'] + (
                        params['max_centroid_distance'] - params['min_centroid_distance']) * params['uniform_vector'][
                                  i * n_dislocations + n, 4]
            random_bearing = int_bearing_vector[i * n_dislocations + n]
            destination_point = geopy.distance.distance(kilometers=random_distance).destination(
                (epi_lat_initial, epi_lon_initial), random_bearing)
            epi_lat_candidate, epi_lon_candidate = destination_point.latitude, destination_point.longitude
            epi_lat_candidate_slab, _ = _find_nearest_val(depth_list[:, 1], epi_lat_candidate)
            epi_lon_candidate_slab, _ = _find_nearest_val(depth_list[:, 0], epi_lon_candidate)

            idx_slab = np.where(
                np.logical_and(depth_list[:, 1] == epi_lat_candidate_slab, depth_list[:, 0] == epi_lon_candidate_slab))[
                0]
            if len(idx_slab) > 0:
                idx_slab = idx_slab[0]
            else:
                # that means that the chosen point surpassed the allowed SSE band. Try with a new one with opposite bearing
                random_bearing = random_bearing + 180
                destination_point = geopy.distance.distance(kilometers=random_distance).destination(
                    (epi_lat_initial, epi_lon_initial), random_bearing)
                epi_lat_candidate, epi_lon_candidate = destination_point.latitude, destination_point.longitude
                epi_lat_candidate_slab, _ = _find_nearest_val(depth_list[:, 1], epi_lat_candidate)
                epi_lon_candidate_slab, _ = _find_nearest_val(depth_list[:, 0], epi_lon_candidate)

                idx_slab = np.where(np.logical_and(depth_list[:, 1] == epi_lat_candidate_slab,
                                                   depth_list[:, 0] == epi_lon_candidate_slab))[0]
                if len(idx_slab) == 0:
                    continue  # never mind...
                else:
                    idx_slab = idx_slab[0]
            # print(epi_lat_candidate_slab,epi_lon_candidate_slab, 'idx_slab',idx_slab)
        else:  # with probability 1-p -> random choice
            idx_slab = random_idx_slab_array[i * n_dislocations + n]

        epi_lat = depth_list[idx_slab, 1]
        epi_lon = depth_list[idx_slab, 0]
        hypo_depth = - depth_list[idx_slab, 2]  # opposite sign for positive depths (Okada, 1985)
        depth_variability = -10 + 20 * params['uniform_vector'][i * n_dislocations + n, 0]
        if hypo_depth > 14.6:
            hypo_depth = hypo_depth + depth_variability
        if hypo_depth < 0:
            raise Exception('Negative depth')
        strike = strike_list[idx_slab, 2]
        dip = dip_list[idx_slab, 2]
        min_slip, max_slip = 1, 50  # mm
        min_rake, max_rake = 80, 100
        if 'rake_range' in params:
            min_rake, max_rake = params['rake_range']
        if 'slip_range' in params:
            min_slip, max_slip = params['slip_range']
        rake = min_rake + (max_rake - min_rake) * params['uniform_vector'][i * n_dislocations + n, 1]
        u = min_slip + (max_slip - min_slip) * params['uniform_vector'][i * n_dislocations + n, 2]  # mm
        L = W = 20  # km (square dislocations)

        if params['correct_latlon']:
            conv_coords = compute_geodesic_km_conversion_array(station_coordinates[:, :2], (epi_lat, epi_lon))
            lon_km, lat_km = conv_coords[:, 0], conv_coords[:, 1]
            displacement = okada85(lon_km, lat_km, 0, 0,
                                   hypo_depth + W / 2 * np.sin(np.deg2rad(dip)), L, W, u, 0, strike, dip, rake)
        else:
            displacement = okada85((station_coordinates[:, 1] - epi_lon) * 111.3194,
                                   (station_coordinates[:, 0] - epi_lat) * 111.3194, 0, 0,
                                   hypo_depth + W / 2 * np.sin(np.deg2rad(dip)), L, W, u, 0, strike, dip, rake)
        displacement_all.append(displacement)
        epi_lat_all.append(epi_lat)
        epi_lon_all.append(epi_lon)
        hypo_depth_all.append(hypo_depth)
        strike_all.append(strike)
        dip_all.append(dip)
        rake_all.append(rake)
        u_all.append(u)
    return displacement_all, epi_lat_all, epi_lon_all, hypo_depth_all, strike_all, dip_all, rake_all, u_all


def synthetic_displacements_stations_cascadia_v4(n, station_coordinates, **kwargs):
    """Similar to synthetic_displacements_stations_cascadia_v3, but:
     - when a dislocation is computed, this will be drawn in a vicinity of existing dislocations with probability p,
    otherwise its position will be randomly generated with probability 1-p."""
    if 'max_depth' in kwargs:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2(max_depth=kwargs['max_depth'])
    elif 'depth_range' in kwargs:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2(depth_range=kwargs['depth_range'])
    else:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2()
    max_n_dislocations = kwargs['max_n_disloc']
    n_random_elements_to_draw = 5  # slip, rake, depth variability, spatial_proba, rnd_distance
    uniform_vector = np.random.uniform(0, 1, (n * max_n_dislocations, n_random_elements_to_draw))
    int_bearing_vector = np.random.randint(0, 360, size=n * max_n_dislocations)
    n_dislocations_array = np.random.randint(low=0, high=1 + max_n_dislocations, size=n)
    random_idx_slab_array = np.random.randint(low=0, high=admissible_depth.shape[0], size=n * max_n_dislocations)
    results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=True)(
        delayed(_synthetic_displacement_stations_cascadia_v4)(i, admissible_depth, admissible_strike, admissible_dip,
                                                              station_coordinates, n_dislocations_array[i],
                                                              random_idx_slab_array, int_bearing_vector,
                                                              uniform_vector=uniform_vector,
                                                              **kwargs) for i in range(n))

    '''for i in range(n):
        for direction in range(3):
            disp_stations[i, :, direction] = results[i][0][direction]
        catalogue[i, :] = results[i][1:]'''

    displacement_list = [results[i][0] for i in range(n)]
    catalogue_list = [results[i][1:] for i in range(n)]
    return displacement_list, catalogue_list


def _sigmoid(x, alpha, beta, x0):
    return alpha / (1 + np.exp(-beta * (x - x0)))


def sigmoidal_rise_time(t, regime_value, duration, center, tol=0.01):
    gamma = regime_value * tol
    beta = 2 / duration * np.log((regime_value - gamma) / gamma)
    return _sigmoid(t, regime_value, beta, center)


def cosinusoidal_rise_time(t, regime_value, duration, center):
    full_template = np.zeros(t.shape)

    # Calculate start and end indices of the template
    template_start_idx = 0
    template_end_idx = duration

    # Calculate global start and end indices
    global_start_idx = max(0, center - duration // 2)
    global_end_idx = min(len(full_template), center + (duration + 1) // 2)

    # Adjust template indices for negative global_start_index
    if center - duration // 2 < 0:
        template_start_idx = max(0, duration // 2 - center)

    # Adjust template indices for exceeding global_end_index
    if center + (duration + 1) // 2 > len(full_template):
        template_end_idx = duration - (center + (duration + 1) // 2 - len(full_template))

    # Generate cosinusoidal template
    cos_template = 1 / 2 * (1 - np.cos(np.pi * np.arange(duration) / duration))

    # Assign template to full_template within valid indices
    full_template[global_start_idx:global_end_idx] = regime_value * cos_template[template_start_idx:template_end_idx]

    # Fill the rest of full_template with regime_value if needed
    if global_end_idx < len(full_template):
        full_template[global_end_idx:] = regime_value

    return full_template


def generate_synthetic_sses(n_samples, window_length, station_codes, station_coordinates, **kwargs):
    """As 'synthetic_sses_v3' but:
    - when a dislocation is computed, this will be drawn in a vicinity of existing dislocations with probability p,
    otherwise its position will be randomly generated with probability 1-p."""
    min_days, max_days = 5, 30
    synthetic_displacement, catalogue = synthetic_displacements_stations_cascadia_v4(n_samples, station_coordinates,
                                                                                     **kwargs)
    transients = np.zeros((n_samples, len(station_codes), window_length, 2))
    random_durations = np.random.randint(low=min_days, high=max_days, size=n_samples * kwargs['max_n_disloc'])
    random_center_values = np.random.randint(low=0, high=window_length, size=n_samples * kwargs['max_n_disloc'])

    transient_time_array = np.linspace(0, window_length, window_length)
    for sample in range(n_samples):
        for station in range(len(station_codes)):
            for direction in range(2):
                n_disloc = len(catalogue[sample][0])  # index 0 is used but all of them are equivalent
                for disloc in range(n_disloc):  # synth disp is a tuple, not np.array -> indexed as [direction][station]
                    transient = cosinusoidal_rise_time(transient_time_array,
                                                       synthetic_displacement[sample][disloc][direction][station],
                                                       random_durations[sample * kwargs['max_n_disloc'] + disloc],
                                                       random_center_values[sample * kwargs['max_n_disloc'] + disloc])
                    transients[sample, station, :, direction] += transient

    return transients, random_durations, synthetic_displacement, catalogue
