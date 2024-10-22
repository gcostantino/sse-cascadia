import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

from FUNCTIONS.functions_slab import GEO_UTM
from st_dbscan_giuseppe.st_dbscan import ST_DBSCAN


def cluster_events(slip, time_array, x_centr_lon, y_centr_lat, n_time_steps, slip_thresh, spatiotemporal=True):
    if spatiotemporal:
        cluster_dict = _space_time_sse_clustering(slip, time_array, x_centr_lon, y_centr_lat, slip_thresh)
        return cluster_dict
    else:
        cluster_list = _space_sse_clustering(slip, x_centr_lon, y_centr_lat, slip_thresh)
        patch_time_matrix = _create_patch_time_matrix(cluster_list, slip, y_centr_lat, n_time_steps)
        bounding_box_polygons = _patch_time_sse_clustering(patch_time_matrix, time_array, y_centr_lat)
        merged_bboxes = _merge_overlapping_polygons(bounding_box_polygons)
        merged_bboxes = [poly.bounds for poly in merged_bboxes]
        return merged_bboxes


def _merge_overlapping_polygons(polygons):
    merged_polygons = []
    while polygons:
        # Pop the first polygon from the list
        current_polygon = polygons.pop(0)
        # Check for overlapping polygons
        overlapping_polygons = [polygon for polygon in polygons if polygon.intersects(current_polygon)]
        # Merge overlapping polygons
        for polygon in overlapping_polygons:
            current_polygon = current_polygon.union(polygon)
            polygons.remove(polygon)
        # Append merged polygon to the result list
        merged_polygons.append(current_polygon)
    return merged_polygons


def _latsort(y_centr_lat):
    latsort = np.argsort(y_centr_lat)[::-1]
    return latsort


def _create_patch_time_matrix(cluster_list, slip, y_centr_lat, n_time_steps):
    n_patches = slip.shape[1]
    patch_time_matrix = np.zeros((n_time_steps, n_patches))  # mask
    patch_time_matrix.fill(np.nan)
    for i in range(len(slip)):
        cluster_i = cluster_list[i]
        for key in cluster_i:
            cluster_idx = cluster_i[key]
            patch_time_matrix[i, cluster_idx] = 1
    patch_time_matrix = patch_time_matrix[:, _latsort(y_centr_lat)]
    return patch_time_matrix


def _space_time_sse_clustering_dev(slip, time_array, x_centr_lon, y_centr_lat, slip_thresh, t_eps=5, s_eps=0.5,
                                   min_samples=50):
    '''slip[slip < slip_thresh] = np.nan
    plt.matshow(slip[:, _latsort(y_centr_lat)].T, vmax=2, aspect='auto', origin='upper')
    plt.colorbar()
    plt.show()'''
    st_points = []
    for i in range(len(slip)):  # time steps
        valid_patches = np.where(slip[i] > slip_thresh)[0]
        for valid_patch_idx in valid_patches:
            # st_points.append([time_array[i], x_centr_lon[valid_patch_idx], y_centr_lat[valid_patch_idx]])
            st_points.append([i, x_centr_lon[valid_patch_idx], y_centr_lat[valid_patch_idx]])
    valid_xyt = np.array(st_points)  # sorted by time
    print(valid_xyt.shape)
    dbscan = ST_DBSCAN(eps1=s_eps, eps2=t_eps, min_samples=min_samples, metric='euclidean',
                       n_jobs=multiprocessing.cpu_count())  # 50 km, 5 days
    # dbscan.fit(valid_xyt)
    dbscan.fit_frame_split(valid_xyt, frame_size=120, frame_overlap=t_eps)  # 4-month framing
    labels = dbscan.labels
    print(labels)
    for cluster_label in set(labels) - {-1}:
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
        plt.scatter(valid_xyt[cluster_indices, 0], valid_xyt[cluster_indices, 2])

    noise_indices = [i for i, label in enumerate(labels) if label == -1]
    plt.scatter(valid_xyt[noise_indices, 0], valid_xyt[noise_indices, 2], color='C0', marker='x')
    plt.show()


def _space_time_sse_clustering(slip, time_array, x_centr_lon, y_centr_lat, slip_thresh, t_eps=5, s_eps=50,
                               min_samples=50):

    x_centr, y_centr = GEO_UTM(x_centr_lon, y_centr_lat)  # km

    st_points = []
    for i in range(len(slip)):  # time steps
        valid_patches = np.where(slip[i] > slip_thresh)[0]
        for valid_patch_idx in valid_patches:
            # st_points.append([time_array[i], x_centr_lon[valid_patch_idx], y_centr_lat[valid_patch_idx]])
            #st_points.append([i, x_centr_lon[valid_patch_idx], y_centr_lat[valid_patch_idx]])
            st_points.append([i, x_centr[valid_patch_idx], y_centr[valid_patch_idx]])
    valid_xyt = np.array(st_points)  # sorted by time

    dbscan = ST_DBSCAN(eps1=s_eps, eps2=t_eps, min_samples=min_samples, temporal_metric='euclidean',
                       spatial_metric='euclidean', n_jobs=multiprocessing.cpu_count())  # 50 km, 5 days
    #dbscan.fit_frame_split(valid_xyt, frame_size=120, frame_overlap=t_eps)  # 4-month framing
    dbscan.fit(valid_xyt)
    labels = dbscan.labels

    '''for cluster_label in set(labels) - {-1}:
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
        plt.scatter(valid_xyt[cluster_indices, 0], valid_xyt[cluster_indices, 2])
        plt.xlim([0, 800])
        #plt.ylim([40, 50])
        plt.ylim([y_centr.min(), y_centr.max()])
        plt.title(f'cluster {cluster_label}, {len(cluster_indices)} elements')
    plt.show()'''

    '''noise_indices = [i for i, label in enumerate(labels) if label == -1]
    plt.scatter(valid_xyt[noise_indices, 0], valid_xyt[noise_indices, 2], color='C0', marker='x')
    plt.xlim([0, 800])
    plt.ylim([40, 50])
    plt.show()'''

    clusters = dict()  # for each cluster: dict() -> key: time, value: list of activated patches at time t
    for cluster_label in set(labels) - {-1}:
        cluster_dict = dict()
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
        for xyt_point in valid_xyt[cluster_indices]:
            cluster_t = int(xyt_point[0])
            if cluster_t not in cluster_dict:
                cluster_dict[cluster_t] = []
            # idx_patch = np.where((x_centr_lon == xyt_point[1]) & (y_centr_lat == xyt_point[2]))[0][0]
            idx_patch = np.where((x_centr == xyt_point[1]) & (y_centr == xyt_point[2]))[0][0]
            cluster_dict[cluster_t].append(idx_patch)
        clusters[cluster_label] = cluster_dict

    '''for key in clusters.keys():
        for t in clusters[key].keys():
            plt.scatter(x_centr_lon[clusters[key][t]], y_centr_lat[clusters[key][t]])
            plt.title(f'cluster #{key + 1}, time: {t}')
            plt.xlim([-130, -120])
            plt.ylim([40, 50])
            plt.show()'''

    '''for key in clusters.keys():
        for t in clusters[key].keys():
            # plt.scatter(x_centr_lon[clusters[key][t]], y_centr_lat[clusters[key][t]])
            plt.scatter(x_centr[clusters[key][t]], y_centr[clusters[key][t]])
        plt.title(f'cluster #{key + 1}')
        plt.xlim([x_centr.min(), x_centr.max()])
        plt.ylim([y_centr.min(), y_centr.max()])
        plt.show()'''
    return clusters


def _space_sse_clustering(slip, x_centr_lon, y_centr_lat, slip_thresh, eps=0.5, min_samples=5):
    cluster_list = []  # each element will be a dictionary containing the patch idx of a cluster element
    for i in range(len(slip)):
        patch_idx = slip[i] > slip_thresh
        valid_xy = np.vstack((x_centr_lon[patch_idx], y_centr_lat[patch_idx])).T
        clusters = dict()
        if len(valid_xy) > 0:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(valid_xy)
            labels = dbscan.labels_
            for cluster_label in set(labels) - {-1}:
                cluster_indices = [j for j, label in enumerate(labels) if label == cluster_label]
                # express the indices as referred to the original 'full' geometry
                clusters[cluster_label] = np.where(patch_idx)[0][cluster_indices]
        cluster_list.append(clusters)
    return cluster_list


def _patch_time_sse_clustering(patch_time_matrix, time_array, y_centr_lat, time_patch_eps=10, patch_time_min_samples=5):
    latsort = _latsort(y_centr_lat)
    valid_xy_patch_time = np.argwhere(~np.isnan(patch_time_matrix))

    dbscan = DBSCAN(eps=time_patch_eps, min_samples=patch_time_min_samples)
    dbscan.fit(valid_xy_patch_time)
    labels = dbscan.labels_

    cluster_polygons = []
    bounding_box_polygons = []

    for cluster_label in set(labels) - {-1}:
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
        cluster_time_patch_idx = valid_xy_patch_time[cluster_indices]
        # build convexHull
        is_flat = np.all(cluster_time_patch_idx[:, 0] == cluster_time_patch_idx[0, 0]) or np.all(
            cluster_time_patch_idx[:, 1] == cluster_time_patch_idx[0, 1])
        if is_flat:
            continue
        # transform points to time and patch latitude
        time_index, patch_index = cluster_time_patch_idx[:, 0], cluster_time_patch_idx[:, 1]
        cluster_time_lat_idx = np.vstack((time_array[time_index], y_centr_lat[latsort][patch_index])).T
        hull = ConvexHull(cluster_time_lat_idx)
        hull_vertices = cluster_time_lat_idx[hull.vertices]
        # build a polygon out of the convex hull
        polygon = Polygon(hull_vertices)
        cluster_polygons.append(polygon)
        # build rectangular bounding box
        minx, miny, maxx, maxy = polygon.bounds
        bounding_box_polygon = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
        bounding_box_polygons.append(bounding_box_polygon)

    return bounding_box_polygons
