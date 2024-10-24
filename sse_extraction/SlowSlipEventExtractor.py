import matplotlib.pyplot as plt
import numpy as np

from denoised_data_utils.DenoisedDataHandler import DenoisedDataHandler
from slip_modeling.ModelAnalyzer import ModelAnalyzer
from sse_extraction.sse_extraction_from_slip import get_events_from_slip_model, refine_durations
from utils.ellipse_fitting import fit_ellipse_mo, ConvergenceError, median_xy_mo


class SlowSlipEventExtractor:
    def __init__(self, slip_thresholds=(0.07, 0.3, 0.5, 0.7), base_sse_info_folder='../../DATA/sse-cascadia/sse_info'):
        self.ma = ModelAnalyzer()
        self.ddh = DenoisedDataHandler()
        self.slip_thresholds = slip_thresholds
        self.base_sse_info_folder = base_sse_info_folder
        self.sse_info_thresh = None
        self.new_duration_dict = None

    def _extract_events_unfiltered(self, load=True, spatiotemporal=True, cut_neg_slip=True):
        sse_info_thresh, _ = get_events_from_slip_model(self.slip_thresholds, self.ma.slip_rates,
                                                        self.ma.signed_slip_rates, self.ma.area,
                                                        self.ddh.corrected_time,
                                                        self.ma.x_centr_lon, self.ma.y_centr_lat,
                                                        len(self.ddh.corrected_time),
                                                        shear_modulus=self.ma.shear_modulus,
                                                        load=load, spatiotemporal=spatiotemporal,
                                                        cut_neg_slip=cut_neg_slip,
                                                        base_folder=self.base_sse_info_folder)
        new_duration_dict = refine_durations(self.slip_thresholds, sse_info_thresh, mo_rate_percentage=.95)
        self.sse_info_thresh = sse_info_thresh
        self.new_duration_dict = new_duration_dict

    def get_extracted_events_unfiltered(self):
        if self.sse_info_thresh is None and self.new_duration_dict is None:
            self._extract_events_unfiltered()
        return self.sse_info_thresh, self.new_duration_dict

    def get_moment_rate_events(self, thresh: float, refined_durations: bool):
        *_, mo_rate_list, _ = self.sse_info_thresh[thresh]
        if refined_durations:
            mo_rate_list_all_events = []
            for i, mo_rate in enumerate(mo_rate_list):
                idx = self.new_duration_dict[thresh][i]
                new_start, new_end = idx
                mo_rate_list_all_events.append(mo_rate[new_start:new_end])
        else:
            mo_rate_list_all_events = mo_rate_list
        return mo_rate_list_all_events

    def get_event_date_idx(self, thresh: float, refined_durations: bool):
        """Returns the absolute event date indexing, to be used with GNSS time array.
        N.W.: here, the end idx is not expressed in slicing notation, to be consistent with the event extraction
        method. When using this in slices, remember to add 1: [start:end + 1]."""
        date_list_idx_all_events = []
        *_, date_idx_list, _, _ = self.sse_info_thresh[thresh]
        if refined_durations:
            for i, date in enumerate(date_idx_list):
                idx = self.new_duration_dict[thresh][i]
                new_start_date = date_idx_list[i][0] + idx[0]
                new_end_date = date_idx_list[i][0] + idx[1] - 1  # NW: end idx NOT in slicing notation
                date_list_idx_all_events.append((new_start_date, new_end_date))
        else:
            date_list_idx_all_events = date_idx_list
        return date_list_idx_all_events

    def _find_patch_center_mo_distribution(self, mo_rate, mo_thresh: float, max_attempts: int = 10):
        attempts = 0
        original_mo_thresh = mo_thresh
        while attempts < max_attempts:
            try:
                partial_mo = np.sum(mo_rate, axis=0)  # partial Mo distribution for the first delta_win days
                ellipse_params = fit_ellipse_mo(self.ma.x_centr_lon, self.ma.y_centr_lat, partial_mo,
                                                mo_thresh=mo_thresh)
                return ellipse_params
            except ConvergenceError:
                mo_thresh -= .05
                attempts += 1
        # last resort : compute the median position
        return median_xy_mo(self.ma.x_centr_lon, self.ma.y_centr_lat, np.sum(mo_rate, axis=0), original_mo_thresh)

    def get_start_end_patch(self, thresh: float, delta_win: int = 3, mo_thresh: float = .5, show=False):
        start_points, end_points = [], []
        self.sse_info_thresh, self.new_duration_dict = self.get_extracted_events_unfiltered()
        mo_rates = self.get_moment_rate_events(thresh, refined_durations=False)
        for ev_idx, mo_rate in enumerate(mo_rates):

            print('event ID:', ev_idx, 'LEN', len(mo_rate))
            start_mo_rate, end_mo_rate = mo_rate[:delta_win], mo_rate[-delta_win:]
            print(start_mo_rate.shape, end_mo_rate.shape)
            xc_start, yc_start, *_ = self._find_patch_center_mo_distribution(start_mo_rate, mo_thresh)
            xc_end, yc_end, *_ = self._find_patch_center_mo_distribution(end_mo_rate, mo_thresh)
            start_points.append((xc_start, yc_start))
            end_points.append((xc_end, yc_end))
            if show:
                plt.scatter(self.ma.x_centr_lon, self.ma.y_centr_lat, c=start_mo_rate)
                plt.scatter(xc_start, yc_start, marker='x', c='red')
                plt.show()

                plt.scatter(self.ma.x_centr_lon, self.ma.y_centr_lat, c=end_mo_rate)
                plt.scatter(xc_end, yc_end, marker='x', c='red')
                plt.show()
        return start_points, end_points

    def visualize_events(self):
        # Code to visualize slow slip events
        pass
