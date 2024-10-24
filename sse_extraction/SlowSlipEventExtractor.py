import numpy as np

from denoised_data_utils.DenoisedDataHandler import DenoisedDataHandler
from slip_modeling.ModelAnalyzer import ModelAnalyzer
from sse_extraction.sse_extraction_from_slip import get_events_from_slip_model, refine_durations


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
        *_, date_idx_list, _, _ = self.sse_info_thresh[thresh]
        if refined_durations:
            date_list_idx_all_events = []
            for i, date in enumerate(date_idx_list):
                idx = self.new_duration_dict[thresh][i]
                new_start_date = date_idx_list[i][0] + idx[0]
                new_end_date = date_idx_list[i][0] + idx[1] - 1  # NW: end idx NOT in slicing notation
                date_list_idx_all_events.append((new_start_date, new_end_date))
        else:
            date_list_idx_all_events = date_idx_list
        return date_list_idx_all_events

    def get_start_end_patch(self, thresh):
        self.sse_info_thresh, self.new_duration_dict = self.get_extracted_events_unfiltered()

    def visualize_events(self):
        # Code to visualize slow slip events
        pass
