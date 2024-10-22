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

    def _extract_events(self, load=True, spatiotemporal=True, cut_neg_slip=True):
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

    def get_extracted_events(self):
        if self.sse_info_thresh is None and self.new_duration_dict is None:
            self._extract_events()
        return self.sse_info_thresh, self.new_duration_dict

    def filter_events(self, criteria):
        # Code to filter extracted events based on some criteria
        pass

    def visualize_events(self):
        # Code to visualize slow slip events
        pass
