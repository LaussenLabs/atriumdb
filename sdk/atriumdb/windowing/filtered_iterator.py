# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from atriumdb.windowing.window import Window
from atriumdb.windowing.dataset_iterator import DatasetIterator
from atriumdb.windowing.windowing_functions import get_signal_dictionary, get_label_dictionary, \
    _get_patient_info_from_cache, _load_patient_cache, get_window_list


class FilteredDatasetIterator(DatasetIterator):
    """
    A specialized iterator for windowed segments of a dataset with additional filtering functionality.

    This iterator extends the functionality of DatasetIterator by allowing custom filtering and altering of data windows on the fly.

    For pure dataset filtering meant for repeated runs it is probably better to filter and save the `DatasetDefinition` object instead.

    This Iterator is best for if you need to modify the window, if you only need to iterate once or if your dataset is so large,
    waiting for `DatasetDefinition.filter` is impractical.

    :param function window_filter_fn: Function to filter windows during iteration. Takes a window object as input and returns
        True if the window should be included in the iteration, and False if it should be omitted. This allows for
        customized filtering based on specific criteria set within the function.
    """
    def __init__(self, sdk, definition, window_duration_ns: int, window_slide_ns: int, num_windows_prefetch: int = None,
                 label_threshold=0.5, shuffle=False, max_cache_duration=None, window_filter_fn=None,
                 patient_history_fields: list = None, label_exact_match=False):
        super().__init__(sdk=sdk, definition=definition, window_duration_ns=window_duration_ns,
                         window_slide_ns=window_slide_ns, num_windows_prefetch=num_windows_prefetch,
                         label_threshold=label_threshold, shuffle=shuffle, max_cache_duration=max_cache_duration,
                         patient_history_fields=patient_history_fields, label_exact_match=label_exact_match)
        self.window_filter_fn = window_filter_fn

    def _load_batch_matrix(self, idx: int):
        batch_index, batch_start_index, batch_end_index, batch_num_windows, batch_size = \
            self._calculate_batch_size(idx)

        window_cache = []

        batch_data = self.batch_info[batch_index]
        patient_info_cache = {}
        patient_history_cache = {}

        for source_index, (source_type, source_identifier, source_batch_start_time, source_batch_end_time, range_start_time, range_end_time, range_num_windows) in enumerate(batch_data):
            device_id, patient_id, query_patient_id = self.unpack_source_info(source_identifier, source_type)

            _load_patient_cache(patient_id, patient_info_cache, patient_history_cache, self.sdk, self.patient_history_fields)

            source_batch_data_dictionary = get_signal_dictionary(
                self.sdk, device_id, query_patient_id, self.window_duration_ns, self.window_slide_ns, self.measures,
                source_batch_start_time, source_batch_end_time, batch_num_windows, range_start_time, range_end_time)

            sliced_labels, threshold_labels = get_label_dictionary(
                self.sdk, device_id, query_patient_id, source_batch_start_time, source_batch_end_time, self.label_sets,
                self.label_threshold, range_num_windows, self.row_period_ns, self.row_size, self.slide_size, label_exact_match=self.label_exact_match)

            patient_history_fields = self.patient_history_fields

            measures = self.measures
            window_duration_ns = self.window_duration_ns
            num_windows_for_batch = range_num_windows

            batch_window_list = get_window_list(device_id, patient_id, measures, source_batch_data_dictionary,
                                                     source_batch_start_time, num_windows_for_batch, window_duration_ns,
                                                     threshold_labels, sliced_labels, patient_history_cache,
                                                     patient_history_fields, patient_info_cache)
            if self.window_filter_fn:
                filtered_window_list = [window for window in batch_window_list if self.window_filter_fn(window)]
                batch_window_list = filtered_window_list

            window_cache.extend(batch_window_list)

        if self.random_gen is not None:
            self.random_gen.shuffle(window_cache)

        if self.random_gen is not None:
            self.random_gen.shuffle(window_cache)

        self.window_cache = window_cache

        self.current_batch_start_index = batch_start_index
        self.current_batch_end_index = batch_end_index
