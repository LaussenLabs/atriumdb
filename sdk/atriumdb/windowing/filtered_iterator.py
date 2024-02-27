import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from atriumdb.windowing.window import Window
from atriumdb.windowing.dataset_iterator import DatasetIterator


class FilteredDatasetIterator(DatasetIterator):
    """
    A specialized iterator for windowed segments of a dataset with additional filtering functionality.

    This iterator extends the functionality of DatasetIterator by allowing custom filtering of data windows. It is
    useful in scenarios where specific criteria need to be applied to determine which data windows are relevant for
    further processing or analysis.

    :param window_filter_fn: Function to filter windows during iteration. Takes a window object as input and returns
        True if the window should be included in the iteration, and False if it should be omitted. This allows for
        customized filtering based on specific criteria set within the function.
    :type window_filter_fn: function, optional
    """
    def __init__(self, sdk, validated_measure_list, validated_label_set_list, validated_sources,
                 window_duration_ns: int, window_slide_ns: int, num_windows_prefetch: int = None,
                 label_threshold=0.5, time_units=None, shuffle=False, max_cache_duration=None, window_filter_fn=None,
                 patient_history_fields: list = None):
        super().__init__(sdk, validated_measure_list, validated_label_set_list, validated_sources,
                         window_duration_ns, window_slide_ns, num_windows_prefetch, label_threshold, time_units,
                         shuffle, max_cache_duration, patient_history_fields)
        self.window_filter_fn = window_filter_fn

    def _load_batch_matrix(self, idx: int):
        batch_index, batch_start_index, batch_end_index, batch_num_windows, batch_size = \
            self._calculate_batch_size(idx)

        window_cache = []
        matrix_cache = []

        num_filtered_windows = 0
        # batch_data = [self.batch_info[batch_index]]
        batch_data = self.batch_info[batch_index]
        patient_info_cache = {}
        patient_history_cache = {}

        for source_index, (source_type, source_identifier, source_batch_start_time, source_batch_end_time, range_start_time, range_end_time, range_num_windows) in enumerate(batch_data):
            range_size = self.row_size + (range_num_windows - 1) * self.slide_size
            # Pre load source matrix and associated times
            source_matrix = np.full((len(self.measures), range_size), np.nan)

            quantized_end_time = source_batch_start_time + (range_size * self.row_period_ns)
            source_time_array = np.arange(source_batch_start_time, quantized_end_time, self.row_period_ns)

            device_id, patient_id, query_patient_id = self.unpack_source_info(source_identifier, source_type)

            self._load_patient_cache(patient_id, patient_info_cache, patient_history_cache)

            source_batch_data_dictionary = self._query_source_data(
                device_id, query_patient_id, source_batch_start_time, source_batch_end_time, range_start_time,
                range_end_time, range_num_windows, range_size, source_matrix)

            windowed_views = sliding_window_view(
                source_matrix, (len(self.measures), self.row_size), axis=None)

            windowed_times = sliding_window_view(source_time_array, self.row_size)

            sliced_views = windowed_views[0][::self.slide_size]
            sliced_times = windowed_times[::self.slide_size]

            sliced_labels, threshold_labels = self._get_source_label_data(
                device_id, query_patient_id, source_batch_start_time, source_batch_end_time, source_time_array)

            for window_i in range(len(sliced_views)):
                signal_dictionary = {}
                for measure in self.measures:
                    measure_id = measure['id']
                    measure_tag = measure['tag']
                    measure_freq_nhz = measure['freq_nhz']
                    measure_freq_hz = float(measure_freq_nhz / (10 ** 9))
                    measure_units = measure['units']

                    window_times = source_batch_data_dictionary[measure_id][0][window_i]

                    window_values = source_batch_data_dictionary[measure_id][1][window_i]
                    measure_expected_count = source_batch_data_dictionary[measure_id][2]

                    signal_dictionary[(measure_tag, measure_freq_hz, measure_units)] = \
                        {
                            'times': np.copy(window_times),
                            'values': np.copy(window_values),
                            'expected_count': measure_expected_count,
                            'actual_count': np.sum(~np.isnan(window_values)),
                            'measure_id': measure_id,
                        }

                label_time_series = np.copy(sliced_labels[window_i]) if \
                    sliced_labels is not None else None

                window_classification = threshold_labels[window_i] if \
                    threshold_labels is not None else None

                window_start_time = int(sliced_times[window_i][0])

                # Get patient info
                window_patient_info = self._get_patient_info_from_cache(
                    patient_id, window_start_time, patient_info_cache, patient_history_cache)

                result_window = Window(
                    signals=signal_dictionary,
                    start_time=window_start_time,
                    device_id=device_id,
                    patient_id=patient_id,
                    label_time_series=label_time_series,
                    label=window_classification,
                    patient_info=window_patient_info
                )

                if not self.window_filter_fn or self.window_filter_fn(result_window):
                    window_cache.append(result_window)
                    matrix_cache.append(np.copy(sliced_views[window_i]))
                else:
                    num_filtered_windows += 1

        if self.random_gen is not None:
            self.random_gen.shuffle(window_cache)
            self.random_gen.shuffle(matrix_cache)

        self.window_cache = window_cache
        self.matrix_cache = matrix_cache

        self.current_batch_start_index = batch_start_index
        self.current_batch_end_index = batch_end_index
