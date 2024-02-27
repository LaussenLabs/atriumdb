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


from bisect import bisect_right
import warnings
import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from atriumdb.windowing.window import Window
from atriumdb.windowing.windowing_functions import get_threshold_labels, find_closest_measurement


class DatasetIterator:
    """
    Iterator over windowed segments of a dataset.

    Allows efficient iterative access to sliding windows of data from different sources (e.g., devices or patients)
    by organizing data into batches and loading one batch at a time.

    :param sdk: SDK object to fetch data
    :type sdk: AtriumSDK
    :param validated_measure_list: List of validated measures with information about each measure
    :type validated_measure_list: list
    :param validated_sources: Dictionary containing sources with associated time ranges
    :type validated_sources: dict
    :param window_duration_ns: Duration of each window in nanoseconds
    :type window_duration_ns: int
    :param window_slide_ns: Interval in nanoseconds by which the window advances in time
    :type window_slide_ns: int
    :param num_windows_prefetch: Number of windows you want to get from AtriumDB at a time. Setting this value
            higher will make decompression faster but at the expense of using more RAM. (default the number of windows
            that gets you closest to 10 million values).
    :type num_windows_prefetch: int, optional
    :param time_units: If you would like the window_duration and window_slide to be specified in units other than
                            nanoseconds you can choose from one of ["s", "ms", "us", "ns"].
    :type time_units: str
    :param shuffle: If True, shuffle the order of windows before iterating. If an integer, it will initialize
                    a seeded random number generator for reproducible shuffling. If False or None, no shuffling occurs.
    :type shuffle: Union[bool, None, int]
    :param max_cache_duration: If specified, no single cache will have a time range larger than this duration.
                               The time range will be split accordingly. The duration must be larger than window_duration_ns.
    :type max_cache_duration: int, optional
    :param list patient_history_fields: A list of patient_history fields you would like returned in the Window object.
    """

    def __init__(self, sdk, validated_measure_list, validated_label_set_list, validated_sources,
                 window_duration_ns: int, window_slide_ns: int, num_windows_prefetch: int = None,
                 label_threshold=0.5, time_units=None, shuffle=False, max_cache_duration=None,
                 patient_history_fields: list = None):
        # AtriumSDK object
        self.sdk = sdk

        # Initialize random number generator if shuffle is specified with a seed
        self.random_gen = random.Random(shuffle) if isinstance(shuffle, int) else random.Random() if shuffle else None

        # Store the label_threshold
        self.label_threshold = label_threshold

        self.patient_history_fields = patient_history_fields

        self.time_units = "ns" if time_units is None else time_units
        self.time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}

        # List of validated measures. Each item is a "measure_info" from sdk data.
        self.measures = validated_measure_list

        # List of validated label sets. Each item is a label_set id from the label_set table.
        self.label_sets = validated_label_set_list

        # Dictionary containing sources. Each source type contains identifiers (device_id/patient_id)
        # and have associated time ranges (or "all").
        self.sources = validated_sources
        self.max_cache_duration = max_cache_duration
        if max_cache_duration is not None:
            assert max_cache_duration >= window_duration_ns, \
                "max_cache_duration must be greater than window_duration_ns"
            self._split_time_ranges(max_cache_duration)

        self.shuffle = shuffle

        # Duration of each window in nanoseconds. Represents the time span of each data segment.
        self.window_duration_ns = int(window_duration_ns)

        # The sliding interval in nanoseconds by which the window advances in time.
        self.window_slide_ns = int(window_slide_ns)

        # Determine the highest frequency among the measures, used to compute matrix row sizes and more.
        self.highest_freq_nhz = max([measure['freq_nhz'] for measure in self.measures])

        # Compute the matrix's row size based on the highest frequency and the window duration in nanoseconds.
        # Determines how many data points fit in a single window.

        # Emitting warnings for row_size, slide_size, and row_period_ns

        if (self.window_duration_ns * self.highest_freq_nhz) % 1e18 != 0:
            warnings.warn(
                f'Given window duration of {window_duration_ns / 1e9} seconds and signal frequency of '
                f'{self.highest_freq_nhz / (10 ** 9)}Hz result in a non-integer number of signals in the window. '
                f'window size / row size will round down'
            )

        if (self.window_slide_ns * self.highest_freq_nhz) % 1e18 != 0:
            warnings.warn(
                f'Given sliding window duration of {window_slide_ns / 1e9} seconds and signal frequency of '
                f'{self.highest_freq_nhz / (10 ** 9)}Hz result in a non-integer number of signals in the slide. '
                f'slide will round down'
            )

        self.row_size = int((self.highest_freq_nhz * self.window_duration_ns) // 1e18)

        # The slide size in terms of matrix rows for the sliding window operation.
        self.slide_size = int((self.highest_freq_nhz * self.window_slide_ns) // 1e18)

        # Time duration between consecutive data points, given the highest frequency.
        self.row_period_ns = int(1e18 // self.highest_freq_nhz)

        # Default number of windows for each batch is determined as 10 million divided by the number of data points in a window.
        # If provided, it uses the given num_windows_prefetch.
        self.max_batch_size = int(10_000_000 // self.row_size) if num_windows_prefetch is None else num_windows_prefetch

        # Lists containing the starting index for each batch and details about each batch respectively.
        # Also, the total length (number of windows) in the dataset.
        self.batch_info, self.batch_first_index, self._length = self._extract_cache_info()

        # Variables to store the start and end indices of the current batch.
        # initialized to -1 so that the first request will always trigger a batch load.
        self.current_batch_start_index = -1
        self.current_batch_end_index = -1
        self.next_cache_index = 0

        # The current batch's matrix containing the batch's data for windowing.
        self.batch_matrix = None

        # Array containing the timestamps corresponding to each value in the batch.
        self.batch_times = None

        # patient_id of current batch (if the data source is a patient) or None.
        self.current_patient_id = None

        # device_id of current batch (if the data source is a device) or None.
        self.current_device_id = None

        # The start time of the current batch.
        self.current_batch_start_time = None

        # The most recent index called on __getitem__.
        self.current_index = None

        # Attribute to store the batch data by measure ID.
        self.batch_data_dictionary = {}

        # Attribute to store the batch label information.
        self.batch_label_time_series = None
        self.batch_label_thresholds = None

        # Window Cache
        self.window_cache = []
        self.matrix_cache = []
        self.cache_window_i = 0

    def _split_time_ranges(self, max_duration):
        for source_type, sources in self.sources.items():
            for source_id, time_ranges in sources.items():
                new_time_ranges = []
                for start, end in time_ranges:
                    while start + max_duration < end:
                        new_time_ranges.append([start, start + max_duration])
                        start += max_duration
                    new_time_ranges.append([start, end])
                self.sources[source_type][source_id] = new_time_ranges

    def _extract_cache_info(self):
        # Flattening the nested dictionary/list structure
        flattened_sources = []
        for source_type, sources in self.sources.items():
            for source_id, time_ranges in sources.items():
                for range_start_time, range_end_time in time_ranges:
                    flattened_sources.append([source_type, source_id, range_start_time, range_end_time])

        # Shuffling if random_gen is not None
        if self.random_gen is not None:
            self.random_gen.shuffle(flattened_sources)

        cache_info = []  # List to hold all batches
        starting_window_index_per_batch = [0]  # Starts with 0 to indicate the first window starts at index 0
        total_number_of_windows = 0  # A counter for the total number of windows across all batches

        current_batch = []
        current_batch_num_windows = 0

        for source in flattened_sources:
            source_type, source_id, range_start_time, range_end_time = source
            num_time_range_windows = 0
            time_range_info_start = cur_window_start = range_start_time
            time_range_info_end = cur_window_end = cur_window_start + self.window_duration_ns

            # While the next window is valid
            while cur_window_start < range_end_time:

                # Increment Window Counters
                num_time_range_windows += 1
                current_batch_num_windows += 1
                total_number_of_windows += 1

                # Update the valid end_time
                time_range_info_end = cur_window_end

                # Check if we've gone over.
                if current_batch_num_windows >= self.max_batch_size:
                    # Add current time_range_info to the batch
                    if num_time_range_windows > 0:
                        time_range_info = [
                            source_type,
                            source_id,
                            time_range_info_start,
                            time_range_info_end,
                            time_range_info_start,
                            min(range_end_time, time_range_info_end),
                            num_time_range_windows,
                        ]
                        current_batch.append(time_range_info)

                    # Add batch to cache list
                    cache_info.append(current_batch)

                    # Reset Batch
                    current_batch = []
                    current_batch_num_windows = 0
                    starting_window_index_per_batch.append(total_number_of_windows)

                    # Separate Time Range
                    time_range_info_start = cur_window_start + self.window_slide_ns
                    num_time_range_windows = 0

                # Locate next possible window
                cur_window_start += self.window_slide_ns
                cur_window_end += self.window_slide_ns

            # Once all the windows have been accounted for, add them to the batch
            if num_time_range_windows > 0:
                time_range_info = [
                    source_type,
                    source_id,
                    time_range_info_start,
                    time_range_info_end,
                    time_range_info_start,
                    min(range_end_time, time_range_info_end),
                    num_time_range_windows,
                ]
                current_batch.append(time_range_info)

        # Add final batch to cache list
        cache_info.append(current_batch)
        starting_window_index_per_batch.append(total_number_of_windows)

        return cache_info, starting_window_index_per_batch, total_number_of_windows

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

                window_cache.append(result_window)
                matrix_cache.append(np.copy(sliced_views[window_i]))

        if self.random_gen is not None:
            self.random_gen.shuffle(window_cache)
            self.random_gen.shuffle(matrix_cache)

        self.window_cache = window_cache
        self.matrix_cache = matrix_cache

        self.current_batch_start_index = batch_start_index
        self.current_batch_end_index = batch_end_index

    def _load_patient_cache(self, patient_id, patient_info_cache, patient_history_cache):
        if patient_id is None:
            return
        # Check if patient id is in the info cache.
        if patient_id not in patient_info_cache:
            patient_info_cache[patient_id] = self.sdk.get_patient_info(patient_id=patient_id)
            if patient_info_cache[patient_id] is None:
                patient_info_cache[patient_id] = {}
            else:
                # Delete Height And Weight From Static Info
                if 'height' in patient_info_cache[patient_id]:
                    del patient_info_cache[patient_id]['height']

                if 'weight' in patient_info_cache[patient_id]:
                    del patient_info_cache[patient_id]['weight']

        if self.patient_history_fields and patient_id not in patient_history_cache:
            patient_history_cache[patient_id] = {}
            for field in ['height', 'weight']:
                if field not in self.patient_history_fields:
                    continue
                patient_history_cache[patient_id][field] = self.sdk.get_patient_history(
                    patient_id=patient_id, field=field)

    def _get_patient_info_from_cache(self, patient_id, window_start_time, patient_info_cache, patient_history_cache):
        window_patient_info = patient_info_cache.get(patient_id, {})
        if self.patient_history_fields:
            for field, history_timeseries in patient_history_cache.get(patient_id, {}).items():
                best_match = find_closest_measurement(window_start_time, history_timeseries)
                if best_match is None:
                    continue

                _, _, _, value, units, time = best_match
                window_patient_info[field] = {
                    'value': value,
                    'units': units,
                    'time': time,
                }
        return window_patient_info

    def unpack_source_info(self, source_id, source_type):
        if source_type == "device_ids":
            device_id = source_id
            patient_id = None
        elif source_type == "patient_ids":
            device_id = None
            patient_id = source_id
        elif source_type == "device_patient_tuples":
            device_id, patient_id = source_id

        else:
            raise ValueError(f"Source type must be either device_ids or patient_ids, not {source_type}")
        # only query by patient if device_id isn't available.
        query_patient_id = patient_id if device_id is None else None
        return device_id, patient_id, query_patient_id

    def _get_source_label_data(self, device_id, query_patient_id, batch_start_time, batch_end_time, batch_time_array):
        sliced_labels, threshold_labels = None, None
        # If labels exist, calculate them
        if len(self.label_sets) > 0:
            # Preallocate label matrix
            label_matrix = np.zeros((len(self.label_sets), len(batch_time_array)), dtype=np.int8)

            # Populate label matrix
            for idx, label_set_id in enumerate(self.label_sets):
                self.sdk.get_label_time_series(
                    label_name_id=label_set_id,
                    device_id=device_id if device_id else None,
                    patient_id=query_patient_id if query_patient_id else None,
                    start_time=batch_start_time,
                    end_time=batch_end_time,
                    timestamp_array=batch_time_array,
                    out=label_matrix[idx]
                )

            # Create label windows
            windowed_label_views = sliding_window_view(
                label_matrix, (len(self.label_sets), self.row_size), axis=None)
            sliced_labels = windowed_label_views[0][::self.slide_size]

            threshold_labels = get_threshold_labels(sliced_labels, label_threshold=self.label_threshold)
        return sliced_labels, threshold_labels

    def _query_source_data(self, device_id, query_patient_id, batch_start_time, batch_end_time, range_start_time,
                           range_end_time, batch_num_windows, batch_size, batch_matrix):
        # Reset and populate the batch data signal dictionary
        source_batch_data_dictionary = {}
        for i, measure in enumerate(self.measures):
            freq_nhz = measure['freq_nhz']
            period_ns = int(1e18 // freq_nhz)
            measure_id = measure['id']

            # Create a time array for this specific measure
            measure_window_size = int((freq_nhz * self.window_duration_ns) // 1e18)
            measure_slide_size = int((freq_nhz * self.window_slide_ns) // 1e18)
            measure_batch_size = measure_window_size + (batch_num_windows - 1) * measure_slide_size
            measure_quantized_end_time = batch_start_time + (measure_batch_size * period_ns)
            measure_filled_time_array = np.arange(batch_start_time, measure_quantized_end_time, period_ns)
            measure_filled_value_array = np.full(measure_filled_time_array.shape, np.nan)

            # Fetch data for this measure and window from the SDK
            data_start_time = max(range_start_time, batch_start_time)
            data_end_time = min(range_end_time, batch_end_time)

            _, measure_sdk_times, measure_sdk_values = self.sdk.get_data(
                measure_id, data_start_time, data_end_time, device_id=device_id, patient_id=query_patient_id)

            # Batch Matrix
            # Convert times to indices on the matrix using vectorized operations
            closest_i_array_matrix = np.floor((measure_sdk_times - batch_start_time) / self.row_period_ns).astype(int)

            # Make sure indices are within bounds
            mask = (closest_i_array_matrix >= 0) & (closest_i_array_matrix < batch_size)
            closest_i_array_matrix = closest_i_array_matrix[mask]

            # Populate the matrix using vectorized operations
            batch_matrix[i, closest_i_array_matrix] = measure_sdk_values[mask]

            # Batch Signals
            # Convert times to indices on the matrix using vectorized operations
            closest_i_array_signals = np.floor((measure_sdk_times - batch_start_time) / period_ns).astype(int)

            # Make sure indices are within bounds
            mask = (closest_i_array_signals >= 0) & (closest_i_array_signals < measure_batch_size)
            closest_i_array_signals = closest_i_array_signals[mask]

            # Populate the arrays using vectorized operations
            measure_filled_value_array[closest_i_array_signals] = measure_sdk_values[mask]
            measure_filled_time_array[closest_i_array_signals] = measure_sdk_times[mask]

            # convert time data from nanoseconds to unit of choice
            # if self.time_units != 'ns':
            #     measure_filled_time_array = measure_filled_time_array / self.time_unit_options[self.time_units]

            # Create Windows
            windowed_measure_times = sliding_window_view(measure_filled_time_array, measure_window_size)
            windowed_measure_values = sliding_window_view(measure_filled_value_array, measure_window_size)

            # Slide the windows
            sliced_windowed_measure_times = windowed_measure_times[::measure_slide_size]
            sliced_windowed_measure_values = windowed_measure_values[::measure_slide_size]

            # Store the measure's time and value arrays in the batch data dictionary
            source_batch_data_dictionary[measure_id] = \
                (sliced_windowed_measure_times, sliced_windowed_measure_values, measure_window_size)
        return source_batch_data_dictionary

    def _calculate_batch_size(self, idx):
        batch_index = bisect_right(self.batch_first_index, idx) - 1
        if batch_index < 0 or len(self.batch_info) <= batch_index:
            raise ValueError(f"index {idx} outside of batched data.")
        batch_start_index = self.batch_first_index[batch_index]
        batch_end_index = self.batch_first_index[batch_index + 1]
        batch_num_windows = batch_end_index - batch_start_index
        batch_size = self.row_size + (batch_num_windows - 1) * self.slide_size
        return batch_index, batch_start_index, batch_end_index, batch_num_windows, batch_size

    def get_current_window_start(self):
        return self.current_batch_start_time + \
               ((self.current_index - self.current_batch_start_index) * self.window_slide_ns)

    def __next__(self) -> Window:
        """
        Fetches a Window object corresponding to the next index, encapsulating multiple signals of varying
        frequencies along with their associated metadata. The Window object returned will have its `signals` attribute
        populated with a dictionary, where each key is a tuple describing the measure tag, the frequency of the
        measure in Hz, and the units of the measure. The value corresponding to each key is another dictionary
        containing the actual data points, expected count, actual count of non-NaN data points, measure ID, and
        the timestamps associated with each data point of the signal.

        :return: A Window object encapsulating the signals and associated metadata for the specified index. The
            structure of the Window object and the included signals dictionary is described in the Window Format section
            of the documentation.
        :rtype: Window
        :raises IndexError: Raised if the index is out of bounds.

        :Example:

        .. code-block:: python

            window_obj = dataset_iterator[5]
            signals_dict = window_obj.signals

            for measure_info, signal_data in signals_dict.items():
                print(f"Measure Info: {measure_info}")
                print(f"Times: {signal_data['times']}")
                print(f"Values: {signal_data['values']}")
                print(f"Expected Count: {signal_data['expected_count']}")
                print(f"Actual Count: {signal_data['actual_count']}")
                print(f"Measure ID: {signal_data['measure_id']}")

        """
        while self.cache_window_i >= len(self.window_cache):
            if self.next_cache_index >= len(self.batch_info):
                # Reset iterator
                self.current_batch_start_index = -1
                self.current_batch_end_index = -1
                self.next_cache_index = 0

                # Attribute to store the batch label information.
                self.batch_label_time_series = None
                self.batch_label_thresholds = None

                # Window Cache
                self.window_cache = []
                self.matrix_cache = []
                self.cache_window_i = 0
                raise StopIteration
            self._load_batch_matrix(self.batch_first_index[self.next_cache_index])
            self.cache_window_i = 0
            self.next_cache_index += 1

        window = self.window_cache[self.cache_window_i]
        self.cache_window_i += 1
        return window

    def __iter__(self):
        """
        Returns the iterator object itself. This method is required to make the class iterable.

        By implementing this method, the DatasetIterator class can be used in loop constructs like for-loops.
        This is useful for iterating over the dataset in a convenient and Pythonic way.

        The __iter__ method is part of the iterator protocol in Python, which requires __iter__ and __next__
        methods to be defined for an object to be considered an iterator. When used in a for-loop, Python
        automatically calls the __iter__ method to obtain an iterator and then repeatedly calls the __next__
        method to retrieve successive items from the iterator.

        :return: Returns the iterator object (self).
        :rtype: DatasetIterator

        :Example:

        .. code-block:: python

            dataset_iterator = DatasetIterator(...)  # Assuming proper initialization
            for window in dataset_iterator:
                # Process each window here
                ...
        """
        return self
