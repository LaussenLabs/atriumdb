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
import pickle
import hashlib
from datetime import datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from atriumdb.windowing.window import Window
from atriumdb.windowing.windowing_functions import get_signal_dictionary, find_closest_measurement, \
    get_label_dictionary, _get_patient_info_from_cache, _load_patient_cache, get_window_list


class DatasetIterator:
    """
    Iterator over windowed segments of a dataset.

    Allows efficient iterative access to sliding windows of data from different sources (e.g., devices or patients)
    by organizing data into batches and loading one batch at a time.

    :param AtriumSDK sdk: SDK object to fetch data
    :param DatasetDefinition definition: DatasetDefinition of the cohort to be iterated over.
    :param int window_duration_ns: Duration of each window in nanoseconds
    :param int window_slide_ns: Interval in nanoseconds by which the window advances in time
    :param int num_windows_prefetch: Number of windows you want to get from AtriumDB at a time. Setting this value
            higher will make decompression faster but at the expense of using more RAM. (default the number of windows
            that gets you closest to 10 million values).
    :param bool | None | int shuffle: If True, shuffle the order of windows before iterating. If an integer, it will initialize
                                        a seeded random number generator for reproducible shuffling. If False or None, no shuffling occurs.
    :param int | None max_cache_duration: If specified, no single cache will have a time range larger than this duration.
                                           The time range will be split accordingly. The duration must be larger than window_duration_ns.
    :param list patient_history_fields: A list of patient_history fields you would like returned in the Window object.
    :param str cache_dir: A directory, if specified, caches the results of _extract_cache_info to speed up future iterations. Setting to None will disable the cache.
    :param bool label_exact_match: If True, labels will be matched exactly as requested, and child labels will not be returned
        when their parent is requested. If False, child labels will be included when their parent is requested.
    """

    def __init__(self, sdk, definition, window_duration_ns: int, window_slide_ns: int, num_windows_prefetch: int = None,
                 label_threshold=0.5, shuffle=False, max_cache_duration=None, patient_history_fields: list = None,
                 label_exact_match=False):
        if not definition.is_validated:
            definition.validate(sdk=sdk)

        self.label_exact_match = label_exact_match
        # Extract validated data from the definition
        validated_data = definition.validated_data_dict
        # List of validated measures. Each item is a "measure_info" from sdk data.
        self.measures = validated_data['measures']
        # List of validated label sets. Each item is a label_set id from the label_set table.
        self.label_sets = validated_data['labels']
        # Dictionary containing sources. Each source type contains identifiers (device_id/patient_id)
        # and have associated time ranges.
        self.sources = validated_data['sources']

        # AtriumSDK object
        self.sdk = sdk

        # Initialize random number generator if shuffle is specified with a seed
        self.random_gen = None
        if isinstance(shuffle, int) and not isinstance(shuffle, bool):
            self.random_gen = random.Random(shuffle)
        elif shuffle:
            self.random_gen = random.Random()

        # Store the label_threshold
        self.label_threshold = label_threshold

        self.patient_history_fields = patient_history_fields

        self.max_cache_duration = max_cache_duration
        if shuffle is not False and max_cache_duration is not None:
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

        if (self.window_duration_ns * self.highest_freq_nhz) % (10 ** 18) != 0:
            warnings.warn(
                f'Given window duration of {window_duration_ns / 1e9} seconds and signal frequency of '
                f'{self.highest_freq_nhz / (10 ** 9)}Hz result in a non-integer number of signals in the window. '
                f'window size / row size will round down'
            )

        if (self.window_slide_ns * self.highest_freq_nhz) % (10 ** 18) != 0:
            warnings.warn(
                f'Given sliding window duration of {window_slide_ns / 1e9} seconds and signal frequency of '
                f'{self.highest_freq_nhz / (10 ** 9)}Hz result in a non-integer number of signals in the slide. '
                f'slide will round down'
            )

        self.row_size = int((self.highest_freq_nhz * self.window_duration_ns) // (10 ** 18))

        # The slide size in terms of matrix rows for the sliding window operation.
        self.slide_size = int((self.highest_freq_nhz * self.window_slide_ns) // (10 ** 18))

        # Time duration between consecutive data points, given the highest frequency.
        self.row_period_ns = int((10 ** 18) // self.highest_freq_nhz)

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
        if len(current_batch) > 0:
            cache_info.append(current_batch)
        starting_window_index_per_batch.append(total_number_of_windows)

        return cache_info, starting_window_index_per_batch, total_number_of_windows

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
                self.label_threshold, range_num_windows, self.row_period_ns, self.row_size, self.slide_size,
                label_exact_match=self.label_exact_match)

            patient_history_fields = self.patient_history_fields

            measures = self.measures
            window_duration_ns = self.window_duration_ns
            num_windows_for_batch = range_num_windows

            batch_window_list = get_window_list(device_id, patient_id, measures, source_batch_data_dictionary,
                                                     source_batch_start_time, num_windows_for_batch, window_duration_ns,
                                                     threshold_labels, sliced_labels, patient_history_cache,
                                                     patient_history_fields, patient_info_cache)

            window_cache.extend(batch_window_list)

        if self.random_gen is not None:
            self.random_gen.shuffle(window_cache)

        self.window_cache = window_cache

        self.current_batch_start_index = batch_start_index
        self.current_batch_end_index = batch_end_index

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

    def __len__(self):
        """
        Returns the total number of Window objects available in the dataset. This method can be accessed using
        Python's built-in `len()` function.

        :return: The total number of Window objects in the dataset.
        :rtype: int

        """
        return self._length
