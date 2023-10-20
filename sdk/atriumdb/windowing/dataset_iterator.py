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

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from atriumdb.windowing.window import Window


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
    """

    def __init__(self, sdk, validated_measure_list, validated_sources, window_duration_ns: int, window_slide_ns: int,
                 num_windows_prefetch: int = None, time_units=None):
        # AtriumSDK object
        self.sdk = sdk

        self.time_units = "ns" if time_units is None else time_units
        self.time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}

        # List of validated measures. Each item is a "measure_info" from sdk data.
        self.measures = validated_measure_list

        # Dictionary containing sources. Each source type contains identifiers (device_id/patient_id)
        # and have associated time ranges (or "all").
        self.sources = validated_sources

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
        self.batch_first_index, self.batch_info, self._length = self._extract_batch_info()

        # Variables to store the start and end indices of the current batch.
        # initialized to -1 so that the first request will always trigger a batch load.
        self.current_batch_start_index = -1
        self.current_batch_end_index = -1

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

    def _extract_batch_info(self):
        # Initialize empty lists to store batch details and starting window indices
        batch_info = []
        batch_first_index = []

        # A global index of windows
        current_index = 0

        # For each source type
        for source_type, sources in self.sources.items():

            # For each source identifier of that type
            for source_id, time_ranges in sources.items():

                # For all time ranges in that source
                for range_start_time, range_end_time in time_ranges:
                    cur_window_start_time = range_start_time
                    true_range_end_time = range_end_time

                    # Adjust the range_end_time to ensure it's aligned to the window_slide_ns
                    total_range_duration = max(range_end_time - cur_window_start_time, self.window_duration_ns)
                    remainder = (total_range_duration - self.window_duration_ns) % self.window_slide_ns
                    if remainder != 0:
                        range_end_time += self.window_slide_ns - remainder

                    # Initialize the batch starting time and the starting index of this batch
                    batch_start_time = cur_window_start_time
                    batch_index_start = current_index

                    # Keep adding windows to the batch until the end of the time range is reached
                    while cur_window_start_time + self.window_duration_ns <= range_end_time:

                        # If the batch reaches its maximum size, finalize this batch and start a new one
                        if self.max_batch_size and (current_index - batch_index_start) >= self.max_batch_size:
                            batch_info.append(
                                [
                                    source_type,
                                    source_id,
                                    batch_start_time,
                                    cur_window_start_time + self.window_duration_ns,
                                    range_start_time,
                                    true_range_end_time,
                                ])

                            batch_first_index.append(batch_index_start)

                            # Reset batch start time and index to current values for the new batch
                            batch_start_time = cur_window_start_time
                            batch_index_start = current_index

                        # Move to the next window by sliding forward
                        cur_window_start_time += self.window_slide_ns
                        # Increment the global window index
                        current_index += 1

                    # After exiting the while loop, if there are remaining windows that haven't been batched,
                    # create a batch for them.
                    if (current_index - batch_index_start) > 0:
                        batch_info.append(
                            [
                                source_type,
                                source_id,
                                batch_start_time,
                                cur_window_start_time + self.window_duration_ns,
                                range_start_time,
                                true_range_end_time,
                            ])
                        batch_first_index.append(batch_index_start)

        # Append the final window count to the batch_first_index list for future batch size math.
        batch_first_index.append(current_index)

        # Return the lists containing batch start indices, batch details, and the total number of windows
        return batch_first_index, batch_info, current_index

    def _load_batch_matrix(self, idx: int):
        batch_index = bisect_right(self.batch_first_index, idx) - 1
        if batch_index < 0 or len(self.batch_info) <= batch_index:
            raise ValueError(f"index {idx} outside of batched data.")

        batch_start_index = self.batch_first_index[batch_index]
        batch_end_index = self.batch_first_index[batch_index + 1]

        batch_num_windows = batch_end_index - batch_start_index
        batch_size = self.row_size + (batch_num_windows - 1) * self.slide_size

        # Get the matrix
        source_type, source_id, batch_start_time, batch_end_time, range_start_time, range_end_time = \
            self.batch_info[batch_index]

        batch_matrix = np.full((len(self.measures), batch_size), np.nan)

        quantized_end_time = batch_start_time + (batch_size * self.row_period_ns)
        batch_time_array = np.arange(batch_start_time, quantized_end_time, self.row_period_ns)

        if source_type == "device_ids":
            device_id = source_id
            patient_id = None
            self.current_patient_id = None
            self.current_device_id = source_id
        elif source_type == "patient_ids":
            device_id = None
            patient_id = source_id
            self.current_patient_id = source_id
            self.current_device_id = None
        elif source_type == "device_patient_tuples":
            device_id, patient_id = source_id
            self.current_patient_id = patient_id
            self.current_device_id = device_id
        else:
            raise ValueError(f"Source type must be either device_ids or patient_ids, not {source_type}")

        self.batch_data_dictionary.clear()

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
                measure_id, data_start_time, data_end_time, device_id=device_id, patient_id=patient_id)

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
            if self.time_units != 'ns':
                measure_filled_time_array = measure_filled_time_array / self.time_unit_options[self.time_units]

            # Create Windows
            windowed_measure_times = sliding_window_view(measure_filled_time_array, measure_window_size)
            windowed_measure_values = sliding_window_view(measure_filled_value_array, measure_window_size)

            # Slide the windows
            sliced_windowed_measure_times = windowed_measure_times[::measure_slide_size]
            sliced_windowed_measure_values = windowed_measure_values[::measure_slide_size]

            # Store the measure's time and value arrays in the batch data dictionary
            self.batch_data_dictionary[measure_id] = \
                (sliced_windowed_measure_times, sliced_windowed_measure_values, measure_window_size)

        windowed_views = sliding_window_view(
            batch_matrix, (len(self.measures), self.row_size), axis=None)

        windowed_times = sliding_window_view(batch_time_array, self.row_size)

        sliced_views = windowed_views[0][::self.slide_size]
        sliced_times = windowed_times[::self.slide_size]

        self.current_batch_start_index = batch_start_index
        self.current_batch_end_index = batch_end_index
        self.batch_matrix = sliced_views
        self.batch_times = sliced_times
        self.current_batch_start_time = batch_start_time

    def get_current_window_start(self):
        return self.current_batch_start_time + \
               ((self.current_index - self.current_batch_start_index) * self.window_slide_ns)

    def __len__(self) -> int:
        """
        Get the total number of windows in the dataset.

        :return: Total number of windows in the dataset
        :rtype: int
        """
        return self._length

    def get_array_matrix(self, idx: int) -> np.ndarray:
        """
        Fetch the window data (numpy matrix) for a given index. By nature of being a matrix, the returned array will
        have equal numbers of values for each row (signal), and therefore gaps are filled with numpy.nan values in
        rows with lower sample frequencies.

        :param idx: Index of the desired window
        :type idx: int
        :return: Array of data corresponding to the given index
        :rtype: np.ndarray
        :raises IndexError: If the index is out of bounds
        """
        if idx < 0:
            idx = self._length - idx

        if idx < 0 or self._length <= idx:
            raise IndexError(f"Index {idx} out of bounds for iterator of length {self._length}")

        if idx < self.current_batch_start_index or self.current_batch_end_index <= idx:
            self._load_batch_matrix(idx)

        self.current_index = idx

        return np.copy(self.batch_matrix[idx - self.current_batch_start_index])

    def get_signal_window(self, idx: int) -> dict:
        """
        Fetch the window data (dictionary of signals) for a given index.

        :param idx: Index of the desired window
        :type idx: int
        :return: Array of data corresponding to the given index
        :rtype: dict
        :raises IndexError: If the index is out of bounds
        """
        if idx < 0:
            idx = self._length - idx

        if idx < 0 or self._length <= idx:
            raise IndexError(f"Index {idx} out of bounds for iterator of length {self._length}")

        if idx < self.current_batch_start_index or self.current_batch_end_index <= idx:
            self._load_batch_matrix(idx)

        self.current_index = idx
        signal_dictionary = {}
        for measure in self.measures:
            measure_id = measure['id']
            measure_tag = measure['tag']
            measure_freq_nhz = measure['freq_nhz']
            measure_units = measure['units']

            window_times = self.batch_data_dictionary[measure_id][0][idx - self.current_batch_start_index]

            window_values = self.batch_data_dictionary[measure_id][1][idx - self.current_batch_start_index]
            measure_expected_count = self.batch_data_dictionary[measure_id][2]

            signal_dictionary[(measure_tag, measure_freq_nhz, measure_units)] = \
                {
                    'times': np.copy(window_times),
                    'values': np.copy(window_values),
                    'expected_count': measure_expected_count,
                    'actual_count': np.sum(~np.isnan(window_values)),
                    'measure_id': measure_id,
                }

        return signal_dictionary

    def __getitem__(self, idx: int) -> Window:
        """
        Fetches a Window object corresponding to the given index, encapsulating multiple signals of varying
        frequencies along with their associated metadata. The Window object returned will have its `signals` attribute
        populated with a dictionary, where each key is a tuple describing the measure tag, the frequency of the
        measure in Hz, and the units of the measure. The value corresponding to each key is another dictionary
        containing the actual data points, expected count, actual count of non-NaN data points, measure ID, and
        the timestamps associated with each data point of the signal.

        :param idx: The index of the desired window. This index must be within the bounds of the available data
            windows. Negative indexing is supported, `-idx = len(iterator) - idx`.
        :type idx: int
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
        signal_dictionary = self.get_signal_window(idx)

        result_window = Window(
            signals=signal_dictionary,
            start_time=self.get_current_window_start(),
            device_id=self.current_device_id,
            patient_id=self.current_patient_id,
        )

        return result_window

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
        idx = 0 if self.current_index is None else self.current_index + 1
        return self.__getitem__(idx)
