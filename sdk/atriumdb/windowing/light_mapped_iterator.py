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

import warnings

import numpy as np
import random
from numpy.lib.stride_tricks import sliding_window_view

from atriumdb.windowing.window import Window
from atriumdb.windowing.dataset_iterator import DatasetIterator
from atriumdb.windowing.windowing_functions import get_threshold_labels, find_closest_measurement, \
    _get_patient_info_from_cache, _load_patient_cache


class LightMappedIterator(DatasetIterator):
    def __init__(self, sdk, definition,
                 window_duration_ns: int, window_slide_ns: int, label_threshold=0.5,
                 shuffle=False, patient_history_fields: list = None, allow_partial_windows = True, label_exact_match=False):
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

        # Store the label_threshold
        self.label_threshold = label_threshold

        self.patient_history_fields = patient_history_fields
        self.allow_partial_windows = allow_partial_windows

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

        # cache for patient data
        self.patient_info_cache = {}
        self.patient_history_cache = {}

        # Process the sources to compute window counts
        self.total_windows = None
        self._process_sources()

        # Shuffle the window indices if shuffle is True
        self.shuffle = shuffle
        self.random_gen = None
        if self.shuffle:
            self.random_gen = random.Random(shuffle) if isinstance(shuffle, int) else random.Random()
            # Create a shuffled array of indices
            self.shuffled_indices = np.arange(self.total_windows, dtype=np.int64)
            self.random_gen.shuffle(self.shuffled_indices)
        else:
            self.shuffled_indices = None

        # Initialize the iterator index
        self._current_idx = 0

    def _process_sources(self):
        # List to hold info about each source and its windows
        self.sources_info = []  # Each item is a dict with source info and num_windows
        total_windows = 0
        self.window_indices = []
        for source_type, sources in self.sources.items():
            for source_id, time_ranges in sources.items():
                for start_time, end_time in time_ranges:
                    duration = end_time - start_time
                    if duration <= 0 or (not self.allow_partial_windows and duration < self.window_duration_ns):
                        continue
                    # Calculate number of windows
                    num_windows = int((duration - self.window_duration_ns) // self.window_slide_ns) + 1
                    if self.allow_partial_windows and ((duration - self.window_duration_ns) % self.window_slide_ns > 0):
                        num_windows += 1

                    if num_windows <= 0:
                        continue
                    # Append info
                    source_info = {
                        'source_type': source_type,
                        'source_id': source_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'num_windows': num_windows,
                        'start_index': total_windows,
                    }
                    self.sources_info.append(source_info)
                    # Record the starting index for np.searchsorted
                    start_index = total_windows
                    self.window_indices.append(start_index)
                    total_windows += num_windows

        self.total_windows = total_windows
        self.window_indices = np.array(self.window_indices)

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of bounds for mapped iterator of size {self.total_windows}")
        # Swap idx with shuffled index if shuffling is enabled
        if self.shuffled_indices is not None:
            idx = self.shuffled_indices[idx]
        idx = int(idx)
        # Find which source the idx corresponds to
        source_idx = np.searchsorted(self.window_indices, idx, side='right') - 1
        source_info = self.sources_info[source_idx]
        window_offset = idx - source_info['start_index']
        window_start_time = source_info['start_time'] + window_offset * self.window_slide_ns
        window_end_time = window_start_time + self.window_duration_ns
        # Now, get data for this window
        return self._get_window(source_info, window_start_time, window_end_time)

    def _get_window(self, source_info, window_start_time, window_end_time):
        # Unpack source_info
        source_type = source_info['source_type']
        source_id = source_info['source_id']
        device_id, patient_id, query_patient_id = self.unpack_source_info(source_id, source_type)
        # Get data for each measure
        signals = {}
        for measure in self.measures:
            measure_id = measure['id']
            freq_nhz = measure['freq_nhz']
            period_ns = int((10 ** 18) // freq_nhz)
            measure_expected_count = int((freq_nhz * self.window_duration_ns) // (10 ** 18))
            measure_times = (np.arange(measure_expected_count) * period_ns) + window_start_time
            measure_values = np.full(measure_times.shape, np.nan)
            if len(measure_values) > 0:
                if measure_expected_count != int(round((window_end_time - window_start_time) / period_ns)):
                    window_end_time = window_start_time + int(
                        round(measure_expected_count * (10 ** 18 / freq_nhz)))
                self.sdk.get_data(
                    measure_id, window_start_time, window_end_time, device_id=device_id, patient_id=query_patient_id,
                    return_nan_filled=measure_values)
            measure_tag = measure['tag']
            measure_freq_hz = freq_nhz / (10 ** 9)
            measure_units = measure['units']
            signals[(measure_tag, measure_freq_hz, measure_units)] = {
                'times': measure_times,
                'values': measure_values,
                'expected_count': measure_expected_count,
                'actual_count': np.sum(~np.isnan(measure_values)),
                'measure_id': measure_id,
            }

        _load_patient_cache(patient_id, self.patient_info_cache, self.patient_history_cache, self.sdk, self.patient_history_fields)
        window_patient_info = _get_patient_info_from_cache(
            patient_id, window_start_time, self.patient_info_cache, self.patient_history_cache)

        label, label_time_series = np.array([]), np.array([])

        if len(self.label_sets) > 0:
            # Preallocate label matrix
            label_time_series_num_samples = (int(self.window_duration_ns) * int(self.highest_freq_nhz)) // (10 ** 18)
            label_time_series = np.zeros((len(self.label_sets), label_time_series_num_samples), dtype=np.int8)
            period_ns = (10 ** 18) / self.highest_freq_nhz
            label_timestamp_array = np.arange(window_start_time, window_end_time, period_ns, dtype=np.int64)

            # Populate label matrix
            for idx, label_set_id in enumerate(self.label_sets):
                self.sdk.get_label_time_series(
                    label_name_id=label_set_id,
                    device_id=device_id if device_id else None,
                    patient_id=query_patient_id if query_patient_id else None,
                    start_time=window_start_time,
                    end_time=window_start_time + self.window_duration_ns,
                    timestamp_array=label_timestamp_array,
                    out=label_time_series[idx],
                    include_descendants=not self.label_exact_match,
                )
            label = (np.mean(label_time_series, axis=1) > self.label_threshold).astype(np.int8)

        # Create Window object
        window = Window(
            signals=signals,
            start_time=window_start_time,
            device_id=device_id,
            patient_id=patient_id,
            label_time_series=label_time_series,
            label=label,
            patient_info=window_patient_info)

        return window

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.total_windows:
            raise StopIteration
        window = self[self._current_idx]
        self._current_idx += 1
        return window
