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
from bisect import bisect_right

from atriumdb.windowing.window import Window


def get_threshold_labels(sliced_labels, label_threshold=0.5):
    # Calculate the percentage of 1s for each label in each window
    percentages = np.mean(sliced_labels, axis=-1)
    # Apply threshold
    return (percentages > label_threshold).astype(int)


def find_closest_measurement(time, measurements):
    """
    Find the measurement with the time value closest, but less than or equal to the given time,
    directly using bisect on the list of tuples.

    :param time: The time (epoch timestamp) to find the closest measurement for.
    :param measurements: A list of tuples containing the measurement value, units, and epoch timestamp.
    :return: The tuple from the measurements list with the closest time less than or equal to the given time.
    """
    # Use bisect_right with a key function that extracts the timestamp part of the tuple
    idx = bisect_right(measurements, time, key=lambda x: x[5])

    if idx > 0:
        return measurements[idx - 1]
    else:
        return None


def get_signal_dictionary(sdk, device_id, query_patient_id, window_duration_ns, window_slide_ns, measures,
                          batch_start_time, batch_end_time, batch_num_windows, range_start_time, range_end_time):
    # Reset and populate the batch data signal dictionary
    source_batch_data_dictionary = {}
    for i, measure in enumerate(measures):
        freq_nhz = measure['freq_nhz']
        period_ns = int((10 ** 18) // freq_nhz)
        measure_id = measure['id']

        # Create a time array for this specific measure
        measure_window_size = int((freq_nhz * window_duration_ns) // (10 ** 18))
        measure_slide_size = int((freq_nhz * window_slide_ns) // (10 ** 18))
        measure_batch_size = measure_window_size + (batch_num_windows - 1) * measure_slide_size
        measure_quantized_end_time = batch_start_time + (measure_batch_size * period_ns)
        measure_filled_time_array = np.arange(batch_start_time, measure_quantized_end_time, period_ns)
        measure_filled_value_array = np.full(measure_filled_time_array.shape, np.nan)

        # If partial windows are allowed, we need to make room for an extra full window,
        # but then only partially populate it. So find just the region where we actually want data
        data_start_time = max(range_start_time, batch_start_time)
        data_end_time = min(range_end_time, batch_end_time)

        start_index = np.searchsorted(measure_filled_time_array, data_start_time, side='left')

        expected_num_values = int(round((data_end_time - data_start_time) / (10 ** 18 / freq_nhz)))
        if expected_num_values > measure_filled_value_array.size - start_index:
            data_end_time = data_start_time + int(
                round((measure_filled_value_array.size - start_index) * (10 ** 18 / freq_nhz)))
            expected_num_values = int(round((data_end_time - data_start_time) / (10 ** 18 / freq_nhz)))

        nan_filled_out = measure_filled_value_array[start_index:start_index + expected_num_values]

        if expected_num_values > 0:
            sdk.get_data(
                measure_id, data_start_time, data_end_time, device_id=device_id, patient_id=query_patient_id,
                return_nan_filled=nan_filled_out)

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


def get_label_data(device_id, query_patient_id, batch_start_time, batch_end_time, batch_time_array, sdk,
                   row_size, slide_size, label_sets, label_threshold, label_exact_match=False):
    sliced_labels, threshold_labels = None, None
    # If labels exist, calculate them
    if len(label_sets) > 0:
        # Preallocate label matrix
        label_matrix = np.zeros((len(label_sets), len(batch_time_array)), dtype=np.int8)

        # Populate label matrix
        for idx, label_set_id in enumerate(label_sets):
            sdk.get_label_time_series(
                label_name_id=label_set_id,
                device_id=device_id if device_id else None,
                patient_id=query_patient_id if query_patient_id else None,
                start_time=batch_start_time,
                end_time=batch_end_time,
                timestamp_array=batch_time_array,
                out=label_matrix[idx],
                include_descendants=not label_exact_match,
            )

        # Create label windows
        windowed_label_views = sliding_window_view(
            label_matrix, (len(label_sets), row_size), axis=None)
        sliced_labels = windowed_label_views[0][::slide_size]

        threshold_labels = get_threshold_labels(sliced_labels, label_threshold=label_threshold)
    return sliced_labels, threshold_labels


def get_label_dictionary(sdk, device_id, query_patient_id, source_batch_start_time, source_batch_end_time, label_sets,
                         label_threshold, range_num_windows, row_period_ns, row_size, slide_size, label_exact_match=False):
    range_size = row_size + (range_num_windows - 1) * slide_size
    quantized_end_time = source_batch_start_time + (range_size * row_period_ns)
    source_time_array = np.arange(source_batch_start_time, quantized_end_time, row_period_ns)
    sliced_labels, threshold_labels = get_label_data(
        device_id, query_patient_id, source_batch_start_time, source_batch_end_time, source_time_array,
        sdk, row_size, slide_size, label_sets, label_threshold, label_exact_match=label_exact_match
    )
    return sliced_labels, threshold_labels

def _get_patient_info_from_cache(patient_id, window_start_time, patient_info_cache, patient_history_cache):
    window_patient_info = patient_info_cache.get(patient_id, {})
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

def get_window_list(device_id, patient_id, validated_measure_list, source_batch_data_dictionary,
                    batch_start_time, num_windows, window_duration_ns, threshold_labels, sliced_labels,
                    patient_history_cache, patient_history_fields, patient_info_cache):
    batch_window_list = []
    window_start_time = batch_start_time
    for window_i in range(num_windows):
        signal_dictionary = {}
        for measure in validated_measure_list:
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

        window_patient_info = patient_info_cache.get(patient_id, {})
        if patient_history_fields:
            window_patient_info = _get_patient_info_from_cache(
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
        batch_window_list.append(result_window)
        window_start_time += window_duration_ns
    return batch_window_list


def _load_patient_cache(patient_id, patient_info_cache, patient_history_cache, sdk, patient_history_fields):
    if patient_id is None:
        return
    # Check if patient id is in the info cache.
    if patient_id not in patient_info_cache:
        patient_info_cache[patient_id] = sdk.get_patient_info(patient_id=patient_id)
        if patient_info_cache[patient_id] is None:
            patient_info_cache[patient_id] = {}
        else:
            # Delete Height And Weight From Static Info
            if 'height' in patient_info_cache[patient_id]:
                del patient_info_cache[patient_id]['height']

            if 'weight' in patient_info_cache[patient_id]:
                del patient_info_cache[patient_id]['weight']

    if patient_history_fields and patient_id not in patient_history_cache:
        patient_history_cache[patient_id] = {}
        for field in ['height', 'weight']:
            if field not in patient_history_fields:
                continue
            patient_history_cache[patient_id][field] = sdk.get_patient_history(
                patient_id=patient_id, field=field)
