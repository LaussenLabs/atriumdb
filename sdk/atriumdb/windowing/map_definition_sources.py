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

import bisect
import warnings
from typing import List, Tuple

from atriumdb.intervals.difference import list_difference
from atriumdb.intervals.intersection import list_intersection
from atriumdb.intervals.union import intervals_union_list


def map_validated_sources(sources: dict, sdk) -> dict:
    # Initialize the new sources dictionary with a new key "device_patient_tuples"
    mapped_sources = {"device_patient_tuples": {}}

    # Extract patient_ids and device_ids dictionaries from the sources dictionary
    patient_ids = sources.get('patient_ids', {})
    device_ids = sources.get('device_ids', {})

    # The device_patient table may reference devices that don't exist in the device table (orphaned
    # mappings), so mapped devices must be checked against the device table before being trusted.
    known_device_ids = None
    warned_device_ids = set()

    def device_exists(device_id):
        nonlocal known_device_ids
        if known_device_ids is None:
            known_device_ids = {int(dev_id) for dev_id in sdk.get_all_devices().keys()}
        return int(device_id) in known_device_ids

    # Function to process ids (either patient_ids or device_ids) and update the mapped_sources dictionary
    def process_ids(ids_dict, id_type):
        for src_id, time_ranges in ids_dict.items():
            union_ranges = []
            device_patient_data = sdk.get_device_patient_data(
                    patient_id_list=[src_id] if id_type == 'patient_ids' else None,
                    device_id_list=[src_id] if id_type == 'device_ids' else None)

            time_sorted_device_patient_data = sorted(device_patient_data, key=lambda x: x[2])

            # Extract start_times and end_times
            device_patient_starts = [entry[2] for entry in time_sorted_device_patient_data]
            device_patient_ends = [entry[3] for entry in time_sorted_device_patient_data]

            for time_range in time_ranges:
                start_time, end_time = time_range
                # Fetch device_patient_data based on id_type
                matching_device_patient_data = find_device_patient_data(
                    time_sorted_device_patient_data, device_patient_starts, device_patient_ends, start_time, end_time)

                # Aggregate the time ranges based on the device and patient IDs
                aggregated_ranges = aggregate_time_ranges(matching_device_patient_data)
                for (device_id, patient_id), ranges in aggregated_ranges.items():
                    if not device_exists(device_id):
                        if device_id not in warned_device_ids:
                            warned_device_ids.add(device_id)
                            warnings.warn(
                                f"device_patient mapping references device id {device_id} which does not "
                                f"exist in the dataset. Ignoring its device-patient mappings.")
                        continue
                    intersected_ranges = list_intersection(ranges, [time_range])
                    if intersected_ranges:
                        key = (device_id, patient_id)
                        if key not in mapped_sources["device_patient_tuples"]:
                            mapped_sources["device_patient_tuples"][key] = intersected_ranges
                        else:
                            mapped_sources["device_patient_tuples"][key].extend(intersected_ranges)
                        # Update the union_ranges list for the current src_id
                        union_ranges.extend(sorted(intersected_ranges))

            # Calculate the union of ranges and update the mapped_sources dictionary with differences for the current src_id
            union_ranges = intervals_union_list(union_ranges).tolist()

            difference_ranges = list_difference(sorted(time_ranges), union_ranges)
            if len(difference_ranges) > 0:
                if id_type not in mapped_sources:
                    mapped_sources[id_type] = {}
                mapped_sources[id_type][src_id] = difference_ranges

    # Process patient_ids and device_ids separately
    process_ids(patient_ids, 'patient_ids')
    process_ids(device_ids, 'device_ids')

    if 'device_patient_tuples' in mapped_sources:
        mapped_sources['device_patient_tuples'] = reorder_dict_by_sublist(mapped_sources['device_patient_tuples'])

    return mapped_sources


def reorder_dict_by_sublist(input_dict):
    # Turn the dictionary into a list of (key, value) pairs.
    dict_items = list(input_dict.items())

    # Sort the list of pairs based on the first element of the first sublist in the values.
    sorted_items = sorted(dict_items, key=lambda item: item[1][0][0])

    # Create a new dictionary using the sorted pairs.
    new_dict = {key: value for key, value in sorted_items}
    return new_dict


def aggregate_time_ranges(device_patient_data: List[Tuple[int, int, int, int]]):
    result = {}
    for device_id, patient_id, start_time, end_time in device_patient_data:
        key = (device_id, patient_id)
        if key not in result:
            result[key] = []
        result[key].append([start_time, end_time])

    # Sort the time ranges for each unique (device_id, patient_id) pair
    for key in result:
        result[key].sort(key=lambda x: x[0])

    return result


def find_device_patient_data(sorted_device_patient_data, starts, ends, start_time, end_time):
    # Rows are sorted by start time only; when mappings overlap (e.g. two devices
    # concurrently mapped to the same patient) the ends are not guaranteed to be
    # in ascending order, so bisecting on ends is invalid. Bisect on starts to cut
    # off rows starting after end_time, then keep every row whose end reaches
    # start_time.
    end_idx = bisect.bisect_right(starts, end_time)
    return [row for row in sorted_device_patient_data[:end_idx] if row[3] >= start_time]