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

    # Function to process ids (either patient_ids or device_ids) and update the mapped_sources dictionary
    def process_ids(ids_dict, id_type):
        for src_id, time_ranges in ids_dict.items():
            union_ranges = []
            for time_range in time_ranges:
                start_time, end_time = time_range
                # Fetch device_patient_data based on id_type
                device_patient_data = sdk.get_device_patient_data(
                    patient_id_list=[src_id] if id_type == 'patient_ids' else None,
                    device_id_list=[src_id] if id_type == 'device_ids' else None,
                    start_time=start_time, end_time=end_time)
                # Aggregate the time ranges based on the device and patient IDs
                aggregated_ranges = aggregate_time_ranges(device_patient_data)
                for (device_id, patient_id), ranges in aggregated_ranges.items():
                    intersected_ranges = list_intersection(ranges, [time_range])
                    if intersected_ranges:
                        key = (device_id, patient_id)
                        if key not in mapped_sources["device_patient_tuples"]:
                            mapped_sources["device_patient_tuples"][key] = intersected_ranges
                        else:
                            mapped_sources["device_patient_tuples"][key].extend(intersected_ranges)
                        # Update the union_ranges list for the current src_id
                        union_ranges.extend(intersected_ranges)

            # Calculate the union of ranges and update the mapped_sources dictionary with differences for the current src_id
            union_ranges = intervals_union_list(union_ranges).tolist()
            for time_range in time_ranges:
                difference_ranges = list_difference([time_range], union_ranges)
                if difference_ranges:
                    if id_type not in mapped_sources:
                        mapped_sources[id_type] = {src_id: difference_ranges}
                    else:
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