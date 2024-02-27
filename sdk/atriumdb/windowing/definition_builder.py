import numpy as np

from atriumdb.adb_functions import get_measure_id_from_generic_measure
from atriumdb.intervals.compact import reverse_compact_list
from atriumdb.intervals.intersection import list_intersection
from atriumdb.intervals.union import intervals_union_list


def build_source_intervals(sdk, measures=None, labels=None, patient_id_list=None, mrn_list=None,
                           device_id_list=None, device_tag_list=None, start_time=None, end_time=None,
                           gap_tolerance=None, merge_strategy=None):
    merge_strategy = "union" if merge_strategy is None else merge_strategy
    gap_tolerance = 0 if gap_tolerance is None else gap_tolerance
    # Check that exactly one source identifier list is provided
    source_lists = [patient_id_list, mrn_list, device_id_list, device_tag_list]
    if sum([source_list is not None for source_list in source_lists]) != 1:
        raise ValueError("Exactly one of patient_id_list, mrn_list, device_id_list, device_tag_list must be provided.")

    # Check that either measures or labels is provided, not both or neither
    if measures is None and labels is None:
        raise ValueError("Either measures or labels must be provided.")
    if measures is not None and labels is not None:
        raise ValueError("Only one of measures or labels should be provided.")

    if mrn_list is not None:
        if mrn_list == "all":
            all_patients = sdk.get_all_patients()
            mrn_list = [patient_info["mrn"] for patient_info in all_patients.values() if patient_info["mrn"] is not None]
        device_patient_list = \
            [(None, sdk.get_patient_id(mrn)) for mrn in mrn_list
             if sdk.get_patient_id(mrn) is not None]
    elif patient_id_list is not None:
        if patient_id_list == "all":
            patient_id_list = list(sdk.get_all_patients().keys())
        device_patient_list = \
            [(None, patient_id) for patient_id in patient_id_list
             if sdk.get_patient_info(patient_id) is not None]
    elif device_tag_list is not None:
        if device_tag_list == "all":
            all_devices = sdk.get_all_devices()
            device_tag_list = [device_info['tag'] for device_info in all_devices.values()]
        device_patient_list = \
            [(sdk.get_device_id(device_tag), None) for device_tag in device_tag_list
             if sdk.get_device_id(device_tag) is not None]
    elif device_id_list is not None:
        if device_id_list == "all":
            device_id_list = list(sdk.get_all_devices().keys())
        device_patient_list = \
            [(device_id, None) for device_id in device_id_list
             if sdk.get_device_info(device_id) is not None]
    else:
        raise ValueError("Exactly one of patient_id_list, mrn_list, device_id_list, device_tag_list must be provided.")

    source_list = next(filter(None, source_lists))  # Get the non-None source list
    source_key = ['patient_ids', 'mrns', 'device_ids', 'device_tags'][source_lists.index(source_list)]
    source_intervals = {'patient_ids': {}, 'mrns': {}, 'device_ids': {}, 'device_tags': {}}

    for device_id, patient_id in device_patient_list:
        interval_list = []
        if measures is not None:
            for measure in measures:
                measure_id = get_measure_id_from_generic_measure(sdk, measure, measure_tag_match_rule="best")
                if not measure_id:
                    continue
                measure_id = measure_id[0]
                interval_list.append(sdk.get_interval_array(
                    measure_id, device_id=device_id, patient_id=patient_id, start=start_time, end=end_time,
                    gap_tolerance_nano=gap_tolerance))

        elif labels is not None:
            for label in labels:
                interval_list.append(get_label_intervals(
                    sdk, label, device_id=device_id, patient_id=patient_id, start=start_time, end=end_time,
                    gap_tolerance=gap_tolerance))

        else:
            raise ValueError("Either measures or labels must be provided.")

        if merge_strategy == "union":
            merged_interval_array = intervals_union_list(interval_list)
        elif merge_strategy == "intersection":
            if len(interval_list) == 0:
                merged_interval_array = np.array([], dtype=np.int64)
            else:
                merged_interval_list = interval_list[0]
                for interval_array in interval_list[1:]:
                    merged_interval_list = list_intersection(merged_interval_list, interval_array)

                merged_interval_array = np.array(reverse_compact_list(merged_interval_list), dtype=np.int64)
        else:
            raise ValueError("merge_strategy must be either union or intersection")

        if merged_interval_array.size > 0:
            merged_interval_array = close_gaps(merged_interval_array, gap_tolerance)
            # Convert to list of dictionaries with "start" and "end" keys
            intervals = [{'start': int(start), 'end': int(end)} for start, end in merged_interval_array]
            # Add these intervals to the correct element in the source_intervals dictionary
            if patient_id is not None:
                source_intervals['patient_ids'][patient_id] = intervals
            elif device_id is not None:
                source_intervals['device_ids'][device_id] = intervals

    return source_intervals


def close_gaps(sorted_2d_array, gap_tolerance):
    if len(sorted_2d_array) == 0:
        return np.array([], dtype=np.int64)

    merged = [sorted_2d_array[0].tolist()]
    last_merged_interval = merged[-1]

    for i in range(1, len(sorted_2d_array)):
        current_interval = sorted_2d_array[i]

        # Check if the current interval is within the gap tolerance of the last merged interval
        if current_interval[0] - last_merged_interval[1] <= gap_tolerance:
            # Update the end of the last merged interval
            last_merged_interval[1] = current_interval[1]
        else:
            # Add the current interval as a new entry
            merged.append(current_interval.tolist())
            last_merged_interval = merged[-1]

    return np.array(merged, dtype=np.int64)


def get_label_intervals(sdk, label_name: str, device_id=None, patient_id=None, start=None, end=None,
                        gap_tolerance=None):
    """
    Retrieves intervals of a specific label from an AtriumSDK object.

    :param AtriumSDK sdk: The AtriumSDK object to interact with.
    :param str label_name: The name of the label to retrieve intervals for.
    :param int device_id: (Optional) Device ID to filter labels.
    :param int patient_id: (Optional) Patient ID to filter labels.
    :param int start: (Optional) The start time in nanoseconds.
    :param int end: (Optional) The end time in nanoseconds.
    :param int gap_tolerance: (Optional) The maximum allowable gap size in the data such that the output considers a
            region continuous. Put another way, the minimum gap size, such that the output of this method will add
            a new row.
    :return: A 2D numpy array of intervals, each row containing the start and end time.
    :rtype: numpy.ndarray
    """

    # Retrieve labels using the sdk
    labels = sdk.get_labels(
        name_list=[label_name],
        device_list=[device_id] if device_id else None,
        patient_id_list=[patient_id] if patient_id else None,
        start_time=start,
        end_time=end
    )

    sorted_labels = sorted(labels, key=lambda x: x['start_time_n'])

    # Process labels to create intervals
    intervals = []
    for label in sorted_labels:
        if len(intervals) > 0 and label['start_time_n'] - intervals[-1][1] <= gap_tolerance:
            intervals[-1][1] = label['end_time_n']
        else:
            intervals.append([label['start_time_n'], label['end_time_n']])

    return np.array(intervals, dtype=np.int64)
