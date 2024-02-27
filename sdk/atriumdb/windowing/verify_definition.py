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
from typing import List, Tuple, Dict

import yaml
import numpy as np

from atriumdb.adb_functions import get_measure_id_from_generic_measure
from atriumdb.intervals.union import intervals_union_list
from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.windowing.map_definition_sources import map_validated_sources


def verify_definition(definition, sdk, gap_tolerance=None, measure_tag_match_rule="best"):
    """
    Verifies and validates a dataset definition against the given AtriumSDK, including measures, label sets,
    and sources.

    Parameters:
    - definition (str | DatasetDefinition): The dataset definition to be verified. This can be either a filename
      (str) pointing to a dataset definition file or an instance of DatasetDefinition.
    - sdk: An AtriumSDK object pointing at the dataset to validate the requested definition against.
    - gap_tolerance (int, optional): The minimum allowed gap (in nanoseconds) in any generated time ranges
        (time ranges are explained below).
    - measure_tag_match_rule (str, optional): "best" or "all" as a strategy for dealing with measure tags where there
        may be multiple measures with the given tag.

    Returns:
    - tuple: A tuple containing three elements:
        1. validated_measure_list (list of dicts): A list of dictionaries, each representing a validated measure.
           Each dictionary includes:
           - 'id': The unique identifier of the measure.
           - 'tag': A string tag associated with the measure.
           - 'freq_nhz': The frequency of the measure in nanohertz (optional).
           - 'units': The units of the measure (optional).
        2. validated_label_set_list (list of int): A list of label set IDs that have been validated against
            the AtriumSDK. Each element is an integer representing the unique identifier of a label set.
        3. mapped_sources (dict): A dictionary representing the validated and mapped sources with the following keys:
           - 'device_patient_tuples': A dictionary where each key is a tuple (device_id, patient_id) and each value
             is a list of time ranges (start_time, end_time) for which the device-patient tuple has data. Both
             start_time and end_time are integers, representing nanosecond precision timestamps since the Unix epoch.
           - 'patient_ids' (optional): A dictionary present only if there are unmatched patient IDs. Each key is a
             patient ID, and each value is a list of time ranges where data could not be matched to devices.
           - 'device_ids' (optional): A dictionary present only if there are unmatched device IDs. Each key is a
             device ID, and each value is a list of time ranges where data could not be matched to patients.

    Raises:
    - ValueError: If the input 'definition' is neither a filename (str) nor an instance of DatasetDefinition, or
      if specified measures or label sets are not found in the AtriumSDK.

    """
    gap_tolerance = 60_000_000_000 if gap_tolerance is None else gap_tolerance  # 1 minute nano default
    # If the input is a string, it is assumed to be a filename
    if isinstance(definition, str):
        definition = DatasetDefinition(definition)
    elif isinstance(definition, DatasetDefinition):
        pass
    else:
        raise ValueError("Input must be either a filename or an instance of DatasetDefinition.")

    # Validate measures
    validated_measure_list = _validate_measures(definition, sdk, measure_tag_match_rule=measure_tag_match_rule)

    # Validate label sets
    validated_label_set_list = _validate_label_sets(definition, sdk)

    # Validate sources
    validated_sources = _validate_sources(definition, sdk, validated_measure_list, gap_tolerance=gap_tolerance)

    mapped_sources = map_validated_sources(validated_sources, sdk)

    return validated_measure_list, validated_label_set_list, mapped_sources


def _validate_measures(definition: DatasetDefinition, sdk, measure_tag_match_rule="best"):
    assert "measures" in definition.data_dict, "definition must have some specified measures"

    measures = definition.data_dict["measures"]

    validated_measure_list = []

    for measure_spec in measures:
        measure_ids = get_measure_id_from_generic_measure(sdk, measure_spec, measure_tag_match_rule=measure_tag_match_rule)

        for measure_id in measure_ids:
            if measure_id is None:
                raise ValueError(f"Measure {measure_spec} not found in AtriumSDK. Must use AtriumSDK.insert_measure "
                                 f"to insert the measure")

            measure_info = sdk.get_measure_info(measure_id)
            validated_measure_info = {
                'id': measure_id,
                'tag': measure_info['tag'],
                'freq_nhz': measure_info.get('freq_nhz'),
                'units': measure_info.get('unit')
            }
            validated_measure_list.append(validated_measure_info)

    return validated_measure_list


def _validate_label_sets(definition: DatasetDefinition, sdk):
    # If there aren't any labels, there's nothing to do.
    if "labels" not in definition.data_dict or len(definition.data_dict['labels']) == 0:
        return []

    labels = definition.data_dict["labels"]

    all_sdk_label_sets = sdk.get_all_label_names()
    all_sdk_label_set_name_to_id_dict = {
        label_info['name']: label_info['id'] for label_info in all_sdk_label_sets.values()}

    validated_label_set_list = []

    for label in labels:
        if label not in all_sdk_label_set_name_to_id_dict:
            raise ValueError(
                f"Label set {label} not found in AtriumSDK. Must use AtriumSDK.insert_label with a valid label if you "
                f"want use the label set {label}. If you don't have any valid labels, but still want"
                f"to include the label set use: AtriumSDK.sql_handler.insert_label_set(name) to introduce a new "
                f"label set.")
        validated_label_set_list.append(all_sdk_label_set_name_to_id_dict[label])

    return validated_label_set_list


def _validate_sources(definition: DatasetDefinition, sdk, validated_measure_list, gap_tolerance=None):
    data_dict = definition.data_dict

    validated_sources_dict = dict()
    for source_type, source_data in data_dict.items():
        if len(source_data) == 0:
            continue
        current_validated_source_dict = dict()
        if source_type == 'mrns':
            source_type = "patient_ids"
            mrn_to_patient_id_map = sdk.get_mrn_to_patient_id_map(list(source_data.keys()))
            for mrn, time_specs in source_data.items():
                if mrn in mrn_to_patient_id_map:
                    patient_id = mrn_to_patient_id_map[mrn]
                    validated_entries = _get_validated_entries(time_specs,
                                                               validated_measure_list, sdk,
                                                               patient_id=patient_id,
                                                               gap_tolerance=gap_tolerance)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[patient_id] = validated_entries
                else:
                    warnings.warn(f"MRN {mrn} not found in database, omitting from cohort data")

        elif source_type == 'patient_ids':
            all_patients = sdk.get_all_patients()
            for patient_id, time_specs in source_data.items():
                if patient_id in all_patients:
                    validated_entries = _get_validated_entries(time_specs,
                                                               validated_measure_list, sdk,
                                                               patient_id=patient_id,
                                                               gap_tolerance=gap_tolerance)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[patient_id] = validated_entries
                else:
                    warnings.warn(f"Patient ID {patient_id} not found in database, omitting from cohort data")

        elif source_type == 'device_tags':
            source_type = 'device_ids'
            all_devices = sdk.get_all_devices()
            tag_to_dev_id_map = {str(device_info['tag']): int(dev_id) for dev_id, device_info in all_devices.items()}
            for device_tag, time_specs in source_data.items():
                if str(device_tag) in tag_to_dev_id_map:
                    device_id = tag_to_dev_id_map[str(device_tag)]
                    validated_entries = _get_validated_entries(time_specs,
                                                               validated_measure_list, sdk,
                                                               device_id=device_id,
                                                               gap_tolerance=gap_tolerance)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[device_id] = validated_entries
                else:
                    warnings.warn(f"Device tag {device_tag} not found in database, omitting from cohort data")

        elif source_type == 'device_ids':
            all_device_ids = list(sdk.get_all_devices().keys())
            for device_id, time_specs in source_data.items():
                if device_id in all_device_ids:
                    validated_entries = _get_validated_entries(time_specs,
                                                               validated_measure_list, sdk,
                                                               device_id=device_id,
                                                               gap_tolerance=gap_tolerance)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[device_id] = validated_entries
                else:
                    warnings.warn(f"Device ID {device_id} not found in database, omitting from cohort data")

        elif source_type in ['measures', 'labels']:
            # Not a source type
            continue
        else:
            raise ValueError(f"Invalid source type {source_type}. Allowed types are: "
                             f"'mrns', 'patient_ids', 'device_tags', 'device_ids'.")

        # Update or create new source dictionary
        if source_type in validated_sources_dict:
            validated_sources_dict[source_type].update(current_validated_source_dict)
        else:
            validated_sources_dict[source_type] = current_validated_source_dict

    return validated_sources_dict


def _get_validated_entries(time_specs, validated_measures, sdk, device_id=None, patient_id=None, gap_tolerance=None):
    gap_tolerance = 60 * 60 * 1_000_000_000 if gap_tolerance is None else gap_tolerance  # 1 hour nano default

    union_intervals = intervals_union_list(
        [sdk.get_interval_array(
            measure_info['id'], device_id=device_id, patient_id=patient_id, gap_tolerance_nano=gap_tolerance)
            for measure_info in validated_measures])

    merged_union_intervals = []
    for start, end in union_intervals:
        if len(merged_union_intervals) > 0 and start - merged_union_intervals[-1][1] <= gap_tolerance:
            merged_union_intervals[-1][1] = end
        else:
            merged_union_intervals.append([start, end])

    union_intervals = np.array(merged_union_intervals, dtype=np.int64)

    if union_intervals.size == 0:
        source_type, source_id = ("device_id", device_id) if device_id is not None else ("patient_id", patient_id)
        warnings.warn(f"{source_type}: {source_id} could not be found over the requested "
                      f"time regions for the specified measures. Skipping")
        return None

    if time_specs == "all":
        return union_intervals.tolist()

    interval_list = []

    for region_data in time_specs:
        if 'time0' in region_data:
            start, end = region_data['time0'] - region_data['pre'], region_data['time0'] + region_data['post']
        elif 'start' in region_data and 'end' in region_data:
            start, end = region_data['start'], region_data['end']
        elif 'start' in region_data:
            start, end = region_data['start'], union_intervals[-1][1]
        elif 'end' in region_data:
            start, end = union_intervals[0][0], region_data['end']
        else:
            raise ValueError("time0, pre, post or start and/or end must be in the region specification.")

        interval_list.append([start, end])

    return interval_list
