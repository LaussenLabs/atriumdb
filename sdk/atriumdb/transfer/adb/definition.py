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
from atriumdb.windowing.definition import DatasetDefinition


def create_dataset_definition_from_verified_data(sdk, validated_measure_list, validated_mapped_sources,
                                                 validated_label_set_list=None, prefer_patient=True,
                                                 patient_id_map=None, file_path_dicts=None, time_shift_nano=None):
    """
    Creates a new DatasetDefinition object based on the verified and validated data.

    Parameters:
    - sdk (AtriumSDK): An AtriumSDK object used for fetching additional information like device tags and label names.
    - validated_measure_list (list of dicts): A list of dictionaries, each representing a validated measure.
    - validated_mapped_sources (dict): A dictionary representing the validated and mapped sources.
    - validated_label_set_list (list of int, optional): A list of label set IDs that have been validated against the
        AtriumSDK.
    - prefer_patient (bool, optional): Determines whether to prefer saving data by patient or by device for the
      device_patient tuples part of the validated_mapped_sources. Defaults to True.
    - time_shift_nano (int, optional): how many nanoseconds to shift time data.

    Returns:
    - DatasetDefinition: A new DatasetDefinition object created from the verified and validated data.

    """

    # Initialize the parameters for the DatasetDefinition
    measures = []
    labels = []
    patient_ids = {}
    device_tags = {}
    patient_id_map = {} if patient_id_map is None else patient_id_map
    file_path_dicts = {} if file_path_dicts is None else file_path_dicts

    # Process measures
    for measure in validated_measure_list:
        measure_info = {"tag": measure['id']}
        if 'freq_nhz' in measure:
            measure_info["freq_hz"] = measure['freq_nhz'] / 1e9  # Convert nanohertz to Hertz
        if 'units' in measure:
            measure_info["units"] = measure['units']
        measures.append(measure_info)

    # Process label sets
    if validated_label_set_list:
        for label_set_id in validated_label_set_list:
            label_name_info = sdk.get_label_name_info(label_set_id)
            if label_name_info:
                labels.append(label_name_info['name'])
            else:
                raise ValueError(f"label set id {label_set_id} not found in dataset.")

    # Process mapped sources
    for device_patient_tuple, time_ranges in validated_mapped_sources.get('device_patient_tuples', {}).items():
        # Time shift
        time_ranges = get_shifted_time_ranges(time_ranges, time_shift_nano)

        device_id, patient_id = device_patient_tuple

        # Map the de-identified patient ids if needed
        patient_id = patient_id_map.get(patient_id, patient_id)

        device_info = sdk.get_device_info(device_id)
        if device_info is None:
            raise ValueError(f"device id {device_id} not found in dataset.")
        device_tag = device_info['tag']

        # Choose to preferentially save data by patient or device based on the prefer_patient flag

        if prefer_patient:
            if patient_id not in patient_ids:
                patient_ids[patient_id] = []
            for time_range in time_ranges:
                patient_ids[patient_id].append({'start': time_range[0], 'end': time_range[1]})
                source_tuple = ('device_patient_tuples', device_patient_tuple, int(time_range[0]), int(time_range[-1]))
                if source_tuple in file_path_dicts and len(file_path_dicts[source_tuple]) > 0:
                    patient_ids[patient_id][-1]['files'] = file_path_dicts[source_tuple]
        else:
            if device_tag not in device_tags:
                device_tags[device_tag] = []
            for time_range in time_ranges:
                device_tags[device_tag].append({'start': time_range[0], 'end': time_range[1]})
                source_tuple = ('device_patient_tuples', device_patient_tuple, int(time_range[0]), int(time_range[-1]))
                if source_tuple in file_path_dicts and len(file_path_dicts[source_tuple]) > 0:
                    device_tags[device_tag][-1]['files'] = file_path_dicts[source_tuple]

    # Handle unmatched device_ids by converting to device_tags and including them in the device_tags dictionary
    for device_id, time_ranges in validated_mapped_sources.get('device_ids', {}).items():
        # Time shift
        time_ranges = get_shifted_time_ranges(time_ranges, time_shift_nano)

        device_info = sdk.get_device_info(device_id)
        if device_info is None:
            raise ValueError(f"Unmatched device id {device_id} not found in dataset.")
        device_tag = device_info['tag']
        if device_tag not in device_tags:
            device_tags[device_tag] = []
        for time_range in time_ranges:
            device_tags[device_tag].append({'start': time_range[0], 'end': time_range[1]})
            source_tuple = ('device_ids', device_id, int(time_range[0]), int(time_range[-1]))
            if source_tuple in file_path_dicts and len(file_path_dicts[source_tuple]) > 0:
                device_tags[device_tag][-1]['files'] = file_path_dicts[source_tuple]

    # Create the new DatasetDefinition object
    new_dataset_definition = DatasetDefinition(measures=measures, patient_ids=patient_ids, device_tags=device_tags,
                                               labels=labels)

    return new_dataset_definition


def get_shifted_time_ranges(time_ranges, time_shift_nano):
    if time_shift_nano is None:
        return time_ranges

    shifted_time_ranges = []
    for start, end in time_ranges:
        shifted_time_ranges.append([start + time_shift_nano, end + time_shift_nano])

    return shifted_time_ranges
