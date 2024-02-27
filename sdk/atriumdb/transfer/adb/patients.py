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


import csv
import time
from pathlib import Path
from typing import Dict, Union
import random


def transfer_patient_info(src_sdk, dest_sdk, patient_id_list=None, mrn_list=None, deidentify=True,
                          patient_info_to_transfer=None, start_time_nano=None, end_time_nano=None,
                          deidentification_functions=None, time_shift_nano=None):
    """
    Transfers patient information, mappings between patients and devices, and patient history records from one
    AtriumSDK instance to another. Depending on the `deidentify` parameter, it may also anonymize
    patient IDs in the transfer process.

   :param AtriumSDK src_sdk: The source SDK instance from which to transfer data.
   :param AtriumSDK dest_sdk: The destination SDK instance to transfer data to.
   :param list patient_id_list: (Optional) A list of patient IDs to transfer. If not provided, `mrn_list`
    will be used to generate this list. If set to "all", transfer all patient info.
   :param list mrn_list: (Optional) A list of medical record numbers (MRNs) corresponding to the patients to transfer.
    If `patient_id_list` is not provided, `mrn_list` will be used to determine patients to transfer.
   :param union[bool, str, Path] deidentify: If True, randomly assign new patient IDs. If a filename (str or Path)
    is provided, use the deidentification table from the file. If False, keep the original patient IDs.
   :param list patient_info_to_transfer: (Optional) A list of patient info keys to transfer.
    If None or "all", transfer all patient info.
   :param int start_time_nano: (Optional) The start time in nanoseconds for patient-device history records.
   :param int end_time_nano: (Optional) The end time in nanoseconds for patient-device history records.
   :param deidentification_functions: (Optional) A dictionary mapping patient info keys to deidentification functions.
   :param int time_shift_nano: (Optional) The number of nanoseconds to shift the times of the data.

   :raises ValueError: If both `patient_id_list` and `mrn_list` are None, or if the provided
    `patient_info_to_transfer` contains invalid keys.

   .. note::
      - If `deidentify` is a filename, it assumes that the CSV format is as follows: original_patient_id,new_patient_id
      - If you use `deidentify=True`, you may also want to restrict what patient information is being transferred
        using patient_info_to_transfer

   Examples:
   ---------

   .. code-block:: python

      from atriumdb import AtriumSDK
      from atriumdb.transfer.adb.patients import transfer_patient_info

      # Initialize source and destination SDK instances
      src_sdk = AtriumSDK(dataset_location="./src_dataset")
      dest_sdk = AtriumSDK(dataset_location="./dest_dataset")

      # Transfer specific patients using their patient IDs
      transfer_patient_info(src_sdk, dest_sdk, patient_id_list=[1234, 5678])

      # Transfer all patients with deidentification
      transfer_patient_info(src_sdk, dest_sdk, patient_id_list="all", deidentify=True)

      # Transfer data using MRNs and specify a deidentification map from a file
      transfer_patient_info(src_sdk, dest_sdk, mrn_list=[123456, 654321], deidentify="deid_map.csv")

      # Transfer all patient info between a specific time range
      transfer_patient_info(src_sdk, dest_sdk, patient_id_list="all", start_time_nano=1617264000000000000, end_time_nano=1617350400000000000)

    """
    patient_id_list = validate_patient_transfer_list(src_sdk, patient_id_list, mrn_list)

    patient_id_map = create_patient_id_map(patient_id_list, deidentify)

    transfer_patient_table(src_sdk, dest_sdk, patient_id_list, patient_id_map, patient_info_to_transfer,
                           deidentification_functions, time_shift_nano, deidentify)

    transfer_patient_device_mapping(src_sdk, dest_sdk, patient_id_map, start_time_nano, end_time_nano, time_shift_nano)

    return patient_id_map


def transfer_patient_device_mapping(src_sdk, dest_sdk, patient_id_map, start_time_nano, end_time_nano,
                                    time_shift_nano=None):
    dest_device_dict = dest_sdk.get_all_devices()

    src_to_dest_dev_ids_dict = {src_sdk.get_device_id(device_info['tag']): dest_dev_id for dest_dev_id, device_info in
                                dest_device_dict.items() if src_sdk.get_device_id(device_info['tag']) is not None}

    device_patient_list = src_sdk.get_device_patient_data(
        device_id_list=list(src_to_dest_dev_ids_dict.keys()), patient_id_list=list(patient_id_map.keys()),
        start_time=start_time_nano, end_time=end_time_nano)

    dest_device_patient_list = []
    for device_id, patient_id, start_time, end_time in device_patient_list:
        end_time = int(time.time_ns()) if end_time is None else end_time
        if time_shift_nano is not None:
            start_time += time_shift_nano
            end_time += time_shift_nano

        dest_device_patient_list.append(
            [src_to_dest_dev_ids_dict[device_id], patient_id_map[patient_id], start_time, end_time])

    if len(dest_device_patient_list) > 0:
        dest_sdk.insert_device_patient_data(device_patient_data=dest_device_patient_list)


def validate_patient_transfer_list(from_sdk, patient_id_list, mrn_list):
    if patient_id_list is None and mrn_list is None:
        raise ValueError("Either patient_id_list or mrn_list must be specified.")
    if patient_id_list == "all":
        patient_id_list = list(from_sdk.get_all_patients().keys())
    patient_id_list = [] if patient_id_list is None else patient_id_list
    assert isinstance(patient_id_list, list), "patient_id_list must be a list of patient ids or the string \"all\""
    if mrn_list is not None:
        patient_id_from_mrn_list = []
        if mrn_list == "all":
            patient_id_from_mrn_list = list(from_sdk.get_all_patients().keys())
        elif isinstance(mrn_list, list):
            mrn_to_patient_id_map = from_sdk.get_mrn_to_patient_id_map(mrn_list)
            patient_id_list.extend([mrn_to_patient_id_map[mrn] for mrn in mrn_list if mrn in mrn_to_patient_id_map])
        else:
            raise ValueError("mrn_list must be a list of patient ids or the string \"all\"")

        patient_id_list.extend(patient_id_from_mrn_list)
    patient_id_list = list(set(patient_id_list))
    patient_id_list.sort()
    return patient_id_list


def create_patient_id_map(patient_id_list, deidentify, overwrite=False):
    if not deidentify:
        # No de-identification needed; return a direct mapping
        return {patient_id: patient_id for patient_id in patient_id_list}

    if isinstance(deidentify, (str, Path)):
        # If deidentify is a file path
        file_path = Path(deidentify)
        if file_path.exists() and not overwrite:
            # If the file exists and we're not overwriting, load the existing map
            patient_id_map = read_csv_to_dict(file_path)
            if validate_csv_dict(patient_id_map, patient_id_list):
                return patient_id_map
            else:
                raise ValueError("Existing de-identification map does not cover all patient IDs.")
        else:
            # If the file doesn't exist or we're overwriting, create a new map
            new_map = generate_new_patient_id_map(patient_id_list)
            write_dict_to_csv(new_map, file_path)
            return new_map
    else:
        # deidentify is True, generate new map without saving
        return generate_new_patient_id_map(patient_id_list)


def generate_new_patient_id_map(patient_id_list):
    new_ids = generate_patient_ids(len(patient_id_list))
    return dict(zip(patient_id_list, new_ids))


def write_dict_to_csv(dictionary: Dict, file_path: Union[str, Path]):
    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def read_csv_to_dict(file_path: Union[str, Path]) -> Dict:
    dictionary = {}
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dictionary[int(row[0])] = int(row[1])
    return dictionary


def validate_csv_dict(csv_dict: Dict, patient_id_list: list) -> bool:
    return all(patient_id in csv_dict for patient_id in patient_id_list)


def generate_patient_ids(number_of_patients):
    # Generate the list of unique patient IDs
    return random.sample(range(10000, 10000 + 2 * number_of_patients), number_of_patients)


def transfer_patient_table(from_sdk, to_sdk, patient_id_list, patient_id_map, patient_info_to_transfer=None,
                           deidentification_functions=None, time_shift_nano=None, deidentify=False):
    """
    Transfers patient information from one SDK to another.

    :param from_sdk: The SDK instance to transfer data from.
    :param to_sdk: The SDK instance to transfer data to.
    :param patient_id_list: List of patient IDs to transfer.
    :param patient_id_map: A mapping of old patient IDs to new patient IDs.
    :param patient_info_to_transfer: List of patient info keys to transfer. If None, all info is transferred.
    :param deidentification_functions: A dictionary mapping patient info keys to deidentification functions.
    :param time_shift_nano: The number of nanoseconds to shift the times of the data.
    :param deidentify: Whether or not to transfer information by default.
    """

    deidentification_functions = {} if deidentification_functions is None else deidentification_functions
    # Retrieve all patients from the from_sdk
    all_patients = from_sdk.get_all_patients()

    if len(all_patients) == 0:
        return

    validate_patient_info_to_transfer(patient_info_to_transfer, all_patients)

    # Iterate over the patient IDs to be transferred
    for patient_id in patient_id_list:
        # Retrieve patient info based on patient_id
        patient_info = all_patients.get(patient_id)

        # If patient info exists, proceed with transfer
        if not patient_info:
            print(f"Warning: Patient ID {patient_id} not found in from_sdk.")
            continue

        # Use patient_id_map to get the new patient ID (if deidentified)
        new_patient_id = patient_id_map.get(patient_id, patient_id)
        dest_patient_info = {}
        for key, value in patient_info.items():
            # Remove unwanted / dynamic keys
            if key in ["id", 'weight', 'height', 'height_units', 'height_time', 'weight_units', 'weight_time']:
                continue

            if deidentify and (not isinstance(patient_info_to_transfer, list) or key not in patient_info_to_transfer):
                continue
            if patient_info_to_transfer is not None and patient_info_to_transfer != "all" and key not in patient_info_to_transfer:
                continue

            # Time shift time values
            if time_shift_nano is not None and key in ['dob', 'first_seen', 'last_updated']:
                value += time_shift_nano

            if key in deidentification_functions:
                value = deidentification_functions[key](value)

            dest_patient_info[key] = value

        # Insert patient info into to_sdk
        to_sdk.insert_patient(patient_id=new_patient_id, **dest_patient_info)

        # Transfer Patient History
        patient_history_list = []
        for field in ['height', 'weight']:
            if deidentify and (not isinstance(patient_info_to_transfer, list) or field not in patient_info_to_transfer):
                continue

            if patient_info_to_transfer is not None and patient_info_to_transfer != "all" and field not in patient_info_to_transfer:
                continue

            patient_history_list.extend(from_sdk.get_patient_history(patient_id=patient_id, field=field))

        for _, query_patient_id, field, value, units, measurement_time in patient_history_list:
            if field in deidentification_functions:
                value = deidentification_functions[field](value)

            if time_shift_nano is not None:
                measurement_time += time_shift_nano

            to_sdk.insert_patient_history(field, value, units, measurement_time, patient_id=new_patient_id)


def validate_patient_info_to_transfer(patient_info_to_transfer, all_patients):
    if patient_info_to_transfer and patient_info_to_transfer != "all":
        if not isinstance(patient_info_to_transfer, list):
            raise ValueError("patient_info_to_transfer, must be a list of patient "
                             "info attributes or the string 'all' (or None)")

        available_patient_info_keys = list(list(all_patients.values())[0].keys())
        for key in patient_info_to_transfer:
            if key not in available_patient_info_keys:
                raise ValueError(
                    f"key {key} from patient_info_to_transfer, not a valid key in patient_info. "
                    f"Must be one of: {available_patient_info_keys}")
