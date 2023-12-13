# patient_info_to_transfer is default "all" or a list of strings equal to the keys/columns in the patient table that we want to transfer

# if deidentify is True, then we randomly assign new patient_ids to the to_sdk from range(10_000, 10_000 + 2 * len(total_transfered_patients)
# deidentify can also be a filename (str or Path) to a deidentification table

# patient_id_list or mrn_list must be not None, if mrn_list is not None, it is converted into a patient_id_list using the from_sdk
import csv
from pathlib import Path
from typing import Dict, Union
import random


def transfer_patient_info(from_sdk, to_sdk, patient_id_list=None, mrn_list=None, deidentify=True,
                          patient_info_to_transfer=None, start_time_nano=None, end_time_nano=None):
    """
    Transfers patient information, patient to device mapping and patient histories
    """
    patient_id_list = validate_patient_transfer_list(from_sdk, patient_id_list, mrn_list)

    patient_id_map = create_patient_id_map(patient_id_list, deidentify)

    transfer_patient_table(from_sdk, to_sdk, patient_id_list, patient_id_map, patient_info_to_transfer)

    transfer_patient_device_mapping(from_sdk, to_sdk, patient_id_list, patient_id_map, start_time_nano, end_time_nano)


def validate_patient_transfer_list(from_sdk, patient_id_list, mrn_list):
    if patient_id_list is None and mrn_list is None:
        # TODO: better Error msg
        raise ValueError("One must be specified, all or list")
    if patient_id_list == "all":
        patient_id_list = list(from_sdk.get_all_patients.keys())
    patient_id_list = [] if patient_id_list is None else patient_id_list
    assert isinstance(patient_id_list, list), "TODO: Good Error Message"
    if mrn_list is not None:
        patient_id_from_mrn_list = []
        if mrn_list == "all":
            patient_id_from_mrn_list = list(from_sdk.get_all_patients.keys())
        elif isinstance(mrn_list, list):
            mrn_to_patient_id_map = from_sdk.get_mrn_to_patient_id_map(mrn_list)
            patient_id_list.extend([mrn_to_patient_id_map[mrn] for mrn in mrn_list if mrn in mrn_to_patient_id_map])
        else:
            raise ValueError("TODO: Write Error")

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


def transfer_patient_table(from_sdk, to_sdk, patient_id_list, patient_id_map, patient_info_to_transfer=None):
    """
    Transfers patient information from one SDK to another.

    :param from_sdk: The SDK instance to transfer data from.
    :param to_sdk: The SDK instance to transfer data to.
    :param patient_id_list: List of patient IDs to transfer.
    :param patient_id_map: A mapping of old patient IDs to new patient IDs.
    :param patient_info_to_transfer: List of patient info keys to transfer. If None, all info is transferred.
    """

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
        if patient_info:
            # Use patient_id_map to get the new patient ID (if deidentified)
            new_patient_id = patient_id_map.get(patient_id, patient_id)

            # Filter patient info if patient_info_to_transfer is specified
            if patient_info_to_transfer and patient_info_to_transfer != "all":
                patient_info = {key: patient_info[key] for key in patient_info_to_transfer if key in patient_info}

            # Insert patient info into to_sdk
            to_sdk.insert_patient(patient_id=new_patient_id, **patient_info)
        else:
            print(f"Warning: Patient ID {patient_id} not found in from_sdk.")


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
