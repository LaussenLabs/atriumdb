from atriumdb import AtriumSDK
from typing import Union, List
from pathlib import Path, PurePath
from tqdm import tqdm

from atriumdb.adb_functions import convert_value_to_nanoseconds
from atriumdb.transfer.csv.export_csv import export_csv_from_sdk
from atriumdb.transfer.csv.import_csv import import_csv_to_sdk

import logging

from atriumdb.transfer.csv.json_file_metadata import export_json_metadata, import_json_metadata

_LOGGER = logging.getLogger(__name__)

ONE_DAY_NANO = 86400 * (10 ** 9)


def import_csv_dataset(sdk: AtriumSDK, directory: Union[str, PurePath]):
    directory_path = Path(directory)
    assert directory_path.is_dir(), f"{directory_path} is not a valid directory."

    file_metadata = import_json_metadata(directory_path)

    for path in directory_path.rglob('*.csv'):
        if str(path.name) not in file_metadata:
            continue
        import_csv_to_sdk(sdk, path, file_metadata[str(path.name)])


def export_csv_dataset(sdk: AtriumSDK, directory: Union[str, PurePath], measure_id_list: List[int] = None,
                       device_id_list: List[int] = None, patient_id_list: List[int] = None, mrn_list: List[int] = None,
                       start: int = None, end: int = None, time_units: str = None, csv_dur=None, by_patient=False,
                       include_scale_factors=False):
    # Check if directory is already a directory, if not, make it so.
    directory_path = Path(directory)
    if not directory_path.is_dir():
        directory_path.mkdir(parents=True, exist_ok=True)

    time_units = 'ns' if time_units is None else time_units

    if start is not None:
        start = convert_value_to_nanoseconds(start, time_units)

    if end is not None:
        end = convert_value_to_nanoseconds(end, time_units)

    csv_dur = ONE_DAY_NANO if csv_dur is None else convert_value_to_nanoseconds(csv_dur, time_units)

    file_metadata = {}

    if by_patient:
        if mrn_list is not None:
            patient_id_list = [] if patient_id_list is None else patient_id_list
            mrn_patient_id_dict = sdk.get_mrn_to_patient_id_map(mrn_list=mrn_list)
            patient_id_list.extend(list(mrn_patient_id_dict.values))
            patient_id_list = list(set(patient_id_list))

        for measure_id in measure_id_list or sdk.get_all_measures().keys():
            for patient_id in patient_id_list or sdk.get_all_patient_ids():
                device_patient_list = sdk.get_device_patient_data(
                    device_id_list=device_id_list, patient_id_list=[patient_id], mrn_list=mrn_list,
                    start_time=start, end_time=end)

                for device_id, _, start_time, end_time in device_patient_list:
                    filename = directory_path / f"{measure_id}_{device_id}_{start_time}_{end_time}.csv"
                    metadata, _ = export_csv_from_sdk(sdk, filename, measure_id, start_time, end_time,
                                                      device_id=device_id, include_scale_factors=include_scale_factors)
                    metadata = {str(Path(filename).name): metadata}
                    file_metadata.update(metadata)
    else:
        for measure_id in tqdm(measure_id_list or sdk.get_all_measures().keys(), position=0, leave=False):
            measure_info = sdk.get_measure_info(measure_id)
            if measure_info is None:
                continue
            measure_tag = measure_info['tag']
            freq = measure_info['freq_nhz']
            unit = measure_info['unit']
            for device_id in tqdm(device_id_list or sdk.get_all_devices().keys(), position=1, leave=False,
                                  desc=f"Measure id: {measure_id}"):
                device_info = sdk.get_device_info(device_id)
                if device_info is None:
                    continue
                device_tag = device_info['tag']

                interval_arr = sdk.get_interval_array(measure_id, device_id=device_id)
                if interval_arr.size == 0:
                    _LOGGER.info(f"No data for measure id {measure_id}, device id {device_id}.")
                    continue
                start = interval_arr[0][0] if start is None else max(start, interval_arr[0][0])
                end = interval_arr[-1][-1] if end is None else min(end, interval_arr[-1][-1])
                for file_start in range(start, end, csv_dur):
                    file_end = file_start + csv_dur
                    filename = directory_path / f"{measure_tag}~{freq}~{unit}~{device_tag}~{file_start}~{file_end}.csv"
                    metadata, _ = export_csv_from_sdk(
                        sdk, filename, measure_id, file_start, file_end, device_id=device_id,
                        include_scale_factors=include_scale_factors)
                    file_metadata.update(metadata)

    export_json_metadata(directory, file_metadata)
