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

from atriumdb import AtriumSDK
from typing import Union, List
from pathlib import Path, PurePath
from tqdm import tqdm

from atriumdb.adb_functions import convert_value_to_nanoseconds
from atriumdb.transfer.formats.export_data import export_data_from_sdk
from atriumdb.transfer.formats.formats import IMPLEMENTED_DATA_FORMATS
from atriumdb.transfer.formats.import_data import import_data_to_sdk

import logging

from atriumdb.transfer.formats.json_file_metadata import export_json_metadata, import_json_metadata

_LOGGER = logging.getLogger(__name__)

ONE_DAY_NANO = 86400 * (10 ** 9)


def import_dataset(sdk: AtriumSDK, directory: Union[str, PurePath], data_format=None):
    directory_path = Path(directory)
    assert directory_path.is_dir(), f"{directory_path} is not a valid directory."

    file_metadata = import_json_metadata(directory_path)

    ext = IMPLEMENTED_DATA_FORMATS[data_format]['ext']
    for path in directory_path.rglob(f'*{ext}'):
        if str(path.name) not in file_metadata:
            continue
        import_data_to_sdk(sdk, path, file_metadata[str(path.name)], data_format=data_format)


def export_dataset(sdk: AtriumSDK, directory: Union[str, PurePath], data_format=None, device_id_list: List[int] = None,
                   patient_id_list: List[int] = None, mrn_list: List[int] = None, start: int = None, end: int = None,
                   time_units: str = None, csv_dur=None, by_patient=False, include_scale_factors=False,
                   measure_id_list: List[int] = None):
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
            patient_id_list.extend(list(mrn_patient_id_dict.values()))
            patient_id_list = list(set(patient_id_list))

        for measure_id in measure_id_list or sdk.get_all_measures().keys():
            for patient_id in patient_id_list or sdk.get_all_patients().keys():
                device_patient_list = sdk.get_device_patient_data(
                    device_id_list=device_id_list, patient_id_list=[patient_id], mrn_list=mrn_list,
                    start_time=start, end_time=end)

                for device_id, _, start_time, end_time in device_patient_list:
                    metadata, _ = export_data_from_sdk(sdk, directory_path, measure_id, start_time, end_time,
                                                                 device_id=device_id,
                                                                 data_format=data_format,
                                                                 include_scale_factors=include_scale_factors)
                    file_metadata.update(metadata)
    else:
        for measure_id in tqdm(measure_id_list or sdk.get_all_measures().keys(), position=0, leave=False):
            if sdk.get_measure_info(measure_id) is None:
                continue

            for device_id in tqdm(device_id_list or sdk.get_all_devices().keys(), position=1, leave=False,
                                  desc=f"Measure id: {measure_id}"):
                if sdk.get_device_info(device_id) is None:
                    continue

                interval_arr = sdk.get_interval_array(measure_id, device_id=device_id)
                if interval_arr.size == 0:
                    _LOGGER.info(f"No data for measure id {measure_id}, device id {device_id}.")
                    continue
                start = int(interval_arr[0][0]) if start is None else max(start, int(interval_arr[0][0]))
                end = int(interval_arr[-1][-1]) if end is None else min(end, int(interval_arr[-1][-1]))
                file_start = start
                while file_start < end:
                    file_end = min(file_start + csv_dur, end)
                    metadata, _ = export_data_from_sdk(sdk, directory_path, measure_id, file_start, file_end,
                                                       device_id=device_id,
                                                       data_format=data_format,
                                                       include_scale_factors=include_scale_factors)
                    file_metadata.update(metadata)
                    file_start += csv_dur

    export_json_metadata(directory, file_metadata)
