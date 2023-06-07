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

from pathlib import PurePath, Path
from typing import Union

from atriumdb import AtriumSDK
import pandas as pd

from atriumdb.transfer.formats.formats import IMPLEMENTED_DATA_FORMATS
from atriumdb.transfer.formats.json_file_metadata import create_json_metadata_dict


def export_data_from_sdk(sdk: AtriumSDK, directory: Union[str, PurePath], measure_id, start_time, end_time,
                         device_id=None, data_format=None, include_scale_factors=False):

    directory = Path(directory)
    data_format = 'csv' if data_format is None else data_format.lower()
    if data_format not in IMPLEMENTED_DATA_FORMATS:
        raise ValueError(f"Unsupported data format '{data_format}', supported formats are "
                         f"{list(IMPLEMENTED_DATA_FORMATS.keys())}")
    ext = IMPLEMENTED_DATA_FORMATS[data_format]['ext']

    device_info = sdk.get_device_info(device_id)
    measure_info = sdk.get_measure_info(measure_id)

    assert device_info is not None, f"device_id {device_id} not found."
    assert measure_info is not None, f"measure_id {measure_id} not found."

    measure_tag, freq, unit = measure_info['tag'], measure_info['freq_nhz'], measure_info['unit']
    device_tag = device_info['tag']

    filename = directory / f"{measure_tag}~{freq}~{unit}~{device_tag}~{start_time}~{end_time}{ext}"
    headers, times, values = sdk.get_data(measure_id, start_time, end_time, device_id=device_id, analog=False)

    if len(headers) == 0:
        return None

    device_tag, measure_tag, freq_nhz, units = get_source_info(sdk, measure_id, device_id)
    scale_m, scale_b = None, None

    if include_scale_factors:
        scale_m, scale_b = get_scale_factors(headers)

    df = create_dataframe(measure_tag, times, values)

    if data_format == 'csv':
        df.to_csv(filename, index=False)
    elif data_format == 'parquet':
        df.to_parquet(filename, index=False, engine='fastparquet')

    metadata = create_json_metadata_dict(
        measure_tag, freq_nhz, units, device_tag, start_time, end_time, scale_m, scale_b, None)

    filename = str(filename.name)

    return {filename: metadata}, df


def get_source_info(sdk, measure_id, device_id):
    measure_info = sdk.get_measure_info(measure_id)
    device_info = sdk.get_device_info(device_id)

    assert measure_info is not None, f"No measure id {measure_id} found"
    assert device_info is not None, f"No device_ id {device_id} found"

    measure_tag = measure_info['tag']
    freq_nhz = measure_info['freq_nhz']
    units = measure_info['unit']
    device_tag = device_info['tag']

    return device_tag, measure_tag, freq_nhz, units


def get_scale_factors(headers):
    scale_m = headers[0].scale_m
    scale_b = headers[0].scale_b

    return scale_m, scale_b


def create_dataframe(measure_tag, times, values):
    data = {f"Epoch Nano": times, f"{measure_tag}": values}
    df = pd.DataFrame(data)

    return df
