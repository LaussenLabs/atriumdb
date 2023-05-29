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
import pandas as pd
from typing import Union, Dict
from pathlib import PurePath

from atriumdb.transfer.formats.formats import IMPLEMENTED_DATA_FORMATS


def import_data_to_sdk(sdk: AtriumSDK, filename: Union[str, PurePath], metadata: Dict[str, Union[str, float, int]],
                       data_format=None):
    data_format = 'csv' if data_format is None else data_format.lower()
    if data_format not in IMPLEMENTED_DATA_FORMATS:
        raise ValueError(f"Unsupported data format '{data_format}', supported formats are "
                         f"{list(IMPLEMENTED_DATA_FORMATS.keys())}")

    if data_format == 'csv':
        df = pd.read_csv(filename)
    elif data_format == 'parquet':
        df = pd.read_parquet(filename, engine='fastparquet')

    device_tag = metadata['device_tag']
    measure_tag = metadata['measure_tag']
    freq_nhz = metadata['freq']
    units = metadata['unit']
    scale_m = metadata.get('scale_m', None)
    scale_b = metadata.get('scale_b', None)

    period_ns = (10 ** 18) // freq_nhz
    times, values = parse_csv_data(df)
    assert times.size > 0, f"No data in file {filename}."

    measure_id, device_id = get_source_identifiers(sdk, device_tag, measure_tag, freq_nhz, units)
    assert measure_id is not None, f"Measure {measure_tag} {freq_nhz} {units} could not be inserted."
    assert device_id is not None, f"Device {device_tag} could not be inserted."

    sdk.write_data_easy(measure_id, device_id, times, values, freq_nhz, scale_m=scale_m, scale_b=scale_b)

    return measure_id, device_id, int(times[0]), int(times[-1] + period_ns)


def parse_csv_column_info(df):
    device_tag, measure_tag, freq_nhz, units, scale_m, scale_b = None, None, None, None, None, None

    # Extract the column names from the dataframe
    col_names = df.columns

    time_col_name = col_names[0]
    val_col_name = col_names[1]

    device_tag = time_col_name.split(" ")[0]

    val_col_split = val_col_name.split(" ")

    measure_tag = val_col_split[0]
    freq_nhz = int(val_col_split[1])
    units = val_col_split[2]
    scale_m = float(val_col_split[3]) if len(val_col_split) > 3 else None
    scale_b = float(val_col_split[4]) if len(val_col_split) > 4 else None

    return device_tag, measure_tag, freq_nhz, units, scale_m, scale_b


def parse_csv_data(df):
    times = df[df.columns[0]].to_numpy()
    values = df[df.columns[1]].to_numpy()

    return times, values


def get_source_identifiers(sdk, device_tag, measure_tag, freq_nhz, units):
    measure_id = sdk.get_measure_id(measure_tag, freq_nhz, units)
    if measure_id is None:
        measure_id = sdk.insert_measure(measure_tag, freq_nhz, units)

    device_id = sdk.get_device_id(device_tag)
    if device_id is None:
        device_id = sdk.insert_device(device_tag)

    return measure_id, device_id
